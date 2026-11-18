/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Application
    relJet

Description
    Finite-volume solver for ideal special-relativistic hydrodynamics,
    run from an OpenFOAM-style case directory:

        system/controlDict                  problem, backend, time control
        system/fvSchemes                    reconstruction and limiter
        system/meshDict                     mesh and boundaries
        system/setFieldsDict                initial states (riemann)
        constant/thermophysicalProperties   gamma
        constant/jetProperties              beam and ambient medium (jet)

    Problems
        riemann   1D two-state shock tube; writes postProcessing/sample
        jet       relativistic jet, 2D axisymmetric or 3D Cartesian; writes
                  VTK/ snapshots and postProcessing/jetHead

    Backends
        cpu       OpenMP (threads from OMP_NUM_THREADS)
        cuda      GPU, when built with RELJET_ENABLE_CUDA

Usage
    relJet [-case dir] [-backend cpu|cuda]

\*---------------------------------------------------------------------------*/

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "CsvOutput.H"
#include "Dictionary.H"
#include "Jet.H"
#include "Solver.H"
#include "VtkOutput.H"

#ifdef RELJET_HAVE_CUDA
#include "CudaSolver.H"
#endif

using namespace reljet;

namespace
{

// * * * * * * * * * * * * * * * * * Case setup  * * * * * * * * * * * * * * //

struct Setup
{
    Grid grid;
    SolverOptions opt;
    PrimField init;
    int axialDir; // jet axis, -1 for the Riemann problem
};


void makeDirectories(const std::string& path)
{
    for (std::size_t pos = path.find('/', 1);; pos = path.find('/', pos + 1))
    {
        mkdir(path.substr(0, pos).c_str(), 0755);
        if (pos == std::string::npos)
        {
            break;
        }
    }
}


std::string caseName(const std::string& caseDir)
{
    std::string dir = caseDir;
    while (dir.size() > 1 && dir.back() == '/')
    {
        dir.pop_back();
    }
    if (dir == ".")
    {
        char buf[4096];
        if (getcwd(buf, sizeof(buf)))
        {
            dir = buf;
        }
    }
    const std::size_t slash = dir.find_last_of('/');
    return slash == std::string::npos ? dir : dir.substr(slash + 1);
}


Dictionary readCase(const std::string& caseDir)
{
    Dictionary dict(caseDir);
    const char* required[] = {
        "system/controlDict",
        "system/fvSchemes",
        "system/meshDict",
        "constant/thermophysicalProperties"};
    const char* optional[] = {"system/setFieldsDict", "constant/jetProperties"};
    for (const char* file : required)
    {
        dict.merge(Dictionary::read(caseDir + "/" + file));
    }
    for (const char* file : optional)
    {
        std::ifstream probe(caseDir + "/" + file);
        if (probe)
        {
            dict.merge(Dictionary::read(caseDir + "/" + file));
        }
    }
    return dict;
}


SolverOptions readSchemes(const Dictionary& dict)
{
    SolverOptions opt;
    opt.cfl = dict.getInRange("maxCo", 1e-6, 0.999);
    opt.reconstruction =
        parseReconstruction(dict.get<std::string>("reconstruction"));
    if (opt.reconstruction == Reconstruction::Muscl)
    {
        opt.limiter = parseLimiter(dict.get<std::string>("limiter"));
    }
    const std::string flux = dict.getOrDefault<std::string>("flux", "HLL");
    if (flux != "HLL")
    {
        throw std::invalid_argument(
            "unknown flux '" + flux + "' (expected HLL)");
    }
    return opt;
}


Prim readState(const Dictionary& dict)
{
    Prim w{};
    w.rho = dict.get<Real>("rho");
    w.p = dict.get<Real>("p");
    w.v[0] = dict.getInRange("v", -1.0, 1.0);
    w.C = dict.getOrDefault<Real>("C", 0.0);
    if (!isAdmissible(w))
    {
        throw std::runtime_error(
            dict.name() + ": need rho > 0, p > 0 and |v| < 1");
    }
    return w;
}


Setup setupRiemann(const Dictionary& dict)
{
    const Grid grid = makeGrid(
        dict.get<int>("nCells"),
        1,
        dict.get<Real>("xMin"),
        dict.get<Real>("xMax"),
        0.0,
        1.0,
        2);
    SolverOptions opt = readSchemes(dict);
    opt.bc.lower[0] = parseBoundaryType(dict.get<std::string>("boundary.xMin"));
    opt.bc.upper[0] = parseBoundaryType(dict.get<std::string>("boundary.xMax"));

    const Real x0 = dict.get<Real>("x0");
    const Prim left = readState(dict.subDict("left"));
    const Prim right = readState(dict.subDict("right"));

    PrimField init(grid);
    for (int i = grid.begin(0); i < grid.end(0); ++i)
    {
        init.store(grid.index(i, 0), grid.center(0, i) < x0 ? left : right);
    }
    return Setup{grid, opt, std::move(init), -1};
}


// Lengths in jet radii. Axisymmetric: r in [0, rMax], z in [0, zMax].
// Cartesian: x, y in [-rMax, rMax], z in [0, zMax].
Setup setupJet(const Dictionary& dict, const JetParams& jet)
{
    const std::string geometry = dict.get<std::string>("geometry");
    const int cellsPerRadius = dict.get<int>("cellsPerRadius");
    const Real rMax = dict.get<Real>("rMax") * jet.radius();
    const Real zMax = dict.get<Real>("zMax") * jet.radius();
    const int nR = int(std::lround(rMax / jet.radius() * cellsPerRadius));
    const int nZ = int(std::lround(zMax / jet.radius() * cellsPerRadius));

    GridSpec spec;
    SolverOptions opt = readSchemes(dict);
    int axialDir = 0;
    if (geometry == "axisymmetric")
    {
        axialDir = 1;
        spec.n[0] = nR;
        spec.n[1] = nZ;
        spec.hi[0] = rMax;
        spec.hi[1] = zMax;
        opt.geometry = Geometry::Cylindrical;
        opt.bc.lower[0] = BoundaryType::Reflecting; // axis
        opt.bc.upper[0] = BoundaryType::Outflow;
    }
    else if (geometry == "cartesian")
    {
        axialDir = 2;
        for (int d = 0; d < 2; ++d)
        {
            spec.n[d] = 2 * nR;
            spec.lo[d] = -rMax;
            spec.hi[d] = rMax;
            opt.bc.lower[d] = opt.bc.upper[d] = BoundaryType::Outflow;
        }
        spec.n[2] = nZ;
        spec.hi[2] = zMax;
    }
    else
    {
        throw std::invalid_argument(
            "geometry must be axisymmetric | cartesian, got '" + geometry +
            "'");
    }
    opt.bc.lower[axialDir] = BoundaryType::JetInlet;
    opt.bc.upper[axialDir] = BoundaryType::Outflow;
    opt.bc.jet = jet.nozzle(axialDir);

    const Grid grid = makeGrid(spec);
    PrimField init(grid);
    const Prim ambient = jet.ambientState();
    for (int idx = 0; idx < grid.numCells(); ++idx)
    {
        init.store(idx, ambient);
    }
    return Setup{grid, opt, std::move(init), axialDir};
}


// Furthest axial position where the tracer exceeds 1/2, on the cell column
// closest to the jet axis.
Real headPosition(const PrimField& w, int axialDir)
{
    const Grid& g = w.grid();
    int ijk[kDims] = {g.begin(0), g.begin(1), g.begin(2)};
    for (int d = 0; d < kDims; ++d)
    {
        if (d != axialDir && !g.collapsed(d) && g.lo[d] < 0)
        {
            ijk[d] = g.begin(d) + g.n[d] / 2; // first cell with centre > 0
        }
    }
    Real head = g.lo[axialDir];
    for (int a = g.begin(axialDir); a < g.end(axialDir); ++a)
    {
        ijk[axialDir] = a;
        if (w.load(g.index(ijk[0], ijk[1], ijk[2])).C > 0.5)
        {
            head = g.center(axialDir, a) + g.dx[axialDir] / 2;
        }
    }
    return head;
}


// * * * * * * * * * * * * * * * * * Time loop * * * * * * * * * * * * * * * //

struct RunControl
{
    std::string caseDir;
    std::string name;
    Real endTime;
    Real writeInterval;
    Real sampleInterval; // jet-head sampling
};


std::string timeName(Real t)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", t);
    return buf;
}


// SolverType is Solver or CudaSolver: same interface, no virtual calls.
template<class SolverType>
void runTimeLoop(SolverType& solver, const Setup& setup, const RunControl& ctl)
{
    const Grid& g = solver.grid();
    const bool jet = setup.axialDir >= 0;

    std::ofstream headLog;
    if (jet)
    {
        makeDirectories(ctl.caseDir + "/postProcessing/jetHead/0");
        makeDirectories(ctl.caseDir + "/VTK");
        headLog.open(ctl.caseDir + "/postProcessing/jetHead/0/jetHead.dat");
        headLog << "# Time    zHead\n";
    }

    int writeIndex = 0;
    auto write = [&](Real t)
    {
        std::string path;
        if (jet)
        {
            path =
                "VTK/" + ctl.name + "_" + std::to_string(writeIndex) + ".vtk";
            writeVtk(ctl.caseDir + "/" + path, solver.prim(), t);
        }
        else
        {
            const std::string dir = "postProcessing/sample/" + timeName(t);
            makeDirectories(ctl.caseDir + "/" + dir);
            path = dir + "/line.csv";
            writeCsv1D(
                ctl.caseDir + "/" + path, solver.prim(), solver.cons(), t);
        }
        ++writeIndex;
        std::printf("    writing %s\n", path.c_str());
    };

    std::printf("Starting time loop\n\n");
    const auto start = std::chrono::steady_clock::now();
    Real t = 0;
    Real nextWrite = ctl.writeInterval;
    Real nextSample = 0;
    long steps = 0;
    if (jet)
    {
        write(t);
    }

    while (t < ctl.endTime * (1 - 1e-12))
    {
        const Real dt = std::min(
            {solver.computeTimeStep(), ctl.endTime - t, nextWrite - t});
        solver.step(dt);
        t += dt;
        ++steps;

        if (jet && t >= nextSample)
        {
            headLog << t << "    "
                    << headPosition(solver.prim(), setup.axialDir) << '\n';
            nextSample += ctl.sampleInterval;
        }

        if (t >= nextWrite * (1 - 1e-12) || t >= ctl.endTime * (1 - 1e-12))
        {
            const double wall = std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() - start)
                                    .count();
            const StepDiagnostics& diag = solver.diagnostics();
            std::printf(
                "Time = %s    (step %ld, deltaT = %.4g)\n",
                timeName(t).c_str(),
                steps,
                dt);
            std::printf(
                "    consToPrim: max iterations %d, reconstruction fallbacks"
                " %ld\n",
                diag.maxC2PIterations,
                diag.reconstructionFallbacks);
            std::printf(
                "    ExecutionTime = %.1f s, %.3g cell updates/s\n",
                wall,
                double(g.numInteriorCells()) * steps / wall);
            write(t);
            headLog.flush();
            std::printf("\n");
            std::fflush(stdout);
            nextWrite += ctl.writeInterval;
        }
    }
    std::printf("End\n");
}


int run(const std::string& caseDir, std::string backend)
{
    const Dictionary dict = readCase(caseDir);
    const IdealGasEos eos{dict.get<Real>("gamma")};
    const std::string problem = dict.get<std::string>("problem");
    if (backend.empty())
    {
        backend = dict.getOrDefault<std::string>("backend", "cpu");
    }

    RunControl ctl;
    ctl.caseDir = caseDir;
    ctl.name = caseName(caseDir);
    ctl.endTime = dict.get<Real>("endTime");
    ctl.writeInterval = dict.getOrDefault<Real>("writeInterval", ctl.endTime);

    std::printf(
        "/*--------------------------------------------------------------------"
        "-------*\\\n"
        "  RelJet | Special-relativistic hydrodynamics for astrophysical jets\n"
        "\\*-------------------------------------------------------------------"
        "--------*/\n"
        "Case       : %s\n",
        ctl.name.c_str());

    Setup setup = [&]()
    {
        if (problem == "riemann")
        {
            return setupRiemann(dict);
        }
        if (problem == "jet")
        {
            const JetParams jet(
                dict.get<Real>("lorentzFactor"),
                dict.get<Real>("densityContrast"),
                dict.get<Real>("ambientDensity"),
                dict.get<Real>("pressure"),
                dict.getOrDefault<Real>("radius", 1.0),
                eos);
            std::printf(
                "Jet        : W = %g (v = %.5f), eta = %g, p = %g, analytic "
                "head"
                " speed %.4f\n",
                jet.lorentzFactor(),
                jet.beamVelocity(),
                jet.densityContrast(),
                jet.pressure(),
                jet.headSpeedEstimate());
            return setupJet(dict, jet);
        }
        throw std::invalid_argument(
            "unknown problem '" + problem + "' (expected riemann | jet)");
    }();
    ctl.sampleInterval =
        dict.getOrDefault<Real>("sampleInterval", ctl.writeInterval / 20);

    for (const std::string& key : dict.unusedKeys())
    {
        std::fprintf(stderr, "Warning: unused entry %s\n", key.c_str());
    }

    const Grid& g = setup.grid;
    std::printf(
        "Mesh       : %d x %d x %d cells%s\n",
        g.n[0],
        g.n[1],
        g.n[2],
        setup.opt.geometry == Geometry::Cylindrical ? " (axisymmetric r, z)"
                                                    : "");

    if (backend == "cpu")
    {
        int threads = 1;
#ifdef _OPENMP
        threads = omp_get_max_threads();
#endif
        std::printf("Backend    : cpu, %d OpenMP thread(s)\n\n", threads);
        Solver solver(setup.grid, eos, setup.opt);
        solver.initialize(setup.init);
        runTimeLoop(solver, setup, ctl);
    }
    else if (backend == "cuda")
    {
#ifdef RELJET_HAVE_CUDA
        std::printf(
            "Backend    : cuda, %s\n\n", CudaSolver::deviceName().c_str());
        CudaSolver solver(setup.grid, eos, setup.opt);
        solver.initialize(setup.init);
        runTimeLoop(solver, setup, ctl);
#else
        throw std::runtime_error(
            "relJet was built without CUDA; configure with "
            "-DRELJET_ENABLE_CUDA=ON");
#endif
    }
    else
    {
        throw std::invalid_argument(
            "unknown backend '" + backend + "' (expected cpu | cuda)");
    }
    return 0;
}

} // namespace


// * * * * * * * * * * * * * * * * * * Main  * * * * * * * * * * * * * * * * //

int main(int argc, char** argv)
{
    std::string caseDir = ".";
    std::string backend;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "-case" && i + 1 < argc)
        {
            caseDir = argv[++i];
        }
        else if (arg == "-backend" && i + 1 < argc)
        {
            backend = argv[++i];
        }
        else
        {
            std::fprintf(
                stderr, "Usage: %s [-case dir] [-backend cpu|cuda]\n", argv[0]);
            return 2;
        }
    }

    try
    {
        return run(caseDir, backend);
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "\n--> RelJet FATAL ERROR: %s\n", e.what());
        return 1;
    }
}

// ************************************************************************* //
