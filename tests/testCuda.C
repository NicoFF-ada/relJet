/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    CUDA backend: agreement with the CPU reference solver, conservation, and
    failure reporting. Built only with RELJET_ENABLE_CUDA.

    CPU and GPU evaluate the same expressions (Kernels.H) but the GPU
    compiler contracts a*b + c into fused multiply-adds, so results agree to a
    tolerance, not bitwise.
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "CudaSolver.H"
#include "Jet.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{

const IdealGasEos kEos{5.0 / 3.0};

SolverOptions mcOptions(BoundaryType all)
{
    SolverOptions opt;
    opt.reconstruction = Reconstruction::Muscl;
    opt.limiter = Limiter::MonotonizedCentral;
    for (int d = 0; d < kDims; ++d)
        opt.bc.lower[d] = opt.bc.upper[d] = all;
    return opt;
}

template<class InitFn>
PrimField fill(const Grid& g, InitFn f)
{
    PrimField w(g);
    for (int k = g.begin(2); k < g.end(2); ++k)
        for (int j = g.begin(1); j < g.end(1); ++j)
            for (int i = g.begin(0); i < g.end(0); ++i)
                w.store(
                    g.index(i, j, k),
                    f(g.center(0, i), g.center(1, j), g.center(2, k)));
    return w;
}

// Largest |a - b| over interior cells, per component scaled by max |a| of
// that component (a pointwise relative error is meaningless for velocities
// that are zero up to round-off).
Real maxScaledDiff(const PrimField& a, const PrimField& b)
{
    const Grid& g = a.grid();
    Real worst = 0;
    for (int c = 0; c < kNumPrim; ++c)
    {
        Real scale = 1e-300, diff = 0;
        for (int k = g.begin(2); k < g.end(2); ++k)
            for (int j = g.begin(1); j < g.end(1); ++j)
                for (int i = g.begin(0); i < g.end(0); ++i)
                {
                    const int idx = g.index(i, j, k);
                    scale = std::fmax(scale, std::fabs(a.at(c, idx)));
                    diff =
                        std::fmax(diff, std::fabs(a.at(c, idx) - b.at(c, idx)));
                }
        worst = std::fmax(worst, diff / scale);
    }
    return worst;
}

// Runs CPU and GPU side by side with the CPU time step and returns the
// largest scaled difference of the primitive fields.
Real compareBackends(
    const Grid& g,
    const SolverOptions& opt,
    const PrimField& init,
    int steps,
    Real* dtDiff = nullptr)
{
    Solver cpu(g, kEos, opt);
    CudaSolver gpu(g, kEos, opt);
    cpu.initialize(init);
    gpu.initialize(init);
    Real worstDt = 0;
    for (int n = 0; n < steps; ++n)
    {
        const Real dt = cpu.computeTimeStep();
        worstDt =
            std::fmax(worstDt, std::fabs(gpu.computeTimeStep() - dt) / dt);
        cpu.step(dt);
        gpu.step(dt);
    }
    if (dtDiff)
        *dtDiff = worstDt;
    return maxScaledDiff(cpu.prim(), gpu.prim());
}

const Prim kLeftP1{10.0, 40.0 / 3.0, {0, 0, 0}, 1.0};
const Prim kRightP1{1.0, 1e-6, {0, 0, 0}, 0.0};

} // namespace

RELJET_TEST(cuda_matches_cpu_shock_tube_1d)
{
    const Grid g = makeGrid(400, 1, 0.0, 1.0, 0.0, 1.0, 2);
    const PrimField init = fill(
        g, [](Real x, Real, Real) { return x < 0.5 ? kLeftP1 : kRightP1; });
    Real dtDiff = 0;
    CHECK(
        compareBackends(
            g, mcOptions(BoundaryType::Outflow), init, 200, &dtDiff) < 1e-11);
    CHECK(dtDiff < 1e-12);
}

RELJET_TEST(cuda_matches_cpu_3d_blast)
{
    GridSpec s;
    s.n[0] = s.n[1] = s.n[2] = 24;
    const Grid g = makeGrid(s);
    const PrimField init = fill(
        g,
        [](Real x, Real y, Real z)
        {
            const Real r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) +
                            (z - 0.5) * (z - 0.5);
            return r2 < 0.04 ? Prim{1.0, 10.0, {0, 0, 0}, 1.0}
                             : Prim{1.0, 0.1, {0.1, -0.2, 0.3}, 0.0};
        });
    CHECK(
        compareBackends(g, mcOptions(BoundaryType::Periodic), init, 20) <
        1e-11);
}

RELJET_TEST(cuda_matches_cpu_axisymmetric_jet)
{
    const JetParams jet(7.0, 0.1, 1.0, 1e-2, 1.0, kEos);
    const Grid g = makeGrid(32, 96, 0.0, 4.0, 0.0, 12.0, 2);
    SolverOptions opt = mcOptions(BoundaryType::Outflow);
    opt.geometry = Geometry::Cylindrical;
    opt.bc.lower[0] = BoundaryType::Reflecting;
    opt.bc.lower[1] = BoundaryType::JetInlet;
    opt.bc.jet = jet.nozzle(1);
    PrimField init(g);
    for (int idx = 0; idx < g.numCells(); ++idx)
        init.store(idx, jet.ambientState());
    CHECK(compareBackends(g, opt, init, 300) < 1e-11);
    opt.flux = FluxScheme::Hllc;
    CHECK(compareBackends(g, opt, init, 300) < 1e-11);
}

RELJET_TEST(cuda_conserves_3d)
{
    GridSpec s;
    s.n[0] = s.n[1] = s.n[2] = 16;
    const Grid g = makeGrid(s);
    CudaSolver gpu(g, kEos, mcOptions(BoundaryType::Periodic));
    gpu.initialize(fill(
        g,
        [](Real x, Real y, Real z)
        {
            return x + y + z < 1.5 ? Prim{1.0, 5.0, {0, 0, 0}, 1.0}
                                   : Prim{0.5, 0.1, {0, 0, 0}, 0.0};
        }));
    auto total = [&](int c)
    {
        const ConsField& u = gpu.cons();
        Real sum = 0;
        for (int k = g.begin(2); k < g.end(2); ++k)
            for (int j = g.begin(1); j < g.end(1); ++j)
                for (int i = g.begin(0); i < g.end(0); ++i)
                    sum += u.at(c, g.index(i, j, k));
        return sum;
    };
    const Real mass0 = total(iD), tau0 = total(iTau);
    for (int n = 0; n < 20; ++n)
        gpu.step(gpu.computeTimeStep());
    CHECK_NEAR(total(iD) / mass0, 1.0, 1e-13);
    CHECK_NEAR(total(iTau) / tau0, 1.0, 1e-13);
}

RELJET_TEST(cuda_reports_c2p_failure)
{
    // A huge time step drives the state inadmissible; the GPU must throw the
    // same kind of error as the CPU, not continue with NaNs.
    const Grid g = makeGrid(64, 1, 0.0, 1.0, 0.0, 1.0, 2);
    const PrimField init = fill(
        g, [](Real x, Real, Real) { return x < 0.5 ? kLeftP1 : kRightP1; });
    CudaSolver gpu(g, kEos, mcOptions(BoundaryType::Outflow));
    gpu.initialize(init);
    bool threw = false;
    try
    {
        gpu.step(50 * gpu.computeTimeStep());
    }
    catch (const std::runtime_error& e)
    {
        threw = std::string(e.what()).find("consToPrim failed") !=
                std::string::npos;
    }
    CHECK(threw);
}

// ************************************************************************* //
