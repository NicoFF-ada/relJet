/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Multi-dimensional solver: 2D/3D Cartesian sweeps, symmetry, conservation,
    thread independence, VTK output.
\*---------------------------------------------------------------------------*/

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Solver.H"
#include "TestFramework.H"
#include "VtkOutput.H"

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

Grid cube(int n0, int n1, int n2)
{
    GridSpec s;
    s.n[0] = n0;
    s.n[1] = n1;
    s.n[2] = n2;
    return makeGrid(s);
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

Real total(const ConsField& u, int c)
{
    const Grid& g = u.grid();
    Real s = 0;
    for (int k = g.begin(2); k < g.end(2); ++k)
        for (int j = g.begin(1); j < g.end(1); ++j)
            for (int i = g.begin(0); i < g.end(0); ++i)
                s += u.at(c, g.index(i, j, k));
    return s * g.dx[0] * g.dx[1] * g.dx[2];
}

const Prim kLeftP1{10.0, 40.0 / 3.0, {0, 0, 0}, 1.0};
const Prim kRightP1{1.0, 1e-6, {0, 0, 0}, 0.0};

// Problem 1 with the discontinuity normal to direction `dir`.
PrimField shockTubeAlong(const Grid& g, int dir)
{
    return fill(
        g,
        [dir](Real x, Real y, Real z)
        {
            const Real s = dir == 0 ? x : dir == 1 ? y : z;
            return s < 0.5 ? kLeftP1 : kRightP1;
        });
}

} // namespace

RELJET_TEST(multid_grid_3d_layout)
{
    const Grid g = cube(4, 3, 5);
    CHECK(g.numActiveDims() == 3);
    CHECK(g.stride(0) == 1 && g.stride(1) == 8 && g.stride(2) == 8 * 7);
    CHECK(g.index(1, 2, 3) == (3 * 7 + 2) * 8 + 1);
    CHECK(g.numInteriorCells() == 60);
    const Grid g2 = makeGrid(4, 3, 0.0, 1.0, 0.0, 1.0, 2);
    CHECK(g2.collapsed(2) && g2.numActiveDims() == 2 && g2.padded(2) == 1);
}

RELJET_TEST(multid_uniform_flow_stays_uniform_3d)
{
    const Grid g = cube(8, 8, 8);
    const Prim w0{1.0, 0.5, {0.3, -0.4, 0.5}, 0.2};
    Solver s(g, kEos, mcOptions(BoundaryType::Periodic));
    s.initialize(fill(g, [&](Real, Real, Real) { return w0; }));
    for (int n = 0; n < 5; ++n)
        s.step(s.computeTimeStep());
    Real maxErr = 0;
    for (int idx = 0; idx < g.numCells(); ++idx)
    {
        const Prim w = s.prim().load(idx);
        maxErr = std::fmax(
            maxErr,
            std::fabs(w.rho - w0.rho) + std::fabs(w.p - w0.p) +
                std::fabs(w.v[2] - w0.v[2]));
    }
    CHECK(maxErr < 1e-12);
}

RELJET_TEST(multid_shock_tube_same_along_every_axis)
{
    // Problem 1 along x (1D), along y (2D grid) and along z (3D grid) with the
    // same fixed time steps: the profiles must agree to round-off.
    const int n = 100, steps = 60;
    const Real dt = 0.002;
    const Grid gx = makeGrid(n, 1, 0.0, 1.0, 0.0, 1.0, 2);
    const Grid gy = makeGrid(4, n, 0.0, 1.0, 0.0, 1.0, 2);
    const Grid gz = cube(4, 4, n);
    const Grid* grids[] = {&gx, &gy, &gz};
    Real profile[3][100];
    for (int dir = 0; dir < 3; ++dir)
    {
        const Grid& g = *grids[dir];
        Solver s(g, kEos, mcOptions(BoundaryType::Outflow));
        s.initialize(shockTubeAlong(g, dir));
        for (int k = 0; k < steps; ++k)
            s.step(dt);
        for (int a = 0; a < n; ++a)
        {
            int ijk[3] = {g.begin(0), g.begin(1), g.begin(2)};
            ijk[dir] = g.begin(dir) + a;
            profile[dir][a] =
                s.prim().load(g.index(ijk[0], ijk[1], ijk[2])).rho;
        }
    }
    Real maxDiff = 0;
    for (int a = 0; a < n; ++a)
        maxDiff = std::fmax(
            maxDiff,
            std::fmax(
                std::fabs(profile[1][a] - profile[0][a]),
                std::fabs(profile[2][a] - profile[0][a])));
    CHECK(maxDiff < 1e-12);
    CHECK(
        profile[0][n / 2 + 5] > 2.0 &&
        profile[0][n / 2 + 5] <
            5.0); // x = 0.555: left plateau, the waves have moved
}

RELJET_TEST(multid_conservation_3d_blast)
{
    // Over-pressured sphere in a periodic box: D, S and tau totals are constant.
    const Grid g = cube(16, 16, 16);
    Solver s(g, kEos, mcOptions(BoundaryType::Periodic));
    s.initialize(fill(
        g,
        [](Real x, Real y, Real z)
        {
            const Real r2 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5) +
                            (z - 0.5) * (z - 0.5);
            return r2 < 0.04 ? Prim{1.0, 10.0, {0, 0, 0}, 1.0}
                             : Prim{1.0, 0.1, {0, 0, 0}, 0.0};
        }));
    const Real mass0 = total(s.cons(), iD), tau0 = total(s.cons(), iTau);
    for (int n = 0; n < 15; ++n)
        s.step(s.computeTimeStep());
    CHECK_NEAR(total(s.cons(), iD) / mass0, 1.0, 1e-13);
    CHECK_NEAR(total(s.cons(), iTau) / tau0, 1.0, 1e-13);
    for (int c : {iS0, iS1, iS2})
        CHECK_NEAR(total(s.cons(), c), 0.0, 1e-13);
    CHECK(
        s.prim()
            .load(g.index(g.begin(0) + 8, g.begin(1) + 8, g.begin(2) + 12))
            .p > 0.1); // blast arrived
}

RELJET_TEST(multid_results_independent_of_thread_count)
{
#ifdef _OPENMP
    const Grid g = cube(32, 32, 16); // above the threading threshold
    auto run = [&](int threads)
    {
        omp_set_num_threads(threads);
        Solver s(g, kEos, mcOptions(BoundaryType::Outflow));
        s.initialize(fill(
            g,
            [](Real x, Real y, Real)
            {
                return x + y < 1 ? Prim{1.0, 1.0, {0.1, 0, 0}, 1.0}
                                 : Prim{0.1, 0.01, {0, 0.2, 0}, 0.0};
            }));
        for (int n = 0; n < 3; ++n)
            s.step(s.computeTimeStep());
        return PrimField(s.prim());
    };
    const int maxThreads = omp_get_max_threads();
    const PrimField a = run(1);
    const PrimField b = run(4);
    omp_set_num_threads(maxThreads);
    bool identical = true;
    for (int c = 0; c < kNumPrim; ++c)
        for (int idx = 0; idx < g.numCells(); ++idx)
            identical = identical && a.at(c, idx) == b.at(c, idx);
    CHECK(identical);
#endif
}

RELJET_TEST(multid_vtk_writes_expected_size)
{
    const Grid g = makeGrid(6, 4, 0.0, 1.0, 0.0, 1.0, 2);
    const PrimField w = fill(
        g,
        [](Real, Real, Real) {
            return Prim{1.0, 1.0, {0, 0, 0}, 0};
        });
    const std::string path = "testMultiD_output.vtk";
    writeVtk(path, w, 0.0);
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    CHECK(in.good());
    const long bytes = static_cast<long>(in.tellg());
    CHECK(
        bytes >
        24 * 4 * 7); // 4 scalars + 3-vector of 24 float cells, plus headers
    in.close();
    std::remove(path.c_str());
}

// ************************************************************************* //
