/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Boundary conditions and the 1D solver loop (first-order reconstruction).
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Solver.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{

const IdealGasEos kEos{5.0 / 3.0};

SolverOptions firstOrder(BoundaryType lo, BoundaryType hi)
{
    SolverOptions opt;
    opt.bc.lower[0] = lo;
    opt.bc.upper[0] = hi;
    return opt;
}

// Shock tube with left/right states split at x = 0.5.
PrimField shockTube(const Grid& g, const Prim& left, const Prim& right)
{
    PrimField w(g);
    for (int i = g.begin(0); i < g.end(0); ++i)
        w.store(g.index(i, 0), g.center(0, i) < 0.5 ? left : right);
    return w;
}

Real total(const ConsField& u, int k)
{
    const Grid& g = u.grid();
    Real s = 0;
    for (int i = g.begin(0); i < g.end(0); ++i)
        s += u.at(k, g.index(i, 0)) * g.dx[0];
    return s;
}

const Prim kLeftP1{10.0, 40.0 / 3.0, {0, 0}, 1.0};
const Prim kRightP1{1.0, 1e-6, {0, 0}, 0.0};

} // namespace

RELJET_TEST(boundary_outflow_and_reflecting_ghosts)
{
    const Grid g = makeGrid(4, 1, 0.0, 1.0, 0.0, 1.0, 2);
    PrimField w(g);
    for (int i = g.begin(0); i < g.end(0); ++i)
        w.store(g.index(i, 0), Prim{1.0 + i, 1.0, {0.1 * i, 0.2}, 0});

    applyBoundaries(
        w,
        BoundaryConditions{
            {BoundaryType::Outflow, BoundaryType::Outflow},
            {BoundaryType::Reflecting, BoundaryType::Outflow}});
    // Lower outflow: both ghosts copy the first interior cell (i = 2).
    CHECK(w.load(1).rho == w.load(2).rho && w.load(0).rho == w.load(2).rho);
    CHECK(w.load(0).v[0] == w.load(2).v[0]);
    // Upper reflecting: ghost 6 mirrors 5, ghost 7 mirrors 4; v[0] flips, v[1] does not.
    CHECK(w.load(6).rho == w.load(5).rho && w.load(7).rho == w.load(4).rho);
    CHECK(
        w.load(6).v[0] == -w.load(5).v[0] && w.load(7).v[0] == -w.load(4).v[0]);
    CHECK(w.load(6).v[1] == w.load(5).v[1]);
    CHECK_THROWS(parseBoundaryType("symmetric"));
}

RELJET_TEST(boundary_periodic_wraps_around)
{
    const Grid g = makeGrid(4, 1, 0.0, 1.0, 0.0, 1.0, 2); // interior 2..5
    PrimField w(g);
    for (int i = g.begin(0); i < g.end(0); ++i)
        w.store(g.index(i, 0), Prim{1.0 + i, 1.0, {0.1, 0}, 0});
    const BoundaryConditions bc{
        {BoundaryType::Periodic, BoundaryType::Outflow},
        {BoundaryType::Periodic, BoundaryType::Outflow}};
    applyBoundaries(w, bc);
    CHECK(
        w.load(1).rho == w.load(5).rho &&
        w.load(0).rho == w.load(4).rho); // lower ghosts = last cells
    CHECK(
        w.load(6).rho == w.load(2).rho &&
        w.load(7).rho == w.load(3).rho); // upper ghosts = first cells
    CHECK(w.load(1).v[0] == 0.1);        // no velocity flip
    CHECK_THROWS(validateBoundaries(BoundaryConditions{
        {BoundaryType::Periodic, BoundaryType::Outflow},
        {BoundaryType::Outflow, BoundaryType::Outflow}}));
}

RELJET_TEST(solver_keeps_uniform_flow_uniform)
{
    const Grid g = makeGrid(50, 1, 0.0, 1.0, 0.0, 1.0, 2);
    const Prim w0{1.0, 0.1, {0.8, 0}, 0.5};
    Solver s(g, kEos, firstOrder(BoundaryType::Outflow, BoundaryType::Outflow));
    s.initialize(shockTube(g, w0, w0));
    for (int n = 0; n < 20; ++n)
        s.step(s.computeTimeStep());
    for (int i = g.begin(0); i < g.end(0); ++i)
    {
        const Prim w = s.prim().load(g.index(i, 0));
        CHECK_NEAR(w.rho, w0.rho, 1e-12);
        CHECK_NEAR(w.p, w0.p, 1e-12);
        CHECK_NEAR(w.v[0], w0.v[0], 1e-12);
    }
}

RELJET_TEST(solver_conserves_with_reflecting_walls)
{
    // Closed box: nothing leaves, so D, S1, tau and DC totals stay fixed.
    const Grid g = makeGrid(200, 1, 0.0, 1.0, 0.0, 1.0, 2);
    Solver s(
        g,
        kEos,
        firstOrder(BoundaryType::Reflecting, BoundaryType::Reflecting));
    s.initialize(shockTube(g, kLeftP1, kRightP1));
    const Real mass0 = total(s.cons(), iD), tau0 = total(s.cons(), iTau),
               dc0 = total(s.cons(), iDC);
    Real t = 0;
    while (t < 0.6)
    { // long enough for the shock to hit the right wall
        const Real dt = s.computeTimeStep();
        s.step(dt);
        t += dt;
    }
    CHECK_NEAR(total(s.cons(), iD) / mass0, 1.0, 1e-13);
    CHECK_NEAR(total(s.cons(), iTau) / tau0, 1.0, 1e-13);
    CHECK_NEAR(total(s.cons(), iDC) / dc0, 1.0, 1e-13);
}

RELJET_TEST(solver_problem1_plateau)
{
    // Marti & Mueller (2003) Problem 1, exact plateau p* = 1.448, v* = 0.714.
    // First order on 800 cells is within ~1% (a 6400-cell run gives 1.4480, 0.7140).
    const Grid g = makeGrid(800, 1, 0.0, 1.0, 0.0, 1.0, 2);
    Solver s(g, kEos, firstOrder(BoundaryType::Outflow, BoundaryType::Outflow));
    s.initialize(shockTube(g, kLeftP1, kRightP1));
    Real t = 0;
    while (t < 0.4)
    {
        const Real dt = std::fmin(s.computeTimeStep(), 0.4 - t);
        s.step(dt);
        t += dt;
    }
    const int i =
        g.begin(0) +
        static_cast<int>(0.7 / g.dx[0]); // x = 0.7: inside the plateau
    const Prim w = s.prim().load(g.index(i, 0));
    CHECK_NEAR(w.p, 1.448, 0.015);
    CHECK_NEAR(w.v[0], 0.714, 0.007);
}

RELJET_TEST(solver_rejects_bad_options)
{
    const Grid g = makeGrid(10, 1, 0.0, 1.0, 0.0, 1.0, 2);
    SolverOptions opt;
    opt.cfl = 1.2;
    CHECK_THROWS(Solver(g, kEos, opt));
    opt.cfl = 0.4;
    CHECK_THROWS(Solver(
        makeGrid(10, 1, 0.0, 1.0, 0.0, 1.0, 1),
        kEos,
        [&]
        {
            SolverOptions o = opt;
            o.reconstruction = Reconstruction::Muscl;
            return o;
        }()));
}

// ************************************************************************* //
