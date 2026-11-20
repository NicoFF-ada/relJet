/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Axisymmetric source terms, jet inlet, JetParams, and an end-to-end
    axisymmetric jet run checked against the analytic head speed.
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Geometry.H"
#include "Jet.H"
#include "Solver.H"
#include "Srhd.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{
const IdealGasEos kEos{5.0 / 3.0};
} // namespace

// ------------------------------------------------------- source terms --

RELJET_TEST(jet_cylindrical_source_at_rest_is_pressure_over_r)
{
    const Prim w{1.0, 0.3, {0, 0, 0}, 0};
    const Cons s = cylindricalSource(w, primToCons(w, kEos), 2.0);
    CHECK_NEAR(s.S[0], 0.15, 1e-15);
    CHECK_NEAR(s.D, 0.0, 0.0);
    CHECK_NEAR(s.S[1], 0.0, 0.0);
    CHECK_NEAR(s.S[2], 0.0, 0.0);
    CHECK_NEAR(s.tau, 0.0, 0.0);
    CHECK_NEAR(s.DC, 0.0, 0.0);
}

RELJET_TEST(jet_cylindrical_source_with_swirl)
{
    const Prim w{0.5, 0.2, {0.3, -0.1, 0.4}, 0.5};
    const Cons u = primToCons(w, kEos);
    const Real r = 0.7;
    const Real rhohW2 = u.tau + u.D + w.p;
    const Cons s = cylindricalSource(w, u, r);
    CHECK_NEAR(s.S[0], (rhohW2 * w.v[2] * w.v[2] + w.p) / r, 1e-14);
    CHECK_NEAR(s.S[2], -rhohW2 * w.v[0] * w.v[2] / r, 1e-14);
    CHECK_NEAR(s.S[1], 0.0, 0.0);
}

RELJET_TEST(jet_cylindrical_static_state_is_well_balanced)
{
    // Uniform pressure at rest must stay exactly at rest in (r, z).
    const Grid g = makeGrid(32, 16, 0.0, 1.0, 0.0, 1.0, 2);
    SolverOptions opt;
    opt.geometry = Geometry::Cylindrical;
    opt.reconstruction = Reconstruction::Muscl;
    opt.limiter = Limiter::MonotonizedCentral;
    opt.bc.lower[0] = BoundaryType::Reflecting;
    PrimField init(g);
    for (int idx = 0; idx < g.numCells(); ++idx)
        init.store(idx, Prim{1.0, 0.7, {0, 0, 0}, 0});
    Solver s(g, kEos, opt);
    s.initialize(init);
    for (int n = 0; n < 20; ++n)
        s.step(s.computeTimeStep());
    Real maxV = 0, maxDp = 0;
    for (int j = g.begin(1); j < g.end(1); ++j)
        for (int i = g.begin(0); i < g.end(0); ++i)
        {
            const Prim w = s.prim().load(g.index(i, j));
            maxV = std::fmax(maxV, std::fabs(w.v[0]) + std::fabs(w.v[1]));
            maxDp = std::fmax(maxDp, std::fabs(w.p - 0.7));
        }
    CHECK(maxV < 1e-13);
    CHECK(maxDp < 1e-13);
}

RELJET_TEST(jet_cylindrical_requires_axis_and_2d)
{
    SolverOptions opt;
    opt.geometry = Geometry::Cylindrical;
    CHECK_THROWS(Solver(
        makeGrid(8, 8, 0.0, 1.0, 0.0, 1.0, 2),
        kEos,
        opt)); // axis not reflecting
}

// ----------------------------------------------------------- jet inlet --

RELJET_TEST(jet_inlet_ghost_inside_and_outside_nozzle)
{
    const JetNozzle nozzle{Prim{0.01, 0.001, {0, 0.99, 0}, 1.0}, 1.0};
    const Prim mirror{1.0, 0.001, {0.1, 0.2, 0.05}, 0.0};
    const Prim in = jetInletGhost(mirror, 0.5, nozzle, 1);
    CHECK(
        in.rho == 0.01 && in.p == 0.001 && in.v[1] == 0.99 && in.C == 1.0 &&
        in.v[0] == 0);
    const Prim out = jetInletGhost(mirror, 1.5, nozzle, 1);
    CHECK(out.rho == 1.0 && out.p == 0.001 && out.C == 0.0);
    CHECK(
        out.v[0] == 0.1 && out.v[1] == -0.2 &&
        out.v[2] == 0.05); // only the axial component flips
}

// ----------------------------------------------------------- JetParams --

RELJET_TEST(jet_params_validates)
{
    CHECK_THROWS(JetParams(1.0, 0.01, 1.0, 1e-3, 1.0, kEos)); // W must exceed 1
    CHECK_THROWS(JetParams(7.0, 0.0, 1.0, 1e-3, 1.0, kEos));  // eta
    CHECK_THROWS(JetParams(7.0, 0.01, -1.0, 1e-3, 1.0, kEos)); // rho_a
    CHECK_THROWS(JetParams(7.0, 0.01, 1.0, 0.0, 1.0, kEos));   // p
    CHECK_THROWS(JetParams(7.0, 0.01, 1.0, 1e-3, 0.0, kEos));  // radius
    CHECK_THROWS(JetParams(7.0, 0.01, 1.0, 1e-3, 1.0, IdealGasEos{2.5}));
}

RELJET_TEST(jet_params_derived_states)
{
    const JetParams jet(7.0, 0.01, 2.0, 1e-3, 1.5, kEos);
    CHECK_NEAR(jet.lorentzFactor(), 7.0, 0.0);
    CHECK_NEAR(jet.radius(), 1.5, 0.0);
    CHECK_NEAR(jet.beamVelocity(), std::sqrt(1 - 1 / 49.0), 1e-15);
    const Prim b = jet.beamState(2);
    CHECK_NEAR(b.rho, 0.02, 1e-15);
    CHECK_NEAR(b.p, 1e-3, 0.0);
    CHECK(b.v[0] == 0 && b.v[1] == 0 && b.C == 1.0);
    CHECK_NEAR(b.v[2], jet.beamVelocity(), 0.0);
    const Prim a = jet.ambientState();
    CHECK(
        a.rho == 2.0 && a.p == 1e-3 && a.v[0] == 0 && a.v[1] == 0 &&
        a.v[2] == 0 && a.C == 0.0);
    const JetNozzle nz = jet.nozzle(1);
    CHECK(
        nz.radius == 1.5 && nz.beam.v[1] == jet.beamVelocity() &&
        nz.beam.v[2] == 0);
}

RELJET_TEST(jet_params_overpressured_beam)
{
    const JetParams jet(3.0, 0.03, 1.0, 0.01, 1.0, kEos, 2.0);
    CHECK_NEAR(jet.pressureRatio(), 2.0, 0.0);
    CHECK_NEAR(jet.beamState(1).p, 0.02, 1e-15);
    CHECK_NEAR(jet.ambientState().p, 0.01, 0.0);
    // The default is a pressure-matched jet.
    CHECK_NEAR(
        JetParams(3.0, 0.03, 1.0, 0.01, 1.0, kEos).beamState(1).p, 0.01, 0.0);
    CHECK_THROWS(JetParams(3.0, 0.03, 1.0, 0.01, 1.0, kEos, 0.0));
    // A hotter beam carries more inertia, so the head moves faster.
    CHECK(
        jet.headSpeedEstimate() >
        JetParams(3.0, 0.03, 1.0, 0.01, 1.0, kEos).headSpeedEstimate());
}

RELJET_TEST(jet_params_head_speed)
{
    const JetParams jet(7.0, 0.01, 1.0, 1e-3, 1.0, kEos);
    const Real hb = kEos.specificEnthalpy(0.01, 1e-3),
               ha = kEos.specificEnthalpy(1.0, 1e-3);
    const Real etaR = 0.01 * hb * 49.0 / ha;
    CHECK_NEAR(
        jet.headSpeedEstimate(),
        jet.beamVelocity() * std::sqrt(etaR) / (1 + std::sqrt(etaR)),
        1e-14);
    // A very light, cold jet is much slower than its beam; a heavy jet keeps up.
    CHECK(JetParams(7.0, 1e-4, 1.0, 1e-6, 1.0, kEos).headSpeedEstimate() < 0.1);
    CHECK(JetParams(7.0, 10.0, 1.0, 1e-3, 1.0, kEos).headSpeedEstimate() > 0.9);
    // Same light jet but hot (p / rho_b = 10, h_b = 26): enthalpy is inertia, so it
    // pushes through the ambient medium far faster than its rest mass suggests.
    CHECK(
        JetParams(7.0, 1e-4, 1.0, 1e-3, 1.0, kEos).headSpeedEstimate() > 0.25);
}

// ------------------------------------------------------------ end to end --

RELJET_TEST(jet_axisymmetric_head_advances_at_estimated_speed)
{
    // Coarse axisymmetric jet (4 cells per radius). After the start-up phase the
    // head moves at roughly the 1D momentum-balance speed.
    const JetParams jet(7.0, 0.1, 1.0, 1e-2, 1.0, kEos);
    const Grid g = makeGrid(16, 64, 0.0, 4.0, 0.0, 16.0, 2);
    SolverOptions opt;
    opt.geometry = Geometry::Cylindrical;
    opt.reconstruction = Reconstruction::Muscl;
    opt.limiter = Limiter::Minmod;
    opt.bc.lower[0] = BoundaryType::Reflecting;
    opt.bc.lower[1] = BoundaryType::JetInlet;
    opt.bc.jet = jet.nozzle(1);
    PrimField init(g);
    for (int idx = 0; idx < g.numCells(); ++idx)
        init.store(idx, jet.ambientState());
    Solver s(g, kEos, opt);
    s.initialize(init);

    auto head = [&]
    {
        Real z = 0;
        for (int j = g.begin(1); j < g.end(1); ++j)
            if (s.prim().load(g.index(g.begin(0), j)).C > 0.5)
                z = g.center(1, j) + g.dx[1] / 2;
        return z;
    };
    auto advanceTo = [&](Real tEnd, Real t)
    {
        while (t < tEnd)
        {
            const Real dt = std::fmin(s.computeTimeStep(), tEnd - t);
            s.step(dt);
            t += dt;
        }
        return t;
    };
    Real t = advanceTo(6.0, 0.0);
    const Real z1 = head();
    t = advanceTo(18.0, t);
    const Real z2 = head();
    const Real vSim = (z2 - z1) / 12.0;
    CHECK(vSim > 0.75 * jet.headSpeedEstimate());
    CHECK(vSim < 1.25 * jet.headSpeedEstimate());
}

// ************************************************************************* //
