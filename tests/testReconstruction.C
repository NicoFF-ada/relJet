/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Limiters and MUSCL reconstruction (Reconstruction.H).
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Reconstruction.H"
#include "Solver.H"
#include "TestFramework.H"

using namespace reljet;

RELJET_TEST(recon_minmod)
{
    CHECK_NEAR(minmod(1.0, 3.0), 1.0, 0.0);
    CHECK_NEAR(minmod(3.0, 1.0), 1.0, 0.0);
    CHECK_NEAR(minmod(-2.0, -0.5), -0.5, 0.0);
    CHECK_NEAR(minmod(1.0, -1.0), 0.0, 0.0); // local extremum
    CHECK_NEAR(minmod(0.0, 2.0), 0.0, 0.0);
}

RELJET_TEST(recon_monotonized_central)
{
    CHECK_NEAR(monotonizedCentral(1.0, 1.0), 1.0, 0.0); // smooth: central slope
    CHECK_NEAR(monotonizedCentral(1.0, 3.0), 2.0, 0.0); // min(2, 6, 2)
    CHECK_NEAR(monotonizedCentral(1.0, 0.2), 0.4, 1e-15); // min(2, 0.4, 0.6)
    CHECK_NEAR(monotonizedCentral(-1.0, -3.0), -2.0, 0.0);
    CHECK_NEAR(monotonizedCentral(1.0, -1.0), 0.0, 0.0);
    CHECK_NEAR(monotonizedCentral(0.0, 2.0), 0.0, 0.0);
}

RELJET_TEST(recon_limited_slope_dispatch)
{
    CHECK_NEAR(limitedSlope(1.0, 3.0, Limiter::Minmod), 1.0, 0.0);
    CHECK_NEAR(limitedSlope(1.0, 3.0, Limiter::MonotonizedCentral), 2.0, 0.0);
}

RELJET_TEST(recon_parse_limiter)
{
    CHECK(parseLimiter("minmod") == Limiter::Minmod);
    CHECK(parseLimiter("MC") == Limiter::MonotonizedCentral);
    CHECK_THROWS(parseLimiter("superbee"));
    CHECK_THROWS(parseLimiter("mc"));
}

RELJET_TEST(recon_muscl_exact_for_linear_data)
{
    // Linear profiles have equal left/right differences: both limiters keep the
    // slope, and the face values sit exactly halfway to the neighbours.
    const Prim wm{1.0, 2.0, {0.1, -0.2}, 0.0};
    const Prim w0{2.0, 3.0, {0.2, -0.1}, 0.5};
    const Prim wp{3.0, 4.0, {0.3, 0.0}, 1.0};
    for (Limiter lim : {Limiter::Minmod, Limiter::MonotonizedCentral})
    {
        const FaceStates f = reconstructMuscl(wm, w0, wp, lim);
        CHECK_NEAR(f.minus.rho, 1.5, 1e-15);
        CHECK_NEAR(f.plus.rho, 2.5, 1e-15);
        CHECK_NEAR(f.minus.p, 2.5, 1e-15);
        CHECK_NEAR(f.plus.p, 3.5, 1e-15);
        CHECK_NEAR(f.minus.v[0], 0.15, 1e-15);
        CHECK_NEAR(f.plus.v[0], 0.25, 1e-15);
        CHECK_NEAR(f.minus.v[1], -0.15, 1e-15);
        CHECK_NEAR(f.plus.v[1], -0.05, 1e-15);
        CHECK_NEAR(f.minus.C, 0.25, 1e-15);
        CHECK_NEAR(f.plus.C, 0.75, 1e-15);
    }
}

RELJET_TEST(recon_muscl_flat_at_extremum)
{
    // w0 is a local maximum in every component: no slope, faces = cell average.
    const Prim wm{1.0, 1.0, {0.0, 0.0}, 0.0};
    const Prim w0{2.0, 2.0, {0.5, 0.5}, 1.0};
    const FaceStates f =
        reconstructMuscl(wm, w0, wm, Limiter::MonotonizedCentral);
    CHECK(f.minus.rho == 2.0 && f.plus.rho == 2.0);
    CHECK(f.minus.p == 2.0 && f.plus.p == 2.0);
    CHECK(f.minus.v[0] == 0.5 && f.plus.v[1] == 0.5);
    CHECK(f.minus.C == 1.0 && f.plus.C == 1.0);
}

RELJET_TEST(recon_muscl_problem1_beats_first_order)
{
    // End-to-end: on 400 cells MUSCL resolves the thin dense shell behind the
    // shock much better (exact peak 5.07; first order reaches only ~4.2).
    const IdealGasEos eos{5.0 / 3.0};
    const Grid g = makeGrid(400, 1, 0.0, 1.0, 0.0, 1.0, 2);
    PrimField init(g);
    for (int i = g.begin(0); i < g.end(0); ++i)
        init.store(
            g.index(i, 0),
            g.center(0, i) < 0.5 ? Prim{10.0, 40.0 / 3.0, {0, 0}, 1}
                                 : Prim{1.0, 1e-6, {0, 0}, 0});
    SolverOptions opt;
    opt.reconstruction = Reconstruction::Muscl;
    opt.limiter = Limiter::MonotonizedCentral;
    Solver s(g, eos, opt);
    s.initialize(init);
    Real t = 0;
    while (t < 0.4)
    {
        const Real dt = std::fmin(s.computeTimeStep(), 0.4 - t);
        s.step(dt);
        t += dt;
    }
    Real shellPeak = 0;
    for (int i = g.begin(0); i < g.end(0); ++i)
        if (g.center(0, i) > 0.7)
            shellPeak = std::fmax(shellPeak, s.prim().load(g.index(i, 0)).rho);
    CHECK(shellPeak > 4.6);
    CHECK(shellPeak < 5.3); // no large overshoot
}

// ************************************************************************* //
