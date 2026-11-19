/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    HLLC Riemann solver (Hllc.H).
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Kernels.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{

const IdealGasEos kEos{5.0 / 3.0};

// Largest |a - b| over all components; infinite if anything is NaN
// (std::fmax would silently skip a NaN).
Real maxDiff(const Cons& a, const Cons& b)
{
    const Real d[] = {
        a.D - b.D,
        a.S[0] - b.S[0],
        a.S[1] - b.S[1],
        a.S[2] - b.S[2],
        a.tau - b.tau,
        a.DC - b.DC};
    Real m = 0;
    for (Real x : d)
    {
        if (!(std::fabs(x) <= m))
            m = std::isnan(x) ? INFINITY : std::fabs(x);
    }
    return m;
}

// Left / right states of a generic Riemann problem along x, with transverse
// velocity so that every component is non-trivial.
const Prim kWL{1.0, 1.0, {0.3, 0.2, -0.1}, 1.0};
const Prim kWR{0.2, 0.95, {-0.2, 0.1, 0.0}, 0.0};

} // namespace


// ----------------------------------------------------------- HLL averages --

RELJET_TEST(hllc_averages_flux_is_the_hll_flux)
{
    const Cons uL = primToCons(kWL, kEos), uR = primToCons(kWR, kEos);
    const Flux fL = physicalFlux(kWL, uL, 0), fR = physicalFlux(kWR, uR, 0);
    const SignalSpeeds sL = signalSpeeds(kWL, kEos, 0);
    const SignalSpeeds sR = signalSpeeds(kWR, kEos, 0);
    const Real lamL = std::fmin(Real(0), std::fmin(sL.minus, sR.minus));
    const Real lamR = std::fmax(Real(0), std::fmax(sL.plus, sR.plus));
    const HllAverages h = hllAverages(uL, fL, uR, fR, lamL, lamR);
    CHECK(maxDiff(h.F, hllFlux(kWL, uL, kWR, uR, kEos, 0)) < 1e-14);
}

RELJET_TEST(hllc_averages_of_equal_states)
{
    const Cons u = primToCons(kWL, kEos);
    const Flux f = physicalFlux(kWL, u, 0);
    const HllAverages h = hllAverages(u, f, u, f, -0.7, 0.9);
    CHECK(maxDiff(h.U, u) < 1e-14);
    CHECK(maxDiff(h.F, f) < 1e-14);
}

// ---------------------------------------------------------------- contact --

RELJET_TEST(hllc_contact_at_rest)
{
    // Pressure equilibrium, density jump, no motion: a contact that must
    // stay put. lambda* = 0 and p* = p.
    const Prim wL{1.0, 1.0, {0, 0, 0}, 1.0}, wR{0.1, 1.0, {0, 0, 0}, 0.0};
    const Cons uL = primToCons(wL, kEos), uR = primToCons(wR, kEos);
    const HllAverages h = hllAverages(
        uL, physicalFlux(wL, uL, 0), uR, physicalFlux(wR, uR, 0), -0.8, 0.9);
    const Real lamStar = contactSpeed(h, 0);
    CHECK_NEAR(lamStar, 0.0, 1e-14);
    CHECK_NEAR(contactPressure(h, lamStar, 0), 1.0, 1e-13);
}

RELJET_TEST(hllc_moving_contact)
{
    const Prim wL{1.0, 0.5, {0.6, 0, 0}, 1.0}, wR{0.05, 0.5, {0.6, 0, 0}, 0.0};
    const Cons uL = primToCons(wL, kEos), uR = primToCons(wR, kEos);
    const HllAverages h = hllAverages(
        uL, physicalFlux(wL, uL, 0), uR, physicalFlux(wR, uR, 0), -0.3, 0.98);
    const Real lamStar = contactSpeed(h, 0);
    CHECK_NEAR(lamStar, 0.6, 1e-13);
    CHECK_NEAR(contactPressure(h, lamStar, 0), 0.5, 1e-13);
}

RELJET_TEST(hllc_contact_speed_without_energy_flux)
{
    // F^E = 0 exactly (fluid at rest everywhere): the textbook root formula
    // divides by zero, the stable one must not.
    HllAverages h{};
    h.U.D = 1.0;
    h.U.tau = 1.5;
    h.F.S[0] = 1.0; // pressure
    CHECK_NEAR(contactSpeed(h, 0), 0.0, 0.0);
}

// ------------------------------------------------------------ star states --

RELJET_TEST(hllc_star_state_of_a_pure_contact_is_unchanged)
{
    // If the contact moves with the fluid (lambda* = v) at the fluid's own
    // pressure (p* = p), the star state IS the state.
    const Prim w = kWL;
    const Cons u = primToCons(w, kEos);
    const Cons s = starState(w, u, -0.9, w.v[0], w.p, 0);
    CHECK(maxDiff(s, u) < 1e-14);
}

RELJET_TEST(hllc_star_state_satisfies_jump_condition)
{
    // Across the left wave: lamL (U* - U) = F* - F with F* the star flux.
    // For D: F*(D) = D* lambda*, so lamL (D* - D) = D* lambda* - D v.
    const Prim w = kWL;
    const Cons u = primToCons(w, kEos);
    const Real lam = -0.8, lamStar = 0.1, pStar = 1.7;
    const Cons s = starState(w, u, lam, lamStar, pStar, 0);
    CHECK_NEAR(lam * (s.D - u.D), s.D * lamStar - u.D * w.v[0], 1e-14);
    CHECK_NEAR(
        lam * (s.S[0] - u.S[0]),
        s.S[0] * lamStar + pStar - (u.S[0] * w.v[0] + w.p),
        1e-13);
    CHECK_NEAR(
        lam * (s.S[1] - u.S[1]), s.S[1] * lamStar - u.S[1] * w.v[0], 1e-14);
    // F(tau) = (tau + p) v, in the star region (tau* + p*) lambda*
    CHECK_NEAR(
        lam * (s.tau - u.tau),
        (s.tau + pStar) * lamStar - (u.tau + w.p) * w.v[0],
        1e-13);
}

// ------------------------------------------------------------------- flux --

RELJET_TEST(hllc_consistency)
{
    const Cons u = primToCons(kWL, kEos);
    CHECK(
        maxDiff(hllcFlux(kWL, u, kWL, u, kEos, 0), physicalFlux(kWL, u, 0)) <
        1e-14);
}

RELJET_TEST(hllc_resolves_a_contact_at_rest_exactly)
{
    // The defining property: no mass flows through a contact at rest. HLL
    // leaks mass through it (numerical diffusion), HLLC does not.
    const Prim wL{1.0, 1.0, {0, 0, 0}, 1.0}, wR{0.1, 1.0, {0, 0, 0}, 0.0};
    const Cons uL = primToCons(wL, kEos), uR = primToCons(wR, kEos);
    const Flux hllc = hllcFlux(wL, uL, wR, uR, kEos, 0);
    const Flux hll = hllFlux(wL, uL, wR, uR, kEos, 0);
    CHECK_NEAR(hllc.D, 0.0, 1e-14);
    CHECK_NEAR(hllc.DC, 0.0, 1e-14);
    CHECK_NEAR(hllc.S[0], 1.0, 1e-13); // only the pressure
    CHECK(std::fabs(hll.D) > 1e-2);    // for contrast
}

RELJET_TEST(hllc_supersonic_upwinding)
{
    const Prim wL{1.0, 0.1, {0.95, 0, 0}, 1.0}, wR{0.5, 0.1, {0.95, 0, 0}, 0.0};
    const Cons uL = primToCons(wL, kEos), uR = primToCons(wR, kEos);
    CHECK(
        maxDiff(hllcFlux(wL, uL, wR, uR, kEos, 0), physicalFlux(wL, uL, 0)) <
        1e-14);
}

RELJET_TEST(hllc_strong_pressure_jump_uses_hll)
{
    const Prim wL{1.0, 1.0, {0, 0, 0}, 1.0}, wR{1.0, 0.5, {0, 0, 0}, 0.0};
    const Cons uL = primToCons(wL, kEos), uR = primToCons(wR, kEos);
    CHECK(
        maxDiff(
            hllcFlux(wL, uL, wR, uR, kEos, 0),
            hllFlux(wL, uL, wR, uR, kEos, 0)) < 1e-14);
}

RELJET_TEST(hllc_any_direction)
{
    // Same problem rotated into direction 1 gives the rotated flux.
    Prim wL = kWL, wR = kWR;
    std::swap(wL.v[0], wL.v[1]);
    std::swap(wR.v[0], wR.v[1]);
    const Flux fx = hllcFlux(
        kWL, primToCons(kWL, kEos), kWR, primToCons(kWR, kEos), kEos, 0);
    const Flux fy =
        hllcFlux(wL, primToCons(wL, kEos), wR, primToCons(wR, kEos), kEos, 1);
    CHECK_NEAR(fx.D, fy.D, 1e-14);
    CHECK_NEAR(fx.S[0], fy.S[1], 1e-14);
    CHECK_NEAR(fx.S[1], fy.S[0], 1e-14);
    CHECK_NEAR(fx.tau, fy.tau, 1e-14);
}

// ************************************************************************* //

// ---------------------------------------------------------- flux choice --

RELJET_TEST(flux_parse_scheme_names)
{
    CHECK(parseFluxScheme("HLL") == FluxScheme::Hll);
    CHECK(parseFluxScheme("HLLC") == FluxScheme::Hllc);
    CHECK_THROWS(parseFluxScheme("hllc")); // names are case-sensitive
    CHECK_THROWS(parseFluxScheme("Roe"));
}

RELJET_TEST(flux_dispatch_calls_the_chosen_solver)
{
    // A weak jump (p differs by 5 %), so HLLC really uses its star states
    // and differs from HLL.
    const Prim wL{1.0, 1.0, {0.1, 0.2, 0}, 1.0};
    const Prim wR{0.3, 1.05, {0.05, -0.1, 0}, 0.0};
    const Cons uL = primToCons(wL, kEos), uR = primToCons(wR, kEos);
    const Flux hll = hllFlux(wL, uL, wR, uR, kEos, 0);
    const Flux hllc = hllcFlux(wL, uL, wR, uR, kEos, 0);
    CHECK(maxDiff(hll, hllc) > 1e-3);
    CHECK(
        maxDiff(numericalFlux(wL, uL, wR, uR, kEos, 0, FluxScheme::Hll), hll) ==
        0);
    CHECK(
        maxDiff(
            numericalFlux(wL, uL, wR, uR, kEos, 0, FluxScheme::Hllc), hllc) ==
        0);
}

RELJET_TEST(flux_near_shock_sees_every_direction)
{
    // 2D field with uniform pressure except a 50 % jump between rows j and
    // j + 1 (a shock running along direction 0).
    const Grid g = makeGrid(6, 6, 0.0, 1.0, 0.0, 1.0, 2);
    PrimField w(g);
    const int jShock = g.begin(1) + 2;
    for (int idx = 0; idx < g.numCells(); ++idx)
        w.store(idx, Prim{1.0, 1.0, {0, 0, 0}, 0});
    for (int j = jShock + 1; j < g.end(1) + g.ng[1]; ++j)
        for (int i = 0; i < g.padded(0); ++i)
            w.store(g.index(i, j), Prim{1.0, 1.5, {0, 0, 0}, 0});

    const int i = g.begin(0) + 2;
    CHECK(kernels::nearShock(w.view(), g, g.index(i, jShock)));
    CHECK(kernels::nearShock(w.view(), g, g.index(i, jShock + 1)));
    CHECK(!kernels::nearShock(w.view(), g, g.index(i, jShock - 1)));
    CHECK(!kernels::nearShock(w.view(), g, g.index(i, jShock + 2)));

    // A 5 % jump is below the threshold
    for (int j = jShock + 1; j < g.end(1) + g.ng[1]; ++j)
        for (int i2 = 0; i2 < g.padded(0); ++i2)
            w.store(g.index(i2, j), Prim{1.0, 1.05, {0, 0, 0}, 0});
    CHECK(!kernels::nearShock(w.view(), g, g.index(i, jShock)));
}
