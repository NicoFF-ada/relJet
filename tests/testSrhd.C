/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Pointwise SRHD relations (Srhd.H).
    These fail until lorentzFactor / primToCons / physicalFlux are implemented.
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Srhd.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{
constexpr double kTol = 1e-13;
const IdealGasEos kEos{5.0 / 3.0};
} // namespace

RELJET_TEST(srhd_lorentz_factor)
{
    CHECK_NEAR(lorentzFactor(Prim{1, 1, {0, 0}, 0}), 1.0, kTol);
    CHECK_NEAR(lorentzFactor(Prim{1, 1, {0.6, 0}, 0}), 1.25, kTol);
    CHECK_NEAR(
        lorentzFactor(Prim{1, 1, {0.3, 0.4}, 0}), 1.0 / std::sqrt(0.75), kTol);
    CHECK_NEAR(
        lorentzFactor(Prim{1, 1, {0, -0.99}, 0}),
        1.0 / std::sqrt(1 - 0.99 * 0.99),
        1e-12);
}

RELJET_TEST(srhd_prim_to_cons_at_rest)
{
    // v = 0: D = rho, S = 0, tau = rho eps = p/(gamma-1) (internal energy only)
    const Cons u = primToCons(Prim{2.0, 3.0, {0, 0}, 0.25}, kEos);
    CHECK_NEAR(u.D, 2.0, kTol);
    CHECK_NEAR(u.S[0], 0.0, kTol);
    CHECK_NEAR(u.S[1], 0.0, kTol);
    CHECK_NEAR(u.tau, 3.0 / (kEos.gamma - 1), kTol);
    CHECK_NEAR(u.DC, 0.5, kTol);
}

RELJET_TEST(srhd_prim_to_cons_moving)
{
    // rho = 1, p = 0.5, v = (0.6, 0): W = 1.25, h = 2.25, rho h W^2 = 3.515625
    const Cons u = primToCons(Prim{1.0, 0.5, {0.6, 0.0}, 1.0}, kEos);
    CHECK_NEAR(u.D, 1.25, kTol);
    CHECK_NEAR(u.S[0], 2.109375, kTol);
    CHECK_NEAR(u.S[1], 0.0, kTol);
    CHECK_NEAR(u.tau, 3.015625 - 1.25, kTol); // E - D
    CHECK_NEAR(u.DC, 1.25, kTol);
}

RELJET_TEST(srhd_prim_to_cons_identity_S_equals_Eplusp_v)
{
    const Prim w{0.1, 0.01, {-0.2, 0.9}, 0.5};
    const Cons u = primToCons(w, kEos);
    for (int d = 0; d < kDims; ++d)
        CHECK_NEAR(u.S[d], (u.tau + u.D + w.p) * w.v[d], 1e-12);
    // tau > 0 always holds for physical states (energy beyond rest mass).
    CHECK(u.tau > 0);
}

RELJET_TEST(srhd_physical_flux)
{
    const Prim w{1.0, 0.5, {0.6, -0.3}, 0.7};
    const Cons u = primToCons(w, kEos);
    for (int dir = 0; dir < kDims; ++dir)
    {
        const Flux f = physicalFlux(w, u, dir);
        const double vd = w.v[dir];
        CHECK_NEAR(f.D, u.D * vd, kTol);
        CHECK_NEAR(f.S[0], u.S[0] * vd + (dir == 0 ? w.p : 0.0), kTol);
        CHECK_NEAR(f.S[1], u.S[1] * vd + (dir == 1 ? w.p : 0.0), kTol);
        CHECK_NEAR(f.tau, u.S[dir] - u.D * vd, kTol);
        CHECK_NEAR(f.DC, u.DC * vd, kTol);
    }
}

RELJET_TEST(srhd_flux_of_static_state_is_pure_pressure)
{
    // A fluid at rest pushes on a face only through pressure.
    const Prim w{3.0, 0.2, {0, 0}, 0};
    const Flux f = physicalFlux(w, primToCons(w, kEos), 0);
    CHECK_NEAR(f.D, 0.0, kTol);
    CHECK_NEAR(f.S[0], 0.2, kTol);
    CHECK_NEAR(f.S[1], 0.0, kTol);
    CHECK_NEAR(f.tau, 0.0, kTol);
}

// ************************************************************************* //
