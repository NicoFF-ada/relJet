/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Signal speeds and the HLL flux (Riemann.H).
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Riemann.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{
constexpr double kTol = 1e-13;
const IdealGasEos kEos{5.0 / 3.0};

void checkFluxEqual(
    reljet::test::Context& ctx, const Flux& a, const Flux& b, double tol)
{
    CHECK_NEAR(a.D, b.D, tol);
    CHECK_NEAR(a.S[0], b.S[0], tol);
    CHECK_NEAR(a.S[1], b.S[1], tol);
    CHECK_NEAR(a.tau, b.tau, tol);
    CHECK_NEAR(a.DC, b.DC, tol);
}
} // namespace

RELJET_TEST(riemann_speeds_at_rest_are_sound_speed)
{
    const Prim w{1.0, 0.1, {0, 0}, 0};
    const double cs = std::sqrt(kEos.soundSpeedSquared(w.rho, w.p));
    for (int dir = 0; dir < kDims; ++dir)
    {
        const SignalSpeeds s = signalSpeeds(w, kEos, dir);
        CHECK_NEAR(s.minus, -cs, kTol);
        CHECK_NEAR(s.plus, cs, kTol);
    }
}

RELJET_TEST(riemann_speeds_1d_velocity_addition)
{
    const Prim w{1.0, 2.0, {0.9, 0}, 0};
    const double c = std::sqrt(kEos.soundSpeedSquared(w.rho, w.p));
    const double v = 0.9;
    const SignalSpeeds s = signalSpeeds(w, kEos, 0);
    CHECK_NEAR(s.plus, (v + c) / (1 + v * c), kTol);
    CHECK_NEAR(s.minus, (v - c) / (1 - v * c), kTol);
}

RELJET_TEST(riemann_speeds_transverse_velocity_slows_waves)
{
    // Fluid moving along the face: waves across it are symmetric but slower
    // than c_s (time dilation of the moving fluid).
    const Prim w{1.0, 1.0, {0, 0.9}, 0};
    const double cs = std::sqrt(kEos.soundSpeedSquared(w.rho, w.p));
    const SignalSpeeds s = signalSpeeds(w, kEos, 0);
    CHECK_NEAR(s.minus, -s.plus, kTol);
    CHECK(s.plus > 0 && s.plus < cs);
}

RELJET_TEST(riemann_speeds_are_subluminal)
{
    for (double v : {-0.999, -0.5, 0.0, 0.7, 0.9999})
    {
        for (double p : {1e-6, 1.0, 1e6})
        {
            const Prim w{1.0, p, {v, 0.3 * std::sqrt(1 - v * v)}, 0};
            for (int dir = 0; dir < kDims; ++dir)
            {
                const SignalSpeeds s = signalSpeeds(w, kEos, dir);
                CHECK(-1 < s.minus && s.minus <= s.plus && s.plus < 1);
            }
        }
    }
}

RELJET_TEST(riemann_hll_is_consistent)
{
    const Prim w{1.0, 0.5, {0.6, -0.3}, 0.7};
    const Cons u = primToCons(w, kEos);
    for (int dir = 0; dir < kDims; ++dir)
        checkFluxEqual(
            ctx,
            hllFlux(w, u, w, u, kEos, dir),
            physicalFlux(w, u, dir),
            1e-12);
}

RELJET_TEST(riemann_hll_upwinds_supersonic_flow)
{
    // Fast, cold flow: every wave moves the same way, so the flux is purely
    // the upstream physical flux.
    const Prim wL{1.0, 0.01, {0.99, 0}, 1.0};
    const Prim wR{0.5, 0.02, {0.98, 0}, 0.0};
    const Cons uL = primToCons(wL, kEos);
    const Cons uR = primToCons(wR, kEos);
    checkFluxEqual(
        ctx, hllFlux(wL, uL, wR, uR, kEos, 0), physicalFlux(wL, uL, 0), kTol);

    Prim mL = wL,
         mR = wR; // mirror image: everything moves left, take the right state
    mL.v[0] = -0.98;
    mR.v[0] = -0.99;
    const Cons vL = primToCons(mL, kEos);
    const Cons vR = primToCons(mR, kEos);
    checkFluxEqual(
        ctx, hllFlux(mL, vL, mR, vR, kEos, 0), physicalFlux(mR, vR, 0), kTol);
}

RELJET_TEST(riemann_hll_symmetric_collision)
{
    // Two identical streams colliding head-on: by symmetry no mass, energy or
    // tracer crosses the face, but momentum flux (pressure) does.
    const Prim wL{1.0, 1.0, {0.5, 0}, 0.4};
    const Prim wR{1.0, 1.0, {-0.5, 0}, 0.4};
    const Flux f =
        hllFlux(wL, primToCons(wL, kEos), wR, primToCons(wR, kEos), kEos, 0);
    CHECK_NEAR(f.D, 0.0, kTol);
    CHECK_NEAR(f.tau, 0.0, kTol);
    CHECK_NEAR(f.DC, 0.0, kTol);
    CHECK_NEAR(f.S[1], 0.0, kTol);
    CHECK(f.S[0] > 1.0);
}

// ************************************************************************* //
