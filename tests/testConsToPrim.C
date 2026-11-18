/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Conserved -> primitive recovery: round trips, failure reporting.
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "ConsToPrim.H"
#include "Srhd.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{

const IdealGasEos kEos{5.0 / 3.0};

Prim stateWith(Real W, Real pOverRho)
{
    const Real v = std::sqrt(1 - 1 / (W * W));
    return Prim{1.0, pOverRho, {0.6 * v, -0.8 * v}, 0.3};
}

} // namespace

RELJET_TEST(c2p_round_trip_jet_regime)
{
    // Lorentz factors and temperatures typical of the jets we plan to run.
    // Tolerance follows the measured conditioning: rel. error ~ 1e-16 W^2 (1 + rho/p).
    for (Real W : {1.0, 1.5, 3.0, 10.0})
    {
        for (Real theta : {1e-4, 1e-2, 1.0, 1e2})
        {
            const Prim w = stateWith(W, theta);
            const C2PResult r = consToPrim(primToCons(w, kEos), kEos);
            CHECK(r.status == C2PStatus::Success);
            const Real tol = 1e-13 + 1e-14 * W * W * (1 + 1 / theta);
            CHECK_NEAR(r.w.p / w.p, 1.0, tol);
            CHECK_NEAR(r.w.rho / w.rho, 1.0, tol);
            CHECK_NEAR(r.w.v[0], w.v[0], tol);
            CHECK_NEAR(r.w.v[1], w.v[1], tol);
            CHECK_NEAR(r.w.C, w.C, 1e-15);
            CHECK(r.iterations <= 60);
        }
    }
}

RELJET_TEST(c2p_converges_on_extreme_states)
{
    // Cold and very fast: accuracy degrades, but the solve must still succeed
    // with an admissible, finite answer.
    for (Real W : {100.0, 1000.0})
    {
        for (Real theta : {1e-8, 1e-4, 1e4})
        {
            const C2PResult r =
                consToPrim(primToCons(stateWith(W, theta), kEos), kEos);
            CHECK(r.status == C2PStatus::Success);
            CHECK(r.w.rho > 0 && r.w.p > 0);
            CHECK(velocitySquared(r.w) < 1);
        }
    }
}

RELJET_TEST(c2p_rejects_invalid_input)
{
    const Cons good = primToCons(stateWith(2.0, 1.0), kEos);
    Cons u = good;
    u.D = 0;
    CHECK(consToPrim(u, kEos).status == C2PStatus::InvalidInput);
    u = good;
    u.D = -1;
    CHECK(consToPrim(u, kEos).status == C2PStatus::InvalidInput);
    u = good;
    u.tau =
        std::sqrt(
            good.D * good.D + good.S[0] * good.S[0] + good.S[1] * good.S[1]) -
        good.D; // E on the boundary
    CHECK(consToPrim(u, kEos).status == C2PStatus::InvalidInput);
    u = good;
    u.tau = 0;
    CHECK(consToPrim(u, kEos).status == C2PStatus::InvalidInput);
    u = good;
    u.S[0] = std::nan("");
    CHECK(consToPrim(u, kEos).status == C2PStatus::InvalidInput);
}

RELJET_TEST(c2p_reports_non_convergence)
{
    C2POptions opt;
    opt.maxIter = 2;
    const C2PResult r =
        consToPrim(primToCons(stateWith(100.0, 1e-6), kEos), kEos, opt);
    CHECK(r.status == C2PStatus::NotConverged);
    CHECK(r.iterations == 2);
}

// ************************************************************************* //
