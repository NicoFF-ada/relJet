/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Conserved-to-primitive recovery: safeguarded Newton iteration on p.

    Unlike primToCons there is no closed form: given (D, S, tau) we must find
    the pressure p that is consistent with the EOS. Everything else follows
    from p:

      Q   = tau + D + p                 (= rho h W^2)
      v   = S / Q
      W   = 1 / sqrt(1 - v^2)
      rho = D / W
      eps = (tau - D W v^2 W / (1 + W) - p W^2 v^2) / (D W)

    The eps formula is the inverse of the tau assembly in primToCons; it never
    forms E = tau + D, so at rest it is exactly tau / D with no cancellation.

    Residual:  f(p) = p_eos(rho(p), eps(p)) - p
    Derivative (Marti & Mueller 2003, Living Rev. Relativ. 6):
               f'(p) ~= v^2 c_s^2 - 1        (always < 0, so the root is unique)

    Method: safeguarded Newton. We keep a bracket [lo, hi] with f(lo) > 0 >= f(hi)
    and take a Newton step only if it lands inside the bracket; otherwise we
    bisect. This cannot diverge, and the iteration count is capped.

    Admissible input: D > 0 and E > sqrt(D^2 + S^2) with E = tau + D, which
    rearranges without cancellation to  tau > 0  and  tau (tau + 2D) > S^2.
    Bracket for such a state:
      lo = 0              : f(0) > 0 exactly when the state is admissible
      hi = (gamma-1) tau  : rho eps <= tau for any state, so p_eos <= hi

    Known limitation: for W >> 1 and p << rho the kinetic term dominates tau,
    and eps is again a small difference of large numbers, so the relative
    accuracy of p degrades (measured in tests/testConsToPrim.C).
\*---------------------------------------------------------------------------*/

#ifndef ConsToPrim_H
#define ConsToPrim_H

#include <cmath>

#include "IdealGasEos.H"
#include "Types.H"

namespace reljet
{

enum class C2PStatus : int
{
    Success = 0,
    NotConverged = 1, // recoverable: input admissible but the solve failed
    InvalidInput =
        2, // D <= 0, tau <= 0, tau (tau + 2D) <= S^2, or non-finite values
};

struct C2POptions
{
    Real relTol =
        1e-13; // stop when |dp| <= relTol * p, or the bracket is that narrow
    int maxIter = 200; // bisection alone needs ~log2(range / tol) steps
};

struct C2PResult
{
    Prim w;
    C2PStatus status;
    int iterations;
};

namespace detail
{

struct C2PEval
{
    Real f;    // residual p_eos - p
    Real dfdp; // approximate derivative
};

RELJET_HD inline C2PEval
c2pResidual(const Cons& u, Real S2, Real p, const IdealGasEos& eos)
{
    const Real Q = u.tau + u.D + p;
    const Real v2 = S2 / (Q * Q);
    const Real W = 1 / std::sqrt(1 - v2);
    const Real W2v2 = W * W * v2;
    const Real rho = u.D / W;
    const Real eps = (u.tau - u.D * W2v2 / (1 + W) - p * W2v2) / (u.D * W);
    const Real pEos = eos.pressure(rho, eps);
    // c_s^2 evaluated at the current iterate; clamp p to keep it finite near p = 0.
    const Real cs2 = eos.soundSpeedSquared(
        rho,
        p > 0      ? p
        : pEos > 0 ? pEos
                   : Real(1e-300));
    return C2PEval{pEos - p, v2 * cs2 - 1};
}

} // namespace detail

RELJET_HD inline C2PResult consToPrim(
    const Cons& u, const IdealGasEos& eos, const C2POptions& opt = C2POptions{})
{
    C2PResult r{};
    r.status = C2PStatus::InvalidInput;

    Real S2 = 0;
    for (int d = 0; d < kDims; ++d)
        S2 += u.S[d] * u.S[d];
    const bool finite = std::isfinite(u.D) && std::isfinite(u.tau) &&
                        std::isfinite(S2) && std::isfinite(u.DC);
    if (!finite || !(u.D > 0) || !(u.tau > 0) ||
        !(u.tau * (u.tau + 2 * u.D) > S2))
        return r;

    Real lo = 0;
    Real hi = (eos.gamma - 1) * u.tau;
    // Initial guess: all of tau is internal energy (exact for a fluid at rest).
    Real p = hi;

    r.status = C2PStatus::NotConverged;
    for (int it = 1; it <= opt.maxIter; ++it)
    {
        r.iterations = it;
        const detail::C2PEval e = detail::c2pResidual(u, S2, p, eos);
        if (e.f > 0)
            lo = p;
        else
            hi = p;

        const Real step = e.f / e.dfdp;
        if (std::fabs(step) <= opt.relTol * p)
        { // Newton has converged at p
            r.status = C2PStatus::Success;
            break;
        }
        p -= step;
        if (!(p > lo && p < hi))
            p = Real(0.5) * (lo + hi); // Newton left the bracket: bisect
        if (hi - lo <= opt.relTol * hi)
        { // bracket has collapsed onto the root
            r.status = C2PStatus::Success;
            break;
        }
    }

    // Assemble the primitive state from the final pressure.
    const Real Q = u.tau + u.D + p;
    for (int d = 0; d < kDims; ++d)
        r.w.v[d] = u.S[d] / Q;
    const Real W = 1 / std::sqrt(1 - S2 / (Q * Q));
    r.w.rho = u.D / W;
    r.w.p = p;
    r.w.C = u.DC / u.D;

    if (r.status == C2PStatus::Success &&
        !(r.w.rho > 0 && r.w.p > 0 && std::isfinite(W)))
        r.status = C2PStatus::NotConverged;
    return r;
}

} // namespace reljet

#endif

// ************************************************************************* //
