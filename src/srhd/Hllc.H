/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    HLLC approximate Riemann solver for SRHD (Mignone & Bodo 2005, MNRAS
    364, 126). HLL plus the contact wave: two intermediate states U*_L and
    U*_R separated by a contact moving at lambda*. Faces with a strong
    pressure jump, or with inadmissible star states, fall back to HLL.

          lamL        lambda*        lamR
             \    U*_L   |   U*_R    /
        U_L   \          |          /   U_R
    -----------\---------+---------/-----------   face at x = 0
\*---------------------------------------------------------------------------*/

#ifndef Hllc_H
#define Hllc_H

#include <cmath>

#include "ConsOps.H"
#include "Riemann.H"

namespace reljet
{

// ----------------------------------------------------------- HLL averages --
// State U_hll and flux F_hll between the fastest waves.

struct HllAverages
{
    Cons U;
    Flux F;
};

RELJET_HD inline HllAverages hllAverages(
    const Cons& uL,
    const Flux& fL,
    const Cons& uR,
    const Flux& fR,
    Real lamL,
    Real lamR)
{
    HllAverages h;
    h.U = (lamR * uR - lamL * uL - (fR - fL)) * (1 / (lamR - lamL));
    h.F =
        (lamR * fL - lamL * fR + lamL * lamR * (uR - uL)) * (1 / (lamR - lamL));
    return h;
}


// --------------------------------------------------------------- contact --
// Contact speed lambda*: root of  F^E lambda*^2 - (E + F^m) lambda* + m = 0
// (Mignone & Bodo 2005, eq. 18), in the form without cancellation as F^E -> 0.

RELJET_HD inline Real contactSpeed(const HllAverages& h, int dir)
{
    const Real E = h.U.tau + h.U.D;  // total energy  E = tau + D
    const Real FE = h.F.tau + h.F.D; // its flux
    const Real m = h.U.S[dir];       // normal momentum
    const Real Fm = h.F.S[dir];      // its flux
    const Real B = E + Fm;
    return 2 * m / (B + std::sqrt(std::fmax(B * B - 4 * FE * m, Real(0))));
}

// Contact pressure p* = F^m - F^E lambda*.
RELJET_HD inline Real
contactPressure(const HllAverages& h, Real lamStar, int dir)
{
    return h.F.S[dir] - (h.F.tau + h.F.D) * lamStar;
}


// ------------------------------------------------------------ star states --
// Star state between the wave lam and the contact lamStar, on the side of w
// (Mignone & Bodo 2005, eq. 16, with the energy equation written for tau).

RELJET_HD inline Cons starState(
    const Prim& w, const Cons& u, Real lam, Real lamStar, Real pStar, int dir)
{
    const Real v = w.v[dir];
    const Real f = (lam - v) / (lam - lamStar); // compression factor
    Cons s;
    s.D = u.D * f;
    s.DC = u.DC * f;
    for (int d = 0; d < kDims; ++d)
        s.S[d] = u.S[d] * f; // tangential momentum is only compressed
    s.S[dir] = (u.S[dir] * (lam - v) + pStar - w.p) / (lam - lamStar);
    s.tau = (u.tau * (lam - v) + pStar * lamStar - w.p * v) / (lam - lamStar);
    return s;
}


// ------------------------------------------------------------------- flux --
// A conserved state from which a physical primitive state can be recovered.
RELJET_HD inline bool admissibleCons(const Cons& u)
{
    Real S2 = 0;
    for (int d = 0; d < kDims; ++d)
        S2 += u.S[d] * u.S[d];
    return u.D > 0 && u.tau > 0 && u.tau * (u.tau + 2 * u.D) > S2;
}

RELJET_HD inline Flux hllcFlux(
    const Prim& wL,
    const Cons& uL,
    const Prim& wR,
    const Cons& uR,
    const IdealGasEos& eos,
    int dir)
{
    // Wave speeds and physical fluxes, exactly as in hllFlux
    const SignalSpeeds sL = signalSpeeds(wL, eos, dir);
    const SignalSpeeds sR = signalSpeeds(wR, eos, dir);
    const Real lamL = std::fmin(Real(0), std::fmin(sL.minus, sR.minus));
    const Real lamR = std::fmax(Real(0), std::fmax(sL.plus, sR.plus));
    const Flux fL = physicalFlux(wL, uL, dir);
    const Flux fR = physicalFlux(wR, uR, dir);

    // Supersonic: every wave moves the same way, pure upwinding
    if (lamL >= 0)
        return fL;
    if (lamR <= 0)
        return fR;

    const HllAverages h = hllAverages(uL, fL, uR, fR, lamL, lamR);

    // Robustness guard 1: strong pressure jumps (shocks) use HLL
    if (std::fabs(wR.p - wL.p) > Real(0.1) * std::fmin(wL.p, wR.p))
        return h.F;

    const Real lamStar = contactSpeed(h, dir);
    const Real pStar = contactPressure(h, lamStar, dir);
    const Cons uStarL = starState(wL, uL, lamL, lamStar, pStar, dir);
    const Cons uStarR = starState(wR, uR, lamR, lamStar, pStar, dir);

    // Robustness guard 2: a contact outside the wave fan, or unphysical star
    // states, use HLL
    if (!(lamStar > lamL && lamStar < lamR && pStar > 0) ||
        !admissibleCons(uStarL) || !admissibleCons(uStarR))
    {
        return h.F;
    }

    // Jump condition across the outer wave on the face side of the contact
    if (lamStar >= 0)
        return fL + lamL * (uStarL - uL);
    return fR + lamR * (uStarR - uR);
}

} // namespace reljet

#endif

// ************************************************************************* //
