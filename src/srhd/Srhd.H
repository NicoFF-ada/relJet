/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Pointwise special-relativistic hydrodynamics relations (c = 1).

    Everything here acts on ONE cell or ONE face state, carries no memory, and
    is RELJET_HD, so the identical code runs inside CUDA kernels: no heap
    allocation, no exceptions, no virtual calls, no I/O.
\*---------------------------------------------------------------------------*/

#ifndef Srhd_H
#define Srhd_H

#include <cmath>
#include <limits>

#include "IdealGasEos.H"
#include "Types.H"

namespace reljet
{

RELJET_HD inline Real velocitySquared(const Prim& w)
{
    Real v2 = 0;
    for (int d = 0; d < kDims; ++d)
        v2 += w.v[d] * w.v[d];
    return v2;
}

// ---------------------------------------------------------------------------
// Lorentz factor W = 1 / sqrt(1 - v^2). Assumes v^2 < 1; the caller
// is responsible for admissibility.
// ---------------------------------------------------------------------------
RELJET_HD inline Real lorentzFactor(const Prim& w)
{
    return 1 / std::sqrt(1 - velocitySquared(w));
}

// ---------------------------------------------------------------------------
// Primitive -> conserved
//
//   D   = rho W
//   S_i = rho h W^2 v_i
//   tau = rho h W^2 - p - D      (total energy E minus rest-mass energy D)
//   DC  = D C
//
// We evolve tau instead of E = tau + D. In cold or slow gas the thermal energy
// is a tiny fraction of E, so storing E and later computing E - D would cancel
// most significant digits. tau is assembled here from non-negative terms only,
// using W - 1 = W^2 v^2 / (1 + W) and W^2 - 1 = W^2 v^2:
//
//   tau = D W v^2 W / (1 + W)  +  rho eps W^2  +  p W^2 v^2
//         (kinetic)               (thermal)       (pressure work)
//
// At rest this is exactly rho eps, with no subtraction at all.
// ---------------------------------------------------------------------------
RELJET_HD inline Cons primToCons(const Prim& w, const IdealGasEos& eos)
{
    const Real v2 = velocitySquared(w);
    const Real W = lorentzFactor(w);
    const Real W2 = W * W;
    const Real h = eos.specificEnthalpy(w.rho, w.p);
    const Real rhohW2 = w.rho * h * W2; // shared by S and tau
    const Real eps = eos.specificInternalEnergy(w.rho, w.p);

    Cons u{};
    u.D = w.rho * W;
    for (int d = 0; d < kDims; ++d)
        u.S[d] = rhohW2 * w.v[d];
    u.tau = u.D * W2 * v2 / (1 + W) + w.rho * eps * W2 + w.p * W2 * v2;
    u.DC = u.D * w.C;
    return u;
}

// ---------------------------------------------------------------------------
// Physical flux in direction `dir`
//
//   F(D)    = D  v_dir
//   F(S_i)  = S_i v_dir + p delta_{i,dir}
//   F(tau)  = (tau + p) v_dir           ( = S_dir - D v_dir )
//   F(DC)   = DC v_dir
//
// F(tau) is the energy flux S_dir minus the rest-mass flux D v_dir; the
// (tau + p) form gives the same value without that subtraction.
// ---------------------------------------------------------------------------
RELJET_HD inline Flux physicalFlux(const Prim& w, const Cons& u, int dir)
{
    Flux f{};
    f.D = u.D * w.v[dir];
    for (int d = 0; d < kDims; ++d)
        f.S[d] =
            u.S[d] * w.v[dir] + w.p * (d == dir); // bool -> Kronecker delta
    f.tau = (u.tau + w.p) * w.v[dir];
    f.DC = u.DC * w.v[dir];
    return f;
}

} // namespace reljet

#endif

// ************************************************************************* //
