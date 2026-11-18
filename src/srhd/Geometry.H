/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Coordinate geometry and the axisymmetric geometric source terms.

    Cartesian: (x, y, z), any of 1D/2D/3D.
    Cylindrical: axisymmetric (r, z) in 2D. Direction 0 is r, direction 1 is z,
    direction 2 (the angle phi) is collapsed: nothing depends on phi, but the
    velocity may still have a phi component v[2] (swirl).

    In cylindrical coordinates the divergence of a vector picks up 1/r factors:

      div F = (1/r) d(r F_r)/dr + dF_z/dz

    The solver handles the (1/r) d(r .)/dr part by weighting r-face fluxes with
    the face radius and dividing by the cell radius. What remains are SOURCE
    terms in the momentum equations that have no Cartesian counterpart. They
    come from the fact that the basis vectors e_r, e_phi rotate as you move
    around the axis.
\*---------------------------------------------------------------------------*/

#ifndef Geometry_H
#define Geometry_H

#include "Types.H"

namespace reljet
{

enum class Geometry
{
    Cartesian,
    Cylindrical
};

// Axisymmetric geometric source terms, with (v_r, v_z, v_phi) = v[0..2]:
//
//   S(S_r)   =  ( rho h W^2 v_phi^2 + p ) / r      pressure + centrifugal force
//   S(S_phi) = -( rho h W^2 v_r v_phi ) / r        angular-momentum conservation
//
// and zero for D, S_z, tau and DC, whose divergence the face-radius weighting
// already makes exact. The p / r term balances the different areas of the
// inner and outer ring faces: for a uniform state at rest the discrete
// divergence (r+ p - r- p) / (r dr) is exactly p / r, because r is the
// midpoint (r+ + r-) / 2 (well-balanced, tested).
RELJET_HD inline Cons cylindricalSource(const Prim& w, const Cons& u, Real r)
{
    // u.S[2] = rho h W^2 v_phi, so both terms reuse the stored momentum.
    const Real sourceR = (u.S[2] * w.v[2] + w.p) / r;
    const Real sourcePhi = -u.S[2] * w.v[0] / r;
    return Cons{0, {sourceR, 0, sourcePhi}, 0, 0};
}

} // namespace reljet

#endif

// ************************************************************************* //
