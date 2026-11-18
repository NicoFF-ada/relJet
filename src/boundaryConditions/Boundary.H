/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Ghost-cell boundary conditions: outflow, reflecting, periodic, jet inlet.

SourceFiles
    Boundary.C
\*---------------------------------------------------------------------------*/

#ifndef Boundary_H
#define Boundary_H

#include <string>

#include "Fields.H"
#include "JetInlet.H"

namespace reljet
{

// Ghost-cell boundary conditions, applied to the primitive field before
// reconstruction. Each of the four domain faces gets its own type:
//
//   Outflow    : zero gradient. Every ghost layer copies the nearest interior
//                cell, so waves leave the domain with little reflection.
//   Reflecting : mirror image. Ghost layer g copies interior layer g (counted
//                from the wall) and flips the velocity component normal to
//                the wall. Models a solid wall and the symmetry axis
//                r = 0 of the axisymmetric jet.
//   Periodic   : the domain wraps around. Lower ghosts copy the last interior
//                cells and upper ghosts the first ones. Must be set on BOTH
//                faces of a direction (checked by validateBoundaries).
//   JetInlet   : jet nozzle in a reflecting wall (JetInlet.H). The
//                distance from the jet axis is measured from coordinate 0 in
//                the other active directions (r itself in cylindrical runs).
enum class BoundaryType
{
    Outflow,
    Reflecting,
    Periodic,
    JetInlet
};

struct BoundaryConditions
{
    BoundaryType lower[kDims]; // face at lo[d]
    BoundaryType upper[kDims]; // face at hi[d]
    JetNozzle jet{};           // used by JetInlet faces
};

// "outflow", "reflecting", "periodic" or "jetInlet"; throws std::invalid_argument otherwise.
BoundaryType parseBoundaryType(const std::string& name);

// Throws std::invalid_argument if a direction is periodic on only one face.
void validateBoundaries(const BoundaryConditions& bc);

// Fills all ghost cells of every non-collapsed direction.
void applyBoundaries(PrimField& w, const BoundaryConditions& bc);

} // namespace reljet

#endif

// ************************************************************************* //
