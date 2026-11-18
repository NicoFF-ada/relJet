/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Legacy-VTK output of the primitive fields.

SourceFiles
    VtkOutput.C
\*---------------------------------------------------------------------------*/

#ifndef VtkOutput_H
#define VtkOutput_H

#include <string>

#include "Fields.H"

namespace reljet
{

// Writes the interior cells as a legacy-VTK binary STRUCTURED_POINTS file
// (cell data: rho, p, Lorentz factor W, tracer C, velocity vector), readable
// by ParaView and VisIt. Values are stored as 32-bit floats, big-endian as the
// legacy format requires. Collapsed directions become one cell thick.
void writeVtk(const std::string& path, const PrimField& prim, Real time);

} // namespace reljet

#endif

// ************************************************************************* //
