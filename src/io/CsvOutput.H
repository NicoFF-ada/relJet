/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    CSV line output of 1D runs.

SourceFiles
    CsvOutput.C
\*---------------------------------------------------------------------------*/

#ifndef CsvOutput_H
#define CsvOutput_H

#include <string>

#include "Fields.H"

namespace reljet
{

// Writes the interior cells of the first interior row (j = begin(1)) as CSV:
// x, primitive variables, conserved variables. Intended for 1D runs and
// quick plotting; 2D output will get its own format.
void writeCsv1D(
    const std::string& path,
    const PrimField& prim,
    const ConsField& cons,
    Real time);

} // namespace reljet

#endif

// ************************************************************************* //
