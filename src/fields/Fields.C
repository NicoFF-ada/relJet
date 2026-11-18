/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Structure-of-arrays fields and their non-owning device-compatible views.
\*---------------------------------------------------------------------------*/

#include "Fields.H"

namespace reljet
{

SoAField::SoAField(const Grid& grid, int numComponents)
    : grid_(grid),
      numComponents_(numComponents),
      data_(std::size_t(numComponents) * grid.numCells(), Real(0))
{
}

} // namespace reljet

// ************************************************************************* //
