/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Ghost-cell boundary conditions: outflow, reflecting, periodic, jet inlet.
\*---------------------------------------------------------------------------*/

#include "Boundary.H"

#include "Kernels.H"

#include <cmath>
#include <stdexcept>
#include <string>

namespace reljet
{

BoundaryType parseBoundaryType(const std::string& name)
{
    if (name == "outflow")
        return BoundaryType::Outflow;
    if (name == "reflecting")
        return BoundaryType::Reflecting;
    if (name == "periodic")
        return BoundaryType::Periodic;
    if (name == "jetInlet")
        return BoundaryType::JetInlet;
    throw std::invalid_argument(
        "unknown boundary type '" + name +
        "' (expected outflow | reflecting | periodic | jetInlet)");
}

void validateBoundaries(const BoundaryConditions& bc)
{
    for (int d = 0; d < kDims; ++d)
    {
        if ((bc.lower[d] == BoundaryType::Periodic) !=
            (bc.upper[d] == BoundaryType::Periodic))
            throw std::invalid_argument(
                "boundaries: periodic must be set on both faces of direction " +
                std::to_string(d));
    }
}

void applyBoundaries(PrimField& w, const BoundaryConditions& bc)
{
    const Grid& grid = w.grid();
    const PrimView view = w.view();
    for (int d = 0; d < kDims; ++d)
    {
        if (grid.collapsed(d))
            continue;
        for (int column = 0; column < kernels::ghostColumns(grid, d); ++column)
            kernels::fillGhostColumn(view, grid, bc, d, column);
    }
}

} // namespace reljet

// ************************************************************************* //
