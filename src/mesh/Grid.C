/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Uniform, cell-centred 1D/2D/3D mesh with ghost layers.
\*---------------------------------------------------------------------------*/

#include "Grid.H"

#include <stdexcept>

namespace reljet
{

Grid makeGrid(const GridSpec& spec)
{
    if (spec.numGhost < 1)
        throw std::invalid_argument("grid: need at least one ghost layer");
    if (spec.n[0] < 2)
        throw std::invalid_argument("grid: direction 0 needs at least 2 cells");
    Grid g{};
    for (int d = 0; d < kDims; ++d)
    {
        if (spec.n[d] < 1)
            throw std::invalid_argument("grid: cell counts must be >= 1");
        if (!(spec.hi[d] > spec.lo[d]))
            throw std::invalid_argument(
                "grid: need hi > lo in every direction");
        g.n[d] = spec.n[d];
        g.ng[d] = (spec.n[d] == 1) ? 0 : spec.numGhost;
        g.lo[d] = spec.lo[d];
        g.hi[d] = spec.hi[d];
        g.dx[d] = (spec.hi[d] - spec.lo[d]) / spec.n[d];
    }
    return g;
}

Grid makeGrid(
    int n0, int n1, Real lo0, Real hi0, Real lo1, Real hi1, int numGhost)
{
    GridSpec s;
    s.n[0] = n0;
    s.n[1] = n1;
    s.lo[0] = lo0;
    s.hi[0] = hi0;
    s.lo[1] = lo1;
    s.hi[1] = hi1;
    s.numGhost = numGhost;
    return makeGrid(s);
}

} // namespace reljet

// ************************************************************************* //
