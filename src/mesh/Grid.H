/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Uniform, cell-centred 1D/2D/3D mesh with ghost layers.

SourceFiles
    Grid.C
\*---------------------------------------------------------------------------*/

#ifndef Grid_H
#define Grid_H

#include "Types.H"

namespace reljet
{

// Uniform, cell-centred, logically rectangular 3D mesh with ghost layers.
//
// Indexing: (i, j, k) are *padded* indices that include ghosts, so the
// interior is i in [begin(0), end(0)), j in [begin(1), end(1)), k likewise.
// Direction 0 varies fastest in memory, then 1, then 2:
//
//     index(i, j, k) = (k * padded(1) + j) * padded(0) + i
//
// so neighbouring CUDA threads (consecutive i) read neighbouring cells.
// stride(d) is the index distance between neighbours in direction d.
//
// Lower-dimensional runs "collapse" a direction: n[d] = 1 and ng[d] = 0.
// A 1D run collapses directions 1 and 2, a 2D run direction 2.
struct Grid
{
    int n[kDims];   // interior cells per direction
    int ng[kDims];  // ghost layers per direction
    Real lo[kDims]; // lower domain edge
    Real hi[kDims]; // upper domain edge
    Real dx[kDims]; // cell width

    RELJET_HD int padded(int d) const { return n[d] + 2 * ng[d]; }
    RELJET_HD int numCells() const { return padded(0) * padded(1) * padded(2); }
    RELJET_HD int index(int i, int j, int k = 0) const
    {
        return (k * padded(1) + j) * padded(0) + i;
    }
    RELJET_HD int stride(int d) const
    {
        return d == 0 ? 1 : d == 1 ? padded(0) : padded(0) * padded(1);
    }
    RELJET_HD int begin(int d) const { return ng[d]; }
    RELJET_HD int end(int d) const { return ng[d] + n[d]; }
    RELJET_HD bool collapsed(int d) const { return n[d] == 1 && ng[d] == 0; }
    RELJET_HD int numActiveDims() const
    {
        return !collapsed(0) + !collapsed(1) + !collapsed(2);
    }
    RELJET_HD long numInteriorCells() const { return long(n[0]) * n[1] * n[2]; }

    // Cell-centre coordinate of padded index i in direction d.
    RELJET_HD Real center(int d, int i) const
    {
        return lo[d] + (i - ng[d] + Real(0.5)) * dx[d];
    }
};

struct GridSpec
{
    int n[kDims] = {1, 1, 1};
    Real lo[kDims] = {0, 0, 0};
    Real hi[kDims] = {1, 1, 1};
    int numGhost = 2;
};

// Builds and validates a grid; throws std::invalid_argument on bad input.
// Directions with n[d] = 1 are collapsed (no ghosts). Direction 0 must be active.
Grid makeGrid(const GridSpec& spec);

// Convenience for 1D/2D grids (direction 2 collapsed on [0, 1]).
Grid makeGrid(
    int n0, int n1, Real lo0, Real hi0, Real lo1, Real hi1, int numGhost);

} // namespace reljet

#endif

// ************************************************************************* //
