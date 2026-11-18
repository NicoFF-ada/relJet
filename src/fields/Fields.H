/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Structure-of-arrays fields and their non-owning device-compatible views.

SourceFiles
    Fields.C
\*---------------------------------------------------------------------------*/

#ifndef Fields_H
#define Fields_H

#include <cstddef>
#include <vector>

#include "Grid.H"
#include "Types.H"

namespace reljet
{

// Structure-of-arrays storage: component k of cell idx lives at
// data[k * numCells + idx]. Each physical component is one contiguous array,
// so consecutive cells (and consecutive GPU threads) touch consecutive
// addresses.
//
// ConsView / PrimView are non-owning views of such storage: a pointer plus
// the cell count. They are trivially copyable and RELJET_HD, so the same
// load/store code runs on host arrays and on device arrays. The owning host
// fields (ConsField, PrimField) hand out views; the CUDA backend builds views
// of its device buffers.
struct ConsView
{
    Real* data;
    int numCells;

    RELJET_HD Real& at(int k, int idx) const
    {
        return data[std::size_t(k) * numCells + idx];
    }

    RELJET_HD Cons load(int idx) const
    {
        return Cons{
            at(iD, idx),
            {at(iS0, idx), at(iS1, idx), at(iS2, idx)},
            at(iTau, idx),
            at(iDC, idx)};
    }

    RELJET_HD void store(int idx, const Cons& u) const
    {
        at(iD, idx) = u.D;
        at(iS0, idx) = u.S[0];
        at(iS1, idx) = u.S[1];
        at(iS2, idx) = u.S[2];
        at(iTau, idx) = u.tau;
        at(iDC, idx) = u.DC;
    }
};

struct PrimView
{
    Real* data;
    int numCells;

    RELJET_HD Real& at(int k, int idx) const
    {
        return data[std::size_t(k) * numCells + idx];
    }

    RELJET_HD Prim load(int idx) const
    {
        return Prim{
            at(iRho, idx),
            at(iP, idx),
            {at(iV0, idx), at(iV1, idx), at(iV2, idx)},
            at(iC, idx)};
    }

    RELJET_HD void store(int idx, const Prim& w) const
    {
        at(iRho, idx) = w.rho;
        at(iP, idx) = w.p;
        at(iV0, idx) = w.v[0];
        at(iV1, idx) = w.v[1];
        at(iV2, idx) = w.v[2];
        at(iC, idx) = w.C;
    }
};

// Owning host storage for one SoA field on a grid.
class SoAField
{
public:
    SoAField(const Grid& grid, int numComponents);

    const Grid& grid() const { return grid_; }
    int numComponents() const { return numComponents_; }
    int numCells() const { return grid_.numCells(); }

    Real* data() { return data_.data(); }
    const Real* data() const { return data_.data(); }
    std::size_t size() const { return data_.size(); }

    Real* component(int k)
    {
        return data_.data() + std::size_t(k) * numCells();
    }
    const Real* component(int k) const
    {
        return data_.data() + std::size_t(k) * numCells();
    }

    Real& at(int k, int idx) { return component(k)[idx]; }
    Real at(int k, int idx) const { return component(k)[idx]; }

private:
    Grid grid_;
    int numComponents_;
    std::vector<Real> data_;
};

class ConsField : public SoAField
{
public:
    explicit ConsField(const Grid& grid) : SoAField(grid, kNumCons) {}

    ConsView view() { return ConsView{data(), numCells()}; }

    Cons load(int idx) const
    {
        return ConsView{const_cast<Real*>(data()), numCells()}.load(idx);
    }
    void store(int idx, const Cons& u) { view().store(idx, u); }
};

class PrimField : public SoAField
{
public:
    explicit PrimField(const Grid& grid) : SoAField(grid, kNumPrim) {}

    PrimView view() { return PrimView{data(), numCells()}; }

    Prim load(int idx) const
    {
        return PrimView{const_cast<Real*>(data()), numCells()}.load(idx);
    }
    void store(int idx, const Prim& w) { view().store(idx, w); }
};

} // namespace reljet

#endif

// ************************************************************************* //
