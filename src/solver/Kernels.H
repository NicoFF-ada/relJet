/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Pointwise operations of one solver step, shared by the CPU and CUDA backends.

    Each function does the work of ONE cell, face or ghost column, so a backend
    only decides how to loop: OpenMP over (k, j) on the CPU, one thread per
    element on the GPU. Both backends therefore evaluate identical
    expressions; results differ only by floating-point contraction (FMA) and
    reduction order.
\*---------------------------------------------------------------------------*/

#ifndef Kernels_H
#define Kernels_H

#include <cmath>

#include "Boundary.H"
#include "ConsOps.H"
#include "ConsToPrim.H"
#include "Fields.H"
#include "Geometry.H"
#include "Grid.H"
#include "NumericalFlux.H"
#include "Reconstruction.H"
#include "Riemann.H"
#include "Srhd.H"

namespace reljet
{

struct SolverOptions
{
    Real cfl = 0.4;
    Reconstruction reconstruction = Reconstruction::FirstOrder;
    Limiter limiter = Limiter::Minmod;
    FluxScheme flux = FluxScheme::Hll;
    Geometry geometry = Geometry::Cartesian;
    BoundaryConditions bc{};
    C2POptions c2p{};
};

// Index range [lo, hi) per direction, with a linear numbering (direction 0
// fastest) so that one GPU thread can be mapped to one element.
struct Box
{
    int lo[kDims];
    int hi[kDims];

    RELJET_HD int extent(int d) const { return hi[d] - lo[d]; }
    RELJET_HD int size() const { return extent(0) * extent(1) * extent(2); }

    RELJET_HD void decode(int n, int& i, int& j, int& k) const
    {
        i = lo[0] + n % extent(0);
        j = lo[1] + (n / extent(0)) % extent(1);
        k = lo[2] + n / (extent(0) * extent(1));
    }
};

// Interior cells.
RELJET_HD inline Box interiorBox(const Grid& g)
{
    Box b{};
    for (int d = 0; d < kDims; ++d)
    {
        b.lo[d] = g.begin(d);
        b.hi[d] = g.end(d);
    }
    return b;
}

// Cells whose LOWER face along d is a computational face: the interior plus
// one cell past the upper boundary, i.e. n[d] + 1 faces.
RELJET_HD inline Box faceBox(const Grid& g, int d)
{
    Box b = interiorBox(g);
    b.hi[d] += 1;
    return b;
}

namespace kernels
{

// --------------------------------------------------------------- geometry --

RELJET_HD inline Real cellRadius(const Grid& g, int i)
{
    return g.center(0, i);
}

// Radius of the lower r-face of cell i.
RELJET_HD inline Real faceRadius(const Grid& g, int i)
{
    return g.lo[0] + (i - g.ng[0]) * g.dx[0];
}

// -------------------------------------------------------------- time step --

// sum_d max |lambda_d| / dx_d for one cell; dt = cfl / max over cells.
RELJET_HD inline Real
signalRate(const PrimView& w, const Grid& g, const IdealGasEos& eos, int idx)
{
    const Prim p = w.load(idx);
    Real rate = 0;
    for (int d = 0; d < kDims; ++d)
    {
        if (g.collapsed(d))
        {
            continue;
        }
        const SignalSpeeds s = signalSpeeds(p, eos, d);
        rate += std::fmax(std::fabs(s.minus), std::fabs(s.plus)) / g.dx[d];
    }
    return rate;
}

// ------------------------------------------------------------- boundaries --

// Interior index along one direction that ghost layer g (1 = next to the
// boundary) copies; lo / hi are the first / last interior indices.
RELJET_HD inline int
ghostSource(BoundaryType type, bool lowerFace, int g, int lo, int hi)
{
    switch (type)
    {
        case BoundaryType::Outflow:
            return lowerFace ? lo : hi;
        case BoundaryType::Reflecting:
        case BoundaryType::JetInlet:
            return lowerFace ? lo + g - 1 : hi - g + 1;
        case BoundaryType::Periodic:
            return lowerFace ? hi - g + 1 : lo + g - 1;
    }
    return lowerFace ? lo : hi; // unreachable for valid enum values
}

RELJET_HD inline Prim ghostState(
    Prim src, BoundaryType type, int dir, Real distance, const JetNozzle& jet)
{
    if (type == BoundaryType::Reflecting)
    {
        src.v[dir] = -src.v[dir];
    }
    else if (type == BoundaryType::JetInlet)
    {
        src = jetInletGhost(src, distance, jet, dir);
    }
    return src;
}

// Number of ghost columns normal to direction d: every padded cell of the two
// other directions, so that edges and corners are filled too.
RELJET_HD inline int ghostColumns(const Grid& g, int d)
{
    return g.padded((d + 1) % kDims) * g.padded((d + 2) % kDims);
}

// Fills the ghosts at both ends of column `column` along direction d.
// Directions must be processed in order 0, 1, 2 (each pass copies the ghosts
// of the previous ones).
RELJET_HD inline void fillGhostColumn(
    const PrimView& w,
    const Grid& g,
    const BoundaryConditions& bc,
    int d,
    int column)
{
    const int a = (d + 1) % kDims;
    const int b = (d + 2) % kDims;
    int ijk[kDims];
    ijk[a] = column % g.padded(a);
    ijk[b] = column / g.padded(a);

    // Distance from the jet axis, which passes through coordinate 0 of the
    // other active directions.
    const Real xa = g.collapsed(a) ? Real(0) : g.center(a, ijk[a]);
    const Real xb = g.collapsed(b) ? Real(0) : g.center(b, ijk[b]);
    const Real distance = std::sqrt(xa * xa + xb * xb);

    auto at = [&](int id)
    {
        ijk[d] = id;
        return g.index(ijk[0], ijk[1], ijk[2]);
    };
    const int lo = g.begin(d);
    const int hi = g.end(d) - 1;
    for (int layer = 1; layer <= g.ng[d]; ++layer)
    {
        const int srcLo = at(ghostSource(bc.lower[d], true, layer, lo, hi));
        w.store(
            at(lo - layer),
            ghostState(w.load(srcLo), bc.lower[d], d, distance, bc.jet));
        const int srcHi = at(ghostSource(bc.upper[d], false, layer, lo, hi));
        w.store(
            at(hi + layer),
            ghostState(w.load(srcHi), bc.upper[d], d, distance, bc.jet));
    }
}

// ----------------------------------------------------------------- fluxes --

// Relative pressure jumps that count as a shock: hybrid HLLC uses HLL above
// 10 % (as the face-normal guard in hllcFlux); PPM is flattened only at
// strong shocks, above 50 %. Both values were chosen by measurement.
constexpr Real kHllcShockJump = 0.1;
constexpr Real kPpmShockJump = 0.5;

// Multi-dimensional shock detector: the pressure of cell idx differs by more
// than `jump` (relative) from a neighbour in any direction.
RELJET_HD inline bool
nearShock(const PrimView& w, const Grid& g, int idx, Real jump)
{
    const Real p = w.load(idx).p;
    for (int d = 0; d < kDims; ++d)
    {
        if (g.collapsed(d))
            continue;
        for (int side = -1; side <= 1; side += 2)
        {
            const Real pn = w.load(idx + side * g.stride(d)).p;
            if (std::fabs(pn - p) > jump * std::fmin(p, pn))
                return true;
        }
    }
    return false;
}

// Reconstructed state at the lower (upperFace = false) or upper face of the
// cell at cellIdx along the direction with neighbour distance `stride`.
// PPM cells next to a strong shock use MUSCL-MC instead (multidimensional
// flattening). Inadmissible states fall back to the cell average and are
// counted.
RELJET_HD inline Prim faceState(
    const PrimView& w,
    const SolverOptions& opt,
    int cellIdx,
    int stride,
    bool upperFace,
    bool flatten,
    int& fallbacks)
{
    const Prim w0 = w.load(cellIdx);
    FaceStates f{};
    switch (opt.reconstruction)
    {
        case Reconstruction::FirstOrder:
            return w0;
        case Reconstruction::Muscl:
            f = reconstructMuscl(
                w.load(cellIdx - stride),
                w0,
                w.load(cellIdx + stride),
                opt.limiter);
            break;
        case Reconstruction::Ppm:
            if (flatten)
            {
                f = reconstructMuscl(
                    w.load(cellIdx - stride),
                    w0,
                    w.load(cellIdx + stride),
                    Limiter::MonotonizedCentral);
                break;
            }
            f = ppmReconstruct(
                w.load(cellIdx - 2 * stride),
                w.load(cellIdx - stride),
                w0,
                w.load(cellIdx + stride),
                w.load(cellIdx + 2 * stride));
            break;
    }
    const Prim& face = upperFace ? f.plus : f.minus;
    if (isAdmissible(face))
    {
        return face;
    }
    ++fallbacks;
    return w0;
}

// Numerical flux through the lower face (along d) of cell (i, ., .) at idx. Radial
// faces of axisymmetric runs are weighted by the face radius (ring area).
RELJET_HD inline Flux faceFlux(
    const PrimView& w,
    const Grid& g,
    const IdealGasEos& eos,
    const SolverOptions& opt,
    int d,
    int i,
    int idx,
    int& fallbacks)
{
    const int stride = g.stride(d);
    const bool ppm = opt.reconstruction == Reconstruction::Ppm;
    const bool flattenL = ppm && nearShock(w, g, idx - stride, kPpmShockJump);
    const bool flattenR = ppm && nearShock(w, g, idx, kPpmShockJump);
    const Prim wL =
        faceState(w, opt, idx - stride, stride, true, flattenL, fallbacks);
    const Prim wR = faceState(w, opt, idx, stride, false, flattenR, fallbacks);

    // HLLC only away from shocks in any direction: the face-normal guard in
    // hllcFlux cannot see a shock running along the face.
    const bool hllc = opt.flux == FluxScheme::Hllc &&
                      !nearShock(w, g, idx - stride, kHllcShockJump) &&
                      !nearShock(w, g, idx, kHllcShockJump);
    const FluxScheme scheme = hllc ? FluxScheme::Hllc : FluxScheme::Hll;
    const Flux f = numericalFlux(
        wL, primToCons(wL, eos), wR, primToCons(wR, eos), eos, d, scheme);
    if (opt.geometry == Geometry::Cylindrical && d == 0)
    {
        return faceRadius(g, i) * f;
    }
    return f;
}

// rhs -= (F(upper face) - F(lower face)) / dx, or / (r dr) for rings.
// flux holds the lower-face flux of each cell; the upper face of this cell
// is the lower face of the next one.
RELJET_HD inline void addFluxDivergence(
    const ConsView& rhs,
    const ConsView& flux,
    const Grid& g,
    const SolverOptions& opt,
    int d,
    int i,
    int idx)
{
    const bool radial = opt.geometry == Geometry::Cylindrical && d == 0;
    const Real scale = radial ? 1 / (cellRadius(g, i) * g.dx[0]) : 1 / g.dx[d];
    Cons r = rhs.load(idx);
    r -= (flux.load(idx + g.stride(d)) - flux.load(idx)) * scale;
    rhs.store(idx, r);
}

// Geometric source of axisymmetric runs (Geometry.H).
RELJET_HD inline void addCylindricalSource(
    const ConsView& rhs,
    const PrimView& w,
    const Grid& g,
    const IdealGasEos& eos,
    int i,
    int idx)
{
    const Prim p = w.load(idx);
    Cons r = rhs.load(idx);
    r += cylindricalSource(p, primToCons(p, eos), cellRadius(g, i));
    rhs.store(idx, r);
}

// -------------------------------------------------- time integration (SSP-RK2) --

// Stage 1: u0 = U^n, u = U^n + dt L(U^n).
RELJET_HD inline void rkStage1(
    const ConsView& u,
    const ConsView& u0,
    const ConsView& rhs,
    Real dt,
    int idx)
{
    const Cons un = u.load(idx);
    u0.store(idx, un);
    u.store(idx, un + dt * rhs.load(idx));
}

// Stage 2: U^{n+1} = 1/2 U^n + 1/2 (U1 + dt L(U1)), with u = U1 on entry.
RELJET_HD inline void rkStage2(
    const ConsView& u,
    const ConsView& u0,
    const ConsView& rhs,
    Real dt,
    int idx)
{
    u.store(
        idx,
        Real(0.5) * u0.load(idx) +
            Real(0.5) * (u.load(idx) + dt * rhs.load(idx)));
}

} // namespace kernels
} // namespace reljet

#endif

// ************************************************************************* //
