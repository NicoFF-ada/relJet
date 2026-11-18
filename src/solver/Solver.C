/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    CPU finite-volume SRHD solver: MUSCL-HLL, SSP-RK2, OpenMP.
\*---------------------------------------------------------------------------*/

#include "Solver.H"

#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>
#include <string>

namespace reljet
{

namespace
{

const char* toString(C2PStatus s)
{
    switch (s)
    {
        case C2PStatus::Success:
            return "success";
        case C2PStatus::NotConverged:
            return "not converged";
        case C2PStatus::InvalidInput:
            return "invalid input";
    }
    return "unknown";
}

} // namespace

void validateSolverOptions(const SolverOptions& opt)
{
    if (!(opt.cfl > 0 && opt.cfl < 1))
        throw std::invalid_argument("solver: cfl must satisfy 0 < cfl < 1");
    if (opt.c2p.maxIter < 1 || !(opt.c2p.relTol > 0))
        throw std::invalid_argument("solver: invalid c2p options");
    validateBoundaries(opt.bc);
}

void validateSolverSetup(
    const Grid& grid, const IdealGasEos& eos, const SolverOptions& opt)
{
    validateIdealGasEos(eos);
    validateSolverOptions(opt);
    for (int d = 0; d < kDims; ++d)
    {
        if (opt.reconstruction == Reconstruction::Muscl && !grid.collapsed(d) &&
            grid.ng[d] < 2)
            throw std::invalid_argument(
                "solver: MUSCL needs at least 2 ghost layers");
    }
    if (opt.geometry == Geometry::Cylindrical)
    {
        if (!grid.collapsed(2))
            throw std::invalid_argument(
                "solver: cylindrical geometry is 2D (r, z); direction 2 must "
                "be collapsed");
        if (grid.lo[0] < 0)
            throw std::invalid_argument(
                "solver: cylindrical geometry needs r >= 0");
        if (grid.lo[0] == 0 && opt.bc.lower[0] != BoundaryType::Reflecting)
            throw std::invalid_argument(
                "solver: the axis r = 0 needs a reflecting boundary");
    }
}

std::string c2pFailureMessage(const Grid& g, int idx, C2PStatus status)
{
    const int i = idx % g.padded(0);
    const int j = (idx / g.padded(0)) % g.padded(1);
    const int k = idx / (g.padded(0) * g.padded(1));
    return "consToPrim failed (" + std::string(toString(status)) +
           ") at cell (" + std::to_string(i - g.begin(0)) + ", " +
           std::to_string(j - g.begin(1)) + ", " +
           std::to_string(k - g.begin(2)) + "), x = (" +
           std::to_string(g.center(0, i)) + ", " +
           std::to_string(g.center(1, j)) + ", " +
           std::to_string(g.center(2, k)) + ")";
}

Solver::Solver(
    const Grid& grid, const IdealGasEos& eos, const SolverOptions& opt)
    : grid_(grid),
      eos_(eos),
      opt_(opt),
      parallel_(
          grid.numInteriorCells() >= kMinCellsForThreads &&
          grid.n[1] * grid.n[2] > 1),
      u_(grid),
      u0_(grid),
      rhs_(grid),
      flux_(grid),
      w_(grid)
{
    validateSolverSetup(grid_, eos_, opt_);
}

void Solver::initialize(const PrimField& w)
{
    const Box b = interiorBox(grid_);
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
        for (int j = b.lo[1]; j < b.hi[1]; ++j)
            for (int i = b.lo[0]; i < b.hi[0]; ++i)
            {
                const int idx = grid_.index(i, j, k);
                const Prim p = w.load(idx);
                if (!isAdmissible(p))
                    throw std::invalid_argument(
                        "solver: inadmissible initial state at cell (" +
                        std::to_string(i) + ", " + std::to_string(j) + ", " +
                        std::to_string(k) + ")");
                w_.store(idx, p);
                u_.store(idx, primToCons(p, eos_));
            }
    applyBoundaries(w_, opt_.bc);
}

Real Solver::computeTimeStep() const
{
    const Box b = interiorBox(grid_);
    const PrimView w{const_cast<Real*>(w_.data()), w_.numCells()};
    Real maxRate = 0;
#pragma omp parallel for collapse(2) if (parallel_) reduction(max : maxRate)
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
        for (int j = b.lo[1]; j < b.hi[1]; ++j)
            for (int i = b.lo[0]; i < b.hi[0]; ++i)
                maxRate = std::max(
                    maxRate,
                    kernels::signalRate(w, grid_, eos_, grid_.index(i, j, k)));
    return opt_.cfl / maxRate;
}

void Solver::recoverPrimitives(const ConsField& u)
{
    const Box b = interiorBox(grid_);
    const PrimView w = w_.view();
    int maxIter = 0;
    int firstFailure =
        INT_MAX; // smallest failing index: deterministic regardless of threads
#pragma omp parallel for collapse(2) if (parallel_) reduction(max              \
                                                              : maxIter)       \
    reduction(min                                                              \
              : firstFailure)
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
        for (int j = b.lo[1]; j < b.hi[1]; ++j)
            for (int i = b.lo[0]; i < b.hi[0]; ++i)
            {
                const int idx = grid_.index(i, j, k);
                const C2PResult r = consToPrim(u.load(idx), eos_, opt_.c2p);
                if (r.status != C2PStatus::Success)
                {
                    firstFailure = std::min(firstFailure, idx);
                    continue;
                }
                maxIter = std::max(maxIter, r.iterations);
                w.store(idx, r.w);
            }
    diag_.maxC2PIterations = maxIter;

    if (firstFailure != INT_MAX)
    { // exceptions must not leave an OpenMP region: report after it
        const C2PResult r = consToPrim(u.load(firstFailure), eos_, opt_.c2p);
        throw std::runtime_error(
            c2pFailureMessage(grid_, firstFailure, r.status));
    }
}

void Solver::computeRhs(const ConsField& u, ConsField& rhs)
{
    recoverPrimitives(u);
    applyBoundaries(w_, opt_.bc);

    const PrimView w = w_.view();
    const ConsView r = rhs.view();
    const ConsView flux = flux_.view();
    const Box b = interiorBox(grid_);

#pragma omp parallel for collapse(2) if (parallel_)
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
        for (int j = b.lo[1]; j < b.hi[1]; ++j)
            for (int i = b.lo[0]; i < b.hi[0]; ++i)
                r.store(grid_.index(i, j, k), Cons{});

    // Unsplit: every active direction adds its flux divergence to the same rhs.
    for (int d = 0; d < kDims; ++d)
    {
        if (grid_.collapsed(d))
            continue;

        const Box faces = faceBox(grid_, d);
        int fallbacks = 0;
#pragma omp parallel for collapse(2) if (parallel_) reduction(+ : fallbacks)
        for (int k = faces.lo[2]; k < faces.hi[2]; ++k)
            for (int j = faces.lo[1]; j < faces.hi[1]; ++j)
                for (int i = faces.lo[0]; i < faces.hi[0]; ++i)
                {
                    const int idx = grid_.index(i, j, k);
                    flux.store(
                        idx,
                        kernels::faceFlux(
                            w, grid_, eos_, opt_, d, i, idx, fallbacks));
                }
        diag_.reconstructionFallbacks += fallbacks;

#pragma omp parallel for collapse(2) if (parallel_)
        for (int k = b.lo[2]; k < b.hi[2]; ++k)
            for (int j = b.lo[1]; j < b.hi[1]; ++j)
                for (int i = b.lo[0]; i < b.hi[0]; ++i)
                    kernels::addFluxDivergence(
                        r, flux, grid_, opt_, d, i, grid_.index(i, j, k));
    }

    if (opt_.geometry == Geometry::Cylindrical)
    {
#pragma omp parallel for collapse(2) if (parallel_)
        for (int k = b.lo[2]; k < b.hi[2]; ++k)
            for (int j = b.lo[1]; j < b.hi[1]; ++j)
                for (int i = b.lo[0]; i < b.hi[0]; ++i)
                    kernels::addCylindricalSource(
                        r, w, grid_, eos_, i, grid_.index(i, j, k));
    }
}

void Solver::step(Real dt)
{
    if (!(dt > 0) || !std::isfinite(dt))
        throw std::invalid_argument("solver: dt must be positive and finite");
    const Box b = interiorBox(grid_);
    const ConsView u = u_.view();
    const ConsView u0 = u0_.view();
    const ConsView rhs = rhs_.view();

    computeRhs(u_, rhs_);
#pragma omp parallel for collapse(2) if (parallel_)
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
        for (int j = b.lo[1]; j < b.hi[1]; ++j)
            for (int i = b.lo[0]; i < b.hi[0]; ++i)
                kernels::rkStage1(u, u0, rhs, dt, grid_.index(i, j, k));

    computeRhs(u_, rhs_);
#pragma omp parallel for collapse(2) if (parallel_)
    for (int k = b.lo[2]; k < b.hi[2]; ++k)
        for (int j = b.lo[1]; j < b.hi[1]; ++j)
            for (int i = b.lo[0]; i < b.hi[0]; ++i)
                kernels::rkStage2(u, u0, rhs, dt, grid_.index(i, j, k));

    // Leave w_ consistent with the new state (for output and the next dt).
    recoverPrimitives(u_);
    applyBoundaries(w_, opt_.bc);
}

} // namespace reljet

// ************************************************************************* //
