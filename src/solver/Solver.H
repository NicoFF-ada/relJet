/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    CPU finite-volume SRHD solver: MUSCL-HLL, SSP-RK2, OpenMP.

    Finite-volume SRHD solver on 1D, 2D or 3D grids (Cartesian), or 2D
    axisymmetric (r, z).

    Semi-discrete form, Cartesian: for each cell,

      dU/dt = L(U) = - sum_d ( F_d(upper face) - F_d(lower face) ) / dx_d

    summed over the active directions d ("unsplit": all directions use the same
    state, no dimensional splitting). F_d is the HLL flux between reconstructed
    states. Every face flux is added to one cell and subtracted from its
    neighbour, so totals change only through the domain boundaries.

    Axisymmetric (Geometry::Cylindrical; direction 0 = r, 1 = z, 2 collapsed):
    a cell is a ring, so the r-faces have areas proportional to r:

      dU/dt = - ( r+ F+ - r- F- ) / (r dr)  -  ( F_z+ - F_z- ) / dz  +  S_geom

    with r+ / r- the face radii, r the cell-centre radius, and S_geom from
    cylindricalSource() (Geometry.H).

    One evaluation of L(U):
      1. consToPrim on every interior cell (fails loudly, see below)
      2. fill ghost cells of the primitive field (boundary conditions)
      3. per active direction: reconstruct face states, HLL flux at every face,
         accumulate the flux difference
      4. geometric source terms (cylindrical only)

    Time integration: SSP-RK2 (Shu & Osher 1988):
      U1      = U^n + dt L(U^n)
      U^{n+1} = 1/2 U^n + 1/2 ( U1 + dt L(U1) )

    Time step (unsplit multi-D CFL condition):
      dt = cfl / max_cells( sum_d max(|lambda_d-|, |lambda_d+|) / dx_d )

    Failure policy: if conserved-to-primitive recovery fails in any cell, the
    step throws std::runtime_error naming the cell and status. Nothing is
    silently patched. Inadmissible reconstructed face states are replaced by
    the cell average and counted in StepDiagnostics::reconstructionFallbacks.

    Parallelism: cell and face loops use OpenMP when built with it, over the
    two outer directions (k, j). Each loop writes only its own cell / face, and
    counters use OpenMP reductions, so results do not depend on the number of
    threads. Small grids (and 1D, where the outer loops have one iteration)
    stay serial: starting a thread team costs more than the work.

SourceFiles
    Solver.C
\*---------------------------------------------------------------------------*/

#ifndef Solver_H
#define Solver_H

#include <string>

#include "Fields.H"
#include "IdealGasEos.H"
#include "Kernels.H"

namespace reljet
{

// Throws std::invalid_argument for invalid CFL, c2p options or boundaries.
void validateSolverOptions(const SolverOptions& opt);

// Full setup check shared by all backends: EOS, options, and their
// compatibility with the grid (ghost layers for MUSCL, axis and
// dimensionality for cylindrical runs). Throws std::invalid_argument.
void validateSolverSetup(
    const Grid& grid, const IdealGasEos& eos, const SolverOptions& opt);

// Text of the exception thrown when primitive recovery fails at padded index
// idx: status, interior cell indices and coordinates.
std::string c2pFailureMessage(const Grid& grid, int idx, C2PStatus status);

struct StepDiagnostics
{
    long reconstructionFallbacks =
        0; // face states replaced by first order (cumulative)
    int maxC2PIterations = 0; // worst cell in the last recovery
};

class Solver
{
public:
    Solver(const Grid& grid, const IdealGasEos& eos, const SolverOptions& opt);

    // Sets the state from primitive values in the interior cells of `w`.
    void initialize(const PrimField& w);

    // Largest stable step for the current state.
    Real computeTimeStep() const;

    // Advances the conserved state by dt with SSP-RK2.
    void step(Real dt);

    const Grid& grid() const { return grid_; }
    const ConsField& cons() const { return u_; }
    const PrimField& prim() const { return w_; }
    const StepDiagnostics& diagnostics() const { return diag_; }

private:
    void recoverPrimitives(
        const ConsField& u); // u -> w_ (interior), throws on failure
    void computeRhs(
        const ConsField& u, ConsField& rhs); // L(u), uses and overwrites w_

    static constexpr long kMinCellsForThreads = 16384;

    Grid grid_;
    IdealGasEos eos_;
    SolverOptions opt_;
    bool parallel_; // use OpenMP threads for this grid

    // All work arrays are allocated once here and reused every step.
    ConsField u_;   // current state U^n
    ConsField u0_;  // copy of U^n during an RK step
    ConsField rhs_; // L(U)
    ConsField
        flux_; // flux through the LOWER face of each cell, one direction at a time
    PrimField w_; // primitives, including ghost cells
    StepDiagnostics diag_;
};

} // namespace reljet

#endif

// ************************************************************************* //
