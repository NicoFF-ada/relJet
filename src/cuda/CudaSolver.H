/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    CUDA backend of the finite-volume SRHD solver.

    CUDA backend: the same SSP-RK2 / MUSCL / HLL scheme as Solver, with all
    fields resident in device memory.

    Every per-cell and per-face operation is the shared code in
    Kernels.H; this class only owns device memory and launches one
    thread per cell, face or ghost column. Host copies of the fields are made
    on demand by prim() / cons() (for output), never inside a step.

    Failure policy is the same as on the CPU: a failed primitive recovery
    throws std::runtime_error with the same message, naming the first failing
    cell.

    This header has no CUDA dependency, so plain C++ code can use the class;
    it is only available when RelJet is built with RELJET_ENABLE_CUDA.

SourceFiles
    CudaSolver.cu
\*---------------------------------------------------------------------------*/

#ifndef CudaSolver_H
#define CudaSolver_H

#include <memory>
#include <string>

#include "Solver.H"

namespace reljet
{

class CudaSolver
{
public:
    CudaSolver(
        const Grid& grid, const IdealGasEos& eos, const SolverOptions& opt);
    ~CudaSolver();
    CudaSolver(CudaSolver&&) noexcept;
    CudaSolver& operator=(CudaSolver&&) noexcept;
    CudaSolver(const CudaSolver&) = delete;
    CudaSolver& operator=(const CudaSolver&) = delete;

    void initialize(const PrimField& w);
    Real computeTimeStep();
    void step(Real dt);

    const Grid& grid() const;
    const PrimField&
    prim(); // device -> host copy if the state changed since the last call
    const ConsField& cons();
    const StepDiagnostics& diagnostics() const;

    // Name and compute capability of the active device, e.g. "Tesla T4 (sm_75)".
    static std::string deviceName();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace reljet

#endif

// ************************************************************************* //
