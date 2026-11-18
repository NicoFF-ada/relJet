/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    CUDA backend of the finite-volume SRHD solver.
\*---------------------------------------------------------------------------*/

#include "CudaSolver.H"

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <vector>

#include "Kernels.H"

namespace reljet
{

namespace
{

void check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("CUDA: ") + what + ": " + cudaGetErrorString(err));
}

void checkLaunch(const char* kernel)
{
    check(cudaGetLastError(), kernel);
}

// Owning device array; move-only.
template<class T>
class DeviceArray
{
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t n) : n_(n)
    {
        check(cudaMalloc(&ptr_, n * sizeof(T)), "cudaMalloc");
    }
    ~DeviceArray() { cudaFree(ptr_); }
    DeviceArray(DeviceArray&& o) noexcept : ptr_(o.ptr_), n_(o.n_)
    {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }
    DeviceArray& operator=(DeviceArray&& o) noexcept
    {
        std::swap(ptr_, o.ptr_);
        std::swap(n_, o.n_);
        return *this;
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    T* get() const { return ptr_; }
    std::size_t size() const { return n_; }

    void upload(const T* host, std::size_t n)
    {
        check(
            cudaMemcpy(ptr_, host, n * sizeof(T), cudaMemcpyHostToDevice),
            "upload");
    }
    void download(T* host, std::size_t n) const
    {
        check(
            cudaMemcpy(host, ptr_, n * sizeof(T), cudaMemcpyDeviceToHost),
            "download");
    }

private:
    T* ptr_ = nullptr;
    std::size_t n_ = 0;
};

constexpr int kBlock = 256;

int blocksFor(int n)
{
    return (n + kBlock - 1) / kBlock;
}

// Linear thread index; false if past the end of the work range.
__device__ inline bool threadIndex(int count, int& n)
{
    n = blockIdx.x * blockDim.x + threadIdx.x;
    return n < count;
}

// ---------------------------------------------------------------- kernels --

__global__ void zeroKernel(ConsView rhs, Grid g, Box b)
{
    int n, i, j, k;
    if (!threadIndex(b.size(), n))
        return;
    b.decode(n, i, j, k);
    rhs.store(g.index(i, j, k), Cons{});
}

__global__ void ghostKernel(PrimView w, Grid g, BoundaryConditions bc, int d)
{
    int column;
    if (!threadIndex(kernels::ghostColumns(g, d), column))
        return;
    kernels::fillGhostColumn(w, g, bc, d, column);
}

__global__ void faceFluxKernel(
    PrimView w,
    ConsView flux,
    Grid g,
    IdealGasEos eos,
    SolverOptions opt,
    int d,
    Box faces,
    unsigned long long* fallbacks)
{
    int n, i, j, k;
    if (!threadIndex(faces.size(), n))
        return;
    faces.decode(n, i, j, k);
    const int idx = g.index(i, j, k);
    int count = 0;
    flux.store(idx, kernels::faceFlux(w, g, eos, opt, d, i, idx, count));
    if (count)
        atomicAdd(fallbacks, static_cast<unsigned long long>(count));
}

__global__ void fluxDivergenceKernel(
    ConsView rhs, ConsView flux, Grid g, SolverOptions opt, int d, Box b)
{
    int n, i, j, k;
    if (!threadIndex(b.size(), n))
        return;
    b.decode(n, i, j, k);
    kernels::addFluxDivergence(rhs, flux, g, opt, d, i, g.index(i, j, k));
}

__global__ void cylindricalSourceKernel(
    ConsView rhs, PrimView w, Grid g, IdealGasEos eos, Box b)
{
    int n, i, j, k;
    if (!threadIndex(b.size(), n))
        return;
    b.decode(n, i, j, k);
    kernels::addCylindricalSource(rhs, w, g, eos, i, g.index(i, j, k));
}

__global__ void
rkStage1Kernel(ConsView u, ConsView u0, ConsView rhs, Real dt, Grid g, Box b)
{
    int n, i, j, k;
    if (!threadIndex(b.size(), n))
        return;
    b.decode(n, i, j, k);
    kernels::rkStage1(u, u0, rhs, dt, g.index(i, j, k));
}

__global__ void
rkStage2Kernel(ConsView u, ConsView u0, ConsView rhs, Real dt, Grid g, Box b)
{
    int n, i, j, k;
    if (!threadIndex(b.size(), n))
        return;
    b.decode(n, i, j, k);
    kernels::rkStage2(u, u0, rhs, dt, g.index(i, j, k));
}

// Primitive recovery. Failing cells are not written; the smallest failing
// index is recorded (deterministic) so the host can report it.
__global__ void recoverKernel(
    ConsView u,
    PrimView w,
    Grid g,
    IdealGasEos eos,
    C2POptions opt,
    Box b,
    int* firstFailure,
    int* maxIter)
{
    __shared__ int blockIter[kBlock];
    int n, i, j, k;
    int iterations = 0;
    if (threadIndex(b.size(), n))
    {
        b.decode(n, i, j, k);
        const int idx = g.index(i, j, k);
        const C2PResult r = consToPrim(u.load(idx), eos, opt);
        if (r.status == C2PStatus::Success)
        {
            w.store(idx, r.w);
            iterations = r.iterations;
        }
        else
        {
            atomicMin(firstFailure, idx);
        }
    }
    blockIter[threadIdx.x] = iterations;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
            blockIter[threadIdx.x] =
                max(blockIter[threadIdx.x], blockIter[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        atomicMax(maxIter, blockIter[0]);
}

// Per-block maximum of the CFL signal rate; the host reduces the blocks.
__global__ void
signalRateKernel(PrimView w, Grid g, IdealGasEos eos, Box b, Real* blockMax)
{
    __shared__ Real rate[kBlock];
    int n, i, j, k;
    Real r = 0;
    if (threadIndex(b.size(), n))
    {
        b.decode(n, i, j, k);
        r = kernels::signalRate(w, g, eos, g.index(i, j, k));
    }
    rate[threadIdx.x] = r;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        if (threadIdx.x < s)
            rate[threadIdx.x] = fmax(rate[threadIdx.x], rate[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        blockMax[blockIdx.x] = rate[0];
}

} // namespace

// ------------------------------------------------------------------- Impl --

struct CudaSolver::Impl
{
    Impl(const Grid& grid, const IdealGasEos& eos, const SolverOptions& opt)
        : grid(grid),
          eos(eos),
          opt(opt),
          interior(interiorBox(grid)),
          u(std::size_t(kNumCons) * grid.numCells()),
          u0(u.size()),
          rhs(u.size()),
          flux(u.size()),
          w(std::size_t(kNumPrim) * grid.numCells()),
          blockMax(blocksFor(interior.size())),
          c2pCounters(2),
          fallbackCounter(1),
          hostW(grid),
          hostU(grid)
    {
        hostBlockMax.resize(blockMax.size());
    }

    ConsView view(const DeviceArray<Real>& a) const
    {
        return ConsView{a.get(), grid.numCells()};
    }
    PrimView primView() const { return PrimView{w.get(), grid.numCells()}; }

    int* firstFailure() const { return c2pCounters.get(); }
    int* maxIter() const { return c2pCounters.get() + 1; }
    unsigned long long* fallbacks() const { return fallbackCounter.get(); }

    void applyBoundaries()
    {
        for (int d = 0; d < kDims; ++d)
        {
            if (grid.collapsed(d))
                continue;
            ghostKernel<<<blocksFor(kernels::ghostColumns(grid, d)), kBlock>>>(
                primView(), grid, opt.bc, d);
            checkLaunch("ghostKernel");
        }
    }

    void recoverPrimitives(const DeviceArray<Real>& state)
    {
        const int init[2] = {INT_MAX, 0};
        c2pCounters.upload(init, 2);
        recoverKernel<<<blocksFor(interior.size()), kBlock>>>(
            view(state),
            primView(),
            grid,
            eos,
            opt.c2p,
            interior,
            firstFailure(),
            maxIter());
        checkLaunch("recoverKernel");
        int result[2];
        c2pCounters.download(result, 2);
        diag.maxC2PIterations = result[1];
        if (result[0] != INT_MAX)
        {
            Real values[kNumCons];
            for (int c = 0; c < kNumCons; ++c)
                check(
                    cudaMemcpy(
                        &values[c],
                        state.get() + std::size_t(c) * grid.numCells() +
                            result[0],
                        sizeof(Real),
                        cudaMemcpyDeviceToHost),
                    "read failing cell");
            const Cons bad = ConsView{values, 1}.load(0);
            throw std::runtime_error(c2pFailureMessage(
                grid, result[0], consToPrim(bad, eos, opt.c2p).status));
        }
    }

    void computeRhs(const DeviceArray<Real>& state)
    {
        recoverPrimitives(state);
        applyBoundaries();
        const int cells = blocksFor(interior.size());
        zeroKernel<<<cells, kBlock>>>(view(rhs), grid, interior);
        checkLaunch("zeroKernel");
        for (int d = 0; d < kDims; ++d)
        {
            if (grid.collapsed(d))
                continue;
            const Box faces = faceBox(grid, d);
            faceFluxKernel<<<blocksFor(faces.size()), kBlock>>>(
                primView(), view(flux), grid, eos, opt, d, faces, fallbacks());
            checkLaunch("faceFluxKernel");
            fluxDivergenceKernel<<<cells, kBlock>>>(
                view(rhs), view(flux), grid, opt, d, interior);
            checkLaunch("fluxDivergenceKernel");
        }
        if (opt.geometry == Geometry::Cylindrical)
        {
            cylindricalSourceKernel<<<cells, kBlock>>>(
                view(rhs), primView(), grid, eos, interior);
            checkLaunch("cylindricalSourceKernel");
        }
    }

    Grid grid;
    IdealGasEos eos;
    SolverOptions opt;
    Box interior;
    DeviceArray<Real> u, u0, rhs, flux, w;
    DeviceArray<Real> blockMax;
    DeviceArray<int> c2pCounters; // first failing index, max iterations
    DeviceArray<unsigned long long>
        fallbackCounter; // reconstruction fallbacks (cumulative)
    std::vector<Real> hostBlockMax;
    PrimField hostW;
    ConsField hostU;
    bool hostWCurrent = false;
    bool hostUCurrent = false;
    StepDiagnostics diag;
};

// ------------------------------------------------------------- CudaSolver --

CudaSolver::CudaSolver(
    const Grid& grid, const IdealGasEos& eos, const SolverOptions& opt)
{
    validateSolverSetup(grid, eos, opt);
    impl_ = std::make_unique<Impl>(grid, eos, opt);
    const unsigned long long zero = 0;
    impl_->fallbackCounter.upload(&zero, 1);
}

CudaSolver::~CudaSolver() = default;
CudaSolver::CudaSolver(CudaSolver&&) noexcept = default;
CudaSolver& CudaSolver::operator=(CudaSolver&&) noexcept = default;

void CudaSolver::initialize(const PrimField& w)
{
    Impl& s = *impl_;
    // Validation and prim -> cons conversion on the host (once), shared with
    // the CPU solver; then upload both fields.
    Solver reference(s.grid, s.eos, s.opt);
    reference.initialize(w);
    s.w.upload(reference.prim().data(), reference.prim().size());
    s.u.upload(reference.cons().data(), reference.cons().size());
    s.hostWCurrent = s.hostUCurrent = false;
}

Real CudaSolver::computeTimeStep()
{
    Impl& s = *impl_;
    signalRateKernel<<<int(s.blockMax.size()), kBlock>>>(
        s.primView(), s.grid, s.eos, s.interior, s.blockMax.get());
    checkLaunch("signalRateKernel");
    s.blockMax.download(s.hostBlockMax.data(), s.hostBlockMax.size());
    const Real maxRate =
        *std::max_element(s.hostBlockMax.begin(), s.hostBlockMax.end());
    return s.opt.cfl / maxRate;
}

void CudaSolver::step(Real dt)
{
    if (!(dt > 0) || !std::isfinite(dt))
        throw std::invalid_argument("solver: dt must be positive and finite");
    Impl& s = *impl_;
    const int cells = blocksFor(s.interior.size());

    s.computeRhs(s.u);
    rkStage1Kernel<<<cells, kBlock>>>(
        s.view(s.u), s.view(s.u0), s.view(s.rhs), dt, s.grid, s.interior);
    checkLaunch("rkStage1Kernel");

    s.computeRhs(s.u);
    rkStage2Kernel<<<cells, kBlock>>>(
        s.view(s.u), s.view(s.u0), s.view(s.rhs), dt, s.grid, s.interior);
    checkLaunch("rkStage2Kernel");

    s.recoverPrimitives(s.u);
    s.applyBoundaries();

    unsigned long long fallbacks = 0;
    s.fallbackCounter.download(&fallbacks, 1);
    s.diag.reconstructionFallbacks = static_cast<long>(fallbacks);
    s.hostWCurrent = s.hostUCurrent = false;
}

const Grid& CudaSolver::grid() const
{
    return impl_->grid;
}

const PrimField& CudaSolver::prim()
{
    Impl& s = *impl_;
    if (!s.hostWCurrent)
        s.w.download(s.hostW.data(), s.hostW.size());
    s.hostWCurrent = true;
    return s.hostW;
}

const ConsField& CudaSolver::cons()
{
    Impl& s = *impl_;
    if (!s.hostUCurrent)
        s.u.download(s.hostU.data(), s.hostU.size());
    s.hostUCurrent = true;
    return s.hostU;
}

const StepDiagnostics& CudaSolver::diagnostics() const
{
    return impl_->diag;
}

std::string CudaSolver::deviceName()
{
    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    return std::string(prop.name) + " (sm_" + std::to_string(prop.major) +
           std::to_string(prop.minor) + ")";
}

} // namespace reljet

// ************************************************************************* //
