/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Smooth density-wave verification problem with an exact solution.
\*---------------------------------------------------------------------------*/

#include "SmoothWave.H"

#include <cmath>

namespace reljet
{

Real SmoothWave::exactRho(Real x, Real t) const
{
    const Real pi = std::acos(Real(-1));
    return rho0 + amp * std::sin(2 * pi * (x - v0 * t));
}

PrimField runSmoothWave(
    const SmoothWave& wave, int n, Reconstruction rec, Limiter lim, Real cfl)
{
    const Grid grid = makeGrid(n, 1, 0.0, 1.0, 0.0, 1.0, 2);

    SolverOptions opt;
    opt.cfl = cfl;
    opt.reconstruction = rec;
    opt.limiter = lim;
    opt.bc = BoundaryConditions{
        {BoundaryType::Periodic, BoundaryType::Outflow},
        {BoundaryType::Periodic, BoundaryType::Outflow}};

    // Initialise with the exact cell AVERAGE of rho (analytic integral of the
    // sine over the cell), so the initial data carry no O(dx^2) sampling error.
    PrimField init(grid);
    const Real pi = std::acos(Real(-1));
    const Real dx = grid.dx[0];
    for (int i = grid.begin(0); i < grid.end(0); ++i)
    {
        const Real xc = grid.center(0, i);
        const Real avgFactor =
            std::sin(pi * dx) /
            (pi * dx); // <sin(2 pi x)> over a cell = factor * sin(2 pi xc)
        const Real rho =
            wave.rho0 + wave.amp * avgFactor * std::sin(2 * pi * xc);
        init.store(grid.index(i, 0), Prim{rho, wave.p0, {wave.v0, 0}, 0});
    }

    Solver solver(grid, wave.eos, opt);
    solver.initialize(init);
    Real t = 0;
    while (t < wave.tEnd)
    {
        const Real dt = std::fmin(solver.computeTimeStep(), wave.tEnd - t);
        solver.step(dt);
        t += dt;
    }
    return solver.prim();
}

} // namespace reljet

// ************************************************************************* //
