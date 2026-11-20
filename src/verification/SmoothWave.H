/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Smooth density-wave verification problem with an exact solution.

    With uniform pressure p0 and velocity v0, any density profile is an exact
    solution of SRHD: nothing pushes on anything, so the profile just moves:

        rho(x, t) = rho0 + amp * sin(2 pi (x - v0 t)),   p = p0,   v = v0.

    (This is a pure "entropy" or contact wave.) On a periodic domain [0, 1] the
    exact solution is known at all times, which makes it the standard test for
    order of accuracy. The default t_end = 1 moves the wave by v0 = 0.5, i.e.
    half a period: a solver that did nothing at all would show a LARGE error,
    whereas after a whole period it would show none.

SourceFiles
    SmoothWave.C
\*---------------------------------------------------------------------------*/

#ifndef SmoothWave_H
#define SmoothWave_H

#include "IdealGasEos.H"
#include "Solver.H"

namespace reljet
{

struct SmoothWave
{
    Real rho0 = 1.0;
    Real amp = 0.5;
    Real p0 = 1.0;
    Real v0 =
        0.5; // W = 1.155: mildly relativistic, all relativistic terms active
    Real tEnd = 1.0;
    IdealGasEos eos{5.0 / 3.0};

    Real exactRho(Real x, Real t) const;

    // Exact average of rho over the cell [xc - dx/2, xc + dx/2]. Finite-volume
    // schemes evolve averages, so errors are measured against this.
    Real exactRhoAverage(Real xc, Real dx, Real t) const;
};

// Runs the wave on n periodic cells to tEnd with the given scheme and returns
// the final primitive field.
PrimField runSmoothWave(
    const SmoothWave& wave,
    int n,
    Reconstruction rec,
    Limiter lim,
    Real cfl = 0.4);

} // namespace reljet

#endif

// ************************************************************************* //
