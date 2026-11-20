/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Convergence study on the smooth density wave (SmoothWave.H).
    Prints the L1 error of rho and the observed order for each scheme.

    Usage: convergenceStudy [maxCells]   (default 1024)
\*---------------------------------------------------------------------------*/

#include <cstdio>
#include <cstdlib>
#include <exception>

#include "Norms.H"
#include "SmoothWave.H"

using namespace reljet;

int main(int argc, char** argv)
{
    const int maxCells = argc > 1 ? std::atoi(argv[1]) : 1024;
    const SmoothWave wave;

    struct Scheme
    {
        const char* name;
        Reconstruction rec;
        Limiter lim;
    };
    const Scheme schemes[] = {
        {"firstOrder", Reconstruction::FirstOrder, Limiter::Minmod},
        {"MUSCL minmod", Reconstruction::Muscl, Limiter::Minmod},
        {"MUSCL MC", Reconstruction::Muscl, Limiter::MonotonizedCentral},
        {"PPM", Reconstruction::Ppm, Limiter::MonotonizedCentral}};

    try
    {
        std::printf(
            "Smooth density wave: rho = %g + %g sin(2 pi (x - %g t)), p = %g, "
            "gamma = %g, t = %g, cfl = 0.4\n\n",
            wave.rho0,
            wave.amp,
            wave.v0,
            wave.p0,
            wave.eos.gamma,
            wave.tEnd);
        for (const Scheme& s : schemes)
        {
            std::printf(
                "%-14s %8s %14s %8s\n", s.name, "cells", "L1(rho)", "order");
            Real prevErr = 0;
            for (int n = 32; n <= maxCells; n *= 2)
            {
                const PrimField w = runSmoothWave(wave, n, s.rec, s.lim);
                const Real err = l1Error(
                    w,
                    iRho,
                    [&](Real x)
                    { return wave.exactRhoAverage(x, 1.0 / n, wave.tEnd); });
                if (prevErr > 0)
                    std::printf(
                        "%-14s %8d %14.4e %8.3f\n",
                        "",
                        n,
                        err,
                        convergenceOrder(prevErr, err, 2.0));
                else
                    std::printf("%-14s %8d %14.4e %8s\n", "", n, err, "-");
                prevErr = err;
            }
            std::printf("\n");
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}

// ************************************************************************* //
