/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Choice of the approximate Riemann solver, read from fvSchemes:

        flux    HLL;     // HLL | HLLC
\*---------------------------------------------------------------------------*/

#ifndef NumericalFlux_H
#define NumericalFlux_H

#include <cmath>
#include <limits>
#include <string>

#include "Hllc.H"
#include "Riemann.H"

namespace reljet
{

enum class FluxScheme
{
    Hll,
    Hllc
};

// "HLL" or "HLLC"; throws std::invalid_argument otherwise.
FluxScheme parseFluxScheme(const std::string& name);

// Flux dispatch, in the same style as limitedSlope: a switch over an enum
// class with no default branch, so -Wswitch flags a missing case.
RELJET_HD inline Flux numericalFlux(
    const Prim& wL,
    const Cons& uL,
    const Prim& wR,
    const Cons& uR,
    const IdealGasEos& eos,
    int dir,
    FluxScheme scheme)
{
    switch (scheme)
    {
        case FluxScheme::Hll:
            return hllFlux(wL, uL, wR, uR, eos, dir);
        case FluxScheme::Hllc:
            return hllcFlux(wL, uL, wR, uR, eos, dir);
    }
    const Real nan = std::numeric_limits<Real>::quiet_NaN();
    return Flux{nan, {nan, nan, nan}, nan, nan}; // out-of-range enum value
}

} // namespace reljet

#endif

// ************************************************************************* //
