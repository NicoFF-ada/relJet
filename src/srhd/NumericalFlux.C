/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Choice of the approximate Riemann solver.
\*---------------------------------------------------------------------------*/

#include "NumericalFlux.H"

#include <stdexcept>

namespace reljet
{

FluxScheme parseFluxScheme(const std::string& name)
{
    if (name == "HLL")
        return FluxScheme::Hll;
    if (name == "HLLC")
        return FluxScheme::Hllc;
    throw std::invalid_argument(
        "unknown flux '" + name + "' (expected HLL | HLLC)");
}

} // namespace reljet

// ************************************************************************* //
