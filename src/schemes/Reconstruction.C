/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Slope limiters and MUSCL reconstruction of primitive variables.
\*---------------------------------------------------------------------------*/

#include "Reconstruction.H"

#include <stdexcept>

namespace reljet
{

Reconstruction parseReconstruction(const std::string& name)
{
    if (name == "firstOrder")
        return Reconstruction::FirstOrder;
    if (name == "MUSCL")
        return Reconstruction::Muscl;
    if (name == "PPM")
        return Reconstruction::Ppm;
    throw std::invalid_argument(
        "unknown reconstruction '" + name +
        "' (expected firstOrder | MUSCL | PPM)");
}

Limiter parseLimiter(const std::string& name)
{
    if (name == "minmod")
        return Limiter::Minmod;
    if (name == "MC")
        return Limiter::MonotonizedCentral;
    throw std::invalid_argument(
        "unknown limiter '" + name + "' (expected minmod | MC)");
}

} // namespace reljet

// ************************************************************************* //
