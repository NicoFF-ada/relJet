/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Gamma-law ideal-gas equation of state.
\*---------------------------------------------------------------------------*/

#include "IdealGasEos.H"

#include <stdexcept>
#include <string>

namespace reljet
{

void validateIdealGasEos(const IdealGasEos& eos)
{
    if (!(eos.gamma > 1 && eos.gamma <= 2))
    {
        throw std::invalid_argument(
            "ideal-gas gamma must satisfy 1 < gamma <= 2 (causality), got " +
            std::to_string(eos.gamma));
    }
}

} // namespace reljet

// ************************************************************************* //
