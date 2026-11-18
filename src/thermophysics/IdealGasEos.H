/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Gamma-law ideal-gas equation of state.

SourceFiles
    IdealGasEos.C
\*---------------------------------------------------------------------------*/

#ifndef IdealGasEos_H
#define IdealGasEos_H

#include "Types.H"

namespace reljet
{

// Gamma-law ideal gas: p = (gamma - 1) rho eps.
//
// Everything the solver needs from an EOS goes through this small interface
// (eps, h, c_s^2), so a Taub-Matthews EOS can be dropped in later with the
// same member functions.
//
// Causality: c_s^2 = gamma p / (rho h) approaches (gamma - 1) as p/rho -> inf,
// so gamma <= 2 guarantees c_s < 1. validateIdealGasEos() enforces this.
struct IdealGasEos
{
    Real gamma;

    RELJET_HD Real pressure(Real rho, Real eps) const
    {
        return (gamma - 1) * rho * eps;
    }

    RELJET_HD Real specificInternalEnergy(Real rho, Real p) const
    {
        return p / ((gamma - 1) * rho);
    }

    // h = 1 + eps + p/rho = 1 + gamma/(gamma-1) * p/rho
    RELJET_HD Real specificEnthalpy(Real rho, Real p) const
    {
        return 1 + gamma / (gamma - 1) * p / rho;
    }

    // Relativistic sound speed squared, c_s^2 = gamma p / (rho h).
    RELJET_HD Real soundSpeedSquared(Real rho, Real p) const
    {
        return gamma * p / (rho * specificEnthalpy(rho, p));
    }
};

// Throws std::invalid_argument unless 1 < gamma <= 2.
void validateIdealGasEos(const IdealGasEos& eos);

} // namespace reljet

#endif

// ************************************************************************* //
