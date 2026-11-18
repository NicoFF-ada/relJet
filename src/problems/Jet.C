/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Physical parameters, states and head-speed estimate of a pressure-matched jet.
\*---------------------------------------------------------------------------*/

#include "Jet.H"

#include <cmath>
#include <stdexcept>
#include <string>

namespace reljet
{

JetParams::JetParams(
    Real lorentzFactor,
    Real densityContrast,
    Real ambientDensity,
    Real pressure,
    Real jetRadius,
    const IdealGasEos& eos)
    : W_(lorentzFactor),
      eta_(densityContrast),
      rhoA_(ambientDensity),
      p_(pressure),
      rJet_(jetRadius),
      eos_(eos),
      vBeam_(0)
{
    if (!(W_ > 1))
        throw std::invalid_argument(
            "jet: Lorentz factor must be > 1, got " + std::to_string(W_));
    if (!(eta_ > 0))
        throw std::invalid_argument(
            "jet: density contrast must be > 0, got " + std::to_string(eta_));
    if (!(rhoA_ > 0))
        throw std::invalid_argument(
            "jet: ambient density must be > 0, got " + std::to_string(rhoA_));
    if (!(p_ > 0))
        throw std::invalid_argument(
            "jet: pressure must be > 0, got " + std::to_string(p_));
    if (!(rJet_ > 0))
        throw std::invalid_argument(
            "jet: radius must be > 0, got " + std::to_string(rJet_));

    validateIdealGasEos(eos_);

    vBeam_ = std::sqrt(1 - 1 / (W_ * W_));
}

Real JetParams::lorentzFactor() const
{
    return W_;
}
Real JetParams::densityContrast() const
{
    return eta_;
}
Real JetParams::pressure() const
{
    return p_;
}
Real JetParams::radius() const
{
    return rJet_;
}
Real JetParams::beamVelocity() const
{
    return vBeam_;
}

Prim JetParams::ambientState() const
{
    Prim a{};
    a.rho = rhoA_;
    a.p = p_;
    return a;
}

Prim JetParams::beamState(int axialDir) const
{
    Prim b{};
    b.rho = eta_ * rhoA_;
    b.p = p_;
    b.v[axialDir] = vBeam_;
    b.C = 1;
    return b;
}

JetNozzle JetParams::nozzle(int axialDir) const
{
    return JetNozzle{beamState(axialDir), rJet_};
}

Real JetParams::headSpeedEstimate() const
{
    const Real rhoB = eta_ * rhoA_;
    const Real hB = eos_.specificEnthalpy(rhoB, p_);
    const Real hA = eos_.specificEnthalpy(rhoA_, p_);
    const Real etaR = rhoB * hB * W_ * W_ / (rhoA_ * hA);
    const Real s = std::sqrt(etaR);
    return vBeam_ * s / (1 + s);
}

} // namespace reljet

// ************************************************************************* //
