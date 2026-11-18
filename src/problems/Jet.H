/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Physical parameters, states and head-speed estimate of a pressure-matched jet.

    Physical parameters of a pressure-matched relativistic jet, stated the way
    jet papers state them (e.g. Marti et al. 1997, ApJ 479, 151):

      W     beam Lorentz factor
      eta   density contrast rho_beam / rho_ambient   (light jets: eta << 1)
      rhoA  ambient density
      p     pressure, the same in beam and ambient ("pressure matched")
      rJet  beam radius (the natural length unit of the problem)

    The ambient medium is at rest. The beam moves along the axial direction.

SourceFiles
    Jet.C
\*---------------------------------------------------------------------------*/

#ifndef Jet_H
#define Jet_H

#include "IdealGasEos.H"
#include "JetInlet.H"
#include "Types.H"

namespace reljet
{

// JetParams holds a validated parameter set: the constructor throws
// std::invalid_argument unless W > 1, eta, rhoA, p, rJet > 0 and the EOS is
// causal, and the data cannot change afterwards. Code that receives a
// JetParams never needs to re-check it.
//
// headSpeedEstimate() is the one-dimensional momentum balance at the jet head
// (Marti et al. 1997), in the frame of the head:
//
//   eta_R  = (rho_b h_b W^2) / (rho_a h_a)
//   v_head = v_beam sqrt(eta_R) / (1 + sqrt(eta_R))
class JetParams
{
public:
    explicit JetParams(
        Real lorentzFactor,
        Real densityContrast,
        Real ambientDensity,
        Real pressure,
        Real jetRadius,
        const IdealGasEos& eos);

    Real lorentzFactor() const;
    Real densityContrast() const;
    Real pressure() const;
    Real radius() const;
    Real beamVelocity() const;

    Prim beamState(int axialDir) const;
    Prim ambientState() const;
    JetNozzle nozzle(int axialDir) const;
    Real headSpeedEstimate() const;

private:
    Real W_;
    Real eta_;
    Real rhoA_;
    Real p_;
    Real rJet_;
    IdealGasEos eos_;
    Real vBeam_; // derived in the constructor
};

} // namespace reljet

#endif

// ************************************************************************* //
