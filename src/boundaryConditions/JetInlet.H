/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Ghost state of the jet-inlet boundary: nozzle inflow inside a wall.

    Jet inlet: the boundary face through which the jet enters the domain.

    The face is normal to the jet axis ("axial direction"): z in 2D
    axisymmetric runs (direction 1), z in 3D Cartesian runs (direction 2).
    On that face, ghost cells closer to the axis than the nozzle radius hold
    the jet beam state; all other ghost cells form a solid wall.

             ambient medium (at rest)
      ^ z
      |      |  jet  |
      |      | beam  |
      +======|=======|======   <- inlet face: wall | nozzle | wall
             r < r_jet
\*---------------------------------------------------------------------------*/

#ifndef JetInlet_H
#define JetInlet_H

#include <cmath>

#include "Types.H"

namespace reljet
{

// Plain data for the boundary loop and the CUDA kernels: no
// validation here, that happens in JetParams (Jet.H).
struct JetNozzle
{
    Prim beam{};     // primitive state injected inside the nozzle
    Real radius = 0; // nozzle radius
};

// Ghost state on the inlet face. Ghosts closer to the axis than the nozzle
// radius hold the beam state: the beam is supersonic along the axis, so all
// characteristics enter the domain and all variables are imposed. All other
// ghosts mirror the interior with the axial velocity reversed (solid wall).
RELJET_HD inline Prim jetInletGhost(
    const Prim& mirror, Real distance, const JetNozzle& nozzle, int axialDir)
{
    if (distance < nozzle.radius)
        return nozzle.beam;               // inflow: impose everything
    Prim wall = mirror;                   // copy: `mirror` is const
    wall.v[axialDir] = -wall.v[axialDir]; // no flow through the wall
    return wall;
}

} // namespace reljet

#endif

// ************************************************************************* //
