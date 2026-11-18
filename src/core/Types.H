/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Basic types: Real, the primitive and conserved states, component indices.

    RELJET_HD marks small pointwise functions that must be callable from both
    host code and CUDA kernels. Under a plain C++ compiler it is empty.
\*---------------------------------------------------------------------------*/

#ifndef Types_H
#define Types_H

#if defined(__CUDACC__)
#define RELJET_HD __host__ __device__
#else
#define RELJET_HD
#endif

namespace reljet
{

using Real = double;

// Number of spatial directions and velocity components. Direction 0 is x
// (Cartesian) or r (axisymmetric), direction 1 is y or z, direction 2 is z or
// phi. Lower-dimensional runs collapse unused directions (one cell, no ghosts)
// and keep the corresponding velocity components at 0.
constexpr int kDims = 3;

// Component order inside the structure-of-arrays fields. Keep these in sync
// with the member order of Cons / Prim below.
enum ConsVar : int
{
    iD = 0,
    iS0 = 1,
    iS1 = 2,
    iS2 = 3,
    iTau = 4,
    iDC = 5,
    kNumCons = 6
};
enum PrimVar : int
{
    iRho = 0,
    iP = 1,
    iV0 = 2,
    iV1 = 3,
    iV2 = 4,
    iC = 5,
    kNumPrim = 6
};

// Primitive state of one cell: what a physicist "measures" in the fluid
// rest frame (rho, p) plus the lab-frame three-velocity and a passive tracer.
struct Prim
{
    Real rho;      // proper rest-mass density
    Real p;        // pressure
    Real v[kDims]; // three-velocity in units of c
    Real C;        // passive scalar: jet-material fraction in [0, 1]
};

// Conserved state of one cell: densities in the lab frame that obey
// d/dt U + div F(U) = S. A face flux has exactly the same layout.
struct Cons
{
    Real D;        // rest-mass density           rho W
    Real S[kDims]; // momentum density            rho h W^2 v_i
    Real tau;      // energy minus rest mass      rho h W^2 - p - D
    Real DC;       // tracer mass density         D C
};
using Flux = Cons;

} // namespace reljet

#endif

// ************************************************************************* //
