/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Arithmetic operators on conserved states and fluxes.

    Lets the solver write its formulas per state ("U1 = U + dt L") instead of
    per component. RELJET_HD and inline: they compile to the same six scalar
    operations as hand-written loops, on the host and in kernels.

    Flux is the same type as Cons (Types.H), so these work on fluxes too.
\*---------------------------------------------------------------------------*/

#ifndef ConsOps_H
#define ConsOps_H

#include "Types.H"

namespace reljet
{

// Free functions, so that both `dt * u` and `u * dt` work; binary operators
// return a new state, compound ones modify their left operand.


RELJET_HD inline Cons operator+(const Cons& a, const Cons& b)
{
    Cons r;
    r.D = a.D + b.D;
    for (int d = 0; d < kDims; ++d)
        r.S[d] = a.S[d] + b.S[d];
    r.tau = a.tau + b.tau;
    r.DC = a.DC + b.DC;
    return r;
}

RELJET_HD inline Cons operator-(const Cons& a, const Cons& b)
{
    Cons r;
    r.D = a.D - b.D;
    for (int d = 0; d < kDims; ++d)
        r.S[d] = a.S[d] - b.S[d];
    r.tau = a.tau - b.tau;
    r.DC = a.DC - b.DC;
    return r;
}

RELJET_HD inline Cons operator*(Real s, const Cons& a)
{
    Cons r;
    r.D = s * a.D;
    for (int d = 0; d < kDims; ++d)
        r.S[d] = s * a.S[d];
    r.tau = s * a.tau;
    r.DC = s * a.DC;
    return r;
}

RELJET_HD inline Cons operator*(const Cons& a, Real s)
{
    return s * a;
}

RELJET_HD inline Cons& operator+=(Cons& a, const Cons& b)
{
    a.D += b.D;
    for (int d = 0; d < kDims; ++d)
        a.S[d] += b.S[d];
    a.tau += b.tau;
    a.DC += b.DC;
    return a; // the reference to the object we just modified
}

RELJET_HD inline Cons& operator-=(Cons& a, const Cons& b)
{
    a.D -= b.D;
    for (int d = 0; d < kDims; ++d)
        a.S[d] -= b.S[d];
    a.tau -= b.tau;
    a.DC -= b.DC;
    return a;
}

} // namespace reljet

#endif

// ************************************************************************* //
