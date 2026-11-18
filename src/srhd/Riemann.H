/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Relativistic signal speeds and the HLL approximate Riemann solver.

    Approximate Riemann solver: the numerical flux through one face, given the
    states just left (L) and right (R) of it.

    At every face two different states meet. Physically this launches waves
    (shocks, rarefactions, a contact) that carry information left and right.
    The finite-volume update needs the flux that passes through the face while
    those waves play out. Solving this exactly is expensive in SRHD, so we use
    HLL: it keeps only the fastest left-going and right-going waves and assumes
    a single averaged state between them.

      lambda_L <= 0 <= lambda_R region diagram (x-t plane, face at x = 0):

               lambda_L          lambda_R
                  \      U_hll      /
                   \               /
            U_L     \             /     U_R
      ---------------\-----+-----/---------------  t = 0
                         face
\*---------------------------------------------------------------------------*/

#ifndef Riemann_H
#define Riemann_H

#include <cmath>
#include <limits>

#include "IdealGasEos.H"
#include "Srhd.H"
#include "Types.H"

namespace reljet
{

struct SignalSpeeds
{
    Real minus; // slowest (most left-going) characteristic speed
    Real plus;  // fastest (most right-going) characteristic speed
};

// Characteristic speeds in direction `dir` (Marti & Mueller 2003, Living
// Rev. Relativ. 6, 7):
//
//              v_d (1 - c^2)  +-  c sqrt( (1 - v^2) (1 - v^2 c^2 - v_d^2 (1 - c^2)) )
//   lambda± = -----------------------------------------------------------------------
//                                       1 - v^2 c^2
//
// with c = c_s from the EOS and v^2 = |v|^2 over all components. For a
// velocity purely along dir this is the relativistic velocity addition
// (v ± c) / (1 ± v c).
RELJET_HD inline SignalSpeeds
signalSpeeds(const Prim& w, const IdealGasEos& eos, int dir)
{
    const Real c2 = eos.soundSpeedSquared(w.rho, w.p);
    const Real v2 = velocitySquared(w);
    const Real vd = w.v[dir];

    const Real center = vd * (1 - c2);
    const Real spread =
        std::sqrt(c2 * (1 - v2) * (1 - v2 * c2 - vd * vd * (1 - c2)));
    const Real denom = 1 - v2 * c2;
    return SignalSpeeds{(center - spread) / denom, (center + spread) / denom};
}

// HLL flux (Harten, Lax & van Leer 1983):
//
//          lamR F_L  -  lamL F_R  +  lamL lamR (U_R - U_L)
//    F  =  -------------------------------------------------
//                          lamR - lamL
//
// with lamL = min(0, lambda-_L, lambda-_R) and lamR = max(0, lambda+_L,
// lambda+_R). Clamping with 0 makes one formula cover supersonic flow in
// either direction (F = F_L or F_R, pure upwinding). For U_L = U_R the flux
// reduces to the physical flux (consistency).
RELJET_HD inline Flux hllFlux(
    const Prim& wL,
    const Cons& uL,
    const Prim& wR,
    const Cons& uR,
    const IdealGasEos& eos,
    int dir)
{
    const SignalSpeeds sigSpeed_L = signalSpeeds(wL, eos, dir);
    const SignalSpeeds sigSpeed_R = signalSpeeds(wR, eos, dir);
    const Real lamL =
        std::fmin(Real(0), std::fmin(sigSpeed_L.minus, sigSpeed_R.minus));
    const Real lamR =
        std::fmax(Real(0), std::fmax(sigSpeed_L.plus, sigSpeed_R.plus));
    const Flux F_L = physicalFlux(wL, uL, dir);
    const Flux F_R = physicalFlux(wR, uR, dir);

    // HLL formula for one component; lamL, lamR are captured, the rest varies.
    auto hll = [&](Real fl, Real fr, Real ul, Real ur)
    {
        return (lamR * fl - lamL * fr + lamL * lamR * (ur - ul)) /
               (lamR - lamL);
    };

    Flux f{};
    f.D = hll(F_L.D, F_R.D, uL.D, uR.D);
    for (int d = 0; d < kDims; ++d)
        f.S[d] = hll(F_L.S[d], F_R.S[d], uL.S[d], uR.S[d]);
    f.tau = hll(F_L.tau, F_R.tau, uL.tau, uR.tau);
    f.DC = hll(F_L.DC, F_R.DC, uL.DC, uR.DC);
    return f;
}

} // namespace reljet

#endif

// ************************************************************************* //
