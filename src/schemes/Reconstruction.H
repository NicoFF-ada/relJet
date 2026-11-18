/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Slope limiters and MUSCL reconstruction of primitive variables.

    Spatial reconstruction: from cell AVERAGES to values at cell FACES.

    A finite-volume code only knows one number per cell (the average). The
    Riemann solver needs the state just left and right of each face. The
    simplest choice copies the cell average to both faces (first order:
    piecewise constant). MUSCL instead fits a straight line through each cell:

               slope s
      minus  /---------\  plus           face value = w0 -+ s/2   (in units of dx)
      ------+     w0    +------
         i-1/2         i+1/2

    The slope comes from the neighbours, but must be LIMITED: near a shock an
    unlimited slope overshoots and creates new extrema (oscillations, and in
    SRHD easily negative pressure or v > 1). A limiter picks a slope from the
    left difference dl = w0 - wm and the right difference dr = wp - w0 such that
    no new extrema appear (the TVD property).

SourceFiles
    Reconstruction.C
\*---------------------------------------------------------------------------*/

#ifndef Reconstruction_H
#define Reconstruction_H

#include <cmath>
#include <limits>
#include <string>

#include "Srhd.H"
#include "Types.H"

namespace reljet
{

enum class Reconstruction
{
    FirstOrder,
    Muscl
};
enum class Limiter
{
    Minmod,
    MonotonizedCentral
};

// "firstOrder" or "MUSCL"; throws std::invalid_argument otherwise.
Reconstruction parseReconstruction(const std::string& name);

// A primitive state is admissible when rho > 0, p > 0 and v^2 < 1.
RELJET_HD inline bool isAdmissible(const Prim& w)
{
    return w.rho > 0 && w.p > 0 && velocitySquared(w) < 1;
}

struct FaceStates
{
    Prim minus; // value at the cell's lower face (i - 1/2)
    Prim plus;  // value at the cell's upper face (i + 1/2)
};

// Slope limiters, from the left difference dl = w0 - wm and the right
// difference dr = wp - w0:
//
//   minmod(dl, dr)              smaller of |dl|, |dr|, keeping the sign
//   monotonizedCentral(dl, dr)  sign * min(2|dl|, 2|dr|, |dl + dr| / 2)
//
// Both return 0 when dl and dr differ in sign (w0 is a local extremum), which
// is what prevents new extrema (TVD). Minmod is the most diffusive TVD
// limiter; MC keeps contacts sharper at the cost of slight overshoots.
RELJET_HD inline Real minmod(Real dl, Real dr)
{
    if (!(dl * dr > 0))
        return 0; // opposite signs or a zero: w0 is an extremum (or flat)
    return std::fabs(dl) < std::fabs(dr) ? dl : dr;
}

RELJET_HD inline Real monotonizedCentral(Real dl, Real dr)
{
    if (!(dl * dr > 0))
        return 0;
    const Real magnitude = std::fmin(
        2 * std::fmin(std::fabs(dl), std::fabs(dr)), std::fabs(dl + dr) / 2);
    return std::copysign(magnitude, dl);
}

// Limiter dispatch. A switch over an enum class with no default branch, so
// -Wswitch flags a missing case; no virtual calls, so it runs in kernels.
// Out-of-range values return NaN, which the admissibility check rejects.
RELJET_HD inline Real limitedSlope(Real dl, Real dr, Limiter lim)
{
    switch (lim)
    {
        case Limiter::Minmod:
            return minmod(dl, dr);
        case Limiter::MonotonizedCentral:
            return monotonizedCentral(dl, dr);
    }
    return std::numeric_limits<
        Real>::quiet_NaN(); // only reachable with an out-of-range enum value
}

// "minmod" or "MC"; throws std::invalid_argument otherwise.
Limiter parseLimiter(const std::string& name);

// MUSCL reconstruction of one cell: face states from the cell (w0) and its
// neighbours (wm, wp), component by component:
//
//   s = limitedSlope(q0 - qm, qp - q0),   minus = q0 - s/2,   plus = q0 + s/2
//
// Primitive variables are reconstructed (not conserved ones): limiting rho,
// p and v directly keeps the face values bounded by their neighbours, so
// faces are physical far more often. The solver still checks admissibility
// and falls back to the cell average (counted in the diagnostics).
RELJET_HD inline FaceStates
reconstructMuscl(const Prim& wm, const Prim& w0, const Prim& wp, Limiter lim)
{
    FaceStates f{};

    // One component: limited slope from the neighbours, then the two face values.
    // `minus` and `plus` are references, so the lambda writes straight into f.
    auto reconstruct = [&](Real qm, Real q0, Real qp, Real& minus, Real& plus)
    {
        const Real slope = limitedSlope(q0 - qm, qp - q0, lim);
        minus = q0 - slope / 2;
        plus = q0 + slope / 2;
    };

    reconstruct(wm.rho, w0.rho, wp.rho, f.minus.rho, f.plus.rho);
    reconstruct(wm.p, w0.p, wp.p, f.minus.p, f.plus.p);
    for (int d = 0; d < kDims; ++d)
        reconstruct(wm.v[d], w0.v[d], wp.v[d], f.minus.v[d], f.plus.v[d]);
    reconstruct(wm.C, w0.C, wp.C, f.minus.C, f.plus.C);
    return f;
}

} // namespace reljet

#endif

// ************************************************************************* //
