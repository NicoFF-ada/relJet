/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Error norms and observed convergence order.

    Verification asks: does the code solve the equations it claims to solve, as
    accurately as the method promises? For a smooth solution, a scheme of order
    k has error E(dx) ~ K dx^k. Halving dx should divide the error by 2^k:
    about 2 for first order, about 4 for second order. Measuring that ratio is
    one of the strictest tests a CFD code can pass: a bug in almost any part of
    the scheme spoils the order, even when the solution "looks right".
\*---------------------------------------------------------------------------*/

#ifndef Norms_H
#define Norms_H

#include <cmath>
#include <limits>

#include "Fields.H"
#include "Types.H"

namespace reljet
{

// Discrete L1 error of one primitive component on a 1D grid (row j =
// begin(1)):  E = sum_i | q_i - q_exact(x_i) | dx.
//
// q_i is a cell average and q_exact(x_i) a point value; they differ by
// O(dx^2), which does not affect orders up to two. `exact` is any callable
// Real -> Real (typically a lambda), taken as a template parameter so the
// call inlines.
template<class ExactFn>
Real l1Error(const PrimField& w, int component, const ExactFn& exact)
{
    const Grid& g = w.grid();
    const int j = g.begin(1);
    Real sum = 0;
    for (int i = g.begin(0); i < g.end(0); ++i)
    {
        const Real q = w.at(component, g.index(i, j));
        sum += std::fabs(q - exact(g.center(0, i))) * g.dx[0];
    }
    return sum;
}

// Observed order of accuracy from two grids with dx_coarse = r dx_fine:
//   k = log(E_coarse / E_fine) / log(r).  NaN unless both errors are positive
// and r > 1.
inline Real convergenceOrder(Real errCoarse, Real errFine, Real refinement)
{
    if (!(errCoarse > 0 && errFine > 0 && refinement > 1))
        return std::numeric_limits<Real>::quiet_NaN();
    return std::log(errCoarse / errFine) / std::log(refinement);
}

} // namespace reljet

#endif

// ************************************************************************* //
