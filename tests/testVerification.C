/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Error norms (Norms.H) and the measured order of accuracy of the solver on
    the smooth density wave.
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Norms.H"
#include "SmoothWave.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{

// Field on [0, 2] with rho = x at cell centres and p = 1 everywhere.
PrimField linearField(int n)
{
    const Grid g = makeGrid(n, 1, 0.0, 2.0, 0.0, 1.0, 2);
    PrimField w(g);
    for (int i = 0; i < g.padded(0); ++i)
        w.store(g.index(i, 0), Prim{g.center(0, i), 1.0, {0, 0}, 0});
    return w;
}

} // namespace

RELJET_TEST(norm_l1_zero_for_exact_field)
{
    const PrimField w = linearField(10);
    CHECK_NEAR(l1Error(w, iRho, [](Real x) { return x; }), 0.0, 1e-15);
}

RELJET_TEST(norm_l1_constant_offset_and_component)
{
    // |error| = 0.25 everywhere on a domain of length 2  =>  E_1 = 0.5.
    const PrimField w = linearField(10);
    CHECK_NEAR(l1Error(w, iRho, [](Real x) { return x + 0.25; }), 0.5, 1e-14);
    CHECK_NEAR(l1Error(w, iRho, [](Real x) { return x - 0.25; }), 0.5, 1e-14);
    CHECK_NEAR(
        l1Error(w, iP, [](Real) { return 3.0; }),
        4.0,
        1e-14); // picks the right component
}

RELJET_TEST(norm_l1_ignores_ghost_cells)
{
    PrimField w = linearField(10);
    w.at(iRho, 0) = 1e6; // a ghost cell
    CHECK_NEAR(l1Error(w, iRho, [](Real x) { return x; }), 0.0, 1e-15);
}

RELJET_TEST(norm_convergence_order)
{
    CHECK_NEAR(convergenceOrder(4e-3, 1e-3, 2.0), 2.0, 1e-14);
    CHECK_NEAR(convergenceOrder(2e-3, 1e-3, 2.0), 1.0, 1e-14);
    CHECK_NEAR(convergenceOrder(9.0, 1.0, 3.0), 2.0, 1e-14);
    CHECK(std::isnan(convergenceOrder(0.0, 1e-3, 2.0)));
    CHECK(std::isnan(convergenceOrder(1e-3, 1e-3, 1.0)));
}

RELJET_TEST(verify_smooth_wave_order_of_accuracy)
{
    // Observed order between 64 and 128 cells on the smooth density wave.
    const SmoothWave wave;
    auto order = [&](Reconstruction rec, Limiter lim)
    {
        auto err = [&](int n)
        {
            return l1Error(
                runSmoothWave(wave, n, rec, lim),
                iRho,
                [&](Real x) { return wave.exactRho(x, wave.tEnd); });
        };
        return convergenceOrder(err(64), err(128), 2.0);
    };
    const Real first = order(Reconstruction::FirstOrder, Limiter::Minmod);
    const Real mc = order(Reconstruction::Muscl, Limiter::MonotonizedCentral);
    CHECK(first > 0.8 && first < 1.2);
    CHECK(mc > 1.8);
}

// ************************************************************************* //
