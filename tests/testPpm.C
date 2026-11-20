/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    PPM reconstruction (Reconstruction.H).
\*---------------------------------------------------------------------------*/

#include <cmath>

#include "Solver.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{

// A smooth, monotone cubic, gentle enough that the MC limiter stays
// inactive, and its exact average over [a, b].
Real cubic(Real x)
{
    return 1 + 0.3 * x + 0.05 * x * x + 0.01 * x * x * x;
}

Real cubicAverage(Real a, Real b)
{
    auto integral = [](Real x)
    {
        return x + 0.15 * x * x + 0.05 / 3 * x * x * x + 0.0025 * x * x * x * x;
    };
    return (integral(b) - integral(a)) / (b - a);
}

// Value of the cell parabola at xi in [0, 1].
Real parabola(Real qL, Real qR, Real q6, Real xi)
{
    return qL + xi * (qR - qL + q6 * (1 - xi));
}

} // namespace

// ------------------------------------------------------------- face value --

RELJET_TEST(ppm_face_value_is_exact_for_cubics)
{
    // Cells of width 1 around the face x = 0: [-2,-1], [-1,0], [0,1], [1,2].
    const Real face = ppmFaceValue(
        cubicAverage(-2, -1),
        cubicAverage(-1, 0),
        cubicAverage(0, 1),
        cubicAverage(1, 2));
    CHECK_NEAR(face, cubic(0), 1e-14);
}

RELJET_TEST(ppm_face_value_stays_between_its_neighbours_at_a_jump)
{
    const Real face = ppmFaceValue(0, 0, 1, 1);
    CHECK(face >= 0 && face <= 1);
    CHECK_NEAR(ppmFaceValue(0, 0, 0, 1), 0.0, 0.0); // flat side of a jump
}

// -------------------------------------------------------------- parabola --

RELJET_TEST(ppm_parabola_keeps_the_cell_average)
{
    // Simpson's rule is exact for a parabola.
    const Real q0 = 0.7, qL = 0.2, qR = 1.5;
    const Real q6 = ppmCurvature(q0, qL, qR);
    const Real average =
        (parabola(qL, qR, q6, 0) + 4 * parabola(qL, qR, q6, 0.5) +
         parabola(qL, qR, q6, 1)) /
        6;
    CHECK_NEAR(average, q0, 1e-15);
    CHECK_NEAR(parabola(qL, qR, q6, 0), qL, 1e-15);
    CHECK_NEAR(parabola(qL, qR, q6, 1), qR, 1e-15);
}

// --------------------------------------------------------------- limiter --

RELJET_TEST(ppm_monotone_parabola_is_unchanged)
{
    const FaceValues f = ppmLimit(0.5, 0.4, 0.6);
    CHECK(f.minus == 0.4 && f.plus == 0.6);
}

RELJET_TEST(ppm_local_extremum_is_flattened)
{
    const FaceValues f = ppmLimit(1.0, 0.5, 0.8); // q0 above both faces
    CHECK(f.minus == 1.0 && f.plus == 1.0);
}

RELJET_TEST(ppm_overshoot_inside_the_cell_is_removed)
{
    // Average close to qR: the unlimited parabola rises above qR inside.
    for (const Real q0 : {0.9, 0.1})
    {
        const FaceValues f = ppmLimit(q0, 0.0, 1.0);
        const Real dq = f.plus - f.minus;
        const Real q6 = ppmCurvature(q0, f.minus, f.plus);
        // Monotone: q'(xi) = dq + q6 (1 - 2 xi) keeps the sign of dq.
        CHECK(dq * (dq + q6) >= -1e-15);
        CHECK(dq * (dq - q6) >= -1e-15);
        CHECK(f.minus >= 0 && f.plus <= 1);
    }
}

// ------------------------------------------------------- reconstruction --

RELJET_TEST(ppm_reconstructs_every_component)
{
    // Linear data in every component: PPM reproduces lines exactly.
    auto state = [](Real i)
    {
        return Prim{
            1 + 0.1 * i,
            2 + 0.05 * i,
            {0.1 + 0.01 * i, -0.2 + 0.02 * i, 0.03 * i},
            0.5 + 0.1 * i};
    };
    const FaceStates f =
        ppmReconstruct(state(-2), state(-1), state(0), state(1), state(2));
    const Prim lo = state(-0.5), hi = state(0.5);
    CHECK_NEAR(f.minus.rho, lo.rho, 1e-15);
    CHECK_NEAR(f.plus.rho, hi.rho, 1e-15);
    CHECK_NEAR(f.minus.p, lo.p, 1e-15);
    CHECK_NEAR(f.plus.p, hi.p, 1e-15);
    for (int d = 0; d < kDims; ++d)
    {
        CHECK_NEAR(f.minus.v[d], lo.v[d], 1e-15);
        CHECK_NEAR(f.plus.v[d], hi.v[d], 1e-15);
    }
    CHECK_NEAR(f.minus.C, lo.C, 1e-15);
    CHECK_NEAR(f.plus.C, hi.C, 1e-15);
}

// ---------------------------------------------------------------- wiring --

RELJET_TEST(ppm_option_and_ghost_layers)
{
    CHECK(parseReconstruction("PPM") == Reconstruction::Ppm);
    CHECK(requiredGhostLayers(Reconstruction::Ppm) == 3);
    CHECK(requiredGhostLayers(Reconstruction::Muscl) == 2);

    SolverOptions opt;
    opt.reconstruction = Reconstruction::Ppm;
    opt.bc.lower[0] = opt.bc.upper[0] = BoundaryType::Outflow;
    const IdealGasEos eos{5.0 / 3.0};
    CHECK_THROWS(Solver(makeGrid(16, 1, 0.0, 1.0, 0.0, 1.0, 2), eos, opt));
    Solver ok(makeGrid(16, 1, 0.0, 1.0, 0.0, 1.0, 3), eos, opt);
}
