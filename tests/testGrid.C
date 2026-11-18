/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.
\*---------------------------------------------------------------------------*/

#include "Fields.H"
#include "Grid.H"
#include "TestFramework.H"

using namespace reljet;

RELJET_TEST(grid_1d_layout)
{
    const Grid g = makeGrid(10, 1, 0.0, 1.0, 0.0, 1.0, 2);
    CHECK(g.collapsed(1));
    CHECK(g.padded(0) == 14);
    CHECK(g.numCells() == 14);
    CHECK(g.begin(0) == 2 && g.end(0) == 12);
    CHECK_NEAR(g.dx[0], 0.1, 1e-15);
    CHECK_NEAR(g.center(0, g.begin(0)), 0.05, 1e-15);
    CHECK_NEAR(g.center(0, g.end(0) - 1), 0.95, 1e-15);
    CHECK_NEAR(g.center(0, 0), -0.15, 1e-15); // ghost cell
}

RELJET_TEST(grid_2d_layout_is_dir0_contiguous)
{
    const Grid g = makeGrid(4, 3, 0.0, 1.0, -1.0, 2.0, 2);
    CHECK(!g.collapsed(1));
    CHECK(g.padded(0) == 8 && g.padded(1) == 7);
    CHECK(g.index(1, 0) - g.index(0, 0) == 1);
    CHECK(g.index(0, 1) - g.index(0, 0) == g.padded(0));
    CHECK_THROWS(makeGrid(0, 1, 0.0, 1.0, 0.0, 1.0, 2));
    CHECK_THROWS(makeGrid(4, 1, 1.0, 0.0, 0.0, 1.0, 2));
}

RELJET_TEST(fields_round_trip_soa)
{
    const Grid g = makeGrid(5, 1, 0.0, 1.0, 0.0, 1.0, 2);
    ConsField u(g);
    const Cons a{1.0, {2.0, 3.0}, 4.0, 5.0};
    u.store(3, a);
    const Cons b = u.load(3);
    CHECK(
        b.D == 1.0 && b.S[0] == 2.0 && b.S[1] == 3.0 && b.tau == 4.0 &&
        b.DC == 5.0);
    // SoA: the tau component of cell 3 sits in the contiguous tau array.
    CHECK(u.component(iTau)[3] == 4.0);
}

// ************************************************************************* //
