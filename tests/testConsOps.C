/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Arithmetic operators on Cons (ConsOps.H).
\*---------------------------------------------------------------------------*/

#include "ConsOps.H"
#include "TestFramework.H"

using namespace reljet;

namespace
{

const Cons kA{1.0, {2.0, 3.0, 4.0}, 5.0, 6.0};
const Cons kB{0.5, {-1.0, 0.25, 8.0}, 2.0, -3.0};

bool equal(const Cons& u, const Cons& v)
{
    return u.D == v.D && u.S[0] == v.S[0] && u.S[1] == v.S[1] &&
           u.S[2] == v.S[2] && u.tau == v.tau && u.DC == v.DC;
}

} // namespace

RELJET_TEST(cons_ops_add)
{
    CHECK(equal(kA + kB, Cons{1.5, {1.0, 3.25, 12.0}, 7.0, 3.0}));
}

RELJET_TEST(cons_ops_subtract)
{
    CHECK(equal(kA - kB, Cons{0.5, {3.0, 2.75, -4.0}, 3.0, 9.0}));
}

RELJET_TEST(cons_ops_scale_both_orders)
{
    const Cons expected{2.0, {4.0, 6.0, 8.0}, 10.0, 12.0};
    CHECK(equal(2.0 * kA, expected));
    CHECK(equal(kA * 2.0, expected));
}

RELJET_TEST(cons_ops_compound_assignment)
{
    Cons u = kA;
    u += kB;
    CHECK(equal(u, kA + kB));
    u -= kB;
    CHECK(equal(u, kA));
    Cons v = kA;
    (v -= kB) -= kB; // chaining needs the returned reference
    CHECK(equal(v, kA - kB - kB));
}

RELJET_TEST(cons_ops_do_not_modify_arguments)
{
    const Cons a = kA, b = kB;
    const Cons r = a - b * 3.0;
    CHECK(equal(a, kA) && equal(b, kB));
    CHECK(equal(r, Cons{-0.5, {5.0, 2.25, -20.0}, -1.0, 15.0}));
}

// ************************************************************************* //
