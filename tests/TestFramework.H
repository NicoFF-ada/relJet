/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Minimal self-registering unit-test framework.

      RELJET_TEST(my_test) { CHECK(1 + 1 == 2); CHECK_NEAR(x, 1.0, 1e-12); }

    Run all tests:          ./relJetTests
    Run matching tests:     ./relJetTests srhd
\*---------------------------------------------------------------------------*/

#ifndef TestFramework_H
#define TestFramework_H

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace reljet::test
{

class Context
{
public:
    void check(bool ok, const char* expr, const char* file, int line)
    {
        if (ok)
            return;
        ++failures_;
        std::printf("    FAIL %s:%d: %s\n", file, line, expr);
    }

    void checkNear(
        double a,
        double b,
        double tol,
        const char* ea,
        const char* eb,
        const char* file,
        int line)
    {
        if (std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol)
            return;
        ++failures_;
        std::printf(
            "    FAIL %s:%d: %s = %.17g, expected %s = %.17g (tol %.3g)\n",
            file,
            line,
            ea,
            a,
            eb,
            b,
            tol);
    }

    int failures() const { return failures_; }

private:
    int failures_ = 0;
};

using TestFn = void (*)(Context&);

struct TestCase
{
    const char* name;
    TestFn fn;
};

inline std::vector<TestCase>& registry()
{
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar
{
    Registrar(const char* name, TestFn fn) { registry().push_back({name, fn}); }
};

} // namespace reljet::test

#define RELJET_TEST(name)                                                      \
    static void name(reljet::test::Context& ctx);                              \
    static const reljet::test::Registrar reljet_registrar_##name(#name, name); \
    static void name([[maybe_unused]] reljet::test::Context& ctx)

#define CHECK(cond) ctx.check((cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol)                                                  \
    ctx.checkNear((a), (b), (tol), #a, #b, __FILE__, __LINE__)
#define CHECK_THROWS(stmt)                                                     \
    do                                                                         \
    {                                                                          \
        bool threw_ = false;                                                   \
        try                                                                    \
        {                                                                      \
            stmt;                                                              \
        }                                                                      \
        catch (...)                                                            \
        {                                                                      \
            threw_ = true;                                                     \
        }                                                                      \
        ctx.check(threw_, "expected exception: " #stmt, __FILE__, __LINE__);   \
    } while (0)

#endif

// ************************************************************************* //
