/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.
\*---------------------------------------------------------------------------*/

#include <cstdio>
#include <cstring>

#include "TestFramework.H"

int main(int argc, char** argv)
{
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int run = 0;
    int failed = 0;
    for (const auto& t : reljet::test::registry())
    {
        if (filter && !std::strstr(t.name, filter))
            continue;
        reljet::test::Context ctx;
        try
        {
            t.fn(ctx);
        }
        catch (const std::exception& e)
        {
            ctx.check(false, e.what(), t.name, 0);
        }
        ++run;
        if (ctx.failures())
            ++failed;
        std::printf("[%s] %s\n", ctx.failures() ? "FAIL" : " ok ", t.name);
    }
    std::printf("\n%d/%d tests passed\n", run - failed, run);
    return failed ? 1 : 0;
}

// ************************************************************************* //
