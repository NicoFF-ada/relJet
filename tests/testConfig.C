/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Dictionary parsing: entries, comments, sub-dictionaries, errors.
\*---------------------------------------------------------------------------*/

#include "Dictionary.H"
#include "TestFramework.H"

using namespace reljet;

RELJET_TEST(dictionary_parses_entries_comments_and_subdicts)
{
    const Dictionary dict = Dictionary::parse(
        "FoamFile { version 2.0; object controlDict; }\n"
        "// line comment\n"
        "gamma   1.6667;   /* block\n comment */\n"
        "nCells 400;\n"
        "limiter MC;\n"
        "left\n{\n    rho 10;\n    p   13.3;\n}\n",
        "test");
    CHECK_NEAR(dict.get<Real>("gamma"), 1.6667, 0.0);
    CHECK(dict.get<int>("nCells") == 400);
    CHECK(dict.get<std::string>("limiter") == "MC");
    CHECK_NEAR(dict.subDict("left").get<Real>("rho"), 10.0, 0.0);
    CHECK_NEAR(dict.get<Real>("left.p"), 13.3, 0.0);
    CHECK(dict.unusedKeys().empty());
}

RELJET_TEST(dictionary_reports_unused_and_rejects_bad_input)
{
    const Dictionary dict = Dictionary::parse(
        "maxCo 0.4;\nmaxCO 0.9;\nnCells 12abc;\ntypo 1;\n", "test");
    CHECK_NEAR(dict.getInRange("maxCo", 0.0, 1.0), 0.4, 0.0);
    CHECK_THROWS(dict.get<int>("nCells"));
    CHECK_THROWS(dict.get<Real>("missing"));
    CHECK_THROWS(dict.getInRange("maxCO", 0.0, 0.5));
    CHECK_NEAR(dict.getOrDefault<Real>("absent", 2.0), 2.0, 0.0);
    CHECK(dict.unusedKeys().size() == 1); // only "typo" was never read
    CHECK_THROWS(Dictionary::parse("a 1;\na 2;\n", "test"));
    CHECK_THROWS(Dictionary::parse("a 1\nb 2;\n{", "test"));
    CHECK_THROWS(Dictionary::parse("sub { a 1; ", "test"));
}

RELJET_TEST(dictionary_merge_rejects_duplicates_across_files)
{
    Dictionary all("case");
    all.merge(Dictionary::parse("FoamFile { object a; }\nx 1;\n", "a"));
    all.merge(Dictionary::parse("FoamFile { object b; }\ny 2;\n", "b"));
    CHECK(all.get<int>("x") == 1 && all.get<int>("y") == 2);
    CHECK_THROWS(all.merge(Dictionary::parse("x 3;\n", "c")));
}

// ************************************************************************* //
