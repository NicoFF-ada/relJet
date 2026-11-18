/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Class
    reljet::Dictionary

Description
    Run-time input in OpenFOAM dictionary syntax:

        keyword    value;      // line comment
        subDict
        {
            keyword    value;
        }

    C-style block comments are accepted as well. Entries of sub-dictionaries
    are addressed as "subDict.keyword" or through subDict(). A FoamFile header
    block is accepted and ignored.

    Every entry that is read is recorded, so unusedKeys() can report typos
    instead of silently running with a default.

SourceFiles
    Dictionary.C
\*---------------------------------------------------------------------------*/

#ifndef Dictionary_H
#define Dictionary_H

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Types.H"

namespace reljet
{
/*---------------------------------------------------------------------------*\
                          Class Dictionary Declaration
\*---------------------------------------------------------------------------*/

class Dictionary
{
public:
    // Constructors

    //- Empty dictionary
    explicit Dictionary(const std::string& name = "");

    //- Parse dictionary text; `name` is used in error messages
    static Dictionary parse(const std::string& text, const std::string& name);

    //- Read a dictionary file; throws if it cannot be opened
    static Dictionary read(const std::string& path);


    // Member Functions

    //- Name of the dictionary (file path, or path/subDict)
    const std::string& name() const { return name_; }

    //- Add all entries of another dictionary; duplicates throw
    void merge(const Dictionary& other);

    //- True if the keyword exists
    bool found(const std::string& key) const;

    //- Sub-dictionary view; reads through it are recorded here too
    Dictionary subDict(const std::string& key) const;

    //- Required entries; throw std::runtime_error if missing or invalid
    template<class T>
    T get(const std::string& key) const;

    //- Optional entries with a default
    template<class T>
    T getOrDefault(const std::string& key, const T& fallback) const
    {
        return found(key) ? get<T>(key) : fallback;
    }

    //- Required real entry that must lie in [lo, hi]
    Real getInRange(const std::string& key, Real lo, Real hi) const;

    //- Entries that were never read, with their source dictionary
    std::vector<std::string> unusedKeys() const;


private:
    struct Entry
    {
        std::string value;
        std::string source;
    };

    const Entry& entry(const std::string& key) const;

    std::string name_;
    std::string prefix_; // "subDict." for sub-dictionary views
    std::shared_ptr<std::map<std::string, Entry>> entries_;
    std::shared_ptr<std::set<std::string>> used_;
};


template<>
Real Dictionary::get<Real>(const std::string& key) const;

template<>
int Dictionary::get<int>(const std::string& key) const;

template<>
std::string Dictionary::get<std::string>(const std::string& key) const;

} // namespace reljet

#endif

// ************************************************************************* //
