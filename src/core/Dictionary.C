/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.
\*---------------------------------------------------------------------------*/

#include "Dictionary.H"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace reljet
{

namespace
{

struct Token
{
    std::string text;
    int line;
};

// Splits dictionary text into words and the punctuation { } ;, dropping
// comments.
std::vector<Token> tokenize(const std::string& text, const std::string& name)
{
    std::vector<Token> tokens;
    std::string word;
    int line = 1;
    auto flush = [&]()
    {
        if (!word.empty())
        {
            tokens.push_back({word, line});
            word.clear();
        }
    };

    for (std::size_t i = 0; i < text.size(); ++i)
    {
        const char c = text[i];
        const char next = i + 1 < text.size() ? text[i + 1] : '\0';
        if (c == '/' && next == '/')
        {
            flush();
            while (i < text.size() && text[i] != '\n')
            {
                ++i;
            }
            --i;
        }
        else if (c == '/' && next == '*')
        {
            flush();
            const std::size_t end = text.find("*/", i + 2);
            if (end == std::string::npos)
            {
                throw std::runtime_error(
                    name + ":" + std::to_string(line) +
                    ": unterminated block comment");
            }
            for (std::size_t k = i; k < end; ++k)
            {
                line += text[k] == '\n';
            }
            i = end + 1;
        }
        else if (c == '{' || c == '}' || c == ';')
        {
            flush();
            tokens.push_back({std::string(1, c), line});
        }
        else if (c == ' ' || c == '\t' || c == '\r' || c == '\n')
        {
            flush();
            line += c == '\n';
        }
        else
        {
            word += c;
        }
    }
    flush();
    return tokens;
}

} // namespace


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Dictionary::Dictionary(const std::string& name)
    : name_(name),
      entries_(std::make_shared<std::map<std::string, Entry>>()),
      used_(std::make_shared<std::set<std::string>>())
{
}


Dictionary Dictionary::parse(const std::string& text, const std::string& name)
{
    Dictionary dict(name);

    const std::vector<Token> tokens = tokenize(text, name);
    std::vector<std::string> scope; // enclosing sub-dictionary names

    std::size_t i = 0;
    auto fail = [&](const std::string& what, int line)
    {
        throw std::runtime_error(
            name + ":" + std::to_string(line) + ": " + what);
    };

    while (i < tokens.size())
    {
        const Token& key = tokens[i];
        if (key.text == "}")
        {
            if (scope.empty())
            {
                fail("unmatched '}'", key.line);
            }
            scope.pop_back();
            ++i;
            continue;
        }
        if (key.text == "{" || key.text == ";")
        {
            fail("expected a keyword, found '" + key.text + "'", key.line);
        }
        if (i + 1 >= tokens.size())
        {
            fail("keyword '" + key.text + "' has no value", key.line);
        }
        if (tokens[i + 1].text == "{")
        {
            scope.push_back(key.text);
            i += 2;
            continue;
        }

        std::string value;
        std::size_t j = i + 1;
        for (; j < tokens.size() && tokens[j].text != ";"; ++j)
        {
            if (tokens[j].text == "{" || tokens[j].text == "}")
            {
                fail("missing ';' after keyword '" + key.text + "'", key.line);
            }
            value += (value.empty() ? "" : " ") + tokens[j].text;
        }
        if (j == tokens.size())
        {
            fail("missing ';' after keyword '" + key.text + "'", key.line);
        }
        if (value.empty())
        {
            fail("keyword '" + key.text + "' has no value", key.line);
        }

        std::string fullKey;
        for (const std::string& s : scope)
        {
            fullKey += s + ".";
        }
        fullKey += key.text;
        if (!dict.entries_->emplace(fullKey, Entry{value, name}).second)
        {
            fail("duplicate keyword '" + fullKey + "'", key.line);
        }
        i = j + 1;
    }
    if (!scope.empty())
    {
        fail(
            "missing '}' for sub-dictionary '" + scope.back() + "'",
            tokens.back().line);
    }
    return dict;
}


Dictionary Dictionary::read(const std::string& path)
{
    std::ifstream in(path);
    if (!in)
    {
        throw std::runtime_error("cannot open dictionary '" + path + "'");
    }
    std::stringstream ss;
    ss << in.rdbuf();
    return parse(ss.str(), path);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Dictionary::merge(const Dictionary& other)
{
    for (const auto& kv : *other.entries_)
    {
        const std::string key = other.prefix_.empty()
                                    ? kv.first
                                    : kv.first.substr(other.prefix_.size());
        if (key.rfind("FoamFile.", 0) == 0)
        {
            continue; // file headers are per file
        }
        if (!entries_->emplace(prefix_ + key, kv.second).second)
        {
            throw std::runtime_error(
                "keyword '" + key + "' is defined in both '" +
                entries_->at(prefix_ + key).source + "' and '" +
                kv.second.source + "'");
        }
    }
}


bool Dictionary::found(const std::string& key) const
{
    return entries_->count(prefix_ + key) != 0;
}


Dictionary Dictionary::subDict(const std::string& key) const
{
    Dictionary sub = *this;
    sub.name_ = name_ + "/" + key;
    sub.prefix_ = prefix_ + key + ".";
    bool any = false;
    for (const auto& kv : *entries_)
    {
        any = any || kv.first.rfind(sub.prefix_, 0) == 0;
    }
    if (!any)
    {
        throw std::runtime_error(
            "sub-dictionary '" + key + "' not found in '" + name_ + "'");
    }
    return sub;
}


const Dictionary::Entry& Dictionary::entry(const std::string& key) const
{
    const auto it = entries_->find(prefix_ + key);
    if (it == entries_->end())
    {
        throw std::runtime_error(
            "keyword '" + key + "' is undefined in dictionary '" + name_ + "'");
    }
    used_->insert(prefix_ + key);
    return it->second;
}


template<>
Real Dictionary::get<Real>(const std::string& key) const
{
    const Entry& e = entry(key);
    std::size_t pos = 0;
    Real v = 0;
    try
    {
        v = std::stod(e.value, &pos);
    }
    catch (const std::exception&)
    {
        pos = 0;
    }
    if (pos == 0 || pos != e.value.size())
    {
        throw std::runtime_error(
            "keyword '" + key + "' in '" + e.source + "' is not a number: '" +
            e.value + "'");
    }
    return v;
}


template<>
int Dictionary::get<int>(const std::string& key) const
{
    const Entry& e = entry(key);
    std::size_t pos = 0;
    int v = 0;
    try
    {
        v = std::stoi(e.value, &pos);
    }
    catch (const std::exception&)
    {
        pos = 0;
    }
    if (pos == 0 || pos != e.value.size())
    {
        throw std::runtime_error(
            "keyword '" + key + "' in '" + e.source + "' is not an integer: '" +
            e.value + "'");
    }
    return v;
}


template<>
std::string Dictionary::get<std::string>(const std::string& key) const
{
    return entry(key).value;
}


Real Dictionary::getInRange(const std::string& key, Real lo, Real hi) const
{
    const Real v = get<Real>(key);
    if (!(v >= lo && v <= hi))
    {
        throw std::runtime_error(
            "keyword '" + key + "' = " + std::to_string(v) + " in '" +
            entry(key).source + "' is outside [" + std::to_string(lo) + ", " +
            std::to_string(hi) + "]");
    }
    return v;
}


std::vector<std::string> Dictionary::unusedKeys() const
{
    std::vector<std::string> out;
    for (const auto& kv : *entries_)
    {
        if (kv.first.rfind(prefix_, 0) != 0 || used_->count(kv.first) ||
            kv.first.rfind("FoamFile.", 0) == 0)
        {
            continue;
        }
        out.push_back(kv.first + " (" + kv.second.source + ")");
    }
    return out;
}

} // namespace reljet

// ************************************************************************* //
