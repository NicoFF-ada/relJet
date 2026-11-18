/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    Legacy-VTK output of the primitive fields.
\*---------------------------------------------------------------------------*/

#include "VtkOutput.H"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace reljet
{

namespace
{

bool hostIsLittleEndian()
{
    const std::uint16_t one = 1;
    unsigned char first;
    std::memcpy(&first, &one, 1);
    return first == 1;
}

void appendBigEndian(std::vector<char>& out, float value)
{
    std::uint32_t bits;
    std::memcpy(&bits, &value, 4);
    if (hostIsLittleEndian())
        bits = __builtin_bswap32(bits);
    char bytes[4];
    std::memcpy(bytes, &bits, 4);
    out.insert(out.end(), bytes, bytes + 4);
}

} // namespace

void writeVtk(const std::string& path, const PrimField& prim, Real time)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
        throw std::runtime_error(
            "output: cannot open '" + path + "' for writing");

    const Grid& g = prim.grid();
    out << "# vtk DataFile Version 3.0\n"
        << "RelJet t=" << time << "\n"
        << "BINARY\nDATASET STRUCTURED_POINTS\n"
        << "DIMENSIONS " << g.n[0] + 1 << ' ' << g.n[1] + 1 << ' ' << g.n[2] + 1
        << '\n'
        << "ORIGIN " << g.lo[0] << ' ' << g.lo[1] << ' ' << g.lo[2] << '\n'
        << "SPACING " << g.dx[0] << ' ' << g.dx[1] << ' ' << g.dx[2] << '\n'
        << "CELL_DATA " << g.numInteriorCells() << '\n';

    // Gathers one quantity over the interior in VTK order (x fastest).
    std::vector<char> buf;
    auto writeField = [&](const char* header, auto value)
    {
        buf.clear();
        for (int k = g.begin(2); k < g.end(2); ++k)
            for (int j = g.begin(1); j < g.end(1); ++j)
                for (int i = g.begin(0); i < g.end(0); ++i)
                    value(prim.load(g.index(i, j, k)));
        out << header;
        out.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        out << '\n';
    };
    auto scalar = [&](const char* name, Real (*f)(const Prim&))
    {
        const std::string header =
            std::string("SCALARS ") + name + " float 1\nLOOKUP_TABLE default\n";
        writeField(
            header.c_str(),
            [&](const Prim& w)
            { appendBigEndian(buf, static_cast<float>(f(w))); });
    };

    scalar("rho", [](const Prim& w) { return w.rho; });
    scalar("p", [](const Prim& w) { return w.p; });
    scalar(
        "W",
        [](const Prim& w)
        {
            return 1 / std::sqrt(
                           1 - (w.v[0] * w.v[0] + w.v[1] * w.v[1] +
                                w.v[2] * w.v[2]));
        });
    scalar("C", [](const Prim& w) { return w.C; });
    writeField(
        "VECTORS velocity float\n",
        [&](const Prim& w)
        {
            for (int d = 0; d < kDims; ++d)
                appendBigEndian(buf, static_cast<float>(w.v[d]));
        });
    if (!out)
        throw std::runtime_error("output: write to '" + path + "' failed");
}

} // namespace reljet

// ************************************************************************* //
