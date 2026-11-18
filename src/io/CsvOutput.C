/*---------------------------------------------------------------------------*\
    RelJet | Special-relativistic hydrodynamics for astrophysical jets
-------------------------------------------------------------------------------
License
    MIT, see LICENSE.

Description
    CSV line output of 1D runs.
\*---------------------------------------------------------------------------*/

#include "CsvOutput.H"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace reljet
{

void writeCsv1D(
    const std::string& path,
    const PrimField& prim,
    const ConsField& cons,
    Real time)
{
    std::ofstream out(path);
    if (!out)
        throw std::runtime_error(
            "output: cannot open '" + path + "' for writing");

    const Grid& g = prim.grid();
    const int j = g.begin(1);
    out << "# time = " << std::setprecision(17) << time << '\n';
    out << "x,rho,p,v0,v1,C,D,S0,S1,tau,DC\n";
    out << std::setprecision(10);
    for (int i = g.begin(0); i < g.end(0); ++i)
    {
        const int idx = g.index(i, j);
        const Prim w = prim.load(idx);
        const Cons u = cons.load(idx);
        out << g.center(0, i) << ',' << w.rho << ',' << w.p << ',' << w.v[0]
            << ',' << w.v[1] << ',' << w.C << ',' << u.D << ',' << u.S[0] << ','
            << u.S[1] << ',' << u.tau << ',' << u.DC << '\n';
    }
}

} // namespace reljet

// ************************************************************************* //
