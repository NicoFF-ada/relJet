#!/usr/bin/env python3
"""Width of the jet/cocoon mixing layer in relJet snapshots.

Usage: mixingLayer.py caseDir [caseDir ...]

For the latest VTK snapshot of each case, prints the volume of partially mixed
material (0.05 < C < 0.95) and the volume of jet material (C > 0.5). A
sharper contact gives a smaller mixed volume for the same jet volume.
Axisymmetric cells are weighted by their ring volume 2 pi r dr dz.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relJetPlot import read_vtk, time_of  # noqa: E402


def latest(case):
    files = glob.glob(os.path.join(case, "VTK", "*.vtk"))
    if not files:
        sys.exit("no VTK files in %s" % case)
    return max(files, key=lambda f: int(re.findall(r"_(\d+)\.vtk$", f)[0]))


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    print("%-28s %8s %12s %12s %8s" % ("case", "t", "mixed vol", "jet vol", "ratio"))
    for case in sys.argv[1:]:
        meta, fields = read_vtk(latest(case))
        C = fields["C"]
        dr, dz = meta["spacing"][0], meta["spacing"][1]
        r = meta["origin"][0] + (np.arange(C.shape[-1]) + 0.5) * dr
        dV = 2 * np.pi * r * dr * dz * np.ones_like(C)
        mixed = dV[(C > 0.05) & (C < 0.95)].sum()
        jet = dV[C > 0.5].sum()
        print("%-28s %8.2f %12.1f %12.1f %8.3f"
              % (os.path.basename(os.path.normpath(case)), time_of(meta), mixed, jet, mixed / jet))


if __name__ == "__main__":
    main()
