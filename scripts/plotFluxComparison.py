#!/usr/bin/env python3
"""HLL against HLLC: a contact at rest, and the jet tracer.

Usage: plotFluxComparison.py fluxComparisonDir output.png

Reads the cases written by tutorials/fluxComparison/Allrun:
contactAtRest{HLL,HLLC} (1D, t = 2) and jet{HLL,HLLC} (latest snapshot).
Left: density across the contact. Right: jet tracer C, HLL above the axis and
HLLC below it, so the two mixing layers can be compared side by side.
"""
import glob
import os
import re
import sys

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relJetPlot import INK, INK_SECONDARY, SERIES, SURFACE, plt, read_vtk, style, time_of  # noqa: E402


def load_line(case):
    files = glob.glob(os.path.join(case, "postProcessing", "sample", "*", "line.csv"))
    if not files:
        sys.exit("no sampled line in %s" % case)
    path = max(files, key=lambda f: float(f.split(os.sep)[-2]))
    return np.genfromtxt(path, delimiter=",", names=True, skip_header=1)


def load_jet(case):
    files = glob.glob(os.path.join(case, "VTK", "*.vtk"))
    if not files:
        sys.exit("no VTK files in %s" % case)
    path = max(files, key=lambda f: int(re.findall(r"_(\d+)\.vtk$", f)[0]))
    return read_vtk(path)


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    root, out = sys.argv[1], sys.argv[2]
    style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.9]})

    # Contact at rest: the exact solution is the initial step at x = 0.5
    ax1.plot([0, 0.5, 0.5, 1], [1, 1, 0.1, 0.1], color=INK, lw=1.2, label="exact")
    for flux, colour, marker in [("HLL", SERIES[0], "o"), ("HLLC", SERIES[1], "s")]:
        d = load_line(os.path.join(root, "contactAtRest" + flux))
        ax1.plot(d["x"], d["rho"], marker, ms=4, mfc="none", mew=1.2,
                 color=colour, label=flux, ls="none")
    ax1.set_xlim(0.2, 0.8)
    ax1.set_xlabel("x")
    ax1.set_ylabel("density")
    ax1.set_title("Contact at rest, first order, t = 2", loc="left")
    ax1.legend(loc="upper right")

    # Jet tracer: HLL mirrored above the axis, HLLC below
    cmap = LinearSegmentedColormap.from_list("tracer", [SURFACE, SERIES[0], "#0d2a4f"])
    (mHll, fHll), (mHllc, fHllc) = (load_jet(os.path.join(root, "jet" + f))
                                    for f in ("HLL", "HLLC"))
    dr, dz = mHll["spacing"][0], mHll["spacing"][1]
    nz, nr = fHll["C"].shape
    zMax = 22
    image = np.vstack([fHllc["C"][:, ::-1].T, fHll["C"].T])  # rows: r from -rMax to rMax
    im = ax2.imshow(image, origin="lower", cmap=cmap, vmin=0, vmax=1,
                    extent=[0, nz * dz, -nr * dr, nr * dr], aspect="equal",
                    interpolation="nearest")
    ax2.set_xlim(0, zMax)
    ax2.set_ylim(-6, 6)
    ax2.axhline(0, color=INK_SECONDARY, lw=0.8)
    ax2.grid(False)
    for y, label in [(4.8, "HLL"), (-5.4, "HLLC")]:
        ax2.text(18.5, y, label, color=INK, fontsize=11, weight="bold")
    ax2.set_xlabel("z  (jet radii)")
    ax2.set_ylabel("r")
    ax2.set_title("Jet tracer C at t = %g (1 = jet material)" % time_of(mHll), loc="left")
    cb = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cb.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
