#!/usr/bin/env python3
"""Plot the output of the convergenceStudy utility (L1 error vs cells).

Usage: convergenceStudy 1024 > convergence.txt
       plotConvergence.py convergence.txt output.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relJetPlot import INK_SECONDARY, SERIES, plt, style  # noqa: E402

MARKERS = ["o", "s", "^"]


def parse(path):
    """Returns {scheme: (cells, errors)} from the convergenceStudy table."""
    schemes, current = {}, None
    for line in open(path):
        parts = line.split()
        if not parts or line.startswith("Smooth"):
            continue
        if parts[0] in ("firstOrder", "MUSCL"):
            current = " ".join(parts[:-3])  # name, then the header words
            schemes[current] = ([], [])
            continue
        if current and parts[0].isdigit():
            schemes[current][0].append(int(parts[0]))
            schemes[current][1].append(float(parts[1]))
    return schemes


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    style()
    schemes = parse(sys.argv[1])
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for n, (name, (cells, err)) in enumerate(schemes.items()):
        cells, err = np.array(cells), np.array(err)
        order = np.log(err[-2] / err[-1]) / np.log(cells[-1] / cells[-2])
        ax.loglog(cells, err, MARKERS[n] + "-", color=SERIES[n], ms=6,
                  label="%s (order %.2f)" % (name, order))

    # Reference slopes, anchored at the coarsest first-order point
    cells, err = map(np.array, list(schemes.values())[0])
    for k, style_ in ((1, "--"), (2, ":")):
        ref = err[0] * 0.5 * (cells / cells[0]) ** (-k)
        ax.loglog(cells, ref, style_, color=INK_SECONDARY, lw=1.0)
        ax.annotate("slope -%d" % k, (cells[-1], ref[-1]), color=INK_SECONDARY,
                    fontsize=8, xytext=(4, 0), textcoords="offset points",
                    va="center")
    ax.set_xlabel("cells")
    ax.set_ylabel(r"$L_1$ error of $\rho$")
    ax.set_title("Smooth density wave, W = 1.15: observed order", loc="left")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(right=cells[-1] * 2.2)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=150)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()
