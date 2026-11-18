#!/usr/bin/env python3
"""Compare 1D relJet results with the exact Riemann solution (Problem 1).

Usage: plotRiemann.py output.png line.csv[:label] [line2.csv:label ...]

The exact solution assumes the Problem 1 states of tutorials/riemannProblem1
(left: rho 10, p 13.33; right: rho 1, p 1e-6; gamma 5/3; x0 = 0.5).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relJetPlot import INK, INK_SECONDARY, SERIES, plt, style  # noqa: E402
from riemannExact import RiemannSolution  # noqa: E402

MARKERS = ["o", "s", "^"]


def load(path):
    with open(path) as f:
        time = float(f.readline().split("=")[1])
    return time, np.genfromtxt(path, delimiter=",", names=True, skip_header=1)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out, runs = sys.argv[1], sys.argv[2:]
    if len(runs) > len(SERIES):
        sys.exit("at most %d runs per plot" % len(SERIES))
    style()

    exact = RiemannSolution((10.0, 40.0 / 3.0, 0.0), (1.0, 1e-6, 0.0), 5.0 / 3.0)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    panels = [("rho", r"density $\rho$", 0), ("p", r"pressure $p$", 1), ("v0", r"velocity $v$", 2)]

    # Zoom on the thin shell between contact and shock, where schemes differ.
    inset = axes[0].inset_axes([0.62, 0.56, 0.36, 0.40])
    t = None
    for n, spec in enumerate(runs):
        path, _, label = spec.partition(":")
        t, d = load(path)
        label = label or os.path.basename(os.path.dirname(path))
        for ax, (name, _, _) in zip(axes, panels):
            ax.plot(d["x"], d[name], MARKERS[n], ms=3.2, mfc="none", mew=1.0,
                    color=SERIES[n], label=label)
        inset.plot(d["x"], d["rho"], MARKERS[n] + "-", ms=3, mfc="none", mew=0.9,
                   lw=0.8, color=SERIES[n])

    x = np.linspace(0, 1, 2001)
    sol = exact.sample(x, t)
    for ax, (_, title, k) in zip(axes, panels):
        ax.plot(x, sol[k], color=INK, lw=1.2, label="exact", zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("x")
        ax.set_xlim(0, 1)
    inset.plot(x, sol[0], color=INK, lw=1.2, zorder=0)
    inset.set_xlim(0.76, 0.86)
    inset.set_ylim(0.5, 6)
    inset.set_title("shell", fontsize=8, loc="left")
    inset.tick_params(labelsize=7)
    axes[0].legend(loc="lower left")
    fig.suptitle(
        "Relativistic blast wave (Marti & Mueller Problem 1), t = %g: "
        "p* = %.3f, v* = %.3f" % (t, exact.p, exact.v),
        color=INK_SECONDARY, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
