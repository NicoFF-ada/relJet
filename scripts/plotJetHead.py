#!/usr/bin/env python3
"""Jet-head position against time, with the 1D momentum-balance estimate.

Usage: plotJetHead.py caseDir output.png

Reads caseDir/postProcessing/jetHead/0/jetHead.dat and the beam parameters
from caseDir/constant/jetProperties and constant/thermophysicalProperties.
The analytic head speed (Marti et al. 1997) is

    eta_R  = rho_b h_b W^2 / (rho_a h_a),   v_head = v_b sqrt(eta_R) / (1 + sqrt(eta_R))
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relJetPlot import INK, SERIES, plt, style  # noqa: E402


def read_dict(path):
    """Flat 'keyword value;' entries of a dictionary file (no sub-dicts)."""
    text = re.sub(r"/\*.*?\*/", "", open(path).read(), flags=re.S)
    text = re.sub(r"//[^\n]*", "", text)
    return {k: v for k, v in re.findall(r"^\s*(\w+)\s+([^;{}\s]+)\s*;", text, flags=re.M)}


def head_speed(case):
    jet = read_dict(os.path.join(case, "constant", "jetProperties"))
    gamma = float(read_dict(os.path.join(case, "constant", "thermophysicalProperties"))["gamma"])
    W, eta = float(jet["lorentzFactor"]), float(jet["densityContrast"])
    rho_a, p = float(jet["ambientDensity"]), float(jet["pressure"])
    rho_b = eta * rho_a
    h = lambda rho: 1 + gamma / (gamma - 1) * p / rho  # noqa: E731
    eta_r = rho_b * h(rho_b) * W * W / (rho_a * h(rho_a))
    v_b = math.sqrt(1 - 1 / (W * W))
    return v_b * math.sqrt(eta_r) / (1 + math.sqrt(eta_r))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    case, output = sys.argv[1], sys.argv[2]
    style()
    t, z = np.loadtxt(os.path.join(case, "postProcessing", "jetHead", "0", "jetHead.dat"), unpack=True)
    v_est = head_speed(case)
    # Early phase (after start-up) and late phase, fitted separately: the
    # head decelerates once it broadens beyond the beam radius.
    early = (t > 0.025 * t[-1]) & (t <= 0.375 * t[-1])
    late = t > 0.7 * t[-1]
    v_early, z_early = np.polyfit(t[early], z[early], 1)
    v_late, z_late = np.polyfit(t[late], z[late], 1)

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(t, z, color=SERIES[0], label="simulation (tracer C > 1/2 on axis)")
    ax.plot(t, v_est * t, "--", color=INK, lw=1.2, label="1D momentum balance, v = %.3f c" % v_est)
    ax.plot(t[early], v_early * t[early] + z_early, ":", color=SERIES[1], lw=2.0,
            label="fit t < %g: v = %.3f c" % (0.375 * t[-1], v_early))
    ax.plot(t[late], v_late * t[late] + z_late, ":", color=SERIES[2], lw=2.0,
            label="fit t > %g: v = %.3f c" % (0.7 * t[-1], v_late))
    ax.set_xlabel("t  (jet radii / c)")
    ax.set_ylabel("head position z  (jet radii)")
    ax.set_title("Jet-head propagation", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print("wrote %s: head speed %.4f early, %.4f late; estimate %.4f (early %+.1f%%)"
          % (output, v_early, v_late, v_est, 100 * (v_early / v_est - 1)))


if __name__ == "__main__":
    main()
