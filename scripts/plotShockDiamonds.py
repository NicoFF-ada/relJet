#!/usr/bin/env python3
"""Shock diamonds of an overpressured axisymmetric jet.

Usage: plotShockDiamonds.py caseDir output.png [snapshot.vtk]

Three panels, mirrored about the jet axis:
  1. pressure, which shows the recollimation shocks;
  2. a synthetic side-on image: emissivity C p^((2a+3)/2) D^(2+a), a = 0.65
     (Gomez et al. 1995), with Doppler factor D, integrated along a line of
     sight perpendicular to the jet. A brightness proxy, not radiative
     transfer;
  3. pressure and Lorentz factor on the axis, with the point where the first
     Mach line from the nozzle lip meets the axis, z = R sqrt(M^2 - 1), and
     the diamond spacing measured by autocorrelation.

Jet parameters are read from caseDir/constant.
"""
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotJet import latest_snapshot  # noqa: E402
from relJetPlot import INK_SECONDARY, SERIES, plt, read_vtk, style, time_of  # noqa: E402

ALPHA = 0.65  # spectral index of the emission proxy


def read_entries(path):
    """Numeric 'key value;' entries of a case dictionary."""
    entries = {}
    for line in open(path):
        m = re.match(r"\s*(\w+)\s+([-+0-9.eE]+)\s*;", line)
        if m:
            entries[m.group(1)] = float(m.group(2))
    return entries


def mach_number(case):
    """Relativistic Mach number W v / (W_s c_s) of the beam."""
    jet = read_entries(os.path.join(case, "constant", "jetProperties"))
    gamma = read_entries(os.path.join(case, "constant", "thermophysicalProperties"))["gamma"]
    W = jet["lorentzFactor"]
    rho = jet["densityContrast"] * jet["ambientDensity"]
    p = jet.get("pressureRatio", 1.0) * jet["pressure"]
    h = 1 + gamma / (gamma - 1) * p / rho
    cs = math.sqrt(gamma * p / (rho * h))
    return W * math.sqrt(1 - 1 / W ** 2) / (cs / math.sqrt(1 - cs * cs)), jet


def synthetic_image(fields, dr, rmax_image):
    """Side-on brightness map I(z, x) of the axisymmetric emissivity."""
    rho_r = fields["p"].shape[1]
    n = min(rho_r, int(rmax_image / dr))
    p, C, W = fields["p"][:, :n], fields["C"][:, :n], fields["W"][:, :n]
    vr = fields["velocity"][:, :n, 0]
    x = (np.arange(-n, n) + 0.5) * dr
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing="ij")  # line of sight along y
    R = np.hypot(X, Y)
    ir = np.minimum((R / dr).astype(int), n - 1)
    inside = R < n * dr
    cos_los = np.where(R > 0, Y / np.maximum(R, 1e-30), 0.0)
    emis = np.clip(C, 0, 1) * p ** ((2 * ALPHA + 3) / 2)
    image = np.empty((p.shape[0], x.size))
    for k in range(p.shape[0]):
        doppler = 1 / (W[k][ir] * (1 - vr[k][ir] * cos_los))
        j = emis[k][ir] * doppler ** (2 + ALPHA)
        image[k] = np.where(inside, j, 0).sum(axis=1) * dr
    return image, n * dr


def diamond_spacing(z, p_axis):
    """Period of the axis pressure from the first autocorrelation peak."""
    q = np.log(p_axis) - np.log(p_axis).mean()
    n = q.size
    ac = np.correlate(q, q, "full")[n - 1:] / np.arange(n, 0, -1)
    dz = z[1] - z[0]
    lag0 = int(2 / dz)
    i = lag0 + int(np.argmax(ac[lag0:n // 2]))
    a, b, c = ac[i - 1], ac[i], ac[i + 1]
    return (i + 0.5 * (a - c) / (a - 2 * b + c)) * dz


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    case, output = sys.argv[1], sys.argv[2]
    snapshot = sys.argv[3] if len(sys.argv) == 4 else latest_snapshot(case)
    meta, fields = read_vtk(snapshot)
    if meta["dims"][2] != 1:
        sys.exit("plotShockDiamonds.py needs an axisymmetric (r, z) case")
    style()
    M, jet = mach_number(case)
    radius = jet.get("radius", 1.0)
    p_amb = jet["pressure"]
    dr, dz = meta["spacing"][0], meta["spacing"][1]
    nz = meta["dims"][1]
    z = (np.arange(nz) + 0.5) * dz
    zmax = nz * dz
    # Show the radii that hold jet material (tracer C > 0.05), at least 4 R.
    has_jet = np.nonzero((fields["C"] > 0.05).any(axis=0))[0]
    r_jet = (has_jet[-1] + 1) * dr if has_jet.size else radius
    rshow = min(meta["dims"][0] * dr, max(4 * radius, 1.15 * r_jet))
    nshow = int(rshow / dr)

    image, rimg = synthetic_image(fields, dr, rshow)
    p_half = fields["p"][:, :nshow] / p_amb
    p_full = np.hstack([p_half[:, ::-1], p_half])

    # The maps are stretched across the axis by `stretch` to keep them legible.
    width, map_height = 11.0, 1.9
    stretch = map_height / (2 * rshow / zmax * width * 0.85)
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(width, 2 * map_height + 4.0),
                             gridspec_kw={"height_ratios": [map_height, map_height, 2.4]})

    ax = axes[0]
    im = ax.imshow(np.log10(p_full).T, origin="lower", cmap="inferno",
                   extent=(0, zmax, -rshow, rshow), aspect="auto", interpolation="bilinear")
    ax.set_title("pressure: recollimation shocks (r stretched %.1f\u00d7)" % stretch,
                 loc="left")
    ax.set_ylabel("r")
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label(r"log$_{10}$ p / p$_{amb}$")
    cb.outline.set_visible(False)

    ax = axes[1]
    ax.set_facecolor("black")
    bright = np.minimum(image / np.percentile(image, 99.9), 1)
    im = ax.imshow(bright.T ** 0.5, origin="lower", cmap="afmhot",
                   extent=(0, zmax, -rimg, rimg), aspect="auto", interpolation="bilinear")
    ax.set_title(r"synthetic side-on image: brightness $\propto$ "
                 r"$\int C\,p^{(2\alpha+3)/2}\,\delta^{2+\alpha}\,dl$, $\alpha$ = %.2f" % ALPHA,
                 loc="left")
    ax.set_ylabel("x")
    cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
    cb.set_label("(brightness)$^{1/2}$, normalised")
    cb.outline.set_visible(False)
    for a in axes[:2]:
        a.grid(False)

    ax = axes[2]
    p_axis = fields["p"][:, 0] / p_amb
    ax.plot(z, p_axis, color=SERIES[0], lw=1.4, label=r"p / p$_{amb}$ on the axis")
    ax.plot(z, fields["W"][:, 0], color=SERIES[1], lw=1.4, label="Lorentz factor W on the axis")
    z_mach = radius * math.sqrt(M * M - 1)
    ax.axvline(z_mach, color=INK_SECONDARY, ls=":", lw=1.0)
    ax.annotate(r"first Mach line: z = R$\sqrt{M_r^2 - 1}$ = %.2f R" % (z_mach / radius),
                (z_mach, 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0),
                textcoords="offset points", va="top", fontsize=8, color=INK_SECONDARY)
    spacing = diamond_spacing(z, fields["p"][:, 0])
    ax.set_title("on the axis: diamond spacing %.1f R (autocorrelation), M$_r$ = %.2f"
                 % (spacing / radius, M), loc="left")
    ax.set_xlim(0, zmax)
    ax.set_xlabel("z  (jet radii)")
    ax.legend(loc="upper right", fontsize=8)
    # an invisible colorbar keeps the z axis aligned with the maps above
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03).ax.set_visible(False)

    fig.suptitle("RelJet: shock diamonds in an overpressured relativistic jet, "
                 "W = %g, K = %g, t = %g" % (jet["lorentzFactor"], jet.get("pressureRatio", 1.0),
                                              time_of(meta)),
                 x=0.01, ha="left", color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=150)
    print("wrote %s (diamond spacing %.2f R, first Mach line at %.2f R)"
          % (output, spacing / radius, z_mach / radius))


if __name__ == "__main__":
    main()
