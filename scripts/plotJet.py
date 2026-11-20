#!/usr/bin/env python3
"""Plot a relJet jet snapshot: density, pressure and Lorentz factor.

Usage: plotJet.py caseDir output.png [snapshot.vtk]

Without a snapshot argument the latest file in caseDir/VTK is used.
Axisymmetric (r, z) data are mirrored about the axis; 3D Cartesian data are
cut through the jet axis (y = 0). The jet axis is drawn horizontally.
"""
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from relJetPlot import INK_SECONDARY, plt, read_vtk, style, time_of  # noqa: E402


def latest_snapshot(case):
    files = glob.glob(os.path.join(case, "VTK", "*.vtk"))
    if not files:
        sys.exit("no VTK snapshots in %s/VTK" % case)
    return max(files, key=lambda f: int(re.findall(r"_(\d+)\.vtk$", f)[0]))


def meridional_slice(meta, fields):
    """Fields on the (z, r) half-plane through the axis, z first."""
    nx, ny, nz = meta["dims"]
    x0, y0, z0 = meta["origin"]
    dx, dy, dz = meta["spacing"]
    out = {}
    if nz == 1:  # axisymmetric: arrays (nz=z, nx=r), mirror to (-r, r)
        for name in ("rho", "p", "W", "C"):
            half = fields[name]  # shape (ny, nx): rows z, columns r
            out[name] = np.hstack([half[:, ::-1], half])
        extent_r = (-(x0 + nx * dx), x0 + nx * dx)
        extent_z = (y0, y0 + ny * dy)
    else:  # 3D: cut y = 0, arrays (nz, ny, nx)
        j = ny // 2
        for name in ("rho", "p", "W", "C"):
            out[name] = fields[name][:, j, :]
        extent_r = (x0, x0 + nx * dx)
        extent_z = (z0, z0 + nz * dz)
    return out, extent_r, extent_z


def plot(meta, fields, output, title_suffix=""):
    data, (r0, r1), (z0, z1) = meridional_slice(meta, fields)
    panels = [("rho", "log10 density", np.log10, "magma"),
              ("p", "log10 pressure", np.log10, "viridis"),
              ("W", "Lorentz factor W", None, "cividis")]
    aspect = (r1 - r0) / (z1 - z0)
    width = 11.0
    fig, axes = plt.subplots(len(panels), 1, figsize=(width, len(panels) * width * aspect * 0.86 + 0.6))
    for ax, (name, label, transform, cmap) in zip(axes, panels):
        img = data[name] if transform is None else transform(np.maximum(data[name], 1e-30))
        im = ax.imshow(img.T, origin="lower", cmap=cmap, aspect="equal",
                       extent=(z0, z1, r0, r1), interpolation="nearest")
        ax.grid(False)
        ax.set_ylabel("r" if meta["dims"][2] == 1 else "x")
        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.03)
        cb.set_label(label)
        cb.outline.set_visible(False)
    axes[-1].set_xlabel("z  (jet radii)")
    fig.suptitle("RelJet: relativistic jet, t = %g%s" % (time_of(meta), title_suffix),
                 x=0.01, ha="left", color=INK_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=130)
    plt.close(fig)


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    case, output = sys.argv[1], sys.argv[2]
    snapshot = sys.argv[3] if len(sys.argv) == 4 else latest_snapshot(case)
    style()
    meta, fields = read_vtk(snapshot)
    plot(meta, fields, output)
    print("wrote", output)


if __name__ == "__main__":
    main()
