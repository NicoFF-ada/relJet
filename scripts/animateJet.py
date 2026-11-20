#!/usr/bin/env python3
"""Animated GIF of the jet density (or pressure) from all snapshots of a case.

Usage: animateJet.py caseDir output.gif [fps] [rho | p]

Needs ffmpeg on the PATH (frames are rendered with matplotlib, then encoded
with a per-file palette).
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotJet import meridional_slice  # noqa: E402
from relJetPlot import plt, read_vtk, style, time_of  # noqa: E402


# log10 colour range per field, fixed over all frames
COLOUR_SCALES = {"rho": ("magma", -2.6, 0.7), "p": ("inferno", -3.5, -0.5)}


def main():
    if len(sys.argv) not in (3, 4, 5):
        sys.exit(__doc__)
    case, output = sys.argv[1], sys.argv[2]
    fps = int(sys.argv[3]) if len(sys.argv) >= 4 else 8
    field = sys.argv[4] if len(sys.argv) == 5 else "rho"
    if field not in COLOUR_SCALES:
        sys.exit(__doc__)
    cmap, vmin, vmax = COLOUR_SCALES[field]
    files = sorted(glob.glob(os.path.join(case, "VTK", "*.vtk")),
                   key=lambda f: int(re.findall(r"_(\d+)\.vtk$", f)[0]))
    if not files:
        sys.exit("no VTK snapshots in %s/VTK" % case)
    style()
    tmp = tempfile.mkdtemp()
    try:
        for n, path in enumerate(files):
            meta, fields = read_vtk(path)
            data, (r0, r1), (z0, z1) = meridional_slice(meta, fields)
            fig, ax = plt.subplots(figsize=(9, 9 * (r1 - r0) / (z1 - z0)))
            ax.imshow(np.log10(data[field]).T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
                      extent=(z0, z1, r0, r1), aspect="equal", interpolation="nearest")
            ax.set_axis_off()
            ax.text(0.01, 0.97, "t = %4.0f" % time_of(meta), transform=ax.transAxes,
                    color="#0b0b0b", fontsize=11, va="top", family="monospace")
            fig.subplots_adjust(0, 0, 1, 1)
            fig.savefig(os.path.join(tmp, "frame_%04d.png" % n), dpi=100)
            plt.close(fig)
        pattern = os.path.join(tmp, "frame_%04d.png")
        palette = os.path.join(tmp, "palette.png")
        subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-i", pattern,
                               "-vf", "palettegen", palette])
        subprocess.check_call(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(fps), "-i", pattern,
                               "-i", palette, "-lavfi", "paletteuse", "-loop", "0", output])
    finally:
        shutil.rmtree(tmp)
    print("wrote", output, "(%d frames)" % len(files))


if __name__ == "__main__":
    main()
