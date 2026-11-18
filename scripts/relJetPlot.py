"""Shared helpers for the RelJet plotting scripts: VTK reader and figure style.

Series colours are the first three slots of a palette validated for colour-
vision deficiency (all pairs distinguishable); exact or reference curves use
the text ink, so identity never rests on colour alone (each series also has
its own marker and a legend entry).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # blue, orange, aqua


def style():
    settings = {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "font.size": 10,
        "axes.titlesize": 11,
        "text.color": INK,
    }
    # Older matplotlib versions lack some keys (e.g. axes.titlecolor).
    plt.rcParams.update({k: v for k, v in settings.items() if k in plt.rcParams})


def read_vtk(path):
    """Reads the legacy binary STRUCTURED_POINTS files written by relJet.

    Returns (meta, fields): meta has dims (cells per direction), origin,
    spacing, title; fields maps names to arrays of shape (nz, ny, nx[, 3]),
    squeezed.
    """
    with open(path, "rb") as f:
        data = f.read()
    fields, meta, pos = {}, {}, 0

    def line():
        nonlocal pos
        end = data.index(b"\n", pos)
        text = data[pos:end].decode()
        pos = end + 1
        return text

    line()
    meta["title"] = line()
    while True:
        text = line()
        key = text.split()[0] if text.strip() else ""
        if key == "DIMENSIONS":
            meta["dims"] = [int(v) - 1 for v in text.split()[1:]]
        elif key in ("ORIGIN", "SPACING"):
            meta[key.lower()] = [float(v) for v in text.split()[1:]]
        elif key == "CELL_DATA":
            ncell = int(text.split()[1])
            break
    nx, ny, nz = meta["dims"]
    while pos < len(data):
        text = line()
        if not text.strip():
            continue
        kind, name = text.split()[:2]
        ncomp = 3 if kind == "VECTORS" else 1
        if kind == "SCALARS":
            line()  # LOOKUP_TABLE
        values = np.frombuffer(data, dtype=">f4", count=ncell * ncomp, offset=pos)
        pos += 4 * ncell * ncomp
        fields[name] = values.reshape(nz, ny, nx, ncomp).squeeze()
    return meta, fields


def time_of(meta):
    """Simulation time from the VTK title line ('RelJet t=...')."""
    for token in meta["title"].replace(",", " ").split():
        if token.startswith("t="):
            return float(token[2:])
    return float("nan")
