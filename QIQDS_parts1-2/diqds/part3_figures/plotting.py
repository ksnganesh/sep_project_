"""Shared Matplotlib styling for the Part 3 figures (paper-style log axes, validated categorical palette)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# categorical slots 1-4 (validated: CVD dE >= 9.1, normal-vision dE >= 22.9 on adjacent pairs; aqua/yellow need labels)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = (ORANGE, BLUE, YELLOW, AQUA)          # order used for the paper's 4-curve figures (Fig. 3: V = 1 first)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"


def setup():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.8, "lines.linewidth": 2.0, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def log_rate_axes(ax, ylim=(1.0, 1e8)):
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_ylabel("Signature rate (signatures per execution)")
    ax.grid(True, which="major", color=GRID, linewidth=0.6, linestyle="--")


def plot_rate(ax, x, rate, *, color, label, ls="-", floor=1.0, end_label=None):
    """Draw the positive part of a rate curve.  Where the key switches on (Figs. 3-6, rising) or off (Fig. 7, falling) the
    curve is closed with a vertical segment down to the axis floor, which is how the paper draws its finite-size thresholds;
    zero-rate points are not drawn."""
    x, rate = np.asarray(x, float), np.asarray(rate, float)
    idx = np.flatnonzero(rate > 0)
    if idx.size == 0:
        return None
    i0, i1 = idx[0], idx[-1]
    xs, ys = list(x[i0:i1 + 1]), list(np.maximum(rate[i0:i1 + 1], floor * 0.999))
    if i0 > 0:
        xs, ys = [xs[0]] + xs, [floor * 0.999] + ys
    if i1 < len(x) - 1:
        xs, ys = xs + [xs[-1]], ys + [floor * 0.999]
    ax.plot(xs, ys, color=color, ls=ls, label=label, solid_capstyle="round")
    k = len(xs) - 1 if i1 == len(x) - 1 else 0          # label the end that stays positive
    return (xs[k], ys[k], end_label) if end_label else None


def end_labels(ax, ends, min_gap_decades: float = 0.32):
    """Direct labels at the right end of log-y curves, pushed apart so they never overlap (text in ink, not series colour)."""
    ends = sorted([e for e in ends if e], key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        ly = np.log10(y)
        if placed and ly - placed[-1] < min_gap_decades:
            ly = placed[-1] + min_gap_decades
        placed.append(ly)
        ax.annotate(text, (x, 10 ** ly), xytext=(5, 0), textcoords="offset points", va="center", fontsize=8.5,
                    color=INK2, annotation_clip=False)


def mark_paper_threshold(ax, xs, *, y=1.0, label="threshold printed in paper"):
    ax.plot(xs, [y * 1.6] * len(xs), ls="none", marker="v", ms=6, mfc="white", mec=INK, mew=1.0, label=label,
            zorder=5, clip_on=False)
