#!/usr/bin/env python
"""Part 3 - the results section (Sec. IV, Figs. 3-7) of Fan et al., PRA 114, 032617 (2026).

    python scripts/part3_figures.py                  # all figures from the cache in results/part3_curves.json (~seconds)
    python scripts/part3_figures.py --recompute      # redo every optimisation (~15-25 min)
    python scripts/part3_figures.py --figs 3 5       # only some figures
    python scripts/part3_figures.py --preset rigorous   # Figs. 3-5 under the theorem-consistent reading instead of PAPER_FIT

Outputs (results/):  fig3_rate_vs_rounds.png ... fig7_photonic_distance.png,  part3_curves.json,  part3_summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from diqds.part2_distribution.presets import PRESETS, threshold_visibility  # noqa: E402
from diqds.part3_figures import plotting as P  # noqa: E402
from diqds.part3_figures.sweeps import Curve, curve_vs_n, curve_vs_v, signature_length  # noqa: E402

RESULTS = ROOT / "results"
CACHE = RESULTS / "part3_curves.json"
SUMMARY = RESULTS / "part3_summary.json"

# Numbers printed in the paper (Sec. IV and captions); used for the comparison table, never as inputs.
PAPER = {
    "fig5_vmin_p0": {"1e+08": 0.8835, "1e+09": 0.8686, "1e+10": 0.8622},
    "fig5_vmin_popt": {"1e+08": 0.8799, "1e+09": 0.8608, "1e+10": 0.8508},
    "fig5_p_legend": {"1e+10": 0.155, "1e+09": 0.090, "1e+08": 0.035},
    "fig7_Lmax_km": {"ideal/1e14": 111.46, "ideal/2e14": 126.53, "real/1e14": 17.82, "real/2e14": 28.87},
}
# Read off the published figures (pixel digitisation of the PDF, +-3 %): used only for the comparison table.
DIGITISED = {
    "fig3_threshold": {"1": 8.83e5, "0.96": 2.39e6, "0.92": 9.39e6, "0.88": 1.46e8},
    "fig3_rate_1e10": {"1": 3.29e7, "0.96": 2.00e7, "0.92": 1.08e7, "0.88": 3.11e6},
}


class Cache:
    def __init__(self, path: Path, recompute: bool):
        self.path = path
        self.data = {} if (recompute or not path.exists()) else json.loads(path.read_text())

    def curve(self, key: str, fn) -> dict:
        if key not in self.data:
            t0 = time.time()
            c = fn()
            self.data[key] = c.to_json() if isinstance(c, Curve) else c
            self.save()
            print(f"    computed {key} in {time.time() - t0:.1f} s")
        return self.data[key]

    def save(self):
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=0, sort_keys=True))


def _fmt(n: float) -> str:
    return f"{n:.0e}"


# ================================================================================================ Figs. 3-5
def fig3(cache: Cache, preset_name: str, summary: dict):
    preset = PRESETS[preset_name]
    plt = P.plt
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    grid = np.geomspace(1e5, 1e10, 61)
    rows, ends = {}, []
    for color, vis in zip(P.SERIES, (1.0, 0.96, 0.92, 0.88)):
        c = cache.curve(f"fig3/{preset_name}/V={vis}/g61",
                        lambda: curve_vs_n(f"V={vis}", grid, vis=vis, preset=preset, p=0.0))
        ends.append(P.plot_rate(ax, c["x"], c["rate"], color=color, label=f"V = {vis:g}", end_label=f"V = {vis:g}"))
        key = f"{vis:g}"
        rows[key] = {"threshold_n": c["threshold"], "rate_1e10": c["rate"][-1],
                     "paper_threshold_n (digitised)": DIGITISED["fig3_threshold"][key],
                     "paper_rate_1e10 (digitised)": DIGITISED["fig3_rate_1e10"][key]}
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e10)
    P.log_rate_axes(ax)
    P.end_labels(ax, ends)
    ax.set_xlabel("Number of rounds n")
    ax.set_title("Fig. 3 - signature rate vs rounds, depolarizing noise, p = 0", fontsize=10, loc="left")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_rate_vs_rounds.png")
    plt.close(fig)
    summary["fig3"] = rows


def fig4(cache: Cache, preset_name: str, summary: dict):
    preset = PRESETS[preset_name]
    plt = P.plt
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    vgrid = np.linspace(0.84, 1.0, 33)
    rows = {}
    for color, ls, pp in zip(P.SERIES, ("-", "--", ":", "-."), (0.0, 0.05, 0.10, 0.15)):
        c = cache.curve(f"fig4/{preset_name}/p={pp}", lambda: curve_vs_v(f"p={pp}", 1e10, vgrid, preset=preset, p=pp))
        P.plot_rate(ax, c["x"], c["rate"], color=color, ls=ls, label=f"p = {pp:g}")
        rows[f"p={pp:g}"] = {"v_min": c["threshold"], "rate_V1": c["rate"][-1]}
    vopt = cache.curve(f"vmin/{preset_name}/popt/1e+10",
                       lambda: {"v": threshold_visibility(10 ** 10, preset, p="opt")})["v"]
    rows["p optimised"] = {"v_min": vopt, "paper": PAPER["fig5_vmin_popt"]["1e+10"]}
    rows["p=0"]["paper"] = PAPER["fig5_vmin_p0"]["1e+10"]
    P.mark_paper_threshold(ax, [PAPER["fig5_vmin_p0"]["1e+10"], PAPER["fig5_vmin_popt"]["1e+10"]])
    ax.set_xlim(0.84, 1.0)
    P.log_rate_axes(ax)
    ax.set_xlabel("Visibility V")
    ax.set_title("Fig. 4 - signature rate vs V, n = 10^10, noisy preprocessing p", fontsize=10, loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig4_rate_vs_visibility_p.png")
    plt.close(fig)
    summary["fig4"] = rows


def fig5(cache: Cache, preset_name: str, summary: dict):
    preset = PRESETS[preset_name]
    plt = P.plt
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    vgrid = np.linspace(0.84, 1.0, 33)
    rows = {}
    for color, n in zip((P.ORANGE, P.BLUE, P.YELLOW), (1e10, 1e9, 1e8)):
        k = _fmt(n)
        p_leg = PAPER["fig5_p_legend"][k]
        c0 = cache.curve(f"fig5/{preset_name}/n={k}/p=0", lambda: curve_vs_v("p=0", n, vgrid, preset=preset, p=0.0))
        c1 = cache.curve(f"fig5/{preset_name}/n={k}/p={p_leg}",
                         lambda: curve_vs_v(f"p={p_leg}", n, vgrid, preset=preset, p=p_leg))
        P.plot_rate(ax, c0["x"], c0["rate"], color=color, label=f"n = {k}, p = 0")
        P.plot_rate(ax, c1["x"], c1["rate"], color=color, ls="--", label=f"n = {k}, p = {p_leg:g}")
        vopt = cache.curve(f"vmin/{preset_name}/popt/{k}",
                           lambda: {"v": threshold_visibility(int(n), preset, p="opt")})["v"]
        rows[k] = {"v_min_p0": c0["threshold"], "paper_p0": PAPER["fig5_vmin_p0"][k],
                   f"v_min_p={p_leg:g}": c1["threshold"], "v_min_popt": vopt, "paper_popt": PAPER["fig5_vmin_popt"][k]}
    P.mark_paper_threshold(ax, list(PAPER["fig5_vmin_p0"].values()) + list(PAPER["fig5_vmin_popt"].values()))
    ax.set_xlim(0.84, 1.0)
    P.log_rate_axes(ax)
    ax.set_xlabel("Visibility V")
    ax.set_title("Fig. 5 - signature rate vs V: p = 0 (solid) and p from the paper's legend (dashed)",
                 fontsize=10, loc="left")
    ax.legend(loc="lower right", fontsize=8.5, ncol=1)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig5_rate_vs_visibility_n.png")
    plt.close(fig)
    summary["fig5"] = rows


# ================================================================================================ Fig. 6
DIGITISED["fig6_threshold"] = {"[73]": 8.09e5, "[74]": 2.23e7, "[75]": 1.85e6}
DIGITISED["fig6_rate_1e10"] = {"[73]": 3.97e7, "[74]": 9.68e6, "[75]": 2.74e7}
DIGITISED["fig6_markers"] = {"star [73] n=1.5e6": 1.3e3, "diamond [75] n=2.4e6": 7.0e2}


def fig6(cache: Cache, preset_name: str, summary: dict):
    from diqds.part3_figures.matter import (DIAMOND_N, EC_MODEL, EXPERIMENTS, PROTOCOL, REF73, REF75, STAR_N,
                                            ref73_threshold_check)
    from diqds.part3_figures.sweeps import KeyLength, signature_rate
    preset = PRESETS[preset_name]
    plt = P.plt
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    grid = np.geomspace(1e5, 1e10, 61)
    rows, ends = {}, []
    colors = {"[73]": P.BLUE, "[74]": P.ORANGE, "[75]": P.YELLOW}
    for e in EXPERIMENTS:
        c = cache.curve(f"fig6/{preset_name}/{e.ref}",
                        lambda: curve_vs_n(e.ref, grid, honest=e.honest, preset=preset, p=0.0, protocol=PROTOCOL,
                                           ec_model=EC_MODEL))
        lab = f"Ref. {e.ref}: {e.platform}, {e.distance} (S = {e.chsh}, Q = {100 * e.qber:.2f} %)"
        ends.append(P.plot_rate(ax, c["x"], c["rate"], color=colors[e.ref], label=lab, end_label=f"Ref. {e.ref}"))
        rows[e.ref] = {"threshold_n": c["threshold"], "hours_to_threshold": e.hours_for(c["threshold"]),
                       "rate_1e10": c["rate"][-1], "paper_threshold_n (digitised)": DIGITISED["fig6_threshold"][e.ref],
                       "paper_rate_1e10 (digitised)": DIGITISED["fig6_rate_1e10"][e.ref]}

    def point(exp, n):
        return cache.curve(f"fig6/{preset_name}/{exp.ref}/point/{n:.3g}", lambda: {"rate": signature_rate(
            KeyLength(preset, 0.0, PROTOCOL, EC_MODEL)(n, honest=exp.honest).ell)})["rate"]
    r_star, r_dia = point(REF73, STAR_N), point(REF75, DIAMOND_N)
    ax.plot([STAR_N], [max(r_star, 1.0)], ls="none", marker="*", ms=14, mfc=P.BLUE, mec=P.INK, mew=0.8,
            label=f"Ref. [73]: n = 1.5e6 (~{REF73.hours_for(STAR_N):.0f} h)", zorder=6)
    ax.plot([DIAMOND_N], [max(r_dia, 1.0)], ls="none", marker="D", ms=8, mfc=P.YELLOW, mec=P.INK, mew=0.8,
            label=f"Ref. [75]: n = 2.4e6 (~{REF75.hours_for(DIAMOND_N):.0f} h, extrapolated)", zorder=6)
    rows["markers"] = {"star [73] n=1.5e6": r_star, "diamond [75] n=2.4e6": r_dia,
                       "paper (digitised)": DIGITISED["fig6_markers"],
                       "hours [73] 1.5e6": REF73.hours_for(STAR_N), "hours [75] 2.4e6": REF75.hours_for(DIAMOND_N)}
    rows["check: omega_thr of Ref. [73] from Eq. (B10)"] = {"computed": ref73_threshold_check(), "Ref. [73]": 0.825538}
    ax.set_xscale("log")
    ax.set_xlim(1e5, 1e10)
    P.log_rate_axes(ax)
    P.end_labels(ax, ends)
    ax.set_xlabel("Number of rounds n")
    ax.set_title("Fig. 6 - matter-qubit experiments, KGP of Ref. [73]", fontsize=10, loc="left")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig6_matter_experiments.png")
    plt.close(fig)
    summary["fig6"] = rows


# ================================================================================================ Fig. 7
DIGITISED["fig7_rate_L0"] = {"ideal/1e14": 1.63e6, "ideal/2e14": 3.35e6, "real/1e14": 9.6e4, "real/2e14": 2.8e5}


def fig7(cache: Cache, preset_name: str, summary: dict):
    from diqds.part3_figures import photonic as ph
    preset = PRESETS[preset_name]
    plt = P.plt

    def table(key, params, Ls, heralds=ph.HERALDS_DEFAULT):
        d = cache.curve(f"fig7/table/{key}", lambda: ph.OperatingTable.from_points(
            ph.operating_curve(params, Ls, heralds=heralds, maxiter=300)).to_json())
        return ph.OperatingTable.from_json(d)

    sources = {
        "ideal": table("ideal", ph.IDEAL_SOURCE, np.arange(0.0, 170.1, 5.0)),
        "real": table(f"real_c13_k{ph.REALISTIC_KAPPA:.4f}", ph.realistic_source("c13"), np.arange(0.0, 60.1, 2.5)),
    }
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    rows = {}
    for src, ls, Lgrid in (("real", "-", np.arange(0.0, 60.1, 2.5)), ("ideal", "--", np.arange(0.0, 170.1, 5.0))):
        for N, color in ((1e14, P.YELLOW), (2e14, P.ORANGE)):
            c = cache.curve(f"fig7/{preset_name}/{src}/{N:.0e}" + (f"/k{ph.REALISTIC_KAPPA:.4f}" if src == "real" else ""),
                            lambda: ph.curve_vs_distance(src, sources[src], N, Lgrid, preset=preset))
            name = "realistic source" if src == "real" else "ideal source"
            P.plot_rate(ax, c["x"], c["rate"], color=color, ls=ls, label=f"{name}, N = {N:.0e}".replace("e+", "e"))
            key = f"{src}/{N:.0e}".replace("+", "")
            rows[key] = {"L_max_km": c["threshold"], "paper_km": PAPER["fig7_Lmax_km"][key], "rate_L0": c["rate"][0],
                         "paper_rate_L0 (digitised)": DIGITISED["fig7_rate_L0"][key]}
    # the other reading of the realistic source, for the record
    expl = table("real_explicit", ph.realistic_source("explicit"), np.arange(0.0, 60.1, 2.5))
    rows["real, explicit-g2 reading (no fit)"] = {
        f"{N:.0e}": cache.curve(f"fig7/{preset_name}/real_explicit/Lmax/{N:.0e}",
                                lambda: {"L": ph.max_distance(expl, N, preset=preset, hi=60.0)})["L"] for N in (1e14, 2e14)}
    rows["realistic kappa (Eq. C13, fitted on N = 1e14)"] = ph.REALISTIC_KAPPA
    for src, tab in sources.items():
        rows[f"{src}: S, Q at L = 0"] = (8.0 * tab.omega[0] - 4.0, tab.qber[0])
        rows[f"{src}: P_H at L = 0"] = tab.PH[0]
    rows["hours for 1e14 rounds at 75 MHz (paper: ~370 h)"] = 1e14 / ph.PhotonicParams().rate_hz / 3600.0
    P.mark_paper_threshold(ax, list(PAPER["fig7_Lmax_km"].values()), label="cut-off printed in paper")
    ax.set_xlim(0, 160)
    P.log_rate_axes(ax, ylim=(1.0, 1e7))
    ax.set_xlabel("Distance L between Alice and Bob (km)")
    ax.set_title("Fig. 7 - all-photonic DIQDS, QD sources: realistic V_HOM = 0.975, g2 = 0.01 (solid);\n"
                 "ideal V_HOM = 1, g2 = 0 (dashed); T = 0.01, eta_l = 0.97, eta_conv = 0.5, L0 = 22 km",
                 fontsize=9, loc="left")
    ax.legend(loc="lower left", bbox_to_anchor=(0.215, 0.03), fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig7_photonic_distance.png")
    plt.close(fig)
    summary["fig7"] = rows


FIGS = {3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", type=int, nargs="*", default=sorted(FIGS))
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--preset", default="paper_fit", choices=sorted(PRESETS))
    a = ap.parse_args(argv)
    P.setup()
    cache = Cache(CACHE, a.recompute)
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() and not a.recompute else {}
    print(f"n_s = {signature_length()} (Eq. 5, Table I);  Figs. 3-5 preset: {a.preset}")
    for f in a.figs:
        t0 = time.time()
        print(f"Fig. {f} ...")
        FIGS[f](cache, a.preset, summary)
        print(f"  done in {time.time() - t0:.1f} s")
    SUMMARY.write_text(json.dumps(summary, indent=1, default=lambda o: float(o) if math.isfinite(o) else None))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
