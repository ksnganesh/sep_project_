#!/usr/bin/env python
"""Part 2 - distribution stage of Fan et al., PRA 114, 032617 (2026): demo + validation.

    python scripts/part2_distribution_demo.py              # uses results/part2_thresholds.json for the slow optimisations (~15 s)
    python scripts/part2_distribution_demo.py --recompute  # redo every optimisation (several minutes)
    python scripts/part2_distribution_demo.py --quick      # skip the optimised-p thresholds and the dense n grid

Outputs (results/):  part2_distribution_validation.png,  part2_thresholds.json
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom

from diqds.part1_messaging import decode_document, encode_document, run_messaging_stage
from diqds.part2_distribution import (LOSS_ONLY_OPT, PAPER_FIT, RIGOROUS, STRICT, WernerDevice, distribute_signature_keys,
                                      plan_kgp, run_kgp, statistics, threshold_rounds, threshold_visibility)
from diqds.part2_distribution.presets import optimum

RESULTS = ROOT / "results"
CACHE = RESULTS / "part2_thresholds.json"

# Minimum visibilities printed in Sec. IV of the paper (discussion of Fig. 5), and those in the repo README (ksnganesh/QIQDS).
NS_PAPER = (10 ** 8, 10 ** 9, 10 ** 10)
PAPER = {"p0": (0.8835, 0.8686, 0.8622), "popt": (0.8799, 0.8608, 0.8508)}
REPO = {"p0": (0.8827, 0.8681, 0.8619), "popt": (0.8788, 0.8601, 0.8503)}
PRESET_LIST = (STRICT, RIGOROUS, LOSS_ONLY_OPT, PAPER_FIT)


class Cache:
    def __init__(self, path: Path, recompute: bool):
        self.path, self.recompute = path, recompute
        self.data = {} if (recompute or not path.exists()) else json.loads(path.read_text())

    def get(self, key: str, fn, allow_compute: bool = True):
        if key in self.data:
            return self.data[key]
        if not allow_compute:
            return None
        self.data[key] = float(fn())
        self.path.parent.mkdir(exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True))
        return self.data[key]


# ------------------------------------------------------------------ 1. terms of Eq. (31)
def section_terms():
    print("=" * 100 + "\n1. Eq. (31) term by term: n = 1e9 rounds, V = 0.95, p = 0, RIGOROUS reading\n" + "=" * 100)
    o = optimum(10 ** 9, 0.95, RIGOROUS, p=0.0)
    print(f"  optimum: gamma = {o.gamma:.4f}   t = {o.t:.4f}   omega_thr = {o.w_thr:.5f}   alpha-1 = {o.a:.2e}   alpha'-1 = {o.a_prime:.2e}")
    ng = o.terms["n_g"]
    for k, v in o.terms.items():
        if k != "ell":
            print(f"    {k:>15}: {v:+.4e}   ({100 * v / ng:+7.3f} % of n g)")
    print(f"    {'ell':>15}: {o.ell:+.4e}   -> {o.ell / o.n:.4f} key bits per round, {o.ell / (3 * 46):.3e} signatures (ns = 46)")
    return o


# ------------------------------------------------------------------ 2. thresholds vs the paper
def section_thresholds(cache: Cache, quick: bool):
    print("\n" + "=" * 100 + "\n2. Minimum visibility for a positive key (Fig. 5 numbers): paper vs repo README vs this implementation\n" + "=" * 100)
    names = [p.name for p in PRESET_LIST]
    table = {}
    for pk, pspec in (("p0", 0.0), ("popt", "opt")):
        if pk == "popt" and quick:
            continue
        print(f"\n  {'noisy preprocessing p = 0' if pk == 'p0' else 'p optimised'}")
        print(f"  {'n':>6} | {'paper':>6} | {'repo':>6} | " + " | ".join(f"{nm:>13}" for nm in names))
        for i, n in enumerate(NS_PAPER):
            row = []
            for pr in PRESET_LIST:
                t0 = time.time()
                v = cache.get(f"vmin/{pr.name}/{pk}/{n:.0e}", lambda: threshold_visibility(n, pr, pspec, tol=2e-6))
                row.append(v)
                table[(pk, n, pr.name)] = v
            print(f"  {n:>6.0e} | {PAPER[pk][i]:>6.4f} | {REPO[pk][i]:>6.4f} | " + " | ".join(f"{v:>13.4f}" for v in row))
            print(f"  {'':>6}   {'':>6}   {'':>6}   " + "   ".join(f"{v - PAPER[pk][i]:>+13.4f}" for v in row) + "   (deviation from paper)")
    return table


# ------------------------------------------------------------------ 3. V_min(n) curves
def section_curves(cache: Cache, quick: bool):
    grid = [1e8, 1e9, 1e10] if quick else [1e7, 3e7, 1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11]
    curves = {}
    for pr in (STRICT, RIGOROUS, PAPER_FIT, LOSS_ONLY_OPT):
        curves[pr.name] = [cache.get(f"vmin/{pr.name}/p0/{n:.0e}", lambda: threshold_visibility(int(n), pr, 0.0, tol=2e-6)) for n in grid]
    return grid, curves


# ------------------------------------------------------------------ 4. protocol end to end
def section_protocol(rng):
    n, vis, ns = 2_000_000, 0.995, 46
    print("\n" + "=" * 100 + f"\n3. Key-generation protocol on real data, then Part 1: n = {n:.0e} rounds per link, V = {vis}, RIGOROUS reading\n" + "=" * 100)
    plan = plan_kgp(n, vis, RIGOROUS)
    print(f"  plan: gamma={plan.gamma:.4f}  omega_thr={plan.omega_thr:.5f}  loss limit={plan.n_loss_max}  win minimum={plan.n_win_min}  "
          f"EC_max={plan.ec_max}  tag={plan.tag_bits} bits  ell={plan.ell}  completeness >= {plan.completeness_lower_bound():.4f}")
    t0 = time.time()
    triples, r_ab, r_ac = distribute_signature_keys(n, vis, ns, rng, plan=plan)
    print(f"  link A-B ({time.time() - t0:.1f} s for both links): accepted={r_ab.accepted} ell={r_ab.ell} keys equal={r_ab.keys_agree}; "
          f"Bell tests {r_ab.n_test} (wins {r_ab.n_win}, losses {r_ab.n_loss}, limit {plan.n_loss_max}); S observed {r_ab.chsh_observed:.4f}")
    print(f"           key errors e_key={r_ab.e_key}, e_test={r_ab.e_test}; EC leak {r_ab.ec_leak} <= {plan.ec_max}; sifted out {r_ab.n_sifted_out}, matched {r_ab.n_matched}")
    print(f"  link A-C: accepted={r_ac.accepted} ell={r_ac.ell} keys equal={r_ac.keys_agree}")
    print(f"  -> {len(triples)} key triples (ell / (3 ns) = {r_ab.ell / (3 * ns):.1f})")
    doc = encode_document(b"DIQDS: signed with device-independent keys", 1000)
    out = run_messaging_stage(doc, *triples[0])
    print(f"  Part 1 messaging stage with these keys: Bob accepts={out.bob_accepts}, Charlie accepts={out.charlie_accepts}, text={decode_document(out.details['signed'].doc)!r}")
    n_ok = sum(run_messaging_stage(encode_document(b"document %d" % i, 1000), *triples[i]).valid for i in range(1, 101))
    print(f"  100 further signatures, each with a fresh key triple: {n_ok}/100 valid")
    print("  failure modes:")
    for label, kw, dev in (("true devices V=0.97 (plan assumed 0.995)", {}, WernerDevice(0.97)),
                           ("real EC code, f_EC = 1.8", {"f_ec": 1.8}, WernerDevice(vis))):
        r = run_kgp(plan, dev, rng, **kw)
        print(f"    {label:<45} -> accepted={r.accepted}: {r.reason} (EC leak {r.ec_leak} vs max {r.ec_max})")
    return plan, r_ab


# ------------------------------------------------------------------ 5. figure
def make_figure(o, grid, curves, plan, r_ab, path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    a = ax[0]
    styles = {"strict": ("C7", "--", "STRICT: $\\beta=g(1)$, loss check only"), "rigorous": ("C0", "-", "RIGOROUS: $\\beta$ optimised, loss+win checks"),
              "loss_only_opt": ("C1", ":", "$\\beta$ optimised, loss check only (= repo README)"), "paper_fit": ("C2", "-.", "PAPER_FIT: $\\beta$ share 0.4")}
    for name, (c, ls, lab) in styles.items():
        a.semilogx(grid, curves[name], color=c, ls=ls, lw=1.8, label=lab)
    a.semilogx(NS_PAPER, PAPER["p0"], "k*", ms=15, zorder=5, label="paper, printed ($p=0$)")
    a.semilogx(NS_PAPER, REPO["p0"], "x", color="m", ms=9, mew=2, zorder=6, label="repo README ($p=0$)")
    a.set_xlabel("number of rounds $n$")
    a.set_ylabel("minimum visibility $V_{\\min}$ for a positive key")
    a.set_title("(a) Finite-size threshold, $p=0$ (Fig. 5 numbers)")
    a.set_ylim(0.855, 0.95)
    a.legend(fontsize=7.5, loc="upper right")
    a.grid(alpha=0.3, which="both")

    b = ax[1]
    ng = o.terms["n_g"]
    margin = 50.0 * math.sqrt(o.n)                                   # the paper's "+50 sqrt(n)" inside EC_max
    pretty = {"second_order_V": "2nd order: $n a V^2$", "second_order_K": "2nd order: $n a^2 K_\\alpha$", "n_gamma": "test rounds: $n\\gamma$",
              "max_entropy_B": "max-entropy: $n a' V'^2$", "theta_s1": "smoothing: $\\vartheta(\\epsilon_s')/a$",
              "theta_s2": "smoothing: $\\vartheta(\\epsilon_s'')/a'$", "eps_EA": "$\\epsilon_{EA}$: $(1/a+1/a')\\log(1/\\epsilon_{EA})$",
              "theta_rest": "$3\\vartheta(\\epsilon_s-\\epsilon_s'-2\\epsilon_s'')$", "hash_tag": "61-bit verification tag",
              "eps_PA": "$2\\log(1/\\epsilon_{PA})$"}
    items = {"EC leakage $n\\,h_{\\rm hon}$ (asymptotic)": (-o.terms["EC_max"] - margin) / ng * 100,
             "EC margin $50\\sqrt{n}$": margin / ng * 100}
    for k, lab in pretty.items():
        items[lab] = -o.terms[k] / ng * 100
    order = sorted(items, key=items.get)
    colors = ["0.55" if lab.startswith("EC leakage") else "C3" for lab in order]
    b.barh(order, [items[k] for k in order], color=colors, alpha=0.85)
    b.set_xscale("log")
    b.set_xlim(1e-5, 1e2)
    b.set_xlabel("cost as % of the leading term $n\\,g(\\omega_{\\rm thr})$   (red: finite-size corrections)")
    b.set_title(f"(b) What limits the key: $n=10^{{{int(round(math.log10(o.n)))}}}$, $V={o.vis}$ (RIGOROUS)\n"
                f"$\\ell/(n g) = {100 * o.ell / ng:.1f}\\,\\%$ survives")
    b.tick_params(axis="y", labelsize=8)
    b.grid(alpha=0.3, which="both", axis="x")

    c = ax[2]
    n, g, w = plan.n, plan.gamma, plan.omega
    lo = int(n * g * (1 - w) * 0.95)
    ks = np.arange(lo, int(plan.n_loss_max * 1.07))
    for vis, col in ((0.995, "C0"), (0.985, "C2"), (0.975, "C1"), (0.965, "C3")):
        wv = statistics(WernerDevice(vis).P)[1]
        mean = n * g * (1 - wv)
        c.plot(ks, binom.pmf(ks, n, g * (1 - wv)) * 1e3, color=col,
               label=f"true devices $V={vis}$: mean {mean:,.0f}, passes {binom.cdf(plan.n_loss_max, n, g * (1 - wv)):.1%}")
    c.axvline(plan.n_loss_max, color="k", ls="--", label=f"abort above {plan.n_loss_max:,}")
    c.axvline(r_ab.n_loss, color="C0", ls=":", label=f"this run: {r_ab.n_loss:,}")
    c.set_xlabel("number of lost CHSH rounds $n_{\\rm loss}$")
    c.set_ylabel("probability  ($\\times10^{-3}$)")
    c.set_title("(c) Parameter estimation: $n=2\\times10^6$, designed for $V=0.995$\n"
                f"honest devices pass with probability $\\geq${plan.completeness_lower_bound():.4f}")
    c.set_ylim(0, 4.6)
    c.legend(fontsize=7.5, loc="upper left")
    c.grid(alpha=0.3)

    fig.suptitle("Part 2 - distribution stage: finite-size bound (Tan et al. Thm 1 / paper App. B) and protocol validation", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recompute", action="store_true", help="ignore the cached optimisations")
    ap.add_argument("--quick", action="store_true", help="skip optimised-p thresholds, coarse n grid")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(2026)
    cache = Cache(CACHE, args.recompute)
    o = section_terms()
    section_thresholds(cache, args.quick)
    grid, curves = section_curves(cache, args.quick)
    plan, r_ab = section_protocol(rng)
    RESULTS.mkdir(exist_ok=True)
    fig_path = RESULTS / "part2_distribution_validation.png"
    make_figure(o, grid, curves, plan, r_ab, fig_path)
    print(f"\nwrote {fig_path.relative_to(ROOT)}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
