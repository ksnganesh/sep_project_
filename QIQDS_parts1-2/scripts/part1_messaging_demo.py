#!/usr/bin/env python
"""Part 1 - messaging stage of Fan et al., PRA 114, 032617 (2026): demo + validation.

    python scripts/part1_messaging_demo.py            # ~30 s
    python scripts/part1_messaging_demo.py --quick    # fewer Monte-Carlo trials (~10 s)

Outputs (in results/):  part1_messaging_validation.png,  part1_kat.json
"""
from __future__ import annotations

import argparse
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

from diqds.part1_messaging import (Bob, Charlie, PartyKeys, TABLE_I, budget, decode_document, encode_document,
                                   eps_forgery, ideal_key_triples, make_bob_tamper, max_guesses,
                                   minimal_signature_length, multiple_of_polynomial, poly_from_key,
                                   product_of_polynomials, random_bits, random_forgery, run_messaging_stage,
                                   sign)
from diqds.part1_messaging.gf2poly import count_irreducibles, irreducibles
from diqds.part1_messaging.kat import write_kat

RESULTS = ROOT / "results"


# ------------------------------------------------------------------ 1. security budget (Eqs. 4-8, Table I)
def section_budget():
    print("=" * 78 + "\n1. Security budget from Table I  (Eqs. 5-8)\n" + "=" * 78)
    b = budget(TABLE_I)
    print(b)
    print(f"\n  ns = 45 would give eps_for = {eps_forgery(45, TABLE_I.nm, TABLE_I.eps_snd):.3e} > {TABLE_I.eps:.0e}")
    print("  sensitivity to the (unstated) choice of eps_sec in Eq. (5):")
    for es in (TABLE_I.eps_snd, 0.0):
        print(f"    eps_sec = {es:.0e}  ->  n_s = "
              f"{minimal_signature_length(TABLE_I.nm, TABLE_I.eps, es, TABLE_I.eps_corr, TABLE_I.eps_prime)}")
    print("  n_s vs message length (cross-check with COMPARISON.md section 4: 42 -> 56 for 1e2 -> 1e6 bits):")
    for nm in (10 ** 2, 10 ** 3, 10 ** 6):
        print(f"    n_m = {nm:>7}  ->  n_s = "
              f"{minimal_signature_length(nm, TABLE_I.eps, TABLE_I.eps_snd, TABLE_I.eps_corr, TABLE_I.eps_prime)}")
    return b


# ------------------------------------------------------------------ 2. one honest run (Fig. 2)
def section_transcript(rng):
    print("\n" + "=" * 78 + "\n2. One honest run of the messaging stage (Fig. 2), ns = 46, nm = 1000\n" + "=" * 78)
    A, B, C = ideal_key_triples(46, rng)
    doc = encode_document(b"Transfer 100 units from Alice to Bob, ref #2026-0001", TABLE_I.nm)
    out = run_messaging_stage(doc, A, B, C)
    for line in out.transcript:
        print("  " + str(line))
    print(f"  Bob accepts: {out.bob_accepts}   Charlie accepts: {out.charlie_accepts}   signature valid: {out.valid}")
    print(f"  signed text: {decode_document(out.details['signed'].doc)!r}")


# ------------------------------------------------------------------ 3. attack battery at paper size
def section_attacks(rng, trials):
    print("\n" + "=" * 78 + f"\n3. Attack battery at paper size (ns = 46, nm = 1000), {trials} runs each\n" + "=" * 78)
    ns, nm = 46, TABLE_I.nm
    rows = []

    def tally(name, expected, fn):
        hits = sum(bool(fn()) for _ in range(trials))
        rows.append((name, hits, expected))
        print(f"  {name:<58} {hits:>4}/{trials}   (expected: {expected})")

    def fresh():
        A, B, C = ideal_key_triples(ns, rng)
        return A, B, C, random_bits(nm, rng)

    def honest():
        A, B, C, d = fresh()
        return run_messaging_stage(d, A, B, C, hash_method="horner").valid

    def forge_random():
        A, B, C, d = fresh()
        dM, t = random_forgery(ns, nm, rng)
        return run_messaging_stage(d, A, B, C, bob_tamper=make_bob_tamper(dM, t), hash_method="horner").charlie_accepts

    def forge_flip():
        A, B, C, d = fresh()

        def flip(sig, doc, y):
            doc[int(rng.integers(nm))] ^= 1
            return sig, doc, y
        return run_messaging_stage(d, A, B, C, bob_tamper=flip, hash_method="horner").charlie_accepts

    def forge_wrong_poly():
        A, B, C, d = fresh()
        A2, _, _ = ideal_key_triples(ns, rng)                       # Bob's guess of p_a: another random irreducible
        dM, t = multiple_of_polynomial(poly_from_key(A2.Z), nm, rng)
        return run_messaging_stage(d, A, B, C, bob_tamper=make_bob_tamper(dM, t), hash_method="horner").charlie_accepts

    def forge_known_poly():
        A, B, C, d = fresh()
        dM, t = multiple_of_polynomial(poly_from_key(A.Z), nm, rng)  # Bob somehow learned Z_a
        return run_messaging_stage(d, A, B, C, bob_tamper=make_bob_tamper(dM, t), hash_method="horner").charlie_accepts

    def repudiation_consistent():
        A, B, C, d = fresh()
        msg, _ = sign(d, A, "horner")
        sig = random_bits(ns, rng) if rng.random() < 0.5 else msg.sig   # Alice sends garbage or a genuine signature
        bob, ch = Bob(B, "horner"), Charlie(C, "horner")
        bob.receive(type(msg)(sig, d))
        ch.record(bob.forward_packet())
        ok_b, _ = bob.verify(ch.reveal())
        ok_c, _ = ch.certify(True)
        return ok_b == ok_c

    def key_mismatch_aborts():
        A, B, C, d = fresh()
        which = "XYZ"[int(rng.integers(3))]
        arr = getattr(C, which).copy()
        arr[int(rng.integers(ns))] ^= 1
        bad = PartyKeys(**{**dict(X=C.X, Y=C.Y, Z=C.Z), which: arr})
        return not run_messaging_stage(d, A, B, bad, hash_method="horner").valid

    tally("honest run: both Bob and Charlie accept", f"{trials}", honest)
    tally("forgery, random (dM, t): Charlie accepts", f"~0  (2^-46 = {2.0**-46:.1e})", forge_random)
    tally("forgery, one flipped document bit: Charlie accepts", "0", forge_flip)
    tally("forgery, dM = multiple of a WRONG guess of p_a: accepts", f"~0  (<= {eps_forgery(46, nm, 0.0):.1e})", forge_wrong_poly)
    tally("forgery, dM = multiple of the TRUE p_a (Prop. 1): accepts", f"{trials}  (Z_a must stay hidden)", forge_known_poly)
    tally("repudiation: Bob and Charlie reach the same verdict", f"{trials}", repudiation_consistent)
    tally("robustness: one wrong key bit -> honest run aborts", f"{trials}", key_mismatch_aborts)
    return rows


# ------------------------------------------------------------------ 4. forging bound at small ns (exact + Monte Carlo)
def preimage_counts_standin(n):
    """#keys Z that the 'first irreducible at or after x^n + Z' stand-in maps to each irreducible polynomial."""
    low = np.array([p & ((1 << n) - 1) for p in irreducibles(n)], dtype=np.int64)
    gaps = np.diff(low, prepend=low[-1] - (1 << n))                    # circular gaps: first polynomial inherits the wrap
    return gaps


def optimal_success(q, n):
    return q + (1.0 - q) * 2.0 ** -n


def section_small_n(rng, mc_scale):
    print("\n" + "=" * 78 + "\n4. Forging bound of Eq. (5) at small ns  (nm = 64): exact vs Monte Carlo vs stand-in f\n" + "=" * 78)
    nm = 64
    ns_list = list(range(8, 17))
    bound, p_shoup, p_standin, ratio = [], [], [], []
    for n in ns_list:
        k = max_guesses(n, nm)
        irr = count_irreducibles(n)
        gaps = np.sort(preimage_counts_standin(n))[::-1]
        assert gaps.sum() == 1 << n
        q_sh = k / irr
        q_st = gaps[:k].sum() / 2.0 ** n
        bound.append(eps_forgery(n, nm, 0.0))
        p_shoup.append(optimal_success(q_sh, n))
        p_standin.append(optimal_success(q_st, n))
        ratio.append(p_standin[-1] / bound[-1])
        print(f"  ns={n:>2}  k={k}  |Irr|={irr:>5}   bound={bound[-1]:.3e}   Shoup f (uniform) = {p_shoup[-1]:.3e} "
              f"({p_shoup[-1]/bound[-1]:.2f} x bound)   stand-in f = {p_standin[-1]:.3e} ({ratio[-1]:.2f} x bound)")

    print("\n  Monte Carlo through the full protocol (Shoup f, real LFSR-Toeplitz hash):")
    mc = {}
    for n, trials in ((8, int(20000 * mc_scale)), (10, int(20000 * mc_scale)), (12, int(40000 * mc_scale))):
        k = max_guesses(n, nm)
        polys = list(irreducibles(n))[:k]
        dM, t = product_of_polynomials(polys, nm, n)
        hits = 0
        for _ in range(trials):
            A, B, C = ideal_key_triples(n, rng)
            doc = random_bits(nm, rng)
            hits += run_messaging_stage(doc, A, B, C, bob_tamper=make_bob_tamper(dM, t),
                                        hash_method="horner").charlie_accepts
        obs = hits / trials
        th = optimal_success(k / count_irreducibles(n), n)
        sig = math.sqrt(th * (1 - th) / trials)
        mc[n] = (obs, sig, th)
        print(f"    ns={n:>2}: observed {obs:.4f} +- {sig:.4f}   theory {th:.4f}   z = {(obs - th) / sig:+.2f}   "
              f"bound {eps_forgery(n, nm, 0.0):.4f}")
    return dict(ns=ns_list, bound=bound, shoup=p_shoup, standin=p_standin, ratio=ratio, mc=mc, nm=nm)


# ------------------------------------------------------------------ 5. figure
def make_figure(b, small, path):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.9), constrained_layout=True)

    # (a) budget vs ns
    a = ax[0]
    ns_grid = np.arange(30, 61)
    a.semilogy(ns_grid, [eps_forgery(int(n), TABLE_I.nm, TABLE_I.eps_snd) for n in ns_grid], "o-", ms=4,
               label=r"$\varepsilon_{\rm for}=2^{-n_s+1}n_m+\varepsilon_{\rm sec}$,  $\varepsilon_{\rm sec}=\varepsilon_{\rm snd}$")
    a.semilogy(ns_grid, [eps_forgery(int(n), TABLE_I.nm, 0.0) for n in ns_grid], ":", color="C0",
               label=r"$2^{-n_s+1}n_m$ alone")
    a.axhline(b.eps_rep, color="C2", ls="-.", label=r"$\varepsilon_{\rm rep}\approx\varepsilon_{\rm rob}\approx 2\times10^{-11}$")
    a.axhline(TABLE_I.eps, color="k", ls="--", label=r"target $\varepsilon=10^{-10}$ (Table I)")
    a.axvline(b.ns, color="gray", lw=1, label=f"minimal $n_s={b.ns}$  ($\\varepsilon_{{\\rm for}}={b.eps_for:.2e}$)")
    a.set_ylim(5e-12, 1e-8)
    a.set_xlabel("signature length $n_s$ (bits)")
    a.set_ylabel("failure probability")
    a.set_title("(a) Security budget, Eqs. (5)-(8), $n_m=10^3$")
    a.legend(fontsize=7.5, loc="upper right")
    a.grid(alpha=0.3, which="both")

    # (b) optimal forgery vs bound at small ns
    s = small
    b_ax = ax[1]
    b_ax.semilogy(s["ns"], s["bound"], "k--", label=r"paper bound $n_m 2^{-n_s+1}$")
    b_ax.semilogy(s["ns"], s["shoup"], "o-", color="C0", label="optimal attack, $f$ = Shoup/BM (uniform $p_a$)")
    b_ax.semilogy(s["ns"], s["standin"], "s-", color="C3", label="optimal attack, stand-in $f$ (non-uniform)")
    xs = sorted(s["mc"])
    b_ax.errorbar(xs, [s["mc"][n][0] for n in xs], yerr=[s["mc"][n][1] for n in xs], fmt="k*", ms=9, capsize=3,
                  label="Monte Carlo, full protocol (Shoup $f$)")
    b_ax.set_xlabel("signature length $n_s$ (bits)")
    b_ax.set_ylabel("forging success probability")
    b_ax.set_title(f"(b) Optimal forging attack, $n_m={s['nm']}$")
    b_ax.legend(fontsize=7.5, loc="lower left")
    b_ax.grid(alpha=0.3, which="both")

    # (c) preimage distribution of f at n = 12
    c = ax[2]
    n = 12
    gaps = preimage_counts_standin(n)
    c.hist(gaps, bins=np.arange(0, gaps.max() + 6, 4), color="C3", alpha=0.75, label="stand-in $f$ (first irreducible $\\geq x^n+Z$)")
    c.bar([n], [count_irreducibles(n)], width=2.2, color="C0", label=f"Shoup/BM $f$: all {count_irreducibles(n)} polynomials have exactly {n}")
    c.set_xlabel(f"number of {n}-bit keys $Z$ mapped to one irreducible polynomial")
    c.set_ylabel("number of polynomials")
    c.set_title(f"(c) How uniform is $p_a=f(Z_a)$?  ($n_s={n}$, exhaustive)")
    c.legend(fontsize=7.5)
    c.grid(alpha=0.3)

    fig.suptitle("Part 1 - messaging stage: security bookkeeping and validation of the forging bound", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer Monte-Carlo trials")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(2026)

    b = section_budget()
    section_transcript(rng)
    section_attacks(rng, trials=100 if args.quick else 300)
    small = section_small_n(rng, mc_scale=0.25 if args.quick else 1.0)

    RESULTS.mkdir(exist_ok=True)
    fig_path = RESULTS / "part1_messaging_validation.png"
    make_figure(b, small, fig_path)
    kat_path = write_kat(RESULTS / "part1_kat.json")
    print(f"\nwrote {fig_path.relative_to(ROOT)} and {kat_path.relative_to(ROOT)}   ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
