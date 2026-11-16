# Part 2 — Distribution stage

Fan et al., *Device-independent quantum digital signatures under realistic conditions*, PRA **114**, 032617 (2026):
Sec. II.A (steps 1–7), Appendix B (Eq. 9 = B12), the depolarizing-noise devices of Sec. IV.

Subpackage `diqds/part2_distribution/` — NumPy, SciPy, Matplotlib; optional QuTiP, Qiskit Aer, SeQUeNCe. Like Part 1 it is an
independent, paper-derived reference that does not touch your existing flat modules. It uses Part 1 only in
`kgp.distribute_signature_keys` (lazy import).

```
python scripts/part2_distribution_demo.py              # ~10 s from the cached optimisations in results/part2_thresholds.json
python scripts/part2_distribution_demo.py --recompute  # redo all 48 optimisations (~3 min)
python -m pytest -q tests/test_part2_distribution.py   # ~1 min; 3 tests need QuTiP / Qiskit Aer / SeQUeNCe and skip otherwise (31 tests)
```

## Paper → code

| Paper | Module | What it does |
|---|---|---|
| App. B, Eq. (9)/(B12) | `finitekey.py` | Tan et al. Thm 1 (Quantum 6, 880), every term separately; written from the source theorem, not from the garbled PDF |
| "numerically optimized" | `optimize.py` | closed form for α′, 1-D search for α, grid-seeded Nelder–Mead over (γ, t, ε split, p, β); `v_min`, `n_min` |
| open choices (below) | `presets.py` | `STRICT`, `RIGOROUS`, `LOSS_ONLY_OPT`, `PAPER_FIT` |
| II.A steps 1–7 | `kgp.py`, `privacy.py` | the protocol on real bit strings; FFT Toeplitz hashing (SciPy) for the tag and privacy amplification |
| Sec. IV honest devices | `devices.py` (+ `quantum_qutip.py`, `quantum_aer.py`) | Werner state + CHSH settings; exact probabilities and sampler |
| "initiated upon a heralding signal" | `heralding.py` | acquisition timeline: NumPy model and SeQUeNCe event-kernel backend |

## Result 1 — an independent re-derivation agrees with your repo

Eq. (31) implemented from Tan et al.'s Theorem 1 gives, for the reading "β optimised, loss count only" (`LOSS_ONLY_OPT`),
V_min = 0.8827 / 0.8681 / 0.8619 (p = 0) and 0.8789 / 0.8601 / 0.8503 (p optimised) at n = 1e8 / 1e9 / 1e10: the README of
`ksnganesh/QIQDS` lists 0.8827 / 0.8681 / 0.8619 and 0.8788 / 0.8601 / 0.8503. Its worked example `signature_rate_werner(1e9, 0.95)`
(ℓ ≈ 2.4e8, γ ≈ 0.008) is matched too: ℓ = 2.377e8, γ = 0.0079. (Numerical agreement only; I could not read `finite_key.py`.)
Also checked: derivatives of r_p vs finite differences (3e-9), convexity of η (tangents are valid lower bounds on the whole quantum
range), E[f_min] = g(ω), and that Tan's variance bound (33) dominates the exact variance everywhere (within 1 % of tight);
ℓ/n → ½(r_p(ω) − h₂(Q′)) as n → ∞ (Tan Eq. 39).

## Result 2 — what the paper leaves open is β, and which counts step 6 bounds

β ∈ [g(0), g(1)] is the free parameter of the min-tradeoff function (Tan Eq. 33); the paper does not list it among the optimised
parameters. It sets the variance term of the entropy-accumulation theorem (minimal at the midpoint, verified) and the thresholds
move by up to 6e-3 in V at n = 1e8. Choosing β < g(1) is only covered by the proof if **both** the win and the loss count are
bounded (Tan Eq. 30), and the win count has the larger variance, so the margin is larger than (B10). Step 6 as written checks only
n_loss, which pairs with β = g(1). Deviation from the paper's printed value in brackets:

| n | paper | repo README | strict | rigorous | loss-only opt | paper-fit |
|---|---|---|---|---|---|---|
| **p = 0** | | | | | | |
| 1e8 | 0.8835 | 0.8827 | 0.8894 (+0.0059) | 0.8847 (+0.0012) | 0.8827 (−0.0008) | 0.8836 (+0.0001) |
| 1e9 | 0.8686 | 0.8681 | 0.8713 (+0.0027) | 0.8690 (+0.0004) | 0.8681 (−0.0005) | 0.8685 (−0.0001) |
| 1e10 | 0.8622 | 0.8619 | 0.8634 (+0.0012) | 0.8623 (+0.0001) | 0.8619 (−0.0003) | 0.8621 (−0.0001) |
| **p optimised** | | | | | | |
| 1e8 | 0.8799 | 0.8788 | 0.8861 (+0.0062) | 0.8811 (+0.0012) | 0.8789 (−0.0010) | 0.8798 (−0.0001) |
| 1e9 | 0.8608 | 0.8601 | 0.8638 (+0.0030) | 0.8612 (+0.0004) | 0.8601 (−0.0007) | 0.8605 (−0.0003) |
| 1e10 | 0.8508 | 0.8503 | 0.8521 (+0.0013) | 0.8509 (+0.0001) | 0.8503 (−0.0005) | 0.8505 (−0.0003) |

* `strict`: β = g(1), loss check only — the protocol as written.
* `rigorous`: β optimised, loss **and** win counts bounded (win-side margin) — consistent with Tan's Theorem 1; **default for the protocol**.
* `loss_only_opt`: β optimised, loss count only — what the README numbers correspond to; the proof does not cover this pairing.
* `paper_fit`: β = g(1) − 0.4 (g(1) − g(0)), loss only. One parameter fitted to the six printed numbers: a calibration, not a derivation.

The printed thresholds sit between `rigorous` and `loss_only_opt`; the spread between natural readings (≈2e-3 at 1e8, ≈1e-4 at
1e10) is the size of what the paper does not pin down. For Figs. 3–5 use `PAPER_FIT` to match the printed numbers or `RIGOROUS`
to stay consistent with the theorem. (`PAPER_FIT` was calibrated on p = 0 at 3 points; the p-optimised points are out of sample.)

## Result 3 — Eq. (B11) as printed is not what produced Figs. 3–5

(B11) as printed gives V_min ≈ 0.929 / 0.917 / 0.912 (β = g(1)); the sifted form
`EC_max = n[(1−γ)/2·h₂(Q′) + γ·h₂((4−S)/8)] + 50√n`, `Q′ = p + (1−2p)Q`, which is Tan's `h_hon` (Eq. 26) with the paper's 50√n,
gives 0.889 / 0.871 / 0.863. The √n coefficient itself matters little (≤ 2e-3 over 0…50 at n = 1e8). This matches the finding in
your README; (B11) as printed corresponds to the single-key-basis protocol of Ref. [73] (Fig. 6, Part 3).

## Result 4 — the protocol on real data, then Part 1

`n = 2·10⁶` rounds per link, `V = 0.995`, `RIGOROUS`: γ = 0.0641, loss limit 19,961, win minimum 108,214, EC_max = 171,898,
ℓ = 170,796 bits, completeness ≥ 0.9987 (exact binomials). Both links accept with equal keys (S observed 2.803 ± 0.008 vs 2.814 expected;
losses 19,131 ≤ 19,961; EC leak 101,444 ≤ 171,898); 1,237 key triples; 101 of 101 tested signatures verify in Part 1's messaging stage.
Failure modes behave as intended: devices of V = 0.97 against a design of 0.995 → "EC budget exceeded"; a code with f_EC = 1.8
→ rejected; a code up to f_EC ≈ 1.7 still fits in EC_max at this n. Finite-size floor: even for V → 1 a positive key needs ≈ 1.0–1.8·10⁶ rounds
(`n_min`: 1.8e6 at V = 0.99, 1.45e6 at 0.999 for `STRICT`; 1.28e6 / 1.03e6 for `RIGOROUS`). Panel (b) of the figure shows what limits the
key at 1e9 rounds: apart from the asymptotic EC leakage (≈ 25 % of n·g, plus 0.45 % for the 50√n margin) the largest costs are n·γ (2.5 %), the second-order term (1.9 %) and the smoothing terms (1.5 %); 66.9 % of n·g survives.

A bug found on the way and fixed: with a single Nelder–Mead start the optimiser ran into the γ boundary at n = 3.2e6, V = 0.995 and
returned a negative key where +3.2e5 exists, which made `n_min` jump. It is now seeded by a (log γ, t) grid (regression test included).

## What I could not run, and simplifications

* **Update (Part 3):** with QuTiP 5.3.1, Qiskit 2.6.2 + qiskit-aer 0.18.2 and SeQUeNCe installed, the three backend tests below
  now run and **pass** (`pytest tests/` : 83 passed, none skipped).
* **QuTiP, Qiskit Aer and SeQUeNCe backends are written but were NOT executed** (no network in the development sandbox, none of the three
  installed). Each is a thin adapter checked against the NumPy reference by a test that skips when the library is missing:
  `quantum_qutip.py` (expects `check_against_numpy`), `quantum_aer.py` (Werner state via `depolarizing_error(1−V, 2)` after the CX;
  statistics vs exact within 5σ), `heralding.sequence_timestamps` (mean gap vs `cycle/p`, both modes). They assume QuTiP ≥ 4.7,
  Qiskit ≥ 1.0 with qiskit-aer ≥ 0.14, SeQUeNCe ≥ 0.6. Please run `python -m pytest -q tests/test_part2_distribution.py` after installing and send me any failure.
  What *is* verified without the libraries is the logic, not the API: a NumPy density-matrix emulation of the QuTiP computation matches the
  closed-form probabilities (signs, conventions); a NumPy emulation of the Aer circuit (h, cx, depolarizing, Ry(−θ), Z-measurement) matches
  too, and the little-endian `'c1c0'` parsing is tested directly; the SeQUeNCe station code runs on a 30-line fake kernel with the same
  Timeline/Entity/Event/Process contract. Whether the real packages accept these calls is what the three skipped tests will tell you.
* **Error correction is an oracle** (ideal leakage `f_EC·[log₂C(m_key, e_key) + log₂C(m_test, e_test)]`), as the paper only bounds EC_max.
  Your repo has a real Cascade; the oracle is the pluggable point.
* PE margins use the k = 3 Gaussian rule of (B10) (win/loss max-variance for `two_sided`), not Tan's exact binomial condition (30);
  completeness is then evaluated exactly. r_p is the closed form (B2) as printed; Tan's numerical bound could be tighter.
* ε budget: `ε_s = (ε_snd − ε_PA − 2ε_h)/2 ≈ 2e-11`, `ε_h = 2⁻⁶¹` (61-bit tag, as in Sec. III.C), ε_EA = ε_PA = 1e-11 (Table I).

## Roadmap — Part 3 (scenario models and Figs. 3–7) — done, see PART3_NOTES.md

Figs. 3–5 are `optimum`/`threshold_*` sweeps; Fig. 6 needs the matter-qubit parameters of Refs. [73–75] and the rate model
(`HeraldingModel`, `ec_model="printed"` for the single-key-basis protocol); Fig. 7 needs the photonic state and heralding
probability of Appendix C/D, which the paper does not fully specify (the table is in the supplement of Ref. [80]).
