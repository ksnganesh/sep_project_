# Part 3 — Results section (Sec. IV, Figs. 3–7)

Fan, Bian, Li, Wen, Yin, Chen, *Device-independent quantum digital signatures under realistic conditions*,
Phys. Rev. A **114**, 032617 (2026): Sec. IV, Figs. 3–7, Appendices C and D.

Subpackage `diqds/part3_figures/` plus `scripts/part3_figures.py`. It reuses Part 1 (n_s = 46, Eq. 4) and Part 2
(Eq. 9 = B12 and its optimiser) unchanged, apart from two backward-compatible switches added to Part 2. Those are `honest=`
(arbitrary device statistics instead of a Werner state) and `protocol=` ('random_key' as in the paper, 'single_key' for the
KGP of Ref. [73]). All 31 Part 2 tests still pass.

```
python scripts/part3_figures.py                 # all five figures from the cache results/part3_curves.json (~10 s)
python scripts/part3_figures.py --recompute     # redo every optimisation (~12 min)
python scripts/part3_figures.py --preset rigorous   # Figs. 3-7 under the theorem-consistent reading instead of PAPER_FIT
python -m pytest -q tests/test_part3_figures.py # 16 tests, ~1 min
```

Outputs: `results/fig3_rate_vs_rounds.png`, `fig4_rate_vs_visibility_p.png`, `fig5_rate_vs_visibility_n.png`,
`fig6_matter_experiments.png`, `fig7_photonic_distance.png`, `part3_summary.json` (every number below), and
`part3_curves.json` (curve data).

## Paper → code

| Paper | Module | What it does |
|---|---|---|
| Eq. (4) with Eq. (9), "parameters numerically optimized" | `sweeps.py` | r_sig = ℓ/(3·46) along n or V; locates the finite-size threshold first and resolves the vertical onset |
| Figs. 3–5, depolarizing noise | `sweeps.curve_vs_n / curve_vs_v` | Werner devices, ideal CHSH settings, p fixed or optimised |
| Fig. 6, "parameters in Refs. [73–75] and KGP in Ref. [73]" | `matter.py` | reported (S, Q) of the three experiments; single-key-basis KGP (η = (1−γ) r_p + γ r_0, Eq. B11 as printed) |
| Fig. 1(b), App. C, App. D, Fig. 7 | `photonic.py` | exact ≤5-photon linear-optics model of the heralded photonic link → P_H(L), (ω, Q)(L); angles optimised |

## Results against the paper

Paper values are the printed numbers where the text gives them. Otherwise they were read off the published figures by
pixel digitisation (±3 %), marked "dig.". All values are in `results/part3_summary.json`.

### Figs. 3–5 (depolarizing noise, `PAPER_FIT` reading of Part 2)

| quantity | this code | paper |
|---|---|---|
| V_min, p = 0, n = 1e8 / 1e9 / 1e10 | 0.8836 / 0.8685 / 0.8621 | 0.8835 / 0.8686 / 0.8622 |
| V_min, p optimised, n = 1e8 / 1e9 / 1e10 | 0.8798 / 0.8605 / 0.8505 | 0.8799 / 0.8608 / 0.8508 |
| V_min at the legend's fixed p (0.035 / 0.090 / 0.155) | 0.87984 / 0.86053 / 0.85048 | (= p-optimised value, to 1e-5) |
| Fig. 3 onset n for V = 1 / 0.96 / 0.92 / 0.88 | 9.09e5 / 2.46e6 / 9.61e6 / 1.50e8 | 8.83e5 / 2.39e6 / 9.39e6 / 1.46e8 (dig.) |
| Fig. 3 r_sig at n = 1e10, V = 1 / 0.96 / 0.92 / 0.88 | 3.45e7 / 2.11e7 / 1.14e7 / 3.28e6 | 3.29e7 / 2.00e7 / 1.08e7 / 3.11e6 (dig.) |

Finding: the p values in the Fig. 5 legend are exactly the threshold-optimal p. The fixed-p and p-optimised thresholds agree
to 1e-5, which confirms that the dashed curves are "p optimised at threshold, then held fixed".

### Fig. 6 (matter-qubit experiments)

| | Ref. [73] ions | Ref. [74] atoms 400 m | Ref. [75] atoms 11 km |
|---|---|---|---|
| reported S, Q used | 2.677, 1.44 % | 2.578, 7.8 % | 2.612, 2.85 % |
| r_sig at n = 1e10 (this / paper dig.) | 4.05e7 / 3.97e7 | 9.92e6 / 9.68e6 | 2.82e7 / 2.74e7 |
| onset n (this / paper dig.) | 1.04e6 / 8.1e5 | 2.94e7 / 2.2e7 | 2.39e6 / 1.85e6 |
| time to onset at the reported event rate | 5.6 h | 6.6e5 h | 1235 h |

* **The KGP of Ref. [73] is identified, and that resolves the B11 puzzle of Part 2.** Eq. (B10) with k = 3 and Ref. [73]'s
  design point (S = 2.64, γ = 13/256, n = 1.5e6) gives ω_thr = 0.825538, the value Ref. [73] quotes. Eq. (B11) as printed
  gives 296,500 bits, which is Ref. [73]'s actual syndrome length m = 296,517. So (B10)/(B11) are the parameter choices of
  Ref. [73]'s single-key-basis protocol. (B11) is right for Fig. 6, and the sifted version is right for Figs. 3–5.
* Large-n rates agree within 2–3 %, so the protocol model (no sifting, B11) is right. The finite-size onsets are later than in
  the paper by a nearly constant factor of 1.29–1.32. Equivalently, about 14 % more √n-order cost appears in the
  single-key case. That difference is not in the printed formulas. Under `LOSS_ONLY_OPT` (β optimised) the factor drops to
  1.16. Removing the −nγ term does not fit all three curves.
* The paper's statements hold. n = 1.5e6 (8 h) is above Ref. [73]'s onset. n = 2.4e6 (≈1240 h at Ref. [75]'s 0.54
  events/s) gives a positive rate (onset 2.39e6). Ref. [74] needs ~3e7 rounds. At 3342 rounds per 75 h that is ~6.6e5 h.
  The paper's caption says "on the order of 10^7 hours", but its own onset of ~2.2e7 rounds gives ~5e5 h at that rate.

### Fig. 7 (all-photonic, quantum-dot sources)

| | this code | paper |
|---|---|---|
| ideal source, cut-off N = 1e14 / 2e14 | **111.21 / 126.32 km** | 111.46 / 126.53 km |
| realistic source, cut-off N = 1e14 / 2e14 | 17.82 (fitted) / **29.68 km** (predicted) | 17.82 / 28.87 km |
| r_sig at L = 0, ideal N = 1e14 / 2e14 | 1.56e6 / 3.24e6 | 1.63e6 / 3.35e6 (dig.) |
| r_sig at L = 0, realistic N = 1e14 / 2e14 | 9.8e4 / 2.84e5 | 9.6e4 / 2.8e5 (dig.) |
| hours for 1e14 rounds at ν = 75 MHz | 370 h | ≈ 370 h |

The model in `photonic.py` was written from Fig. 1(b), Appendices C and D and nothing else:

* Each station has an H and a V photon in one spatial mode. They pass through T beam splitters, then lossy channels
  η_C = η_conv e^{−L/2L0}. The CHS beam splitter t sits in the ± basis, followed by threshold detectors.
* Local detection along a polarisation angle has efficiency η_l and no-click binning. The honest device may also relabel
  outputs.
* Partial distinguishability is modelled as a mixture with pairwise visibility V_HOM. The multiphoton term is an extra,
  distinguishable photon.
* The amplitudes are computed exactly in second quantisation for up to 5 photons. The herald conditioning uses a Möbius
  (OR-) convolution over the four click bits.

Checks (tests): the network is lossless with its loss modes; ideal optics give S = 2√2; the heralded state is
|+−⟩ − t/(1−t)|−+⟩ (Eq. D1) for t = 0.3; and P_H = T²η_C²/2.

* **Herald pattern.** Eq. (D1)'s state comes from the single click pattern {d1−, d2+}. The other pattern gives the mirror
  state t ↔ 1−t. With that one pattern, the ideal-source cut-offs land 0.25 km from the paper with no fitted parameter.
  Accepting both patterns doubles P_H and moves every cut-off out by ~15 km.
* **Why doubling N adds less than L0 ln 2 = 15.25 km.** The paper has +15.07 km (ideal) and +11.05 km (realistic). In the
  model, false heralds from three photons in the channels (∝ T(1−η_C)) grow with L, so the per-round quality degrades with
  distance. That gives +15.11 km (ideal) and +11.9 km (realistic) with no tuning. The effect is stronger for the
  realistic source because it operates close to the asymptotic threshold.
* **Realistic source (the one fitted parameter).** The paper gives V_HOM = 0.975 and g2 = 0.01, and Eq. (C13)
  V = V⁽⁰⁾ − κ g2 with 1 < κ < 3. It does not say how g2 enters the model ("we adopt a modified approach"). Two readings:
  * `explicit`: the extra photon is simulated (as in Ref. [80]). Cut-offs are 12.85 / 24.25 km, short by 5 km.
  * `c13` (default): V_eff = 0.975 − κ·0.01 and no extra photon. κ = 1.5446, inside the paper's range, is fitted to the
    printed 17.82 km. The N = 2e14 cut-off is then a prediction: 29.68 vs 28.87 km. Both L = 0 rates agree to 2 %.
* Optimised honest statistics at L = 0: ideal S = 2.631, Q = 3.48 %. Realistic (c13) S = 2.450, Q = 6.65 %. With
  η_l = 0.97 the ideal case uses the standard CHSH angles. The realistic optimum tilts Alice's angles by about −0.18 rad
  because of the no-click bias. Noisy preprocessing (p optimised) is used, as the text suggests for photonic links.

## Choices the paper does not state (all explicit in the code)

1. Finite-size reading: `PAPER_FIT` (Part 2, β share 0.4) for every figure, so Figs. 3–7 share one calibration that was fixed
   on Fig. 5 alone. `--preset rigorous` regenerates everything under the theorem-consistent reading (κ of Fig. 7 stays the
   value fitted under `PAPER_FIT`).
2. Fig. 6 uses each experiment's headline (S, Q) and its average event rate for the time axis. The paper writes "1 m" for
   Ref. [73], whose ions are about 2 m apart. That does not matter here.
3. Fig. 7: one herald pattern (Eq. D1). α = β = V_HOM, as in Ref. [80]. The CHS detectors are perfect, since their losses
   sit in η_conv. There are no dark counts, because the paper lists none. g2 is treated via C13 with κ fitted (see above).
4. The rates are signatures per protocol execution, un-floored, as in Eq. (4).

## Status of the optional quantum backends (Part 2)

QuTiP 5.3.1, Qiskit 2.6.2 with qiskit-aer 0.18.2, and SeQUeNCe are now installed in the reference environment. The three
backend tests of Part 2, written but never run before, **pass** with them.
