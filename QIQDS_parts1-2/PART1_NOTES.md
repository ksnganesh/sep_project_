# Part 1 — Messaging stage

Fan, Bian, Li, Wen, Yin, Chen, *Device-independent quantum digital signatures under realistic conditions*,
Phys. Rev. A **114**, 032617 (2026): Sec. II.B, Appendix A, Sec. III, Eqs. (4)–(8), Table I.

Self-contained subpackage `diqds/part1_messaging/` (NumPy + SciPy + Matplotlib only — nothing quantum is needed
until Part 2). It does not touch the existing flat modules (`diqds/messaging.py`, `hashing.py`, `gf2.py`,
`security.py`); it is an independent, paper-derived reference you can diff against.

```
python scripts/part1_messaging_demo.py          # ~15 s: budget, Fig. 2 transcript, attacks, figure, KAT
python -m pytest -q tests/test_part1_messaging.py
```

## Paper → code

| Paper | Module | What it does |
|---|---|---|
| App. A, Eqs. (A1)–(A12) | `lfsr_toeplitz.py` | `H=(s, Ws, …, W^{m-1}s)` three ways: literal powers of the companion matrix `W`, Toeplitz form `H[i,j]=S_{n-i+j}` (`scipy.linalg.toeplitz`), and Horner/LFSR form `M_W(W)s` |
| II.B.1 `p_a=f(Z_a)` | `polygen.py`, `gf2poly.py` | f from **Ref. [32], App. B.1** (Shoup minimal polynomial + Berlekamp–Massey), Rabin test Eq. (13) |
| II.B.1–3, Fig. 2 | `protocol.py`, `keys.py` | sign / verify / certify; Charlie cannot reveal `{X_c,Y_c,Z_c}` before recording Bob's packet |
| III.A forgery | `attacks.py` | random forgery, known-`p_a` forgery (Prop. 1), product-of-guesses (optimal attack) |
| Eqs. (4)–(8), Table I | `security.py` | `r_sig`, `ε_for`, `ε_rep`, `ε_rob`, `ε`, minimal `n_s` |

## Verification (all reproduced by the demo/tests)

| Statement | Check | Result |
|---|---|---|
| `H` is Toeplitz and equals `M_W(W)s` (A9–A11) | 3 independent implementations, up to `(n,m)=(46,1000)` | identical |
| `p(x)` divides `M(x)` ⇒ `h(M)=0` (A12, Prop. 1) | random multiples, `n=46, m=1000` | holds for every `s` |
| `p ∤ M ⇒ s ↦ h(M)` is a bijection (why each digest has probability `2^-n`) | exhaustive, `n=8` | 256/256 distinct |
| `f(Z)` uniform on irreducibles | exhaustive `n=5,8,10,11` (+ other public moduli) | every polynomial has **exactly n** preimages |
| Random forgery succeeds w.p. `2^-ns` | 300 runs, `n_s=46` | 0 accepted |
| Bob must not learn `Z_a` (Prop. 1) | attacker who knows `p_a` | 300/300 forged |
| Repudiation (III.B) | 300 garbage/genuine signatures | Bob and Charlie agree 300/300 |
| Robustness (III.C) | one wrong key bit | aborts 300/300 |
| Eq. (5) bound | exact + Monte Carlo through the full protocol, `n_s=8…16`, `n_m=64` | optimal attack = 0.38–0.51 × bound; MC within 1.1 σ of theory |
| Table I → `n_s` | Eq. (5) | `n_s=46`, `ε_for=7.84e-11`, `ε_rep=ε_rob=2.0e-11`, `ε=7.84e-11 ≤ 1e-10`, 138 key bits/signature |

`n_s=46`, `ε=7.84e-11` and `n_s: 42→56` for `n_m: 1e2→1e6` agree with `COMPARISON.md` §4 of the repo.

## The polynomial generator `f` (open item in the README)

The paper only says f is "based on Ref. [32]". Ref. [32] (arXiv:2301.01132, App. B.1) specifies it: fix a public
irreducible `f0(x)` of degree `n`, read the `n` key bits as an element `g` of GF(2^n), form `a_i = [g^i mod f0](0)`
for `i<2n`, and run Berlekamp–Massey; the result is the minimal polynomial of `g`. Each degree-`n` irreducible
polynomial is the minimal polynomial of exactly `n` elements, so `p_a` is **exactly uniform** over irreducibles — what
the bound `2^{-ns+1}·n_m` of Eq. (5) assumes.

The stand-in described in the README ("first irreducible at or after `x^n+Z`") is *not* uniform: a polynomial's
probability is the gap before it. Because Bob's best attack multiplies the `⌊(n_m-1)/n_s⌋` most likely polynomials
into one forged document, this inflates his success probability. Computed exactly (panel b/c of
`results/part1_messaging_validation.png`): 1.1–2.6 × the paper's bound for `n_s=9…16`, tending to grow with `n_s`.
An extreme-value (exponential-gap) estimate suggests roughly 10 × at `n_s=46` — an estimate, not a computation.
No reproduced *number* changes (`n_s`, `r_sig` do not depend on `f`); what changes is whether the bound is true.

Degenerate keys: with probability ≈ `2^{-n_s/2}` (≈1.2e-7 at 46) `g` lies in a proper subfield and the minimal
polynomial has degree `< n_s`; `DegenerateKeyError` is raised and the key block is skipped (`keys.split_pairwise_keys`).

## Choices the paper does not state

1. **`ε_sec` in Eq. (5)** is bounded by `ε_snd` (Table I). With `ε_sec=0` one gets `n_s=45`.
2. **`n_s`** = smallest integer with `ε_for ≤ ε`. **`r_sig`** is returned un-floored (`floor=True` for whole signatures).
3. **Key segmentation:** one signature uses 3·`n_s` consecutive bits of `K_AB` and of `K_AC`, cut `(X|Y|Z)`;
   `X_a=X_c`, `Z_a=Z_c` come from `K_AC`, `Y_a=Y_b⊕Y_c`. Bob's `X_b`, `Z_b` are never used (as in the text).
4. **Variant:** Fan et al. derive the polynomial from `Z_a` and use `X_a` as LFSR state, `Y_a` as one-time pad;
   Ref. [32] instead sends an encrypted random `p_a`. The code follows Fan et al.; Ref. [32] is used for `f` and the bounds.
5. **Symbols garbled in the PDF text:** `ε'` (classical-message error probability, ≤1e-11) was confirmed from Ref. [32] Sec. IV.3.
6. **Document padding** (`bits.encode_document`, injective 1-0* padding) is an addition so byte strings can be signed.
7. Bit order is MSB-first everywhere (`bits.py`); the public modulus for `n=46` is `x^46+x+1` (any choice gives the same distribution).

## Checking the existing modules against this one

`results/part1_kat.json` holds known-answer vectors (a hand-checkable `n=4` toy hash with its explicit `H`,
`f(Z)` vectors, paper-size hashes, a full Fig. 2 run, the Table I budget). Feed the stored inputs to
`diqds.hashing` / `diqds.messaging` / `gf2.irreducible_from_key` and compare with the stored outputs;
`diqds.part1_messaging.kat.check_kat(json.load(...))` does this for this package.

## Roadmap (proposal)

| Part | Content | Libraries |
|---|---|---|
| 2 Distribution | data acquisition, sifting, noisy preprocessing, EC, parameter estimation, PA; key length Eq. (9)/(B12) | NumPy/SciPy (bound + optimisation), **QuTiP** (Werner state, CHSH, heralded state D1), **Qiskit Aer** (round sampling with noise), **SeQUeNCe** (heralding timing, event rates, 8 h / 624 h / 75 MHz claims) |
| 3 Final graphs | Figs. 3–7 | Matplotlib on Part 2 output + `security.signature_rate` |
