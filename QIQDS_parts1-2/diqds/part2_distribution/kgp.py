"""Key-generation protocol (KGP) of the distribution stage, Sec. II.A steps 1-7, run on real bit strings.

    1. parameters      n, gamma, omega_thr, p, EC_max          <- ``plan_kgp`` (finite-key optimiser, Eq. 31)
    2. acquisition     X in {0,1}; Y in {2,3} w.p. gamma (Bell-test round) else Y in {0,1} (key round); outputs A, B
    3. sifting         key round with X != Y:  A = B = 0
    4. noisy preproc.  matched key round: A <- A xor F, Pr[F = 1] = p
    5. error correction  Alice -> Bob; stop when EC_max is reached; then a 2-universal tag of log2(1/eps_h) bits is compared
    6. parameter estimation  score U_i = [A~_i xor B_i = X_i (Y_i - 2)] on Bell-test rounds; abort unless n_loss <= n gamma (1 - w_thr)
                             (and, for the two-sided readings, n_win >= n gamma w_thr)
    7. privacy amplification  K_A = T A,  K_B = T A~  with a random Toeplitz T of ell rows

Honest simplification (the paper only *bounds* the EC leakage by EC_max and does not specify a code): step 5 uses an ORACLE
decoder.  Bob's first guess is B (matched key round), 0 (sifted round) or B xor X(Y-2) (Bell-test round); the oracle then
reveals the corrections at the information-theoretic cost  f_EC * [log2 C(m_key, e_key) + log2 C(m_test, e_test)]  bits (two
error classes, since X, Y are public).  If that exceeds EC_max the exchange stops and Bob keeps his uncorrected guess, so the
tag check fails.  A real code would pay f_EC > 1 (the repo's Cascade does this for real).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from scipy.special import gammaln
from scipy.stats import binom

from .devices import WernerDevice
from .finitekey import Budget, werner_honest
from .optimize import Optimum, PSpec
from .presets import RIGOROUS, Preset, optimum
from .privacy import random_seed, toeplitz_hash


def log2_binom(m: int, e: int) -> float:
    if e <= 0 or e >= m:
        return 0.0
    return float((gammaln(m + 1) - gammaln(e + 1) - gammaln(m - e + 1)) / math.log(2.0))


@dataclass(frozen=True)
class KGPPlan:
    """Step 1: everything fixed before the first round."""
    n: int
    vis: float                  # visibility the thresholds were designed for
    preset: str
    gamma: float
    p: float
    omega: float                # honest CHSH winning probability
    omega_thr: float
    ec_max: int
    ell: int                    # secret-key length from Eq. (31), floored (<= 0: no key)
    tag_bits: int
    two_sided: bool
    n_loss_max: int
    n_win_min: int
    optimum: Optimum = field(repr=False, compare=False)

    def completeness_lower_bound(self) -> float:
        """Pr[honest devices pass parameter estimation] >= 1 - (loss tail) - (win tail), from the exact binomials."""
        q_loss = self.gamma * (1.0 - self.omega)
        ok = binom.cdf(self.n_loss_max, self.n, q_loss)
        if not self.two_sided:
            return float(ok)
        ok_win = binom.sf(self.n_win_min - 1, self.n, self.gamma * self.omega)
        return float(max(0.0, 1.0 - (1.0 - ok) - (1.0 - ok_win)))


def plan_kgp(n: int, vis: float, preset: Preset = RIGOROUS, p: PSpec = 0.0, budget: Budget = Budget(), **kw) -> KGPPlan:
    o = optimum(n, vis, preset, p=p, budget=budget, **kw)
    hon = werner_honest(vis)
    return KGPPlan(n=n, vis=vis, preset=preset.name, gamma=o.gamma, p=o.p, omega=hon.omega, omega_thr=o.w_thr,
                   ec_max=int(math.floor(o.ec_max)), ell=int(math.floor(o.ell)), tag_bits=budget.tag_bits,
                   two_sided=preset.two_sided,
                   n_loss_max=int(math.floor(n * o.gamma * (1.0 - o.w_thr))),
                   n_win_min=int(math.ceil(n * o.gamma * o.w_thr)), optimum=o)


@dataclass
class KGPResult:
    accepted: bool
    reason: str
    ell: int
    key_a: Optional[np.ndarray] = None
    key_b: Optional[np.ndarray] = None
    n_test: int = 0
    n_win: int = 0
    n_loss: int = 0
    n_matched: int = 0
    n_sifted_out: int = 0
    n_flipped: int = 0
    e_key: int = 0
    e_test: int = 0
    ec_leak: int = 0
    ec_max: int = 0
    tag_ok: bool = False
    chsh_observed: float = float("nan")

    @property
    def keys_agree(self) -> bool:
        return self.key_a is not None and self.key_b is not None and bool(np.array_equal(self.key_a, self.key_b))


def run_kgp(plan: KGPPlan, device: WernerDevice, rng: np.random.Generator, *, f_ec: float = 1.0,
            ell_override: Optional[int] = None) -> KGPResult:
    """One execution of steps 2-7.  ``ell_override`` bypasses Eq. (31) (mechanics tests at sizes where the bound gives no key)."""
    n = plan.n
    # 2. data acquisition
    x = rng.integers(0, 2, size=n, dtype=np.uint8)
    test = rng.random(n) < plan.gamma
    y = np.where(test, rng.integers(2, 4, size=n), rng.integers(0, 2, size=n)).astype(np.uint8)
    a, b = device.sample(x, y, rng)

    # 3. sifting
    key_round = ~test
    sifted_out = key_round & (x != y)
    matched = key_round & (x == y)
    a[sifted_out] = 0
    b[sifted_out] = 0

    # 4. noisy preprocessing
    n_flipped = 0
    if plan.p > 0.0:
        flips = rng.random(n) < plan.p
        a[matched] ^= flips[matched].astype(np.uint8)
        n_flipped = int(flips[matched].sum())

    ti = np.flatnonzero(test)
    res = KGPResult(accepted=False, reason="", ell=0, n_test=int(ti.size), n_matched=int(matched.sum()),
                    n_sifted_out=int(sifted_out.sum()), n_flipped=n_flipped, ec_max=plan.ec_max)
    # observed CHSH value from Alice's TRUE outcomes before EC (diagnostic only; a real run would use A~)
    if ti.size:
        win_true = ((a[ti] ^ b[ti]) == (x[ti] & (y[ti] - 2)))
        res.chsh_observed = 8.0 * float(win_true.mean()) - 4.0

    # 5. error correction (oracle with ideal leakage) + verification tag
    a0 = b.copy()
    a0[ti] = b[ti] ^ (x[ti] & (y[ti] - 2))
    err = a != a0
    res.e_key, res.e_test = int(err[matched].sum()), int(err[ti].sum())
    res.ec_leak = int(math.ceil(f_ec * (log2_binom(res.n_matched, res.e_key) + log2_binom(res.n_test, res.e_test))))
    a_tilde = a.copy() if res.ec_leak <= plan.ec_max else a0
    tag_seed = random_seed(n, plan.tag_bits, rng)
    res.tag_ok = bool(np.array_equal(toeplitz_hash(a, tag_seed, plan.tag_bits),
                                     toeplitz_hash(a_tilde, tag_seed, plan.tag_bits)))

    # 6. parameter estimation (on A~)
    win = ((a_tilde[ti] ^ b[ti]) == (x[ti] & (y[ti] - 2)))
    res.n_win = int(win.sum())
    res.n_loss = res.n_test - res.n_win
    if not res.tag_ok:
        res.reason = "tag mismatch" if res.ec_leak <= plan.ec_max else "EC budget exceeded"
        return res
    if res.n_loss > plan.n_loss_max:
        res.reason = "too many losses"
        return res
    if plan.two_sided and res.n_win < plan.n_win_min:
        res.reason = "too few wins"
        return res

    # 7. privacy amplification
    ell = plan.ell if ell_override is None else int(ell_override)
    if ell <= 0:
        res.reason = "finite-size bound gives no key"
        return res
    pa_seed = random_seed(n, ell, rng)
    res.key_a = toeplitz_hash(a, pa_seed, ell)
    res.key_b = toeplitz_hash(a_tilde, pa_seed, ell)
    res.ell, res.accepted, res.reason = ell, True, "ok"
    return res


# ---------------------------------------------------------------------------------------------- glue to Part 1
def distribute_signature_keys(n: int, vis: float, ns: int, rng: np.random.Generator, *,
                              preset: Preset = RIGOROUS, p: PSpec = 0.0,
                              device_factory: Callable[[float], WernerDevice] = WernerDevice,
                              f_ec: float = 1.0, ell_override: Optional[int] = None,
                              plan: Optional[KGPPlan] = None) -> Tuple[Optional[List[tuple]], KGPResult, KGPResult]:
    """Run the KGP on the Alice-Bob and the Alice-Charlie link and cut the two keys into Part 1's per-signature key triples
    (3 ns bits of each link per signature, Eq. 4).  Returns (triples or None, result_AB, result_AC)."""
    from diqds.part1_messaging import split_pairwise_keys     # Part 1 (messaging stage)

    plan = plan or plan_kgp(n, vis, preset, p)
    r_ab = run_kgp(plan, device_factory(vis), rng, f_ec=f_ec, ell_override=ell_override)
    r_ac = run_kgp(plan, device_factory(vis), rng, f_ec=f_ec, ell_override=ell_override)
    if not (r_ab.accepted and r_ac.accepted):
        return None, r_ab, r_ac
    triples = split_pairwise_keys(r_ab.key_a, r_ab.key_b, r_ac.key_a, r_ac.key_b, ns)
    return triples, r_ab, r_ac
