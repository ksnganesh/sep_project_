"""Finite-size secret-key length of the distribution stage.

Appendix B of Fan et al. (PRA 114, 032617) transcribes Theorem 1 of Tan, Sekatski, Bancal, Schwonnek, Renner,
Sangouard, Lim, "Improved DIQKD protocols with finite-size analysis", Quantum 6, 880 (2022), arXiv:2012.08714.
The PDF text of the paper is garbled; this module is written from the source theorem (Tan et al. Eqs. 24-33).

    l_key <= n g(w_thr) - n (a ln2 / 2) V^2 - n a^2 K_alpha - n gamma - n (a'/4) V'^2          a = alpha-1, a' = alpha'-1
             - theta(e_s')/a - theta(e_s'')/a' - (1/a + 1/a') log2(1/e_EA) - 3 theta(e_s - e_s' - 2 e_s'')
             - EC_max - ceil(log2(1/e_h)) - 2 log2(1/e_PA) + 2                                       (Tan Eq. 31 = paper Eq. B12)

Notation clash: ``V`` in the EAT terms is the *variance term* (``v_eat`` here); the Werner-state visibility is ``vis``.

Single-round entropy bound (paper B1-B4)
    r_0(w)  = 1 - h2((1 + sqrt(16w^2 - 16w + 3)) / 2)
    r_p(w)  = r_0(w) + h2((1 + sqrt(1 + p(1-p)(64w^2 - 64w + 8))) / 2)
    eta(w)  = (1-gamma)/2 r_p(w) + gamma r_0(w)           (concave/convex check: ``eta_is_convex``)
    g_t(w)  = eta(t) + (w - t) eta'(t)                    affine tangent = the affine g of Tan Eq. (28)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

SQRT2 = math.sqrt(2.0)
OMEGA_MIN = (2.0 - SQRT2) / 4.0            # (1 - 1/sqrt2)/2: smallest quantum CHSH winning probability
OMEGA_MAX = (2.0 + SQRT2) / 4.0            # Tsirelson
OMEGA_CLASSICAL = 0.75
V_PRIME = 2.0 * math.log2(5.0)             # V' = 2 log 5 = 2 log(1 + 2 dim B'),  dim B' = 2
LN2 = math.log(2.0)


# ---------------------------------------------------------------- entropy bound
def h2(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * math.log2(x) - (1.0 - x) * math.log2(1.0 - x)


def _u(w: float) -> float:
    return 16.0 * w * w - 16.0 * w + 3.0   # = (S/2)^2 - 1 with S = 8w - 4


def r0(w: float) -> float:
    """Eq. (B3): CHSH-based bound without noisy preprocessing (0 below the classical bound)."""
    u = _u(w)
    if u <= 0.0:
        return 0.0
    return 1.0 - h2(0.5 * (1.0 + math.sqrt(min(u, 1.0))))


def rp(w: float, p: float) -> float:
    """Eq. (B2): bound for the noisy-preprocessed key bit, flip probability p.
    Below the classical bound (w <= 3/4) Eve may know A exactly but not the private flip, so the bound is h2(p) -- which is also the
    limit of (B2) as w -> 3/4 (it equals 0 for p = 0)."""
    u = _u(w)
    if u <= 0.0:
        return h2(p)
    z2 = (1.0 - 2.0 * p) ** 2 + 4.0 * p * (1.0 - p) * min(u, 1.0)
    return 1.0 - h2(0.5 * (1.0 + math.sqrt(min(u, 1.0)))) + h2(0.5 * (1.0 + math.sqrt(z2)))


def _d_one_minus_h2(root: float, droot: float) -> float:
    """d/dw [1 - h2((1+root)/2)] = +(root'/2) log2((1+root)/(1-root))   (h2'(y) = log2((1-y)/y))."""
    if root >= 1.0 - 1e-15 or droot == 0.0:
        return 0.0
    return 0.5 * droot * math.log2((1.0 + root) / (1.0 - root))


def dr0(w: float) -> float:
    u = _u(w)
    if u <= 0.0:
        return 0.0
    x = math.sqrt(min(u, 1.0 - 1e-15))
    return _d_one_minus_h2(x, (16.0 * w - 8.0) / x)           # x' = (16 w - 8)/x


def drp(w: float, p: float) -> float:
    u = _u(w)
    if u <= 0.0:
        return 0.0
    u = min(u, 1.0 - 1e-15)
    z = math.sqrt((1.0 - 2.0 * p) ** 2 + 4.0 * p * (1.0 - p) * u)
    dz = 2.0 * p * (1.0 - p) * (32.0 * w - 16.0) / z if z > 0 else 0.0
    return dr0(w) - _d_one_minus_h2(z, dz)                    # r_p = r_0 + h2(.): its derivative is minus that of (1 - h2)


PROTOCOLS = ("random_key", "single_key")


def _key_fraction(gamma: float, protocol: str) -> float:
    """Fraction of rounds whose key bit carries the noisy-preprocessed entropy r_p.

    'random_key' : the paper's protocol (Sec. II.A, Tan et al.): Bob's key input is random in {0,1}, only X = Y survives
                   sifting -> (1 - gamma)/2                                                       (Eq. B1 as printed)
    'single_key' : the KGP of Ref. [73] (Nadlinger et al.): key rounds use the fixed inputs X = 0, Y = 2, no sifting
                   -> (1 - gamma).  Used for the matter-qubit scenario of Fig. 6 ("KGP in Ref. [73]")."""
    if protocol == "random_key":
        return 0.5 * (1.0 - gamma)
    if protocol == "single_key":
        return 1.0 - gamma
    raise ValueError(f"unknown protocol {protocol!r}")


def eta(w: float, gamma: float, p: float, protocol: str = "random_key") -> float:
    """Eq. (B1) on the quantum range above the classical bound (test rounds always contribute gamma r_0)."""
    return _key_fraction(gamma, protocol) * rp(w, p) + gamma * r0(w)


def deta(w: float, gamma: float, p: float, protocol: str = "random_key") -> float:
    return _key_fraction(gamma, protocol) * drp(w, p) + gamma * dr0(w)


def eta_is_convex(gamma: float, p: float, lo: float = 0.7501, hi: float = OMEGA_MAX - 1e-4, m: int = 400) -> bool:
    """Tangents are valid lower bounds on [lo, hi] iff eta is convex there (checked by second differences)."""
    xs = [lo + (hi - lo) * i / m for i in range(m + 1)]
    ys = [eta(x, gamma, p) for x in xs]
    return all(ys[i - 1] - 2 * ys[i] + ys[i + 1] >= -1e-12 for i in range(1, m))


# ---------------------------------------------------------------- min-tradeoff function (Tan Eqs. 28, 33, 68)
@dataclass(frozen=True)
class MinTradeoff:
    gamma: float
    t: float
    p: float
    beta: float
    slope: float
    intercept: float            # g(w) = intercept + slope * w

    def g(self, w: float) -> float:
        return self.intercept + self.slope * w

    @property
    def max_f(self) -> float:   # Max(f_min), Eq. (33)
        return self.g(1.0) / self.gamma + (1.0 - 1.0 / self.gamma) * self.beta

    @property
    def min_f(self) -> float:   # Min_{Q_f}(f_min), Eq. (33)
        return self.g(OMEGA_MIN)

    @property
    def var_f(self) -> float:   # upper bound on Var_{Q_f}(f_min), Eq. (33)
        d0, d1 = self.beta - self.g(0.0), self.beta - self.g(1.0)
        lo, hi = min(d0 * d0, d1 * d1), max(d0 * d0, d1 * d1)
        return (2.0 - SQRT2) / (4.0 * self.gamma) * lo + (2.0 + SQRT2) / (4.0 * self.gamma) * hi


def min_tradeoff(gamma: float, t: float, p: float, beta: float | None = None, protocol: str = "random_key") -> MinTradeoff:
    """Tangent of eta at t; beta defaults to g(1) (then only the 'loss' check of Fan et al. is needed, Tan Eq. 70)."""
    slope = deta(t, gamma, p, protocol)
    intercept = eta(t, gamma, p, protocol) - slope * t
    g1 = intercept + slope
    return MinTradeoff(gamma, t, p, g1 if beta is None else beta, slope, intercept)


# ---------------------------------------------------------------- EAT correction terms (Tan Eq. 32)
def theta(eps: float) -> float:
    """vartheta_eps <= log2(2/eps^2): the bound used for the numerics in Tan et al."""
    return math.log2(2.0 / (eps * eps))


def v_eat(mt: MinTradeoff) -> float:
    return math.sqrt(mt.var_f + 2.0) + math.log2(33.0)


def k_alpha(a: float, mt: MinTradeoff) -> float:
    """K_alpha with alpha = 1 + a."""
    delta = 4.0 + mt.max_f - mt.min_f                 # 2 log 4 = 4
    if a * delta > 700.0 or delta > 700.0:
        return float("inf")
    return (2.0 ** (a * delta) / (6.0 * (1.0 - a) ** 3 * LN2)) * math.log(2.0 ** delta + math.e ** 2) ** 3


# ---------------------------------------------------------------- budget of the epsilons (Table I)
@dataclass(frozen=True)
class Budget:
    eps_snd: float = 5.0e-11     # soundness of the DIQKD protocol (Table I)
    eps_EA: float = 1.0e-11
    eps_PA: float = 1.0e-11
    eps_h: float = 2.0 ** -61    # 61-bit verification tag (paper Sec. III.C: eps_corr <= 2^-61)
    hash_factor: int = 2         # Tan: sound = max{eps_EA, eps_PA + 2 eps_s} + 2 eps_h ; paper's wording suggests 3

    @property
    def eps_s(self) -> float:
        return max((self.eps_snd - self.hash_factor * self.eps_h - self.eps_PA) / 2.0, 0.0)

    @property
    def tag_bits(self) -> int:
        return math.ceil(math.log2(1.0 / self.eps_h))


# ---------------------------------------------------------------- honest devices and protocol parameters
@dataclass(frozen=True)
class Honest:
    """Honest-device summary used by the protocol parameters (B10, B11)."""
    omega: float        # CHSH winning probability  w_exp = (4+S)/8
    qber: float         # bit-error rate of matched key rounds before noisy preprocessing


def werner_honest(vis: float) -> Honest:
    """Werner state, ideal CHSH measurements: S = 2 sqrt2 V, Q = (1-V)/2 (Fig. 3 caption)."""
    return Honest(omega=(2.0 + SQRT2 * vis) / 4.0, qber=(1.0 - vis) / 2.0)


def omega_threshold(n: int, gamma: float, omega: float, k: float = 3.0, two_sided: bool = False) -> float:
    """Eq. (B10): w_thr = 1 - q_thr/gamma, q_thr = q + k sqrt(q(1-q)/n), q = gamma (1 - w)  ==  w - k sqrt(q(1-q)/n)/gamma.

    This sizes the margin on the LOSS count only, which is all that is needed when beta = g(1) (Tan Eq. 70).
    For beta < g(1) the proof needs the loss AND the win count above/below the same w_thr (Tan Eq. 30); the win count
    has the larger variance, so ``two_sided=True`` takes the larger of the two standard deviations."""
    q_loss = gamma * (1.0 - omega)
    var = q_loss * (1.0 - q_loss)
    if two_sided:
        q_win = gamma * omega
        var = max(var, q_win * (1.0 - q_win))
    return omega - k * math.sqrt(var / n) / gamma


def ec_max_bits(n: int, gamma: float, p: float, hon: Honest, model: str = "sifted", c_sqrt: float = 50.0) -> float:
    """Bound on the bits spent on error correction.

    'printed'    : Eq. (B11) as printed, n[(1-g) h2(Q) + g h2((4-S)/8)] + 50 sqrt(n)   (single key basis, Ref. [73]; no p, no sifting)
    'single_key' : n[(1-g) h2(Q') + g h2((4-S)/8)] + 50 sqrt(n): (B11) with noisy preprocessing; equals 'printed' at p = 0
    'sifted'     : n[(1-g)/2 h2(Q') + g h2((4-S)/8)] + 50 sqrt(n),  Q' = p + (1-2p) Q   (Tan Eqs. 24/26 with the paper's 50 sqrt n)
    """
    h_test = h2(1.0 - hon.omega)
    if model == "printed":
        return n * ((1.0 - gamma) * h2(hon.qber) + gamma * h_test) + c_sqrt * math.sqrt(n)
    if model == "single_key":
        qprime = p + (1.0 - 2.0 * p) * hon.qber
        return n * ((1.0 - gamma) * h2(qprime) + gamma * h_test) + c_sqrt * math.sqrt(n)
    if model == "sifted":
        qprime = p + (1.0 - 2.0 * p) * hon.qber
        return n * (0.5 * (1.0 - gamma) * h2(qprime) + gamma * h_test) + c_sqrt * math.sqrt(n)
    raise ValueError(f"unknown ec_model {model!r}")


# ---------------------------------------------------------------- Eq. (31)
def key_length_terms(n: int, mt: MinTradeoff, w_thr: float, a: float, a_prime: float,
                     eps_s1: float, eps_s2: float, ec_max: float, budget: Budget = Budget()) -> Dict[str, float]:
    """Every term of Eq. (31); ``ell`` is their sum (not floored)."""
    eps_r = budget.eps_s - eps_s1 - 2.0 * eps_s2
    if not (eps_s1 > 0 and eps_s2 > 0 and eps_r > 0):
        raise ValueError("need eps_s' + 2 eps_s'' < eps_s")
    L_EA = math.log2(1.0 / budget.eps_EA)
    V = v_eat(mt)
    t = {
        "n_g": n * mt.g(w_thr),
        "second_order_V": -n * a * LN2 / 2.0 * V * V,
        "second_order_K": -n * a * a * k_alpha(a, mt),
        "n_gamma": -n * mt.gamma,
        "max_entropy_B": -n * (a_prime / 4.0) * V_PRIME ** 2,
        "theta_s1": -theta(eps_s1) / a,
        "theta_s2": -theta(eps_s2) / a_prime,
        "eps_EA": -(1.0 / a + 1.0 / a_prime) * L_EA,
        "theta_rest": -3.0 * theta(eps_r),
        "EC_max": -ec_max,
        "hash_tag": -float(budget.tag_bits),
        "eps_PA": -2.0 * math.log2(1.0 / budget.eps_PA),
        "const": 2.0,
    }
    t["ell"] = sum(t.values())
    return t
