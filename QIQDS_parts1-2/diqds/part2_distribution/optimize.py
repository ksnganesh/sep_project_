"""Optimise Eq. (31) over the free parameters, as done for the paper's Figs. 3-5.

Paper (Sec. IV): "eps_s, eps_s', eps_s'', eps_h, alpha, alpha', t and gamma were numerically optimized" and, for noisy
preprocessing, p is optimised separately.  omega_thr (B10) and EC_max (B11) follow from the honest devices, so they are
*not* free: they move with gamma and p.  We take beta = g(1) (Tan: optimum is "very close to g(1)"; then only the
loss check of the paper's step 6 is needed, Tan Eq. 70).

Closed forms used inside the optimiser:
    alpha' :  the alpha'-dependent terms are n a' V'^2/4 + (theta'' + log 1/eps_EA)/a'  ->  a'* = (2/V') sqrt((theta''+L)/n)
    alpha  :  1-D bounded minimisation of  n a ln2/2 V^2 + n a^2 K_alpha + (theta' + L)/a
Outer problem (Nelder-Mead, multi-start):  x = (log10 gamma, t, z1, z2[, p]),  with the epsilon shares
(eps_s', 2 eps_s'', remainder) = softmax(z1, z2, 0) * eps_s.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

from .finitekey import (Budget, Honest, LN2, OMEGA_MAX, V_PRIME, ec_max_bits, k_alpha, key_length_terms, min_tradeoff,
                        omega_threshold, theta, v_eat, werner_honest)

PSpec = Union[float, str]          # a fixed p, or "opt"


@dataclass
class Optimum:
    ell: float
    n: int
    vis: float
    gamma: float
    t: float
    p: float
    a: float
    a_prime: float
    eps_s1: float
    eps_s2: float
    w_thr: float
    ec_max: float
    terms: Dict[str, float] = field(default_factory=dict)
    x: Optional[np.ndarray] = None

    @property
    def alpha(self) -> float:
        return 1.0 + self.a

    @property
    def alpha_prime(self) -> float:
        return 1.0 + self.a_prime


def _shares(z1: float, z2: float):
    m = max(z1, z2, 0.0)
    e1, e2, e3 = math.exp(z1 - m), math.exp(z2 - m), math.exp(-m)
    s = e1 + e2 + e3
    return e1 / s, e2 / s, e3 / s


def _honest(vis, honest: Optional[Honest]) -> Honest:
    """Honest-device statistics: an explicit ``Honest`` (any device, Part 3) or the Werner state of visibility ``vis``."""
    return honest if honest is not None else werner_honest(vis)


def evaluate(n: int, vis: float, gamma: float, t: float, p: float, z1: float, z2: float, *,
             sb: float = 0.0, ec_model: str = "sifted", pe_mode: str = "loss", budget: Budget = Budget(),
             k_sigma: float = 3.0, c_sqrt: float = 50.0, honest: Optional[Honest] = None,
             protocol: str = "random_key") -> Optimum:
    """Key length for given outer parameters, with alpha and alpha' optimised (closed form / 1-D search).
    ``honest`` overrides the Werner statistics of ``vis``; ``protocol`` selects the entropy bound (finitekey.eta)."""
    gamma = min(max(float(gamma), 1e-9), 0.45)
    t = min(max(float(t), 0.7502), OMEGA_MAX - 1e-5)
    p = min(max(float(p), 0.0), 0.5)
    z1, z2, sb = float(z1), float(z2), float(sb)
    hon = _honest(vis, honest)
    mt0 = min_tradeoff(gamma, t, p, protocol=protocol)
    sb = min(max(sb, 0.0), 1.0)
    mt = mt0 if sb == 0.0 else min_tradeoff(gamma, t, p, beta=mt0.g(1.0) - sb * (mt0.g(1.0) - mt0.g(0.0)),
                                            protocol=protocol)
    w_thr = omega_threshold(n, gamma, hon.omega, k_sigma, two_sided=(pe_mode == "two_sided"))
    ec = ec_max_bits(n, gamma, p, hon, ec_model, c_sqrt)
    s1, s2, _ = _shares(z1, z2)
    eps1, eps2 = budget.eps_s * s1, budget.eps_s * s2 / 2.0
    L = math.log2(1.0 / budget.eps_EA)
    V = v_eat(mt)

    a_prime = min((2.0 / V_PRIME) * math.sqrt((theta(eps2) + L) / n), 0.999 * 2.0 / V_PRIME)
    th1 = theta(eps1)
    a0 = math.sqrt(2.0 * (th1 + L) / (n * V * V * LN2))
    lo, hi = math.log(a0) - 4.0, min(math.log(a0) + 4.0, math.log(0.99))

    def f(la: float) -> float:
        a_ = math.exp(la)
        k_ = k_alpha(a_, mt)
        if not math.isfinite(k_):
            return 1e300
        return n * a_ * LN2 / 2.0 * V * V + n * a_ * a_ * k_ + (th1 + L) / a_

    a = math.exp(minimize_scalar(f, bounds=(lo, hi), method="bounded", options={"xatol": 1e-7}).x)
    terms = key_length_terms(n, mt, w_thr, a, a_prime, eps1, eps2, ec, budget)
    return Optimum(ell=terms["ell"], n=n, vis=vis, gamma=gamma, t=t, p=p, a=a, a_prime=a_prime,
                   eps_s1=eps1, eps_s2=eps2, w_thr=w_thr, ec_max=ec, terms=terms)


def optimize_key_length(n: int, vis: float, *, p: PSpec = 0.0, beta: Union[str, float] = "g1",
                        ec_model: str = "sifted", pe_mode: str = "loss", budget: Budget = Budget(), k_sigma: float = 3.0, c_sqrt: float = 50.0,
                        x0: Optional[Sequence[float]] = None, maxiter: int = 800, honest: Optional[Honest] = None,
                        protocol: str = "random_key") -> Optimum:
    """Maximise ell_key over (gamma, t, eps split[, p][, beta]).  beta = "g1": beta = g(1) (single 'loss' check);
    beta = "opt": beta in [g(0), g(1)] is optimised too (needs the two-sided check of Tan Eq. 30);
    beta = float s in [0, 1]: beta = g(1) - s (g(1) - g(0)) held fixed.
    ``honest`` (Part 3) replaces the Werner statistics of ``vis`` by arbitrary (omega, Q); ``protocol`` = 'random_key' (paper)
    or 'single_key' (KGP of Ref. [73])."""
    opt_p = isinstance(p, str)
    opt_b = isinstance(beta, str) and beta == "opt"
    fixed_sb = 0.0 if (isinstance(beta, str) and beta == "g1") else (None if opt_b else float(beta))
    ip, ib = 4, (5 if opt_p else 4)
    if opt_p and p != "opt":
        raise ValueError("p must be a float or 'opt'")
    hon = _honest(vis, honest)

    def neg(x) -> float:
        pp = x[ip] if opt_p else p
        sb = x[ib] if opt_b else fixed_sb
        try:
            r = evaluate(n, vis, 10.0 ** x[0], x[1], pp, x[2], x[3], sb=sb, ec_model=ec_model, pe_mode=pe_mode,
                         budget=budget, k_sigma=k_sigma, c_sqrt=c_sqrt, honest=honest, protocol=protocol)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 1e9
        pen = 0.0
        if not (-8.5 < x[0] < -0.35):
            pen += 1e3 * (abs(x[0] + 4.0) - 4.0)
        if opt_p and not (0.0 <= x[ip] <= 0.5):
            pen += 1e3 * (max(-x[ip], x[ip] - 0.5))
        if opt_b and not (0.0 <= x[ib] <= 1.0):
            pen += 1e3 * (max(-x[ib], x[ib] - 1.0))
        val = -r.ell / n + pen
        return val if math.isfinite(val) else 1e9

    # global initialisation: coarse (log10 gamma, t) grid with the other variables at neutral values
    grid = []
    t_hi = min(hon.omega, OMEGA_MAX - 1e-3)
    for lg in np.arange(-4.2, -0.79, 0.3):
        for t in np.linspace(0.7602, t_hi, 12):
            xg = np.array([lg, t, 1.0, 1.0] + ([0.05] if opt_p else []) + ([0.4] if opt_b else []), float)
            grid.append((neg(xg), xg))
    grid.sort(key=lambda g: g[0])
    starts = ([np.array(x0, float)] if x0 is not None else []) + [grid[0][1], grid[1][1]]

    def run(xs):
        steps = [0.3, 0.004, 0.7, 0.7] + ([0.04] if opt_p else []) + ([0.2] if opt_b else [])
        simplex = [xs] + [xs + np.eye(len(xs))[i] * st for i, st in enumerate(steps)]
        return minimize(neg, xs, method="Nelder-Mead",
                        options={"initial_simplex": np.array(simplex), "xatol": 1e-6, "fatol": 1e-15, "maxiter": maxiter})

    best = None
    for xs in starts:
        res = run(xs)
        if best is None or res.fun < best.fun:
            best = res
    res = run(best.x)                      # restart from the best point to avoid a prematurely collapsed simplex
    if res.fun < best.fun:
        best = res
    x = best.x
    pp = min(max(x[ip], 0.0), 0.5) if opt_p else float(p)
    out = evaluate(n, vis, 10.0 ** x[0], x[1], pp, x[2], x[3], sb=(x[ib] if opt_b else fixed_sb), ec_model=ec_model,
                   pe_mode=pe_mode, budget=budget, k_sigma=k_sigma, c_sqrt=c_sqrt, honest=honest, protocol=protocol)
    out.x = x
    return out


def v_min(n: int, *, p: PSpec = 0.0, ec_model: str = "sifted", budget: Budget = Budget(),
          lo: float = 0.80, hi: float = 0.9995, tol: float = 1e-6, **kw) -> float:  # kw: beta, pe_mode, k_sigma, c_sqrt
    """Smallest Werner visibility with a positive finite-size key at n rounds (the threshold of Figs. 3-5)."""
    state = {"x": None}

    def f(vis: float) -> float:
        o = optimize_key_length(n, vis, p=p, ec_model=ec_model, budget=budget, x0=state["x"], **kw)
        state["x"] = o.x
        return o.ell / n

    if f(hi) <= 0.0:
        return float("nan")
    return brentq(f, lo, hi, xtol=tol, rtol=1e-9)


def n_min(vis: float, *, p: PSpec = 0.0, ec_model: str = "sifted", budget: Budget = Budget(),
          lo: float = 1e4, hi: float = 1e14, rel: float = 1e-3, **kw) -> float:
    """Smallest number of rounds with a positive finite-size key at visibility ``vis`` (geometric bisection)."""
    state = {"x": None}

    def positive(n: float) -> bool:
        o = optimize_key_length(int(n), vis, p=p, ec_model=ec_model, budget=budget, x0=state["x"], **kw)
        state["x"] = o.x
        return o.ell > 0.0

    if not positive(hi):
        return float("nan")
    while hi / lo > 1.0 + rel:
        mid = math.sqrt(lo * hi)
        if positive(mid):
            hi = mid
        else:
            lo = mid
    return hi
