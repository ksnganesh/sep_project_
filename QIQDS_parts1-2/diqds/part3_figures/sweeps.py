"""Signature-rate sweeps behind Figs. 3-7 of Fan et al., PRA 114, 032617 (2026).

Every curve in Sec. IV is the same pipeline:

    finite-size key length ell (Eq. 9 = B12, optimised; Part 2)  ->  r_sig = ell / (3 n_s)  (Eq. 4; n_s = 46 from Eq. 5, Part 1)

The curves have a hard finite-size threshold (ell crosses 0), so a plain log grid would draw the onset as a slanted line.
``curve_vs_n`` / ``curve_vs_v`` therefore locate the threshold first and pack extra points just above it, which is what makes
the vertical rise of the paper's figures visible.  Optimisations along a curve are warm-started from the previous point.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import brentq

from diqds.part1_messaging.security import TABLE_I, budget as messaging_budget
from diqds.part2_distribution.finitekey import Honest
from diqds.part2_distribution.optimize import Optimum, PSpec, optimize_key_length
from diqds.part2_distribution.presets import PAPER_FIT, Preset


@lru_cache(maxsize=None)
def signature_length() -> int:
    """n_s from Eq. (5) with Table I (= 46)."""
    return messaging_budget(TABLE_I).ns


def signature_rate(ell: float) -> float:
    """Eq. (4): r_sig = ell / (3 n_s); 0 when no key."""
    return max(ell, 0.0) / (3.0 * signature_length())


@dataclass
class Curve:
    label: str
    x: np.ndarray
    rate: np.ndarray
    threshold: float                      # x at which ell = 0 (nan if never positive in range)
    params: dict = field(default_factory=dict)
    optima: List[Optimum] = field(default_factory=list, repr=False)

    def to_json(self) -> dict:
        return {"label": self.label, "x": [float(v) for v in self.x], "rate": [float(v) for v in self.rate],
                "threshold": float(self.threshold), "params": self.params,
                "gamma": [float(o.gamma) for o in self.optima], "p": [float(o.p) for o in self.optima]}


class KeyLength:
    """ell(n) for fixed honest statistics, warm-started.  ``honest_fn(x)`` maps the swept variable to (n, Honest|vis)."""

    def __init__(self, preset: Preset = PAPER_FIT, p: PSpec = 0.0, protocol: str = "random_key",
                 ec_model: Optional[str] = None):
        self.preset, self.p, self.protocol = preset, p, protocol
        self.ec_model = ec_model or preset.ec_model
        self._x0 = None

    def __call__(self, n: float, vis: float = 1.0, honest: Optional[Honest] = None) -> Optimum:
        o = optimize_key_length(int(round(n)), vis, p=self.p, beta=self.preset.beta, pe_mode=self.preset.pe_mode,
                                ec_model=self.ec_model, honest=honest, protocol=self.protocol, x0=self._x0)
        if o.ell > 0:
            self._x0 = o.x
        return o


def _threshold(f: Callable[[float], float], lo: float, hi: float, log: bool, tol: float) -> float:
    """Root of f (ell) between lo (f<=0) and hi (f>0)."""
    if f(hi) <= 0:
        return float("nan")
    if f(lo) > 0:
        return lo
    if log:
        r = brentq(lambda u: f(math.exp(u)), math.log(lo), math.log(hi), xtol=tol)
        return math.exp(r)
    return brentq(f, lo, hi, xtol=tol)


def curve_vs_n(label: str, n_grid: Sequence[float], *, vis: float = 1.0, honest: Optional[Honest] = None,
               preset: Preset = PAPER_FIT, p: PSpec = 0.0, protocol: str = "random_key",
               ec_model: Optional[str] = None, dense: int = 14) -> Curve:
    """Signature rate versus the number of rounds n (Figs. 3 and 6)."""
    kl = KeyLength(preset, p, protocol, ec_model)
    n_grid = np.asarray(sorted(n_grid), float)
    thr = _threshold(lambda n: kl(n, vis, honest).ell, n_grid[0], n_grid[-1], log=True, tol=1e-5)
    xs = list(n_grid)
    if math.isfinite(thr):
        xs += list(thr * (1.0 + np.geomspace(1e-5, 0.5, dense)))      # pack points just above the threshold
        xs = sorted(x for x in set(xs) if x <= n_grid[-1])
    kl._x0 = None
    optima, rates = [], []
    for n in xs:
        o = kl(n, vis, honest)
        optima.append(o)
        rates.append(signature_rate(o.ell))
    return Curve(label, np.asarray(xs), np.asarray(rates), thr,
                 {"vis": vis, "preset": preset.name, "p": p, "protocol": protocol,
                  "honest": None if honest is None else [honest.omega, honest.qber]}, optima)


def curve_vs_v(label: str, n: float, v_grid: Sequence[float], *, preset: Preset = PAPER_FIT, p: PSpec = 0.0,
               dense: int = 14) -> Curve:
    """Signature rate versus the Werner visibility V at fixed n (Figs. 4 and 5)."""
    kl = KeyLength(preset, p)
    v_grid = np.asarray(sorted(v_grid), float)
    hi = v_grid[-1]
    # walk down from V = 1 to keep the warm start in the positive-key region, then bracket
    thr = _threshold(lambda v: kl(n, v).ell, v_grid[0], min(hi, 0.9995), log=False, tol=1e-6)
    xs = list(v_grid)
    if math.isfinite(thr):
        xs += list(thr + (hi - thr) * np.geomspace(1e-5, 0.08, dense))
        xs = sorted(x for x in set(xs) if x <= hi)
    kl._x0 = None
    optima, rates = [], []
    for v in xs[::-1]:
        o = kl(n, v)
        optima.append(o)
        rates.append(signature_rate(o.ell))
    return Curve(label, np.asarray(xs), np.asarray(rates[::-1]), thr,
                 {"n": n, "preset": preset.name, "p": p}, optima[::-1])
