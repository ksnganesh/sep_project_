"""Signature rate and security bounds of the messaging stage (Sec. II.B, Sec. III, Table I).

    (4)  r_sig = ell / (3 ns)
    (5)  eps_for = 2**(-ns+1) * nm + eps_sec
    (6)  eps_rep = 2 eps'
    (7)  eps_rob = 2 eps_corr + 2 eps'
    (8)  eps     = max{eps_rob, eps_rep, eps_for}

Symbols that are garbled in the PDF text (eps', eps_for, ...) were identified with the help of
Ref. [32] (Li et al., PR Applied 20, 044011), whose Sec. IV.3 states the same bounds:
eps' is "the probability that error occurs in classical message transmission", set to 1e-11.

Choices that are ours (the paper does not state them):
  * eps_sec in Eq. (5) is bounded by the soundness of the distribution stage, eps_sec <= eps_snd;
  * ns is the smallest integer with eps_for <= eps   ("we use Eq. (5) to determine the signature length");
  * r_sig is returned un-floored by default (smooth curves); ``floor=True`` gives whole signatures.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Table1:
    """Table I of the paper plus the constants quoted in Sec. III.C."""
    eps: float = 1.0e-10          # QDS security parameter
    eps_snd: float = 5.0e-11      # soundness of the underlying DIQKD protocol
    nm: int = 1_000               # message length (bits)
    eps_EA: float = 1.0e-11       # entropy-accumulation parameter   (used in Part 2)
    eps_PA: float = 1.0e-11       # privacy-amplification parameter  (used in Part 2)
    eps_corr: float = 2.0 ** -61  # eps_corr <= 2^-61  (VHASH, Ref. [73])
    eps_prime: float = 1.0e-11    # eps': classical-message error probability, assumed <= 1e-11


TABLE_I = Table1()


def signature_rate(ell: float, ns: int, floor: bool = False) -> float:
    """Eq. (4): signatures per protocol execution for final keys of length ell."""
    r = ell / (3 * ns)
    return float(math.floor(r)) if floor else r


def eps_forgery(ns: int, nm: int, eps_sec: float) -> float:
    """Eq. (5)."""
    return nm * 2.0 ** (1 - ns) + eps_sec


def eps_repudiation(eps_prime: float) -> float:
    """Eq. (6)."""
    return 2.0 * eps_prime


def eps_robustness(eps_corr: float, eps_prime: float) -> float:
    """Eq. (7)."""
    return 2.0 * eps_corr + 2.0 * eps_prime


def eps_total(ns: int, nm: int, eps_sec: float, eps_corr: float, eps_prime: float) -> float:
    """Eq. (8)."""
    return max(eps_robustness(eps_corr, eps_prime),
               eps_repudiation(eps_prime),
               eps_forgery(ns, nm, eps_sec))


def minimal_signature_length(nm: int, eps: float, eps_sec: float, eps_corr: float, eps_prime: float) -> int:
    """Smallest ns with eps_for <= eps (and the other two bounds below eps)."""
    slack = eps - eps_sec
    if slack <= 0:
        raise ValueError("eps_sec must be smaller than the target eps")
    if eps_repudiation(eps_prime) > eps or eps_robustness(eps_corr, eps_prime) > eps:
        raise ValueError("eps_rep / eps_rob already exceed the target eps; no ns can fix that")
    ns = max(1, math.ceil(1 + math.log2(nm / slack)))
    while eps_forgery(ns, nm, eps_sec) > eps:
        ns += 1
    while ns > 1 and eps_forgery(ns - 1, nm, eps_sec) <= eps:
        ns -= 1
    return ns


@dataclass(frozen=True)
class SecurityBudget:
    ns: int
    nm: int
    eps_for: float
    eps_rep: float
    eps_rob: float
    eps_target: float
    eps_sec: float

    @property
    def eps(self) -> float:
        return max(self.eps_for, self.eps_rep, self.eps_rob)

    @property
    def key_bits_per_signature(self) -> int:
        return 3 * self.ns

    def __str__(self) -> str:
        ok = "OK" if self.eps <= self.eps_target else "VIOLATED"
        return (f"n_s = {self.ns} bits  (n_m = {self.nm}, {self.key_bits_per_signature} key bits per signature)\n"
                f"  eps_for = 2^(-ns+1)*nm + eps_sec = {self.eps_for:.3e}   (eps_sec = {self.eps_sec:.1e})\n"
                f"  eps_rep = 2 eps'                  = {self.eps_rep:.3e}\n"
                f"  eps_rob = 2 eps_corr + 2 eps'     = {self.eps_rob:.3e}\n"
                f"  eps = max{{...}} = {self.eps:.3e}  vs target {self.eps_target:.1e}  -> {ok}")


def budget(params: Table1 = TABLE_I, eps_sec: float | None = None, ns: int | None = None) -> SecurityBudget:
    """Security budget for the given parameters; ns defaults to the minimal admissible length."""
    eps_sec = params.eps_snd if eps_sec is None else eps_sec
    if ns is None:
        ns = minimal_signature_length(params.nm, params.eps, eps_sec, params.eps_corr, params.eps_prime)
    return SecurityBudget(ns=ns, nm=params.nm,
                          eps_for=eps_forgery(ns, params.nm, eps_sec),
                          eps_rep=eps_repudiation(params.eps_prime),
                          eps_rob=eps_robustness(params.eps_corr, params.eps_prime),
                          eps_target=params.eps, eps_sec=eps_sec)
