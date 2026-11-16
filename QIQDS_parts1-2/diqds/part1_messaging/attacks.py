"""Bob's forging strategies from Sec. III.A, expressed as ``bob_tamper`` hooks for
``protocol.run_messaging_stage``.

Bob intercepts {Sig, Doc} and forwards {Sig xor t, Doc xor dM}.  Because
h(M xor dM) xor Y = h(M) xor h(dM) xor Y = Sig xor h(dM), Charlie accepts iff h(dM) = t.

  random_forgery            random (dM, t): success probability 2**(-ns)                (paper: "trivial strategy")
  multiple_of_polynomial    dM(x) = p(x) q(x): hashes to 0 for EVERY initial state when p = p_a  (Proposition 1, Eq. A12)
  product_of_polynomials    dM(x) = prod_i p_i(x): one message that covers up to nm/ns guesses p_i of p_a
                            (the optimal attack behind the bound 2**(-ns+1) * nm of Eq. 5), t = 0
"""
from __future__ import annotations

from typing import Callable, Iterable, Tuple

import numpy as np

from .bits import random_bits
from .gf2poly import clmul, deg
from .lfsr_toeplitz import polynomial_message


def random_forgery(ns: int, nm: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    dM = random_bits(nm, rng)
    while not dM.any():
        dM = random_bits(nm, rng)
    return dM, random_bits(ns, rng)


def multiple_of_polynomial(p_int: int, nm: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """dM(x) = p(x) q(x) with deg dM <= nm-1, q random and non-zero; target digest t = 0."""
    n = deg(p_int)
    qbits = nm - n
    if qbits < 1:
        raise ValueError("document too short to hold a multiple of p(x)")
    q = int.from_bytes(rng.bytes((qbits + 7) // 8), "little") & ((1 << qbits) - 1) | 1
    return polynomial_message(clmul(p_int, q), nm), np.zeros(n, dtype=np.uint8)


def product_of_polynomials(polys: Iterable[int], nm: int, ns: int) -> Tuple[np.ndarray, np.ndarray]:
    prod = 1
    for p in polys:
        prod = clmul(prod, p)
    return polynomial_message(prod, nm), np.zeros(ns, dtype=np.uint8)


def max_guesses(ns: int, nm: int) -> int:
    """Largest number of distinct degree-ns polynomials whose product fits in an nm-bit document."""
    return (nm - 1) // ns


def make_bob_tamper(dM: np.ndarray, t: np.ndarray) -> Callable:
    """Hook for ``run_messaging_stage``: forward {Sig xor t, Doc xor dM, Y_b}."""
    def tamper(sig, doc, y_b):
        return sig ^ t, doc ^ dM, y_b
    return tamper
