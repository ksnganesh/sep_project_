"""The polynomial-generation function  p_a = f(Z_a)  of the messaging stage.

Fan et al. (PRA 114, 032617), Sec. II.B.1 only say that f is "based on Ref. [32]".
Ref. [32] = Li et al., Phys. Rev. Applied 20, 044011 (arXiv:2301.01132), App. B.1
specifies it (after Shoup, 1996):

  1. fix a PUBLIC irreducible polynomial f0(x) of degree n  (defines GF(2**n));
  2. read the n key bits Z as a random element g(x) of GF(2**n);
  3. form the sequence  a_i = [g**i mod f0](0),  i = 0 .. 2n-1
     (constant coefficient of g**i mod f0);
  4. run Berlekamp-Massey on (a_i); the result is the minimal polynomial of g.

Why this is the right sampler: the minimal polynomial of g is irreducible, and each
degree-n irreducible polynomial is the minimal polynomial of exactly n field elements
(its n roots).  Hence, for a uniform n-bit Z that is not in a proper subfield,
f(Z) is EXACTLY uniform over the irreducible polynomials of degree n - the property the
forging bound  2**(-ns+1) * nm  of Eq. (5) relies on.  It needs only n random bits.

With probability ~ 2**(-n/2) the element g lies in a proper subfield and the minimal
polynomial has degree < n.  Such a key is *degenerate*; we raise ``DegenerateKeyError``
(the key pool simply skips it, see ``keys.py``).
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Tuple

import numpy as np

from .bits import bits_to_int, int_to_bits
from .gf2poly import deg, default_modulus, is_irreducible, mulmod


class DegenerateKeyError(ValueError):
    """Z lies in a proper subfield of GF(2**n): its minimal polynomial has degree < n."""


# --------------------------------------------------------------------------
# Berlekamp-Massey over GF(2)
# --------------------------------------------------------------------------
def berlekamp_massey_gf2(seq: Sequence[int]) -> Tuple[List[int], int]:
    """Shortest LFSR generating ``seq`` (Massey 1969).

    Returns ``(c, L)`` with ``c = [1, c_1, ..., c_L]`` such that
    ``seq[k] = c_1*seq[k-1] + ... + c_L*seq[k-L]`` (mod 2) for all k >= L.
    """
    N = len(seq)
    c = [0] * (N + 1)
    b = [0] * (N + 1)
    c[0] = b[0] = 1
    L, m = 0, 1
    for i in range(N):
        d = int(seq[i])
        for j in range(1, L + 1):
            d ^= c[j] & int(seq[i - j])
        if d == 0:
            m += 1
            continue
        t = c[:]
        for j in range(N + 1 - m):
            c[j + m] ^= b[j]
        if 2 * L <= i:
            L = i + 1 - L
            b = t
            m = 1
        else:
            m += 1
    return c[: L + 1], L


def characteristic_polynomial(c: Sequence[int], L: int) -> int:
    """x**L + c_1 x**(L-1) + ... + c_L as a GF(2)[x] int."""
    return sum(int(c[j]) << (L - j) for j in range(L + 1))


def element_min_poly(g: int, modulus: int) -> int:
    """Minimal polynomial over GF(2) of the element g of GF(2)[x]/(modulus)  [Ref. 32, App. B.1]."""
    n = deg(modulus)
    seq, y = [], 1
    for _ in range(2 * n):
        seq.append(y & 1)                  # a_i = coefficient of x**0 in g**i mod modulus
        y = mulmod(y, g, modulus)
    c, L = berlekamp_massey_gf2(seq)
    return characteristic_polynomial(c, L)


# --------------------------------------------------------------------------
# f(Z)
# --------------------------------------------------------------------------
@lru_cache(maxsize=1 << 14)
def _poly_from_key_int(z: int, n: int, modulus: int) -> int:
    P = element_min_poly(z, modulus)
    if deg(P) != n:
        raise DegenerateKeyError(f"key lies in a proper subfield (min. polynomial degree {deg(P)} < {n})")
    return P


def poly_from_key(z_bits, modulus: int | None = None) -> int:
    """f(Z) as a GF(2)[x] int: the monic irreducible polynomial of degree n = len(Z)."""
    n = len(z_bits)
    modulus = default_modulus(n) if modulus is None else modulus
    if deg(modulus) != n:
        raise ValueError("modulus must have degree len(Z)")
    return _poly_from_key_int(bits_to_int(z_bits), n, modulus)


def f(z_bits, modulus: int | None = None) -> np.ndarray:
    """The paper's  p_a = f(Z_a):  the ns-bit string (p_{n-1}, ..., p_1, p_0)."""
    n = len(z_bits)
    P = poly_from_key(z_bits, modulus)
    return int_to_bits(P & ((1 << n) - 1), n)


def key_is_degenerate(z_bits, modulus: int | None = None) -> bool:
    try:
        poly_from_key(z_bits, modulus)
        return False
    except DegenerateKeyError:
        return True


# --------------------------------------------------------------------------
# NOT the paper's f: "first irreducible polynomial at or after x**n + Z".
# Kept only to quantify how much a non-uniform f degrades the forging bound.
# --------------------------------------------------------------------------
def standin_poly_from_key(z_bits) -> int:
    n = len(z_bits)
    mask, c = (1 << n) - 1, bits_to_int(z_bits)
    for _ in range(1 << n):
        p = (1 << n) | c
        if is_irreducible(p):
            return p
        c = (c + 1) & mask
    raise RuntimeError("no irreducible polynomial found")
