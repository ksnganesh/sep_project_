"""Arithmetic in GF(2)[x]; a polynomial is a Python int (bit i = coeff. of x**i).

Needed for
  * Rabin's irreducibility test, Eq. (13) of Ref. [32] (Li et al., PR Applied 20, 044011),
  * the Shoup minimal-polynomial generator f(Z)   (see ``polygen.py``),
  * checking Proposition 1 / Eq. (A12) of the paper (p(x) | M(x)  =>  h_{p,s}(M) = 0),
  * counting irreducible polynomials (the denominator in the forging bound).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterator, List


def deg(a: int) -> int:
    """Degree of ``a`` (the zero polynomial has degree -1)."""
    return a.bit_length() - 1


def clmul(a: int, b: int) -> int:
    """Carry-less product in GF(2)[x] (no reduction)."""
    if a.bit_length() < b.bit_length():
        a, b = b, a
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def pmod(a: int, m: int) -> int:
    """Remainder of ``a`` modulo ``m`` in GF(2)[x]."""
    dm = deg(m)
    if dm < 0:
        raise ZeroDivisionError("modulus is the zero polynomial")
    da = deg(a)
    while da >= dm:
        a ^= m << (da - dm)
        da = deg(a)
    return a


def mulmod(a: int, b: int, m: int) -> int:
    """``a*b mod m`` with interleaved reduction (deg m >= 1)."""
    dm = deg(m)
    a = pmod(a, m)
    r = 0
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if (a >> dm) & 1:
            a ^= m
    return r


def pgcd(a: int, b: int) -> int:
    while b:
        a, b = b, pmod(a, b)
    return a


def x_pow_2k_mod(k: int, m: int) -> int:
    """``x**(2**k) mod m`` by k squarings."""
    r = pmod(2, m)
    for _ in range(k):
        r = mulmod(r, r, m)
    return r


def prime_factors(n: int) -> List[int]:
    out, d = [], 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def is_irreducible(p: int) -> bool:
    """Rabin's test, Eq. (13) of Ref. [32].

    p (degree n) is irreducible over GF(2) iff
        x**(2**n) = x  (mod p)   and   gcd(x**(2**(n/d)) - x, p) = 1
    for every prime divisor d of n.
    """
    n = deg(p)
    if n < 1:
        return False
    if n == 1:
        return True
    if p & 1 == 0:                      # x | p
        return False
    if bin(p).count("1") % 2 == 0:      # p(1) = 0  =>  (x+1) | p
        return False
    if x_pow_2k_mod(n, p) != pmod(2, p):
        return False
    for d in prime_factors(n):
        h = x_pow_2k_mod(n // d, p) ^ 2          # x**(2**(n/d)) - x   (mod p)
        if pgcd(p, h) != 1:
            return False
    return True


def _mobius(d: int) -> int:
    mu, primes = 1, prime_factors(d)
    for q in primes:
        if (d // q) % q == 0:
            return 0
        mu = -mu
    return mu


@lru_cache(maxsize=None)
def count_irreducibles(n: int) -> int:
    """Number of monic irreducible polynomials of degree n over GF(2):
    (1/n) * sum_{d | n} mu(d) 2**(n/d)."""
    total = sum(_mobius(d) * (1 << (n // d)) for d in range(1, n + 1) if n % d == 0)
    assert total % n == 0
    return total // n


def irreducibles(n: int) -> Iterator[int]:
    """All monic irreducible polynomials of degree n in increasing order
    (brute force; intended for n <= ~18)."""
    if n == 1:
        yield 0b10
        yield 0b11
        return
    top = 1 << n
    for c in range(1, top, 2):
        p = top | c
        if is_irreducible(p):
            yield p


@lru_cache(maxsize=None)
def default_modulus(n: int) -> int:
    """Public irreducible polynomial of degree n defining GF(2**n) for the
    Shoup generator: the numerically smallest irreducible x**n + c.
    (The distribution of the generated polynomial does not depend on this choice.)"""
    top = 1 << n
    for c in range(1, top, 2):
        if is_irreducible(top | c):
            return top | c
    raise RuntimeError(f"no irreducible polynomial of degree {n}")


def poly_str(p: int) -> str:
    terms = []
    for i in range(deg(p), -1, -1):
        if (p >> i) & 1:
            terms.append("1" if i == 0 else ("x" if i == 1 else f"x^{i}"))
    return " + ".join(terms) if terms else "0"
