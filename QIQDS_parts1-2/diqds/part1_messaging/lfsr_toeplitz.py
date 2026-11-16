"""LFSR-based Toeplitz hash h_{p,s}(M) = H_{n x m} M  (Appendix A of the paper, Eqs. A1-A12).

Notation (paper -> code)
    p = (p_{n-1}, ..., p_0)      irreducible polynomial p(x) = x^n + p_{n-1}x^{n-1} + ... + p_0
    s = (S_n, ..., S_1)^T        initial LFSR state (the key X)
    M = (M_0, ..., M_{m-1})^T    document
    W                            n x n companion matrix of p(x)                  (A4)
    s_j = W^j s                  LFSR state after j steps                        (A7, A8)
    H = (s_0, s_1, ..., s_{m-1})                                                 (A9)
    h(M) = H M = M_W(W) s,  M_W(x) = sum_j M_j x^j                              (A10, A11)
    p(x) | M_W(x)  =>  h(M) = 0                                                 (A12)

Three equivalent evaluations are provided and cross-checked in the tests:
    toeplitz_matrix_from_powers : literal (A7)-(A9), columns W^j s
    toeplitz_matrix             : H[i, j] = S_{n-i+j}  (the sequence form), built with scipy.linalg.toeplitz
    hash_horner                 : M_W(W) s by Horner's rule = what an LFSR does in hardware; O(m) steps
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

from .bits import as_bits, bits_to_int, int_to_bits


def companion_matrix(p) -> np.ndarray:
    """W of Eq. (A4): first row p, ones on the sub-diagonal."""
    p = as_bits(p)
    n = len(p)
    W = np.zeros((n, n), dtype=np.uint8)
    W[0, :] = p
    W[1:, :-1] = np.eye(n - 1, dtype=np.uint8)
    return W


def lfsr_sequence(p, s, length: int) -> np.ndarray:
    """S_1, ..., S_length (array index k-1 holds S_k), Eq. (A6):
    S_{n+1} = p . s  and in general  S_k = sum_{j=0}^{n-1} p_j S_{k-n+j}."""
    p, s = as_bits(p), as_bits(s)
    n = len(p)
    S = np.zeros(max(length, n), dtype=np.uint8)
    S[:n] = s[::-1]                      # s = (S_n, ..., S_1)  ->  S = (S_1, ..., S_n)
    pj = p[::-1]                         # pj[j] = p_j
    for k in range(n, length):
        S[k] = np.bitwise_xor.reduce(S[k - n:k] & pj)
    return S[:length]


def toeplitz_matrix(p, s, m: int) -> np.ndarray:
    """H_{n x m} with H[i, j] = S_{n-i+j}: first column s = (S_n..S_1), first row (S_n..S_{n+m-1})."""
    p, s = as_bits(p), as_bits(s)
    n = len(p)
    S = lfsr_sequence(p, s, n + m - 1)
    return toeplitz(s, S[n - 1:n + m - 1]).astype(np.uint8)


def toeplitz_matrix_from_powers(p, s, m: int) -> np.ndarray:
    """Literal Eqs. (A7)-(A9): columns s_j = W^j s (mod 2)."""
    W = companion_matrix(p).astype(np.int64)
    v = as_bits(s).astype(np.int64)
    cols = []
    for _ in range(m):
        cols.append(v.copy())
        v = (W @ v) % 2
    return np.stack(cols, axis=1).astype(np.uint8)


def hash_matrix(p, s, doc) -> np.ndarray:
    """h_{p,s}(M) = H M (mod 2) via the explicit n x m matrix."""
    doc = as_bits(doc)
    H = toeplitz_matrix(p, s, len(doc)).astype(np.int64)
    return ((H @ doc.astype(np.int64)) % 2).astype(np.uint8)


def hash_horner(p, s, doc) -> np.ndarray:
    """h_{p,s}(M) = M_W(W) s by Horner's rule (Eqs. A10-A11):  r <- W r  xor  M_j s,  j = m-1 .. 0.
    State kept as an MSB-first int: bit (n-1-k) = state[k]; W r = (r >> 1) | parity(p & r) << (n-1)."""
    p, s, doc = as_bits(p), as_bits(s), as_bits(doc)
    n = len(p)
    p_int, s_int, r = bits_to_int(p), bits_to_int(s), 0
    for bit in doc[::-1]:
        fb = bin(p_int & r).count("1") & 1
        r = (r >> 1) | (fb << (n - 1))
        if bit:
            r ^= s_int
    return int_to_bits(r, n)


def hash_digest(p, s, doc, method: str = "matrix") -> np.ndarray:
    if method == "matrix":
        return hash_matrix(p, s, doc)
    if method == "horner":
        return hash_horner(p, s, doc)
    raise ValueError(f"unknown method {method!r}")


def message_polynomial(doc) -> int:
    """M_W(x) = sum_j M_j x^j as a GF(2)[x] int (bit j = M_j)."""
    v = 0
    for j, b in enumerate(as_bits(doc)):
        v |= int(b) << j
    return v


def polynomial_message(poly: int, m: int) -> np.ndarray:
    """Inverse of ``message_polynomial``: the m-bit document (M_0..M_{m-1}) of a polynomial of degree < m."""
    if poly.bit_length() > m:
        raise ValueError("polynomial does not fit in m bits")
    return np.array([(poly >> j) & 1 for j in range(m)], dtype=np.uint8)
