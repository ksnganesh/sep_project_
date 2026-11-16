"""Two-universal hashing for steps 5 and 7 (verification tag, privacy amplification).

A random Toeplitz matrix T (out_len x n), defined by a seed of out_len + n - 1 bits, T[i, j] = s[i - j + n - 1],
is a two-universal family.  y = T x (mod 2) is the central part of the linear convolution of s and x, so it costs
O((n + out_len) log(n + out_len)) with an FFT instead of O(n * out_len): this is what makes megabit-scale privacy
amplification feasible.  (Part 1 used an LFSR-generated Toeplitz matrix for the signature digest; here the seed is simply
random and public.)
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz
from scipy.signal import fftconvolve


def seed_length(n: int, out_len: int) -> int:
    return out_len + n - 1


def toeplitz_matrix_explicit(seed_bits, n: int, out_len: int) -> np.ndarray:
    """T (out_len x n) with T[i, j] = s[i - j + n - 1] -- reference for small sizes."""
    s = np.asarray(seed_bits, dtype=np.uint8)
    if len(s) != seed_length(n, out_len):
        raise ValueError("seed has the wrong length")
    return toeplitz(s[n - 1:n - 1 + out_len], s[:n][::-1]).astype(np.uint8)


def toeplitz_hash(x_bits, seed_bits, out_len: int, check_precision: bool = True) -> np.ndarray:
    """y = T x mod 2 via FFT convolution (float64; rounding error is checked)."""
    x = np.asarray(x_bits, dtype=np.uint8)
    s = np.asarray(seed_bits, dtype=np.uint8)
    n = len(x)
    if len(s) != seed_length(n, out_len):
        raise ValueError(f"seed must have out_len + n - 1 = {seed_length(n, out_len)} bits, got {len(s)}")
    full = fftconvolve(s.astype(np.float64), x.astype(np.float64), mode="full")
    y = full[n - 1:n - 1 + out_len]
    r = np.rint(y)
    if check_precision and out_len > 0 and np.max(np.abs(y - r)) > 0.3:
        raise FloatingPointError("FFT rounding error too large; split the hash into blocks")
    return (r.astype(np.int64) & 1).astype(np.uint8)


def random_seed(n: int, out_len: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=seed_length(n, out_len), dtype=np.uint8)
