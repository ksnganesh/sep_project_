"""Bit-string conventions used throughout Part 1 (messaging stage).

All bit strings are 1-D ``numpy.uint8`` arrays holding 0/1, most-significant
bit first, i.e. the array ``[b0, b1, ..., b_{n-1}]`` is the integer
``b0*2**(n-1) + ... + b_{n-1}``.

Where the paper writes a vector explicitly we keep the paper's order:

* ``p = (p_{n-1}, ..., p_1, p_0)``  -> ``p[0]`` is the coefficient of ``x**(n-1)``
* ``s = (S_n, S_{n-1}, ..., S_1)``  -> ``s[0]`` is the *newest* LFSR element ``S_n``
* ``M = (M_0, M_1, ..., M_{m-1})``  -> ``M[j]`` is the coefficient of ``x**j`` in M(x)
"""
from __future__ import annotations

import numpy as np


def as_bits(x) -> np.ndarray:
    """Return ``x`` as a fresh uint8 0/1 array."""
    return (np.asarray(x, dtype=np.uint8) & 1).copy()


def random_bits(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, 2, size=n, dtype=np.uint8)


def bits_to_int(bits) -> int:
    """MSB-first bit array -> Python int."""
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def int_to_bits(x: int, n: int) -> np.ndarray:
    """Python int -> MSB-first bit array of length n."""
    if x < 0 or x >> n:
        raise ValueError(f"{x} does not fit in {n} bits")
    return np.array([(x >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def bits_to_str(bits) -> str:
    return "".join(str(int(b)) for b in bits)


def str_to_bits(s: str) -> np.ndarray:
    return np.array([int(c) for c in s], dtype=np.uint8)


def bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits) -> bytes:
    bits = as_bits(bits)
    if len(bits) % 8:
        raise ValueError("bit length must be a multiple of 8")
    return np.packbits(bits).tobytes()


# --------------------------------------------------------------------------
# Document encoding.  NOT part of the paper (which simply assumes an n_m-bit
# document); added so that real byte strings can be signed.  The padding is
# injective ("1" then zeros), so two different byte strings never map to the
# same n_m-bit document.
# --------------------------------------------------------------------------
def encode_document(data: bytes, nm: int) -> np.ndarray:
    bits = bytes_to_bits(data)
    if len(bits) + 1 > nm:
        raise ValueError(f"document of {len(bits)} bits does not fit in n_m={nm} bits")
    out = np.zeros(nm, dtype=np.uint8)
    out[: len(bits)] = bits
    out[len(bits)] = 1
    return out


def decode_document(doc_bits) -> bytes:
    doc_bits = as_bits(doc_bits)
    ones = np.flatnonzero(doc_bits)
    if len(ones) == 0:
        raise ValueError("no padding marker found")
    return bits_to_bytes(doc_bits[: ones[-1]])
