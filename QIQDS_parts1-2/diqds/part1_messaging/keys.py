"""Key material for the messaging stage (Sec. II.B).

After the distribution stage Alice shares a final key K_AB with Bob and a final key K_AC
with Charlie (both of length ell).  Each party cuts its key(s) into ns-bit segments and
holds three of them per signature:

    Alice   {X_a, Y_a, Z_a}      Bob   {X_b, Y_b, Z_b}      Charlie   {X_c, Y_c, Z_c}

with the perfect-correlation constraints of the paper

    X_a = X_c        Y_a = Y_b xor Y_c        Z_a = Z_c.

Segment convention (our reading; the paper does not spell it out): one signature consumes
3*ns consecutive bits of K_AB and of K_AC, cut as (X | Y | Z).  Bob's and Charlie's own
segments are the (X|Y|Z) cut of K_AB and K_AC respectively; Alice takes X_a, Z_a from K_AC
and Y_a = Y_b xor Y_c.  (X_b and Z_b are never used in verification, matching the text.)
The cost of 3*ns key bits per signature and per link is what gives r_sig = ell / (3 ns), Eq. (4).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .bits import as_bits, random_bits
from .polygen import key_is_degenerate


@dataclass(frozen=True)
class PartyKeys:
    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray

    @property
    def ns(self) -> int:
        return len(self.X)


def key_bits_per_signature(ns: int) -> int:
    return 3 * ns


def _cut(block: np.ndarray, ns: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return block[:ns].copy(), block[ns:2 * ns].copy(), block[2 * ns:3 * ns].copy()


def triple_from_blocks(ab_alice, ab_bob, ac_alice, ac_charlie, ns: int):
    """Build (Alice, Bob, Charlie) keys from one 3*ns-bit block of each pairwise key."""
    _, yb_a, _ = _cut(as_bits(ab_alice), ns)
    xb, yb, zb = _cut(as_bits(ab_bob), ns)
    xc_a, yc_a, zc_a = _cut(as_bits(ac_alice), ns)
    xc, yc, zc = _cut(as_bits(ac_charlie), ns)
    alice = PartyKeys(X=xc_a, Y=yb_a ^ yc_a, Z=zc_a)
    return alice, PartyKeys(xb, yb, zb), PartyKeys(xc, yc, zc)


def split_pairwise_keys(k_ab_alice, k_ab_bob, k_ac_alice, k_ac_charlie, ns: int,
                        skip_degenerate: bool = True) -> List[tuple]:
    """Cut the final keys of the distribution stage into per-signature key triples.

    ``floor(ell / (3 ns))`` triples are produced (before skipping the ~2**(-ns/2) fraction of
    triples whose Z is degenerate for the polynomial generator, see ``polygen.py``)."""
    L = min(len(k_ab_alice), len(k_ab_bob), len(k_ac_alice), len(k_ac_charlie))
    w = key_bits_per_signature(ns)
    out = []
    for j in range(L // w):
        sl = slice(j * w, (j + 1) * w)
        t = triple_from_blocks(k_ab_alice[sl], k_ab_bob[sl], k_ac_alice[sl], k_ac_charlie[sl], ns)
        if skip_degenerate and key_is_degenerate(t[0].Z):
            continue
        out.append(t)
    return out


def ideal_key_triples(ns: int, rng: np.random.Generator, skip_degenerate: bool = True):
    """Perfectly correlated, perfectly secret key triple (what an ideal distribution stage delivers)."""
    while True:
        blk = [random_bits(3 * ns, rng) for _ in range(4)]
        k_ab, k_ac = blk[0], blk[1]
        alice, bob, charlie = triple_from_blocks(k_ab, k_ab, k_ac, k_ac, ns)
        if not (skip_degenerate and key_is_degenerate(alice.Z)):
            return alice, bob, charlie


def satisfies_correlations(alice: PartyKeys, bob: PartyKeys, charlie: PartyKeys) -> bool:
    """X_a = X_c,  Y_a = Y_b xor Y_c,  Z_a = Z_c."""
    return (np.array_equal(alice.X, charlie.X)
            and np.array_equal(alice.Y, bob.Y ^ charlie.Y)
            and np.array_equal(alice.Z, charlie.Z))
