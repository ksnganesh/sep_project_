"""Honest devices of the depolarizing-noise model (Sec. IV, Figs. 3-5).

Alice and Bob share the Werner state  V |Phi+><Phi+| + (1-V) I/4,  |Phi+> = (|00>+|11>)/sqrt2.  Measurements are in the X-Z plane,
sigma(theta) = cos(theta) Z + sin(theta) X:

    Alice  x = 0, 1          theta_A = 0, pi/2
    Bob    y = 0, 1          theta_B = 0, pi/2           (key settings: the SAME bases as Alice's two settings)
           y = 2, 3          theta_B = pi/4, -pi/4        (ideal CHSH settings, Bell-test rounds)

so that P(a, b | x, y) = (1 + (-1)^(a+b) V cos(theta_x - phi_y)) / 4.  The CHSH game is won iff a xor b = x (y-2); this gives
S = 2 sqrt2 V, winning probability w = (4 + S)/8 and matched-key-round error Q = (1-V)/2 (Fig. 3 caption).

``joint_probabilities`` is the single interface the other backends (QuTiP, Qiskit Aer) override or are checked against.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np

ALICE_ANGLES = (0.0, math.pi / 2)
BOB_ANGLES = (0.0, math.pi / 2, math.pi / 4, -math.pi / 4)


def werner_probabilities(vis: float, alice=ALICE_ANGLES, bob=BOB_ANGLES) -> np.ndarray:
    """P[x, y, a, b] for the Werner state with visibility ``vis``."""
    P = np.zeros((len(alice), len(bob), 2, 2))
    for x, th in enumerate(alice):
        for y, ph in enumerate(bob):
            c = vis * math.cos(th - ph)
            for a in (0, 1):
                for b in (0, 1):
                    P[x, y, a, b] = 0.25 * (1.0 + (-1) ** (a + b) * c)
    return P


def statistics(P: np.ndarray) -> Tuple[float, float, float]:
    """(S, w, Q) of a joint-probability table: CHSH value, winning probability, matched-key-round error."""
    w = 0.0
    for x in (0, 1):
        for yp in (0, 1):
            w += 0.25 * sum(P[x, 2 + yp, a, b] for a in (0, 1) for b in (0, 1) if (a ^ b) == (x & yp))
    q = 0.5 * sum(P[z, z, a, b] for z in (0, 1) for a in (0, 1) for b in (0, 1) if a != b)
    return 8.0 * w - 4.0, w, q


class WernerDevice:
    """Exact-probability sampler (NumPy).  Subclasses may replace ``self.P`` or ``sample``."""

    name = "numpy"

    def __init__(self, vis: float):
        self.vis = float(vis)
        self.P = werner_probabilities(self.vis)

    def joint_probabilities(self) -> np.ndarray:
        return self.P

    @property
    def chsh(self) -> float:
        return statistics(self.P)[0]

    @property
    def omega(self) -> float:
        return statistics(self.P)[1]

    @property
    def qber(self) -> float:
        return statistics(self.P)[2]

    def sample(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Outcomes (a, b) for the input arrays x in {0,1}, y in {0,1,2,3} (rounds are i.i.d., so we batch by setting)."""
        n = len(x)
        a = np.empty(n, dtype=np.uint8)
        b = np.empty(n, dtype=np.uint8)
        for xx in (0, 1):
            for yy in range(4):
                idx = np.flatnonzero((x == xx) & (y == yy))
                if idx.size == 0:
                    continue
                cdf = np.cumsum(self.P[xx, yy].reshape(4))            # outcome index c = 2a + b
                c = np.searchsorted(cdf, rng.random(idx.size) * cdf[-1], side="right").clip(0, 3)
                a[idx] = c >> 1
                b[idx] = c & 1
        return a, b
