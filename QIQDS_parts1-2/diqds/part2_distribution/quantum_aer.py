"""Qiskit Aer backend for the honest devices (optional: ``pip install qiskit qiskit-aer``).

One round = a circuit  h(0); cx(0,1); [two-qubit depolarizing error after the cx]; ry(-theta_x) on qubit 0; ry(-phi_y) on qubit 1;
measure.  With  depolarizing_error(1 - V, 2): rho -> V rho + (1-V) I/4  the prepared state is exactly the Werner state, and
measuring Z after Ry(-theta) is a measurement of sigma(theta) = cos(theta) Z + sin(theta) X.  Qubit 0 is Alice, qubit 1 is Bob;
Qiskit prints classical bits little-endian, so the 2-bit string 'c1c0' is read as b = s[-2], a = s[-1].

Rounds with the same (x, y) are i.i.d., so they are run as one batch of shots and scattered back to their round indices.

NOTE: written against the Qiskit 1.x/2.x + qiskit-aer API; it has NOT been executed in the sandbox this package was developed in
(no Qiskit there).  ``tests/test_part2_distribution.py`` compares its statistics with the exact NumPy model once installed.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .devices import ALICE_ANGLES, BOB_ANGLES, WernerDevice


def available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
        return True
    except Exception:
        return False


def parse_memory(memory) -> Tuple[np.ndarray, np.ndarray]:
    """Per-shot 2-bit strings 'c1c0' (Qiskit prints classical bits little-endian) -> (a, b):
    c0 holds the result of qubit 0 (Alice), c1 that of qubit 1 (Bob)."""
    bits = np.frombuffer("".join(memory).encode(), dtype=np.uint8).reshape(-1, 2) - ord("0")
    return bits[:, 1].astype(np.uint8), bits[:, 0].astype(np.uint8)


class AerWernerDevice(WernerDevice):
    name = "aer"

    def __init__(self, vis: float, seed: Optional[int] = None):
        super().__init__(vis)
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, depolarizing_error

        noise = NoiseModel()
        noise.add_quantum_error(depolarizing_error(1.0 - self.vis, 2), ["cx"], [0, 1])
        self._backend = AerSimulator(noise_model=noise)
        self._seed = seed
        self._circuits = {}
        for x, th in enumerate(ALICE_ANGLES):
            for y, ph in enumerate(BOB_ANGLES):
                qc = QuantumCircuit(2, 2)
                qc.h(0)
                qc.cx(0, 1)
                qc.ry(-th, 0)
                qc.ry(-ph, 1)
                qc.measure([0, 1], [0, 1])
                self._circuits[(x, y)] = transpile(qc, self._backend, optimization_level=0)

    def sample(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        a = np.empty(len(x), dtype=np.uint8)
        b = np.empty(len(x), dtype=np.uint8)
        for (xx, yy), circuit in self._circuits.items():
            idx = np.flatnonzero((x == xx) & (y == yy))
            if idx.size == 0:
                continue
            seed = int(rng.integers(0, 2 ** 31 - 1)) if self._seed is None else self._seed + 97 * xx + 13 * yy
            memory = self._backend.run(circuit, shots=int(idx.size), memory=True, seed_simulator=seed).result().get_memory()
            a[idx], b[idx] = parse_memory(memory)
        return a, b
