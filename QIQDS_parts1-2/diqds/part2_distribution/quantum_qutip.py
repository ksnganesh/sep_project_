"""QuTiP backend for the honest devices (optional dependency: ``pip install qutip``).

Builds the two-qubit state and the projective measurements as ``Qobj`` and obtains P(a, b | x, y) = Tr[(P_a^x (x) P_b^y) rho]
with ``qutip.expect``.  It must reproduce ``devices.werner_probabilities`` exactly (up to floating point); ``check_against_numpy``
does that comparison.  NOTE: written against the stable QuTiP 4/5 API (``basis``, ``tensor``, ``ket2dm``, ``sigmaz``, ``sigmax``,
``qeye``, ``expect``); it has NOT been executed in the sandbox this package was developed in (no QuTiP there) - run
``tests/test_part2_distribution.py`` after installing QuTiP.

QuTiP matters more in Part 3, where the heralded photonic state (Eq. D1 plus source imperfections) is a general two-qubit state.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .devices import ALICE_ANGLES, BOB_ANGLES, WernerDevice, werner_probabilities


def available() -> bool:
    try:
        import qutip  # noqa: F401
        return True
    except Exception:
        return False


def bell_phi_plus():
    import qutip as qt
    return (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()


def werner_state(vis: float):
    """V |Phi+><Phi+| + (1-V) I/4 as a density operator."""
    import qutip as qt
    return vis * qt.ket2dm(bell_phi_plus()) + (1.0 - vis) * qt.tensor(qt.qeye(2), qt.qeye(2)) / 4.0


def projector(theta: float, outcome: int):
    """(1 + (-1)^outcome sigma(theta)) / 2 with sigma(theta) = cos(theta) Z + sin(theta) X."""
    import qutip as qt
    sigma = np.cos(theta) * qt.sigmaz() + np.sin(theta) * qt.sigmax()
    return 0.5 * (qt.qeye(2) + (-1) ** outcome * sigma)


def joint_probabilities(rho, alice: Sequence[float] = ALICE_ANGLES, bob: Sequence[float] = BOB_ANGLES) -> np.ndarray:
    """P[x, y, a, b] = Tr[(P_a(theta_x) (x) P_b(phi_y)) rho] for an arbitrary two-qubit state ``rho``."""
    import qutip as qt
    P = np.zeros((len(alice), len(bob), 2, 2))
    for x, th in enumerate(alice):
        for y, ph in enumerate(bob):
            for a in (0, 1):
                for b in (0, 1):
                    P[x, y, a, b] = float(np.real(qt.expect(qt.tensor(projector(th, a), projector(ph, b)), rho)))
    return P


class QutipWernerDevice(WernerDevice):
    name = "qutip"

    def __init__(self, vis: float):
        super().__init__(vis)
        self.P = joint_probabilities(werner_state(vis))


def check_against_numpy(vis: float = 0.93, atol: float = 1e-12) -> bool:
    return bool(np.allclose(joint_probabilities(werner_state(vis)), werner_probabilities(vis), atol=atol))
