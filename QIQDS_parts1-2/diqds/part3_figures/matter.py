"""Matter-qubit scenario of Fig. 6: the experimentally reported parameters of Refs. [73-75] run through the KGP of Ref. [73].

Paper, Sec. IV: "we conduct simulations based on the experimentally reported parameters in Refs. [73-75] and KGP in Ref. [73]".
The KGP of Ref. [73] (Nadlinger et al., Nature 607, 682) differs from the paper's own protocol of Sec. II.A in one place:
its key rounds use the FIXED inputs X = 0, Y = 2 (Bell-test rounds w.p. gamma), so there is no basis sifting.  In the
finite-key bound this means (finitekey.eta, protocol='single_key')

    eta(w) = (1 - gamma) r_p(w) + gamma r_0(w)        instead of   (1 - gamma)/2 r_p(w) + gamma r_0(w)   (Eq. B1)
    EC_max = n[(1 - gamma) h(Q) + gamma h((4 - S)/8)] + 50 sqrt(n)                                       (Eq. B11 as printed)

so Eq. (B11) - which does not fit Figs. 3-5 - is exactly the error-correction budget of this KGP.  Check: with Ref. [73]'s own
design point (S = 2.64, gamma = 13/256, n = 1.5e6) Eq. (B10) with k = 3 gives omega_thr = 0.825538, the value quoted by Ref. [73].

Reported values used (each paper's headline numbers):
    Ref. [73]  88Sr+ ions, 2 m apart (paper: "1 m")   S = 2.677(6), Q = 1.44(2) %   1.5e6 rounds in ~8 h       [Nadlinger 2022]
    Ref. [74]  87Rb atoms, 400 m apart                S = 2.578(75), Q = 7.8(9) %   3342 rounds in 75 h        [Zhang 2022]
    Ref. [75]  87Rb atoms, 11 km of fibre             S = 2.612(31), Q = 2.85(2) %  1.208e6 events in 624 h   [Lu 2026]
"""
from __future__ import annotations

from dataclasses import dataclass

from diqds.part2_distribution.finitekey import Honest, omega_threshold


@dataclass(frozen=True)
class Experiment:
    ref: str
    platform: str
    distance: str
    chsh: float
    qber: float
    rounds: float            # heralded rounds actually collected
    hours: float             # over this many hours

    @property
    def honest(self) -> Honest:
        return Honest(omega=(4.0 + self.chsh) / 8.0, qber=self.qber)

    @property
    def rounds_per_hour(self) -> float:
        return self.rounds / self.hours

    def hours_for(self, n: float) -> float:
        """Acquisition time for n heralded rounds at the reported average event rate (stable operation assumed)."""
        return n / self.rounds_per_hour


REF73 = Experiment("[73]", "88Sr+ ions", "~1-2 m", 2.677, 0.0144, 1.5e6, 8.0)
REF74 = Experiment("[74]", "87Rb atoms", "400 m", 2.578, 0.078, 3342, 75.0)
REF75 = Experiment("[75]", "87Rb atoms", "11 km fibre", 2.612, 0.0285, 1.208e6, 624.0)
EXPERIMENTS = (REF73, REF74, REF75)

# Fig. 6 markers (paper caption)
STAR_N = 1.5e6           # Ref. [73]: ~8 h of operation
DIAMOND_N = 2.4e6        # Ref. [75]: extrapolated, ~1248 h

PROTOCOL = "single_key"
EC_MODEL = "printed"


def ref73_threshold_check() -> float:
    """omega_thr of Ref. [73] from Eq. (B10): S = 2.64, gamma = 13/256, n = 1.5e6, k = 3  ->  0.825538."""
    return omega_threshold(1_500_000, 13.0 / 256.0, (4.0 + 2.64) / 8.0, k=3.0)
