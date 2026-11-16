"""Timeline of the heralded rounds (Sec. II.A.2: "each [round] initiated upon receipt of a valid heralding signal").

``HeraldingModel`` is the minimal description needed here: one entanglement attempt every ``cycle_time`` seconds, each succeeding
(heralding signal) with probability ``p_herald``.  Part 3 derives p_herald and the cycle time of the matter-qubit experiments
(Refs. [73-75]) and of the photonic scheme; this module only turns them into timestamps / acquisition times.

* NumPy model: the number of attempts needed for n heralds is n + NegativeBinomial(n, p).
* SeQUeNCe backend (optional: ``pip install sequence``): the same process on SeQUeNCe's discrete-event kernel (``Timeline``,
  ``Entity``, ``Event``, ``Process``; times are integer picoseconds).  ``mode="heralds"`` schedules only the successful attempts
  (geometric waiting times; n events), ``mode="attempts"`` schedules every attempt (n/p events: small cases only).
  NOTE: written from the SeQUeNCe kernel API (v0.6/0.7); it has NOT been executed in the sandbox this package was developed in.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class HeraldingModel:
    p_herald: float            # probability that one attempt produces a heralding signal
    cycle_time: float          # seconds per attempt

    def mean_time_per_round(self) -> float:
        return self.cycle_time / self.p_herald

    def acquisition_time(self, n_rounds: int, rng: Optional[np.random.Generator] = None) -> float:
        """Total time (s) to collect n heralded rounds.  Without ``rng`` the expectation n * cycle / p."""
        if rng is None:
            return n_rounds * self.mean_time_per_round()
        failures = int(rng.negative_binomial(n_rounds, self.p_herald))
        return (n_rounds + failures) * self.cycle_time

    def timestamps(self, n_rounds: int, rng: np.random.Generator) -> np.ndarray:
        """Times (s) of the n heralding signals."""
        waits = rng.geometric(self.p_herald, size=n_rounds).astype(np.float64) * self.cycle_time
        return np.cumsum(waits)


# ------------------------------------------------------------------ SeQUeNCe backend
def sequence_available() -> bool:
    try:
        import sequence.kernel.timeline  # noqa: F401
        return True
    except Exception:
        return False


def sequence_timestamps(n_rounds: int, model: HeraldingModel, seed: int = 0, mode: str = "heralds") -> np.ndarray:
    """Heralding times (s) produced on SeQUeNCe's event kernel; distributionally identical to ``HeraldingModel.timestamps``."""
    from sequence.kernel.entity import Entity
    from sequence.kernel.event import Event
    from sequence.kernel.process import Process
    from sequence.kernel.timeline import Timeline

    cycle_ps = max(1, int(round(model.cycle_time * 1e12)))

    class HeraldingStation(Entity):
        def __init__(self, name, timeline):
            super().__init__(name, timeline)
            self.rng = np.random.default_rng(seed)
            self.stamps = []

        def init(self):
            self._schedule(1 if mode == "attempts" else int(self.rng.geometric(model.p_herald)))

        def _schedule(self, cycles: int):
            self.timeline.schedule(Event(self.timeline.now() + cycles * cycle_ps, Process(self, "fire", [])))

        def fire(self):
            if mode == "attempts":
                if self.rng.random() < model.p_herald:
                    self.stamps.append(self.timeline.now())
                if len(self.stamps) < n_rounds:
                    self._schedule(1)
            else:
                self.stamps.append(self.timeline.now())
                if len(self.stamps) < n_rounds:
                    self._schedule(int(self.rng.geometric(model.p_herald)))

    tl = Timeline()
    station = HeraldingStation("herald", tl)
    tl.init()
    tl.run()
    return np.asarray(station.stamps, dtype=np.float64) * 1e-12
