"""Named readings of the choices the paper leaves open (see PART2_NOTES.md).

The paper lists the optimised parameters (eps_s, eps_s', eps_s'', eps_h, alpha, alpha', t, gamma) but not beta, the free
parameter of the min-tradeoff function, nor which counts step 6 must bound.  beta decides how large the variance term of the
entropy-accumulation theorem is, and the thresholds of Figs. 3-5 are sensitive to it (up to 6e-3 in V at n = 1e8):

    STRICT         beta = g(1), loss count only      the protocol exactly as written (step 6 checks only n_loss); provably consistent
    RIGOROUS       beta optimised, loss AND win      Tan Eq. (30): for beta < g(1) both counts must be bounded; the win side has
                   counts bounded                    the larger variance, so the margin is larger.  Default for the protocol.
    LOSS_ONLY_OPT  beta optimised, loss count only   numerically equal to the thresholds in the repo README; not covered by the proof
    PAPER_FIT      beta = g(1) - 0.4 (g(1)-g(0)),    a one-parameter calibration on the six printed thresholds (<= 1e-4 for p = 0,
                   loss count only                   <= 3e-4 with optimised p); an empirical fit, NOT a derivation
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

from .finitekey import Budget
from .optimize import Optimum, PSpec, n_min, optimize_key_length, v_min


@dataclass(frozen=True)
class Preset:
    name: str
    beta: Union[str, float]
    pe_mode: str                  # "loss" (B10: margin on the loss count) | "two_sided" (loss and win counts, Tan Eq. 30)
    ec_model: str = "sifted"
    note: str = ""

    @property
    def two_sided(self) -> bool:
        return self.pe_mode == "two_sided"


STRICT = Preset("strict", "g1", "loss", note="beta = g(1); step 6 as written (loss count only)")
RIGOROUS = Preset("rigorous", "opt", "two_sided", note="beta optimised; loss and win counts checked (Tan Eq. 30)")
LOSS_ONLY_OPT = Preset("loss_only_opt", "opt", "loss", note="beta optimised, loss count only (equals the repo README thresholds)")
PAPER_FIT = Preset("paper_fit", 0.4, "loss", note="beta share 0.4, loss count only; calibrated on the paper's printed thresholds")

PRESETS: Dict[str, Preset] = {p.name: p for p in (STRICT, RIGOROUS, LOSS_ONLY_OPT, PAPER_FIT)}


def optimum(n: int, vis: float, preset: Preset = RIGOROUS, p: PSpec = 0.0, budget: Budget = Budget(), **kw) -> Optimum:
    """Maximal finite-size key length for Werner devices of visibility ``vis`` under the given reading."""
    return optimize_key_length(n, vis, p=p, beta=preset.beta, pe_mode=preset.pe_mode,
                               ec_model=kw.pop("ec_model", preset.ec_model), budget=budget, **kw)


def threshold_visibility(n: int, preset: Preset = RIGOROUS, p: PSpec = 0.0, budget: Budget = Budget(), **kw) -> float:
    return v_min(n, p=p, beta=preset.beta, pe_mode=preset.pe_mode, ec_model=kw.pop("ec_model", preset.ec_model),
                 budget=budget, **kw)


def threshold_rounds(vis: float, preset: Preset = RIGOROUS, p: PSpec = 0.0, budget: Budget = Budget(), **kw) -> float:
    return n_min(vis, p=p, beta=preset.beta, pe_mode=preset.pe_mode, ec_model=kw.pop("ec_model", preset.ec_model),
                 budget=budget, **kw)
