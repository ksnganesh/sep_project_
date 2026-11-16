"""Part 2 of the reproduction of Fan et al., PRA 114, 032617 (2026): the distribution stage.

Paper section -> module
    II.A steps 1-7 (key-generation protocol)       kgp.py (+ privacy.py, devices.py, heralding.py)
    App. B  finite-size key length, Eq. (9)/(B12)   finitekey.py (Tan et al. Thm 1), optimize.py, presets.py
    IV      depolarizing-noise devices              devices.py;  quantum_qutip.py / quantum_aer.py are optional backends
"""
from .devices import WernerDevice, statistics, werner_probabilities  # noqa: F401
from .finitekey import Budget, Honest, MinTradeoff, min_tradeoff, werner_honest  # noqa: F401
from .heralding import HeraldingModel  # noqa: F401
from .kgp import (KGPPlan, KGPResult, distribute_signature_keys, plan_kgp, run_kgp)  # noqa: F401
from .optimize import Optimum, n_min, optimize_key_length, v_min  # noqa: F401
from .presets import (LOSS_ONLY_OPT, PAPER_FIT, PRESETS, RIGOROUS, STRICT, Preset, optimum,  # noqa: F401
                      threshold_rounds, threshold_visibility)
from .privacy import toeplitz_hash, toeplitz_matrix_explicit  # noqa: F401
