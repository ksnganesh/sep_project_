"""Part 3 (Sec. IV, Figs. 3-7) of Fan et al., PRA 114, 032617 (2026): the numbers the figures rest on.

    python -m pytest -q tests/test_part3_figures.py        (~2 min)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diqds.part2_distribution import finitekey as fk  # noqa: E402
from diqds.part2_distribution.presets import PAPER_FIT, threshold_rounds, threshold_visibility  # noqa: E402
from diqds.part3_figures import matter, photonic as ph  # noqa: E402
from diqds.part3_figures.sweeps import KeyLength, signature_length, signature_rate  # noqa: E402


# ------------------------------------------------------------------------------------------------ shared pipeline
def test_signature_length_is_46():
    assert signature_length() == 46                         # Eq. (5) with Table I; r_sig = ell / 138


def test_random_key_protocol_unchanged_by_generalisation():
    """The 'protocol' switch must leave the paper's own protocol (Eq. B1) exactly as in Part 2."""
    for w in (0.8, 0.83, 0.85):
        assert fk.eta(w, 0.01, 0.1) == fk.eta(w, 0.01, 0.1, "random_key")
        assert math.isclose(fk.eta(w, 0.01, 0.0, "single_key"), fk.r0(w))      # no sifting: r_0 on every round


# ------------------------------------------------------------------------------------------------ Figs. 3-5
@pytest.mark.parametrize("n,printed", [(10 ** 8, 0.8835), (10 ** 10, 0.8622)])
def test_fig5_thresholds_p0(n, printed):
    assert abs(threshold_visibility(n, PAPER_FIT, p=0.0) - printed) < 3e-4


def test_fig5_legend_p_is_the_threshold_optimum():
    """The dashed curves of Fig. 5 use p = 0.155 / 0.090 / 0.035: these are the p that minimise V_min (to 1e-5)."""
    v_leg = threshold_visibility(10 ** 9, PAPER_FIT, p=0.090)
    v_opt = threshold_visibility(10 ** 9, PAPER_FIT, p="opt")
    assert abs(v_leg - v_opt) < 2e-5
    assert abs(v_opt - 0.8608) < 5e-4


def test_fig3_threshold_v1():
    """Digitised from the published Fig. 3: V = 1 becomes positive at n ~ 8.8e5."""
    assert 8.0e5 < threshold_rounds(1.0, PAPER_FIT, p=0.0, lo=1e5, hi=1e8) < 1.0e6


# ------------------------------------------------------------------------------------------------ Fig. 6
def test_b10_and_b11_are_the_kgp_of_ref73():
    """Eq. (B10) with k = 3 gives Ref. [73]'s omega_thr = 0.825538 and Eq. (B11) as printed gives its syndrome length
    m = 296,517 bits (S = 2.64, Q = 1.8 %, gamma = 13/256, n = 1.5e6): (B10)/(B11) are the KGP of Ref. [73]."""
    assert abs(matter.ref73_threshold_check() - 0.825538) < 1e-6
    hon = fk.Honest(omega=(4 + 2.64) / 8, qber=0.018)
    ec = fk.ec_max_bits(1_500_000, 13 / 256, 0.0, hon, "printed")
    assert abs(ec - 296_517) / 296_517 < 2e-3
    assert fk.ec_max_bits(1_500_000, 13 / 256, 0.0, hon, "single_key") == ec    # identical at p = 0


@pytest.mark.parametrize("exp,digitised", [(matter.REF73, 3.97e7), (matter.REF74, 9.68e6), (matter.REF75, 2.74e7)])
def test_fig6_large_n_rates(exp, digitised):
    kl = KeyLength(PAPER_FIT, 0.0, matter.PROTOCOL, matter.EC_MODEL)
    r = signature_rate(kl(1e10, honest=exp.honest).ell)
    assert abs(r / digitised - 1) < 0.06


def test_fig6_ref75_extrapolated_point_is_positive():
    """Paper: n = 2.4e6 (~1248 h at Ref. [75]'s event rate) 'yields a positive signature rate'."""
    kl = KeyLength(PAPER_FIT, 0.0, matter.PROTOCOL, matter.EC_MODEL)
    assert kl(2.4e6, honest=matter.REF75.honest).ell > 0
    assert abs(matter.REF75.hours_for(2.4e6) - 1248) < 15


# ------------------------------------------------------------------------------------------------ Fig. 7 photonic model
def test_transfer_matrix_is_a_lossless_network():
    U = ph.transfer_matrix(0.01, 0.3, 0.5)
    assert np.allclose(U @ U.T, np.eye(4))


def test_heralded_state_is_eq_d1():
    """Ideal optics, small T: the pattern {d1-, d2+} heralds |+-> - t/(1-t)|-+>  (Eq. D1).  Measuring both photons in the
    +/- basis (theta = pi/2 projects on |+>) gives P(+,-) = 1 / (1 + (t/(1-t))^2)."""
    t = 0.3
    src = ph.HeraldedSource.build(ph.PhotonicParams(T=1e-3, t=t, eta_l=1.0, v_hom=1.0, g2=0.0), 0.0)
    _, P = src.joint([math.pi / 2], [math.pi / 2])
    expected = 1.0 / (1.0 + (t / (1 - t)) ** 2)
    assert abs(P[0, 0, 0, 1] - expected) < 2e-3                    # Alice clicks on +, Bob does not (he has -)
    assert P[0, 0, 0, 0] < 2e-3 and P[0, 0, 1, 1] < 2e-3


def test_ideal_photonic_link_violates_chsh_maximally():
    src = ph.HeraldedSource.build(ph.PhotonicParams(T=1e-3, eta_l=1.0, v_hom=1.0, g2=0.0), 0.0)
    PH, P = src.joint([0.0, math.pi / 2], [math.pi / 4, -math.pi / 4])
    E = P[:, :, 0, 0] + P[:, :, 1, 1] - P[:, :, 0, 1] - P[:, :, 1, 0]
    assert abs(abs(E[0, 0] + E[0, 1] + E[1, 0] - E[1, 1]) - 2 * math.sqrt(2)) < 1e-2
    assert abs(PH / (1e-3 ** 2 * 0.5 ** 2) - 0.5) < 5e-3          # one pattern: P_H = T^2 eta_C^2 / 2


def test_imperfections_degrade_the_link():
    ops = {}
    for name, prm in (("ideal", ph.IDEAL_SOURCE), ("explicit", ph.realistic_source("explicit"))):
        ops[name] = ph.optimise_angles(ph.HeraldedSource.build(prm, 0.0), maxiter=200)
    assert 2.60 < ops["ideal"].chsh < 2.66 and ops["ideal"].qber < 0.04        # eta_l = 0.97 with no-click binning
    assert ops["explicit"].chsh < ops["ideal"].chsh - 0.1
    assert ops["explicit"].r_inf < 0.25 * ops["ideal"].r_inf


def test_fig7_ideal_source_distances():
    """Paper: 111.46 km (N = 1e14) and 126.53 km (N = 2e14) for V_HOM = 1, g2 = 0 - reproduced with no fitted parameter."""
    tab = ph.OperatingTable.from_points(ph.operating_curve(ph.IDEAL_SOURCE, np.arange(95.0, 141.0, 5.0), maxiter=200))
    L1, L2 = (ph.max_distance(tab, N, lo=95.0, hi=140.0) for N in (1e14, 2e14))
    assert abs(L1 - 111.46) < 1.0 and abs(L2 - 126.53) < 1.0
    assert abs((L2 - L1) - 15.07) < 0.5                            # ~ L0 ln 2, slightly less: false heralds grow with L
