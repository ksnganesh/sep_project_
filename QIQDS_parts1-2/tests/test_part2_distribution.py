"""Tests for Part 2 (distribution stage) of the reproduction of Fan et al., PRA 114, 032617 (2026).

Run:  python -m pytest -q QIQDS_parts1-2/tests/test_part2_distribution.py          (~2 min; the QuTiP / Qiskit Aer / SeQUeNCe tests skip if the
                                                                      library is not installed)
                                                                      uv run pytest -v QIQDS_parts1-2/tests/
"""
import dataclasses
import heapq
import itertools
import math
import sys
import types

import numpy as np

from diqds.part1_messaging import encode_document, run_messaging_stage
from diqds.part2_distribution import (LOSS_ONLY_OPT, PAPER_FIT, RIGOROUS, STRICT, Budget, HeraldingModel, WernerDevice,
                                      distribute_signature_keys, min_tradeoff, n_min, optimize_key_length, plan_kgp,
                                      run_kgp, statistics, threshold_visibility, toeplitz_hash, toeplitz_matrix_explicit,
                                      v_min, werner_honest, werner_probabilities)
from diqds.part2_distribution import finitekey as fk
from diqds.part2_distribution import heralding, quantum_aer, quantum_qutip
from diqds.part2_distribution.optimize import evaluate
from diqds.part2_distribution.presets import optimum
from diqds.part2_distribution.privacy import random_seed

RNG = np.random.default_rng(777)


def skip(msg):
    try:
        import pytest
    except ImportError:
        import unittest
        raise unittest.SkipTest(msg)
    pytest.skip(msg)


# ---------------------------------------------------------------- single-round entropy bound (B1-B4)
def test_entropy_bound_limits():
    assert fk.r0(0.75) == 0.0 and fk.r0(0.7) == 0.0
    assert math.isclose(fk.rp(0.7, 0.2), fk.h2(0.2)) and math.isclose(fk.rp(0.7501, 0.2), fk.h2(0.2), rel_tol=2e-3)   # continuous at 3/4
    assert math.isclose(fk.r0(fk.OMEGA_MAX), 1.0)
    for w in (0.78, 0.82, 0.85):
        assert math.isclose(fk.rp(w, 0.0), fk.r0(w))                  # no preprocessing
        assert math.isclose(fk.rp(w, 0.5), 1.0)                       # fully randomised bit
        assert fk.rp(w, 0.1) >= fk.r0(w) - 1e-12                      # noisy preprocessing never lowers the bound
    ws = np.linspace(0.7501, fk.OMEGA_MAX - 1e-4, 50)
    assert np.all(np.diff([fk.r0(w) for w in ws]) > 0)


def test_derivatives_match_finite_differences():
    for p in (0.0, 0.05, 0.2, 0.4):
        for w in (0.76, 0.80, 0.83, 0.85):
            h = 1e-6
            for f, df, args in ((fk.r0, fk.dr0, ()), (fk.rp, fk.drp, (p,))):
                num = (f(w + h, *args) - f(w - h, *args)) / (2 * h)
                assert abs(num - df(w, *args)) < 1e-6 * max(1.0, abs(num))
            for g in (0.01, 0.2):
                num = (fk.eta(w + h, g, p) - fk.eta(w - h, g, p)) / (2 * h)
                assert abs(num - fk.deta(w, g, p)) < 1e-6 * max(1.0, abs(num))


def test_eta_convex_and_tangent_is_a_lower_bound():
    for gamma in (0.005, 0.05, 0.3):
        for p in (0.0, 0.1, 0.3):
            assert fk.eta_is_convex(gamma, p)
            for t in (0.77, 0.82, 0.85):
                mt = min_tradeoff(gamma, t, p)
                for w in np.linspace(fk.OMEGA_MIN, fk.OMEGA_MAX - 1e-6, 200):
                    # eta(w) for w <= 3/4 is (1-gamma)/2 h2(p): Eve may know A but not the private flip
                    assert mt.g(w) <= fk.eta(w, gamma, p) + 1e-9       # valid min-tradeoff function on the whole quantum range
                assert math.isclose(mt.g(t), fk.eta(t, gamma, p))     # touches at t


def _true_variance(mt, w):
    g = mt.gamma
    f = np.array([mt.beta, mt.g(1.0) / g + (1 - 1 / g) * mt.beta, mt.g(0.0) / g + (1 - 1 / g) * mt.beta])
    q = np.array([1 - g, g * w, g * (1 - w)])
    mean = float((q * f).sum())
    return mean, float((q * f * f).sum() - mean ** 2)


def test_variance_bound_dominates_exact_variance():
    """Tan Eq. (33) is an upper bound on Var_{Q_f}(f_min); E[f_min] = g(w) exactly."""
    ws = np.linspace(fk.OMEGA_MIN, fk.OMEGA_MAX, 801)
    for gamma in (0.005, 0.05, 0.4):
        for t in (0.78, 0.84):
            base = min_tradeoff(gamma, t, 0.0)
            for s in (0.0, 0.3, 0.5, 1.0):
                mt = min_tradeoff(gamma, t, 0.0, beta=base.g(1.0) - s * (base.g(1.0) - base.g(0.0)))
                stats = [_true_variance(mt, w) for w in ws]
                assert max(v for _, v in stats) <= mt.var_f + 1e-9
                assert all(abs(m - mt.g(w)) < 1e-9 for (m, _), w in zip(stats, ws))
                assert mt.max_f >= max(mt.beta, mt.g(1.0) / gamma + (1 - 1 / gamma) * mt.beta) - 1e-9


def test_midpoint_beta_minimises_variance_bound():
    base = min_tradeoff(0.01, 0.83, 0.0)
    d = base.g(1.0) - base.g(0.0)
    vs = [min_tradeoff(0.01, 0.83, 0.0, beta=base.g(1.0) - s * d).var_f for s in np.linspace(0, 1, 101)]
    assert abs(int(np.argmin(vs)) - 50) <= 1
    assert math.isclose(min(vs), d * d / (4 * 0.01), rel_tol=1e-9)
    assert math.isclose(vs[0], (2 + math.sqrt(2)) / (4 * 0.01) * d * d, rel_tol=1e-9)


def test_budget_and_hash_tag():
    b = Budget()
    assert math.isclose(b.eps_s, 2.0e-11, rel_tol=1e-6)               # eps_snd = 5e-11 = eps_PA + 2 eps_s (+ 2 eps_h)
    assert b.tag_bits == 61


def test_key_length_terms_sum_and_signs():
    o = optimum(10 ** 9, 0.95, STRICT)
    t = o.terms
    assert math.isclose(sum(v for k, v in t.items() if k != "ell"), t["ell"], rel_tol=1e-12)
    assert t["n_g"] > 0 and all(v <= 0 for k, v in t.items() if k not in ("n_g", "const", "ell"))
    assert math.isclose(o.ell, 2.2787e8, rel_tol=5e-3)                # value from the first independent evaluation


def test_asymptotic_rate_is_the_devetak_winter_value():
    """Tan Eq. (39): l/n -> (1/2)(r_p(w) - h2(Q')) as n -> infinity (gamma -> 0)."""
    vis, hon = 0.95, werner_honest(0.95)
    o = optimize_key_length(10 ** 17, vis, p=0.0, beta="g1")
    assert abs(o.ell / o.n - 0.5 * (fk.rp(hon.omega, 0.0) - fk.h2(hon.qber))) < 5e-4


def test_ec_models_and_threshold_helpers():
    hon = werner_honest(0.9)
    assert fk.ec_max_bits(10 ** 8, 0.01, 0.0, hon, "printed") > fk.ec_max_bits(10 ** 8, 0.01, 0.0, hon, "sifted")
    assert math.isclose(fk.ec_max_bits(10 ** 8, 0.01, 0.0, hon, "sifted", c_sqrt=0.0),
                        10 ** 8 * (0.5 * 0.99 * fk.h2(hon.qber) + 0.01 * fk.h2(1 - hon.omega)))
    assert fk.ec_max_bits(10 ** 8, 0.01, 0.2, hon, "sifted") > fk.ec_max_bits(10 ** 8, 0.01, 0.0, hon, "sifted")
    n, g, w = 10 ** 9, 0.01, hon.omega
    q = g * (1 - w)
    assert math.isclose(fk.omega_threshold(n, g, w), 1 - (q + 3 * math.sqrt(q * (1 - q) / n)) / g)   # Eq. (B10)
    assert fk.omega_threshold(n, g, w, two_sided=True) < fk.omega_threshold(n, g, w)


# ---------------------------------------------------------------- optimiser
def test_optimiser_is_robust_near_the_finite_size_floor():
    """Regression: a single Nelder-Mead start ran into the gamma boundary at n = 3.2e6 and returned a negative key."""
    o = optimize_key_length(3_200_000, 0.995, p=0.0, beta="g1")
    assert o.ell > 2.5e5
    best = max(evaluate(3_200_000, 0.995, 10 ** lg, t, 0.0, 1.0, 1.0).ell
               for lg in np.arange(-2.0, -0.9, 0.2) for t in np.linspace(0.80, 0.85, 12))
    assert o.ell >= best - 1e-6 * abs(best)


def test_key_length_monotone_in_visibility_and_rounds():
    ells = [optimize_key_length(10 ** 9, v, p=0.0, beta="g1").ell for v in (0.90, 0.95, 0.99)]
    assert ells[0] < ells[1] < ells[2]
    ells = [optimize_key_length(n, 0.97, p=0.0, beta="g1").ell / n for n in (10 ** 7, 10 ** 9, 10 ** 11)]
    assert ells[0] < ells[1] < ells[2]


def test_noisy_preprocessing_helps_in_high_noise_and_hurts_in_low_noise():
    """Sec. IV: 'noisy preprocessing improves performance in high-noise regimes, at the cost of low-noise scenarios'."""
    n = 10 ** 9
    hi = (optimize_key_length(n, 0.872, p="opt", beta="g1").ell, optimize_key_length(n, 0.872, p=0.0, beta="g1").ell)
    assert hi[0] > max(hi[1], 0.0)
    lo = (optimize_key_length(n, 0.99, p=0.15, beta="g1").ell, optimize_key_length(n, 0.99, p=0.0, beta="g1").ell)
    assert lo[1] > lo[0]


def test_thresholds_against_paper_and_repo_p0():
    """n = 1e9, p = 0.  Paper (printed): 0.8686.  Repo README: 0.8681.  The paper lies between the two natural readings."""
    n = 10 ** 9
    v = {p.name: threshold_visibility(n, p, 0.0, tol=1e-6) for p in (STRICT, RIGOROUS, LOSS_ONLY_OPT, PAPER_FIT)}
    assert abs(v["loss_only_opt"] - 0.8681) < 2e-4                    # identical to the repo's value
    assert abs(v["paper_fit"] - 0.8686) < 3e-4
    assert abs(v["rigorous"] - 0.8690) < 3e-4
    assert abs(v["strict"] - 0.8713) < 3e-4
    assert v["loss_only_opt"] < v["paper_fit"] < v["rigorous"] < v["strict"]
    assert v["loss_only_opt"] < 0.8686 < v["rigorous"]                # the printed value is bracketed


def test_printed_b11_is_inconsistent_with_figs_3_to_5():
    """Eq. (B11) as printed (no sifting factor) gives thresholds around 0.91-0.93, far from the printed 0.862-0.884."""
    v = v_min(10 ** 10, p=0.0, beta="g1", ec_model="printed", tol=1e-6)
    assert 0.90 < v < 0.93
    assert v_min(10 ** 10, p=0.0, beta="g1", ec_model="sifted", tol=1e-6) < 0.8645


def test_finite_size_floor_in_rounds():
    n1 = n_min(0.99, p=0.0, beta="g1", rel=2e-2)
    n2 = n_min(0.999, p=0.0, beta="g1", rel=2e-2)
    assert 1.5e6 < n1 < 2.2e6 and 1.2e6 < n2 < n1                      # ~1e6 rounds even for almost perfect devices


# ---------------------------------------------------------------- devices
def test_werner_probabilities_and_statistics():
    for vis in (0.8, 0.95, 1.0):
        P = werner_probabilities(vis)
        assert np.allclose(P.sum(axis=(2, 3)), 1.0)
        assert np.allclose(P.sum(axis=3), 0.5) and np.allclose(P.sum(axis=2), 0.5)      # no signalling, uniform marginals
        S, w, q = statistics(P)
        hon = werner_honest(vis)
        assert math.isclose(S, 2 * math.sqrt(2) * vis, abs_tol=1e-12)
        assert math.isclose(w, hon.omega, abs_tol=1e-12) and math.isclose(q, hon.qber, abs_tol=1e-12)
    assert statistics(werner_probabilities(0.70))[0] < 2.0 < statistics(werner_probabilities(0.72))[0]   # 1/sqrt2 threshold


def test_sampling_matches_probabilities():
    dev = WernerDevice(0.9)
    N = 400_000
    x = RNG.integers(0, 2, N, dtype=np.uint8)
    y = RNG.integers(0, 4, N, dtype=np.uint8)
    a, b = dev.sample(x, y, RNG)
    for xx in (0, 1):
        for yy in range(4):
            m = (x == xx) & (y == yy)
            for aa in (0, 1):
                for bb in (0, 1):
                    p = dev.P[xx, yy, aa, bb]
                    obs = float(((a[m] == aa) & (b[m] == bb)).mean())
                    assert abs(obs - p) < 5 * math.sqrt(p * (1 - p) / m.sum())


# ---------------------------------------------------------------- privacy amplification
def test_toeplitz_fft_matches_explicit_matrix():
    for n, m in ((8, 3), (200, 17), (3000, 40)):
        x = RNG.integers(0, 2, n, dtype=np.uint8)
        s = random_seed(n, m, RNG)
        T = toeplitz_matrix_explicit(s, n, m).astype(np.int64)
        assert np.array_equal(toeplitz_hash(x, s, m), ((T @ x.astype(np.int64)) % 2).astype(np.uint8))


def test_toeplitz_hash_is_linear_and_precise_at_scale():
    n, m = 1_200_000, 61
    x1, x2 = (RNG.integers(0, 2, n, dtype=np.uint8) for _ in range(2))
    s = random_seed(n, m, RNG)
    h = toeplitz_hash(x1 ^ x2, s, m)
    assert np.array_equal(h, toeplitz_hash(x1, s, m) ^ toeplitz_hash(x2, s, m))
    for i in (0, 7, 60):                                               # independent exact check of single output bits
        assert int(np.dot(s[i:i + n][::-1].astype(np.int64), x1.astype(np.int64)) & 1) == int(toeplitz_hash(x1, s, m)[i])


# ---------------------------------------------------------------- protocol mechanics
def test_plan_is_self_consistent():
    plan = plan_kgp(2_000_000, 0.995, RIGOROUS)
    o = plan.optimum
    assert plan.ell == math.floor(o.ell) > 100_000
    assert plan.n_loss_max == math.floor(plan.n * plan.gamma * (1 - plan.omega_thr))
    assert plan.n_win_min == math.ceil(plan.n * plan.gamma * plan.omega_thr)
    hon = werner_honest(0.995)
    assert plan.ec_max > plan.n * (0.5 * (1 - plan.gamma) * fk.h2(hon.qber) + plan.gamma * fk.h2(1 - hon.omega))
    assert plan.completeness_lower_bound() > 0.99
    assert plan.tag_bits == 61 and plan.two_sided
    assert not plan_kgp(2_000_000, 0.995, STRICT).two_sided


def test_sifting_and_noisy_preprocessing_statistics():
    n, p = 300_000, 0.1
    plan = plan_kgp(n, 0.99, RIGOROUS, p=p)
    r = run_kgp(plan, WernerDevice(0.99), RNG, ell_override=200)
    exp_key = n * (1 - plan.gamma)
    assert abs(r.n_test - n * plan.gamma) < 5 * math.sqrt(n * plan.gamma * (1 - plan.gamma))
    for cnt in (r.n_matched, r.n_sifted_out):                          # half of the key rounds are sifted out
        assert abs(cnt - exp_key / 2) < 5 * math.sqrt(exp_key / 4)
    assert abs(r.n_flipped - p * r.n_matched) < 5 * math.sqrt(p * (1 - p) * r.n_matched)
    q_eff = p + (1 - 2 * p) * werner_honest(0.99).qber                 # Q' = p + (1-2p) Q
    assert abs(r.e_key / r.n_matched - q_eff) < 5 * math.sqrt(q_eff * (1 - q_eff) / r.n_matched)


def test_honest_run_mechanics():
    n = 300_000
    plan = plan_kgp(n, 0.99, STRICT)
    assert plan.ell < 0                                                # below the finite-size floor ...
    r = run_kgp(plan, WernerDevice(0.99), RNG)
    assert not r.accepted and r.reason == "finite-size bound gives no key"
    r = run_kgp(plan, WernerDevice(0.99), RNG, ell_override=500)       # ... mechanics only: bypass Eq. (31)
    assert r.accepted and r.ell == 500 and r.tag_ok and r.keys_agree
    assert r.n_loss <= plan.n_loss_max and r.ec_leak <= plan.ec_max
    assert abs(r.chsh_observed - 2 * math.sqrt(2) * 0.99) < 0.05


def test_degraded_devices_and_ec_budget_are_caught():
    plan = plan_kgp(2_000_000, 0.995, RIGOROUS)
    r = run_kgp(plan, WernerDevice(0.97), RNG)
    assert not r.accepted and r.reason in ("EC budget exceeded", "too many losses", "too few wins")
    r = run_kgp(dataclasses.replace(plan, ec_max=1000), WernerDevice(0.995), RNG)
    assert not r.accepted and r.reason == "EC budget exceeded"
    r = run_kgp(plan, WernerDevice(0.995), RNG, f_ec=1.8)              # a code 80 % above the Shannon limit exceeds EC_max here
    assert not r.accepted
    r = run_kgp(plan, WernerDevice(0.995), RNG, f_ec=1.05)
    assert r.accepted and r.keys_agree
    r = run_kgp(dataclasses.replace(plan, n_loss_max=100), WernerDevice(0.995), RNG)
    assert not r.accepted and r.reason == "too many losses"
    r = run_kgp(dataclasses.replace(plan, n_win_min=plan.n), WernerDevice(0.995), RNG)
    assert not r.accepted and r.reason == "too few wins"


def test_end_to_end_device_independent_signatures():
    """Werner pairs -> two KGP links -> key triples -> Part 1 signing/verification."""
    rng = np.random.default_rng(31)
    triples, r_ab, r_ac = distribute_signature_keys(2_000_000, 0.995, 46, rng, preset=RIGOROUS)
    assert r_ab.accepted and r_ac.accepted and r_ab.keys_agree and r_ac.keys_agree
    assert r_ab.ell == r_ac.ell and len(triples) >= 1200 and len(triples) <= r_ab.ell // 138
    for i in range(25):
        out = run_messaging_stage(encode_document(b"doc %d" % i, 1000), *triples[i])
        assert out.valid
    # a key triple is single-use: re-using triple 0 for a different document still verifies (symmetric keys), but the keys of
    # two different links differ
    assert not np.array_equal(r_ab.key_a, r_ac.key_a)


# ---------------------------------------------------------------- heralding timeline
def test_numpy_acquisition_time():
    m = HeraldingModel(p_herald=0.02, cycle_time=1e-6)
    rng = np.random.default_rng(5)
    times = np.array([m.acquisition_time(2000, rng) for _ in range(400)])
    mean = 2000 * m.cycle_time / m.p_herald
    sd = math.sqrt(2000 * (1 - m.p_herald)) / m.p_herald * m.cycle_time        # negative binomial standard deviation
    assert abs(times.mean() - mean) < 4 * sd / math.sqrt(400)
    assert math.isclose(m.acquisition_time(2000), mean, rel_tol=1e-12)
    ts = m.timestamps(5000, rng)
    assert np.all(np.diff(ts) > 0) and abs(np.diff(ts).mean() - m.mean_time_per_round()) < 0.1 * m.mean_time_per_round()


# ---------------------------------------------------------------- logic of the optional backends, checked WITHOUT the libraries
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)


def _phi_plus_dm():
    v = np.zeros(4, dtype=complex)
    v[0] = v[3] = 1 / math.sqrt(2)
    return np.outer(v, v.conj())


def test_closed_form_matches_density_matrix_emulation_of_the_qutip_computation():
    """quantum_qutip computes Tr[(P_a(theta) (x) P_b(phi)) rho] with Qobj; here the same Born-rule computation with NumPy matrices
    must reproduce devices.werner_probabilities (the cos(theta - phi) closed form), validating signs and conventions."""
    def proj(theta, o):
        return 0.5 * (_I2 + (-1) ** o * (math.cos(theta) * _Z + math.sin(theta) * _X))
    from diqds.part2_distribution.devices import ALICE_ANGLES, BOB_ANGLES
    for vis in (0.7, 0.93, 1.0):
        rho = vis * _phi_plus_dm() + (1 - vis) * np.eye(4) / 4
        P = np.zeros((2, 4, 2, 2))
        for x, th in enumerate(ALICE_ANGLES):
            for y, ph in enumerate(BOB_ANGLES):
                for a in (0, 1):
                    for b in (0, 1):
                        P[x, y, a, b] = np.real(np.trace(np.kron(proj(th, a), proj(ph, b)) @ rho))
        assert np.allclose(P, werner_probabilities(vis), atol=1e-12)


def test_aer_circuit_logic_emulated_in_numpy_and_memory_parsing():
    """Emulate h; cx; depolarizing(1-V); ry(-theta) (x) ry(-phi); measure Z (x) Z with NumPy and compare with the exact model;
    check the little-endian 'c1c0' parsing used by AerWernerDevice.sample."""
    from diqds.part2_distribution.devices import ALICE_ANGLES, BOB_ANGLES

    def ry(t):
        return np.array([[math.cos(t / 2), -math.sin(t / 2)], [math.sin(t / 2), math.cos(t / 2)]], dtype=complex)
    vis = 0.91
    rho = (1 - (1 - vis)) * _phi_plus_dm() + (1 - vis) * np.eye(4) / 4          # depolarizing_error(1 - V, 2) after the CX
    P = werner_probabilities(vis)
    for x, th in enumerate(ALICE_ANGLES):
        for y, ph in enumerate(BOB_ANGLES):
            U = np.kron(ry(-th), ry(-ph))                                        # qubit 0 = Alice, qubit 1 = Bob
            probs = np.real(np.diag(U @ rho @ U.conj().T))                       # index 2a + b
            assert np.allclose(probs.reshape(2, 2), P[x, y], atol=1e-12)
    a, b = quantum_aer.parse_memory(["01", "10", "11", "00"])                    # 'c1c0': '01' -> c0 = 1 -> a = 1, b = 0
    assert a.tolist() == [1, 0, 1, 0] and b.tolist() == [0, 1, 1, 0]


def _fake_sequence_modules():
    """Minimal stand-in for sequence.kernel.{timeline,entity,event,process} with the same contract (picosecond integer times)."""
    class Timeline:
        def __init__(self, stop_time=float("inf"), formalism="ket_vector"):
            self.time, self.events, self.entities, self._c = 0, [], [], itertools.count()

        def add_entity(self, e):
            self.entities.append(e)

        def now(self):
            return self.time

        def schedule(self, ev):
            assert ev.time >= self.time
            heapq.heappush(self.events, (ev.time, next(self._c), ev))

        def init(self):
            for e in self.entities:
                e.init()

        def run(self):
            while self.events:
                t, _, ev = heapq.heappop(self.events)
                self.time = t
                ev.process.run()

    class Entity:
        def __init__(self, name, timeline):
            self.name, self.timeline = name, timeline
            timeline.add_entity(self)

    class Process:
        def __init__(self, owner, activation_method, activation_method_args):
            self.owner, self.method, self.args = owner, activation_method, activation_method_args

        def run(self):
            getattr(self.owner, self.method)(*self.args)

    class Event:
        def __init__(self, time, process, priority=0):
            self.time, self.process, self.priority = time, process, priority

    mods = {}
    for name, cls in (("timeline", Timeline), ("entity", Entity), ("process", Process), ("event", Event)):
        m = types.ModuleType(f"sequence.kernel.{name}")
        setattr(m, cls.__name__, cls)
        mods[f"sequence.kernel.{name}"] = m
    mods["sequence"] = types.ModuleType("sequence")
    mods["sequence.kernel"] = types.ModuleType("sequence.kernel")
    return mods


def test_sequence_station_logic_on_a_fake_event_kernel():
    """The station code of heralding.sequence_timestamps run on a minimal fake kernel (same Timeline/Entity/Event/Process contract)."""
    fake = _fake_sequence_modules()
    saved = {k: sys.modules.get(k) for k in fake}
    sys.modules.update(fake)
    try:
        m = HeraldingModel(p_herald=0.2, cycle_time=1e-8)
        for mode in ("heralds", "attempts"):
            ts = heralding.sequence_timestamps(4000, m, seed=3, mode=mode)
            assert len(ts) == 4000 and np.all(np.diff(ts) > 0)
            assert abs(np.diff(ts).mean() - m.mean_time_per_round()) < 0.08 * m.mean_time_per_round()
            assert np.allclose(ts / m.cycle_time, np.rint(ts / m.cycle_time), atol=1e-6)      # heralds fall on attempt boundaries
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


# ---------------------------------------------------------------- optional backends (skip when not installed)
def test_sequence_backend_matches_numpy_model():
    if not heralding.sequence_available():
        skip("SeQUeNCe not installed")
    m = HeraldingModel(p_herald=0.2, cycle_time=1e-8)
    for mode in ("heralds", "attempts"):
        ts = heralding.sequence_timestamps(4000, m, seed=1, mode=mode)
        assert len(ts) == 4000 and np.all(np.diff(ts) > 0)
        assert abs(np.diff(ts).mean() - m.mean_time_per_round()) < 0.06 * m.mean_time_per_round()


def test_qutip_backend_matches_numpy_model():
    if not quantum_qutip.available():
        skip("QuTiP not installed")
    assert quantum_qutip.check_against_numpy(0.93)
    dev = quantum_qutip.QutipWernerDevice(0.9)
    assert np.allclose(dev.P, werner_probabilities(0.9), atol=1e-12)
    assert math.isclose(dev.chsh, 2 * math.sqrt(2) * 0.9, abs_tol=1e-9)


def test_aer_backend_matches_numpy_model():
    if not quantum_aer.available():
        skip("Qiskit Aer not installed")
    vis, N = 0.92, 60_000
    dev = quantum_aer.AerWernerDevice(vis)
    x = RNG.integers(0, 2, N, dtype=np.uint8)
    y = RNG.integers(0, 4, N, dtype=np.uint8)
    a, b = dev.sample(x, y, RNG)
    P = werner_probabilities(vis)
    for xx in (0, 1):
        for yy in range(4):
            m = (x == xx) & (y == yy)
            for aa in (0, 1):
                for bb in (0, 1):
                    p = P[xx, yy, aa, bb]
                    obs = float(((a[m] == aa) & (b[m] == bb)).mean())
                    assert abs(obs - p) < 5 * math.sqrt(p * (1 - p) / m.sum()) + 1e-3
