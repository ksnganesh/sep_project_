"""Fully photonic DIQDS with quantum-dot single-photon sources: Fig. 1(b), Appendices C and D, Fig. 7.

The paper describes the architecture and the imperfections but gives no formula for the heralded statistics or for P_H
("the table is in the supplement of Ref. [80]").  This module computes them EXACTLY within an explicit model, by
second-quantised linear optics of up to five photons:

  sources     each station s in {A, B} has two QD sources emitting one photon each, H and V, into the same spatial mode
              (Fig. 1b, App. D).  Multiphoton emission (App. C.2): a source emits a second photon with probability P2,
              g2 = 2 P2 / (P1 + 2 P2)^2, P1 + P2 = 1 (C10, C12); to first order in P2 at most one extra photon is present and it
              is fully distinguishable from all others (as in Ref. [80]; this generates the visibility drop of Eq. C13 rather
              than inserting it by hand).
  indistinguishability (App. C.1)  every photon is, independently, in a common temporal mode with probability x or in a
              private mode otherwise; then the HOM visibility of any pair is x^2 = V_HOM (C6).  Local and cross-station
              visibilities are taken equal (alpha = beta = V_HOM), as in Ref. [80].
  network     a_{s,P} -> sqrt(1-T) l_{s,P} + sqrt(T) c_{s,P};  c -> sqrt(eta_C) c' + sqrt(1-eta_C) loss,
              eta_C = eta_conv exp(-L / 2 L0)  (D2);  at the CHS c'_A, c'_B (in the +/- polarisation basis) meet on a beam
              splitter of transmittance t: c'_A -> sqrt(t) d1 + sqrt(1-t) d2,  c'_B -> sqrt(1-t) d1 - sqrt(t) d2.
  herald      threshold detectors d1+, d1-, d2+, d2-; a round is heralded by exactly the click pattern {d1-, d2+}.  For one
              photon per station this projects the local photons onto  -(1-t)|+-> + t|-+>  ~  |+-> - t/(1-t)|-+>,  i.e.
              Eq. (D1); for t = 1/2 (App. D) the singlet.  (The pattern {d1+, d2-} heralds the mirror state t <-> 1-t;
              ``HERALDS_BOTH`` accepts both, doubling P_H.)  Two photons from the SAME station bunch in the +/- basis
              (a_H^dag a_V^dag = (a_+^dag2 - a_-^dag2)/2) and cannot give a +/- coincidence unless they are distinguishable:
              this is why local indistinguishability matters, and why T -> 0 suppresses 3-photon false heralds.
  local       polarisation measurement along |theta> = cos(theta/2)|H> + sin(theta/2)|V> with a single threshold detector
              of efficiency eta_l (App. D: source + detection), click -> 0, no click -> 1 (no-click binning), plus a
              classical relabelling bit per Bob setting chosen by the honest device.

Outputs per distance L: P_H (D3: n_H = P_H N), and the honest statistics (omega, Q) for optimised measurement angles, which
feed the same finite-size machinery as Figs. 3-5 (the paper's protocol of Sec. II.A with noisy preprocessing).

Source imperfections, two readings (the paper names both ingredients but not how they are combined):
  * ``explicit``  : V_HOM through the distinguishability model above and g2 through an explicit extra photon (Ref. [80]).
  * ``c13``       : Eq. (C13), V = V_HOM - kappa g2 with 1 < kappa < 3, and no explicit extra photon ("we adopt a modified
                    approach").  kappa is not given; ``REALISTIC_KAPPA`` is fitted to the N = 1e14 distance (17.82 km) and the
                    N = 2e14 distance is then a prediction.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

from diqds.part2_distribution.finitekey import Honest, h2, rp

# ------------------------------------------------------------------------------------------------ parameters (App. D, Fig. 7)
@dataclass(frozen=True)
class PhotonicParams:
    T: float = 0.01                # local beam splitter towards the CHS
    t: float = 0.5                 # CHS beam splitter
    L0: float = 22.0               # km, attenuation length (D2)
    eta_conv: float = 0.5          # wavelength conversion (D2)
    eta_l: float = 0.97            # local efficiency (source extraction + detection)
    v_hom: float = 0.975           # HOM visibility (C6)
    g2: float = 0.01               # second-order correlation (C11)
    rate_hz: float = 75e6          # photon generation rate nu (Sec. IV: 1e14 rounds ~ 370 h)

    def eta_c(self, L: float) -> float:
        return self.eta_conv * math.exp(-L / (2.0 * self.L0))

    @property
    def p2(self) -> float:
        """P2 from g2 = 2 P2 / (1 + P2)^2 (C12 with P1 + P2 = 1)."""
        if self.g2 <= 0:
            return 0.0
        g = self.g2
        return ((1.0 - g) - math.sqrt(1.0 - 2.0 * g)) / g      # small root of g P2^2 + (2g - 2) P2 + g = 0

    @property
    def x_good(self) -> float:
        return math.sqrt(self.v_hom)


IDEAL_SOURCE = PhotonicParams(v_hom=1.0, g2=0.0)
REALISTIC_SOURCE = PhotonicParams(v_hom=0.975, g2=0.01)          # explicit reading; see realistic_source()
# Eq. (C13) V = V_HOM - kappa g2, 1 < kappa < 3: kappa fitted so that the N = 1e14 cut-off is the printed 17.82 km
# (bisection, one herald pattern, PAPER_FIT finite-size reading).  The N = 2e14 cut-off (paper 28.87 km) is then a prediction.
REALISTIC_KAPPA = 1.5446

# mode bookkeeping ---------------------------------------------------------------------------------------------------------
INPUTS = (("A", "H"), ("A", "V"), ("B", "H"), ("B", "V"))
N_OUT = 12          # d1+, d1-, d2+, d2-, loss AH, AV, BH, BV, local AH, AV, BH, BV
D_MODES, LOSS_MODES, LOCAL_MODES = range(0, 4), range(4, 8), range(8, 12)
PATTERN_D1P_D2M = 0b1001           # bits: d1+ = 1, d1- = 2, d2+ = 4, d2- = 8
PATTERN_D1M_D2P = 0b0110
HERALDS_BOTH = (PATTERN_D1P_D2M, PATTERN_D1M_D2P)
# Eq. (D1) prepares |+-> - t/(1-t) |-+>: that is the state of the single pattern {d1-, d2+} (the other pattern gives the mirror
# image t <-> 1-t).  Accepting only this pattern is therefore the paper's specification; with it the ideal-source distances of
# Fig. 7 come out at 111.2 / 126.3 km (paper: 111.46 / 126.53) with no fitted parameter.  Accepting both doubles P_H (+15 km).
HERALDS_DEFAULT = (PATTERN_D1M_D2P,)


def transfer_matrix(T: float, eta_c: float, t: float) -> np.ndarray:
    """4 x 12 single-photon amplitudes input -> output (rows orthonormal: a lossless network with loss modes)."""
    U = np.zeros((4, N_OUT))
    a, c = math.sqrt(T * eta_c / 2.0), math.sqrt(T * (1.0 - eta_c))
    st, s1t = math.sqrt(t), math.sqrt(1.0 - t)
    for i, (s, P) in enumerate(INPUTS):
        sign = 1.0 if P == "H" else -1.0                 # H = (+ + -)/sqrt2,  V = (+ - -)/sqrt2
        to_d1, to_d2 = (st, s1t) if s == "A" else (s1t, -st)
        U[i, 0], U[i, 1] = a * to_d1, sign * a * to_d1
        U[i, 2], U[i, 3] = a * to_d2, sign * a * to_d2
        U[i, 4 + i] = c
        U[i, 8 + i] = math.sqrt(1.0 - T)
    return U


@lru_cache(maxsize=None)
def local_basis(k: int) -> Tuple[Tuple[int, int, int, int], ...]:
    """Occupations (nAH, nAV, nBH, nBV) with total <= k."""
    return tuple(o for o in itertools.product(range(k + 1), repeat=4) if sum(o) <= k)


def species_states(U: np.ndarray, inputs: Tuple[int, ...]) -> np.ndarray:
    """rho[m] (16 x d x d): unnormalised local state of one species (photons sharing a temporal mode) given the click mask m
    of the CHS detectors produced by THIS species, summed over everything else that is not observed (loss modes, multiple
    photons in a clicking detector).  Exact bosonic amplitudes: A(n) = sqrt(prod n!) sum_{assignments -> n} prod U."""
    k = len(inputs)
    basis = local_basis(k)
    index = {b: j for j, b in enumerate(basis)}
    amps: Dict[Tuple[int, ...], float] = {}
    rows = U[list(inputs)]
    nz = [np.flatnonzero(np.abs(r) > 0) for r in rows]
    for assign in itertools.product(*nz):
        amp = 1.0
        for r, m in zip(rows, assign):
            amp *= r[m]
        occ = [0] * N_OUT
        for m in assign:
            occ[m] += 1
        key = tuple(occ)
        amps[key] = amps.get(key, 0.0) + amp
    ext: Dict[Tuple[int, ...], np.ndarray] = {}
    for occ, a in amps.items():
        a *= math.sqrt(float(np.prod([math.factorial(n) for n in occ])))
        e = occ[:8]
        v = ext.setdefault(e, np.zeros(len(basis)))
        v[index[occ[8:]]] += a
    rho = np.zeros((16, len(basis), len(basis)))
    for e, v in ext.items():
        m = sum(1 << j for j in range(4) if e[j] > 0)
        rho[m] += np.outer(v, v)
    return rho


# ------------------------------------------------------------------------------------------------ local measurement operators
@lru_cache(maxsize=None)
def _binom(n: int, k: int) -> int:
    return math.comb(n, k)


def rotation_fock(theta: float, N: int) -> np.ndarray:
    """R[K, i]: amplitude of |K photons in theta, N-K in theta_perp> in |i H, N-i V>."""
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    R = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        j = N - i
        norm = math.sqrt(math.factorial(i) * math.factorial(j))
        for k1 in range(i + 1):
            for k2 in range(j + 1):
                K = k1 + k2
                coef = _binom(i, k1) * c ** k1 * (-s) ** (i - k1) * _binom(j, k2) * s ** k2 * c ** (j - k2)
                R[K, i] += coef * math.sqrt(math.factorial(K) * math.factorial(N - K)) / norm
    return R


@lru_cache(maxsize=None)
def _block_index(k: int, side: str):
    """For local_basis(k): index pairs (a, b) coupled by a one-sided operator, with (N, i, j) to read the block entry."""
    basis = local_basis(k)
    rows, cols, Ns, Is, Js = [], [], [], [], []
    for a, ba in enumerate(basis):
        for b, bb in enumerate(basis):
            if side == "A":
                if ba[2:] != bb[2:] or ba[0] + ba[1] != bb[0] + bb[1]:
                    continue
                N, i, j = ba[0] + ba[1], ba[0], bb[0]
            else:
                if ba[:2] != bb[:2] or ba[2] + ba[3] != bb[2] + bb[3]:
                    continue
                N, i, j = ba[2] + ba[3], ba[2], bb[2]
            rows.append(a); cols.append(b); Ns.append(N); Is.append(i); Js.append(j)
    return len(basis), np.array(rows), np.array(cols), np.array(Ns), np.array(Is), np.array(Js)


def no_click_operator(k: int, theta: float, eta_l: float, side: str) -> np.ndarray:
    """(1 - eta_l)^{n_theta} on one side (A: Alice's local modes, B: Bob's), identity on the other, in local_basis(k).
    n_theta counts photons in |theta> = cos(theta/2)|H> + sin(theta/2)|V>; each is detected with probability eta_l."""
    d, rows, cols, Ns, Is, Js = _block_index(k, side)
    blocks = np.zeros((k + 1, k + 1, k + 1))
    for N in range(k + 1):
        R = rotation_fock(theta, N)
        blocks[N, :N + 1, :N + 1] = R.T @ np.diag([(1.0 - eta_l) ** K for K in range(N + 1)]) @ R
    M = np.zeros((d, d))
    M[rows, cols] = blocks[Ns, Is, Js]
    return M


# ------------------------------------------------------------------------------------------------ heralded statistics
_ZETA = np.array([[1.0 if (mp & ~m) == 0 else 0.0 for mp in range(16)] for m in range(16)])   # Z[m, m'] = [m' subset m]
_MOBIUS = np.linalg.inv(_ZETA)


@dataclass
class HeraldedSource:
    """All species states of one setting (L, source) and the configuration weights."""
    params: PhotonicParams
    L: float
    heralds: Tuple[int, ...] = HERALDS_DEFAULT
    species: Dict[Tuple[int, ...], np.ndarray] = field(default_factory=dict, repr=False)
    configs: List[Tuple[float, Tuple[Tuple[int, ...], ...]]] = field(default_factory=list, repr=False)

    @classmethod
    def build(cls, params: PhotonicParams, L: float, heralds: Sequence[int] = HERALDS_DEFAULT) -> "HeraldedSource":
        U = transfer_matrix(params.T, params.eta_c(L), params.t)
        x, p2 = params.x_good, params.p2
        p1 = 1.0 - p2
        configs = []
        for good in itertools.product((True, False), repeat=4):
            w_good = float(np.prod([x if g else 1.0 - x for g in good]))
            if w_good == 0.0:
                continue
            G = tuple(i for i in range(4) if good[i])
            base = ([G] if G else []) + [(i,) for i in range(4) if not good[i]]
            extras = [(None, p1 ** 4)] + ([(i, p2 * p1 ** 3) for i in range(4)] if p2 > 0 else [])
            for e, w_e in extras:
                sp = tuple(base + ([(e,)] if e is not None else []))
                configs.append((w_good * w_e, sp))
        species = {}
        for _, sp in configs:
            for s in sp:
                if s not in species:
                    species[s] = species_states(U, s)
        return cls(params, L, tuple(heralds), species, configs)

    # ---------------------------------------------------------------------------------- probabilities for given settings
    def joint(self, alice: Sequence[float], bob: Sequence[float]) -> Tuple[float, np.ndarray]:
        """(P_H, P[x, y, a, b]) for Alice angles ``alice`` and Bob angles ``bob`` (outcome 0 = click, 1 = no click)."""
        nx, ny = len(alice), len(bob)
        eta = self.params.eta_l
        # per species and mask: Tr(rho), Tr(rho NA_x), Tr(rho NB_y), Tr(rho NA_x NB_y)
        tr = {}
        for s, rho in self.species.items():
            k = len(s)
            NA = [no_click_operator(k, th, eta, "A") for th in alice]
            NB = [no_click_operator(k, ph, eta, "B") for ph in bob]
            t_i = np.einsum("mii->m", rho)
            t_a = np.stack([np.einsum("mij,ji->m", rho, A) for A in NA], axis=1)                 # (16, nx)
            t_b = np.stack([np.einsum("mij,ji->m", rho, B) for B in NB], axis=1)                 # (16, ny)
            t_ab = np.stack([np.stack([np.einsum("mij,ji->m", rho, A @ B) for B in NB], axis=1) for A in NA], axis=1)
            tr[s] = (t_i, t_a, t_b, t_ab)
        tot_i, tot_a, tot_b, tot_ab = 0.0, np.zeros(nx), np.zeros(ny), np.zeros((nx, ny))
        hmask = np.zeros(16)
        hmask[list(self.heralds)] = 1.0
        for w, sp in self.configs:
            zi = np.ones(16)
            za = np.ones((16, nx))
            zb = np.ones((16, ny))
            zab = np.ones((16, nx, ny))
            for s in sp:
                t_i, t_a, t_b, t_ab = tr[s]
                zi = zi * (_ZETA @ t_i)
                za = za * (_ZETA @ t_a)
                zb = zb * (_ZETA @ t_b)
                zab = zab * np.einsum("mn,nxy->mxy", _ZETA, t_ab)
            tot_i += w * hmask @ (_MOBIUS @ zi)
            tot_a += w * hmask @ (_MOBIUS @ za)
            tot_b += w * hmask @ (_MOBIUS @ zb)
            tot_ab += w * np.einsum("m,mn,nxy->xy", hmask, _MOBIUS, zab)
        PH = tot_i
        P = np.zeros((nx, ny, 2, 2))
        P[:, :, 1, 1] = tot_ab / PH
        P[:, :, 1, 0] = (tot_a[:, None] - tot_ab) / PH
        P[:, :, 0, 1] = (tot_b[None, :] - tot_ab) / PH
        P[:, :, 0, 0] = 1.0 - P[:, :, 1, 1] - P[:, :, 1, 0] - P[:, :, 0, 1]
        return float(PH), P

    def herald_probability(self) -> float:
        return self.joint([0.0], [0.0])[0]


# ------------------------------------------------------------------------------------------------ honest statistics
def stats_from_joint(P: np.ndarray, flips: Sequence[int]) -> Tuple[float, float, Tuple[float, float]]:
    """omega (CHSH win prob., Bob test settings y = 2, 3) and the key-round error (y = 0, 1 matched to x = 0, 1).
    ``flips[y]`` relabels Bob's outcome for setting y (honest-device choice)."""
    def p_eq(x, y, target):           # P(a xor b' = target)
        f = flips[y]
        return sum(P[x, y, a, b] for a in (0, 1) for b in (0, 1) if (a ^ b ^ f) == target)
    omega = 0.25 * sum(p_eq(x, y, x & (y - 2)) for x in (0, 1) for y in (2, 3))
    q0 = 1.0 - p_eq(0, 0, 0)
    q1 = 1.0 - p_eq(1, 1, 0)
    return omega, 0.5 * (q0 + q1), (q0, q1)


def asymptotic_rate(omega: float, qber: float, p_grid=np.linspace(0.0, 0.45, 46)) -> Tuple[float, float]:
    """max_p 1/2 [r_p(omega) - h2(p + (1-2p) Q)]  (gamma -> 0 limit of the paper's protocol); returns (rate, p)."""
    best = (-1.0, 0.0)
    if omega <= 0.75:                  # r_0 depends on S^2; a violation with S < -2 must be relabelled, not used as is
        return best
    for p in p_grid:
        r = 0.5 * (rp(omega, p) - h2(p + (1.0 - 2.0 * p) * qber))
        if r > best[0]:
            best = (r, float(p))
    return best


@dataclass
class Operating:
    L: float
    PH: float
    omega: float
    qber: float
    q_bases: Tuple[float, float]
    angles_alice: Tuple[float, float]
    angles_bob: Tuple[float, float, float, float]
    flips: Tuple[int, int, int, int]
    r_inf: float
    p_inf: float

    @property
    def honest(self) -> Honest:
        return Honest(omega=self.omega, qber=self.qber)

    @property
    def chsh(self) -> float:
        return 8.0 * self.omega - 4.0


def _best_key(P: np.ndarray, z: int, angle: float):
    """Bob's key setting for basis z: his angle aligned (column z+4 of P) or orthogonal (column z), and his relabelling."""
    cands = []
    for col, ang in ((z, angle + math.pi), (z + 4, angle)):
        for f in (0, 1):
            q = 1.0 - sum(P[z, col, a, b] for a in (0, 1) for b in (0, 1) if (a ^ b ^ f) == 0)
            cands.append((q, ang, f))
    return min(cands)


def _evaluate(src: HeraldedSource, v, flips_test):
    a0, a1, b2, b3 = v
    PH, P = src.joint([a0, a1], [a0 + math.pi, a1 + math.pi, b2, b3, a0, a1])
    keys = [_best_key(P, z, (a0, a1)[z]) for z in (0, 1)]
    om, _, _ = stats_from_joint(P[:, :4], (0, 0) + tuple(flips_test))
    q = 0.5 * (keys[0][0] + keys[1][0])
    return PH, om, q, keys


def optimise_angles(src: HeraldedSource, x0: Sequence[float] | None = None, flips0=None, maxiter: int = 400) -> Operating:
    """Measurement angles (and Bob's relabelling bits) maximising the asymptotic key rate - "the measurement angles are
    optimized to maximize the signature rate" (Fig. 7 caption).  Bob's key angle (aligned / orthogonal to Alice's) and
    relabelling are chosen per basis; the test-setting relabelling is searched over its four values."""
    # no-click binning breaks the theta <-> theta + pi symmetry, so both orientations of Bob's test settings are tried
    starts = [np.asarray(x0, float)] if x0 is not None else [
        np.array([0.0, math.pi / 2, math.pi / 4, -math.pi / 4]),
        np.array([0.0, math.pi / 2, math.pi / 4 + math.pi, -math.pi / 4 + math.pi])]
    flip_set = [tuple(flips0)] if flips0 is not None else [(0, 0), (1, 1), (0, 1), (1, 0)]
    best = None
    for ft in flip_set:
        def neg(v):
            _, om, q, _ = _evaluate(src, v, ft)
            return -asymptotic_rate(om, q)[0]
        for s0 in starts:
            res = minimize(neg, s0, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7, "maxiter": maxiter})
            if best is None or res.fun < best[0]:
                best = (res.fun, res.x, ft)
    _, v, ft = best
    PH, om, q, keys = _evaluate(src, v, ft)
    r, p = asymptotic_rate(om, q)
    a0, a1, b2, b3 = (float(u) for u in v)
    return Operating(src.L, PH, om, q, (keys[0][0], keys[1][0]), (a0, a1), (keys[0][1], keys[1][1], b2, b3),
                     (keys[0][2], keys[1][2]) + tuple(ft), r, p)


def operating_curve(params: PhotonicParams, Ls: Sequence[float], heralds: Sequence[int] = HERALDS_DEFAULT,
                    maxiter: int = 300) -> List[Operating]:
    """Operating points along L, each angle optimisation warm-started from the previous distance."""
    out, x0, f0 = [], None, None
    for L in Ls:
        src = HeraldedSource.build(params, float(L), heralds)
        op = optimise_angles(src, x0=x0, flips0=f0, maxiter=maxiter if x0 is None else max(60, maxiter // 4))
        x0 = [op.angles_alice[0], op.angles_alice[1], op.angles_bob[2], op.angles_bob[3]]
        f0 = op.flips[2:]
        out.append(op)
    return out


# ------------------------------------------------------------------------------------------------ Fig. 7: rate versus distance
@dataclass
class OperatingTable:
    """Operating points on an L grid; quality (omega, Q) is interpolated linearly, log P_H linearly (P_H ~ eta_C^2)."""
    L: np.ndarray
    PH: np.ndarray
    omega: np.ndarray
    qber: np.ndarray

    @classmethod
    def from_points(cls, ops: Sequence[Operating]) -> "OperatingTable":
        return cls(np.array([o.L for o in ops]), np.array([o.PH for o in ops]),
                   np.array([o.omega for o in ops]), np.array([o.qber for o in ops]))

    def at(self, L: float) -> Tuple[float, Honest]:
        ph = float(np.exp(np.interp(L, self.L, np.log(self.PH))))
        return ph, Honest(omega=float(np.interp(L, self.L, self.omega)), qber=float(np.interp(L, self.L, self.qber)))

    def to_json(self) -> dict:
        return {"L": self.L.tolist(), "PH": self.PH.tolist(), "omega": self.omega.tolist(), "qber": self.qber.tolist()}

    @classmethod
    def from_json(cls, d: dict) -> "OperatingTable":
        return cls(*(np.asarray(d[k], float) for k in ("L", "PH", "omega", "qber")))


def key_at_distance(table: OperatingTable, L: float, N: float, *, preset=None, p="opt", x0=None):
    """Finite-size key (Eq. 9, the paper's protocol) from n_H = P_H(L) N heralded rounds (Eq. D3)."""
    from diqds.part2_distribution.optimize import optimize_key_length
    from diqds.part2_distribution.presets import PAPER_FIT
    preset = preset or PAPER_FIT
    ph, hon = table.at(L)
    n_h = ph * N
    o = optimize_key_length(max(int(n_h), 1000), 1.0, p=p, beta=preset.beta, pe_mode=preset.pe_mode,
                            ec_model=preset.ec_model, honest=hon, protocol="random_key", x0=x0)
    return n_h, o


def max_distance(table: OperatingTable, N: float, *, preset=None, p="opt", lo: float = 0.0, hi: float | None = None) -> float:
    """Largest L with a positive key (bisection on the sign of ell; the key decreases monotonically with L)."""
    hi = float(table.L[-1]) if hi is None else hi
    f = lambda L: key_at_distance(table, L, N, preset=preset, p=p)[1].ell
    if f(lo) <= 0:
        return float("nan")
    if f(hi) > 0:
        return hi
    return brentq(f, lo, hi, xtol=1e-3)


def realistic_source(model: str = "c13", kappa: float | None = None, v_hom: float = 0.975, g2: float = 0.01) -> PhotonicParams:
    """The realistic QD source of Fig. 7 (V_HOM = 0.975, g2 = 0.01) under one of the two readings (module docstring)."""
    if model == "explicit":
        return PhotonicParams(v_hom=v_hom, g2=g2)
    if model == "c13":
        k = REALISTIC_KAPPA if kappa is None else kappa
        return PhotonicParams(v_hom=v_hom - k * g2, g2=0.0)
    raise ValueError(model)


def curve_vs_distance(label: str, table: OperatingTable, N: float, Ls: Sequence[float], *, preset=None, p="opt",
                      dense: int = 14):
    """Signature rate versus distance (Fig. 7) with the cut-off located and resolved."""
    from .sweeps import Curve, signature_rate
    Ls = np.asarray(sorted(Ls), float)
    Lmax = max_distance(table, N, preset=preset, p=p, lo=float(Ls[0]), hi=float(Ls[-1]))
    xs = list(Ls)
    if math.isfinite(Lmax) and Lmax < Ls[-1]:
        xs = sorted(set([x for x in xs if x < Lmax] + list(Lmax - (Lmax - Ls[0]) * np.geomspace(1e-5, 0.08, dense))
                        + [x for x in xs if x > Lmax]))
    optima, rates, x0 = [], [], None
    for L in xs:
        n_h, o = key_at_distance(table, L, N, preset=preset, p=p, x0=x0)
        if o.ell > 0:
            x0 = o.x
        optima.append(o)
        rates.append(signature_rate(o.ell))
    return Curve(label, np.asarray(xs), np.asarray(rates), Lmax, {"N": N}, optima)
