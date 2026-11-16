"""Tests for Part 1 (messaging stage) of the reproduction of Fan et al., PRA 114, 032617 (2026).

Run:  python -m pytest -q QIQDS_parts1-2/tests/test_part1_messaging.py
uv run pytest -v QIQDS_parts1-2/tests/
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from diqds.part1_messaging import (Bob, Charlie, DegenerateKeyError, PartyKeys, ProtocolOrderError, TABLE_I,
                                   budget, encode_document, decode_document, eps_forgery, f, ideal_key_triples,
                                   int_to_bits, key_bits_per_signature, make_bob_tamper, max_guesses,
                                   minimal_signature_length, multiple_of_polynomial, poly_from_key,
                                   product_of_polynomials, random_bits, random_forgery, run_messaging_stage,
                                   satisfies_correlations, signature_rate, split_pairwise_keys)
from diqds.part1_messaging.bits import bits_to_int
from diqds.part1_messaging.gf2poly import (clmul, count_irreducibles, default_modulus, deg, irreducibles,
                                           is_irreducible, pmod, poly_str)
from diqds.part1_messaging.kat import build_kat, check_kat
from diqds.part1_messaging.lfsr_toeplitz import (companion_matrix, hash_horner, hash_matrix, lfsr_sequence,
                                                 message_polynomial, polynomial_message, toeplitz_matrix,
                                                 toeplitz_matrix_from_powers)
from diqds.part1_messaging.polygen import (berlekamp_massey_gf2, characteristic_polynomial, standin_poly_from_key)
from diqds.part1_messaging import security

RNG = np.random.default_rng(12345)


def _p_bits(P, n):
    return int_to_bits(P & ((1 << n) - 1), n)


# ---------------------------------------------------------------- GF(2)[x]
def _brute_irreducible(p):
    n = deg(p)
    return n >= 1 and all(pmod(p, c) != 0 for d in range(1, n // 2 + 1) for c in range(1 << d, 1 << (d + 1)))


def test_rabin_test_matches_trial_division():
    for n in range(2, 10):
        for c in range(1 << n):
            p = (1 << n) | c
            assert is_irreducible(p) == _brute_irreducible(p), bin(p)


def test_irreducible_count_formula():
    for n in range(2, 13):
        assert count_irreducibles(n) == sum(1 for _ in irreducibles(n))
    # the paper's signature length: lower bound 2^(n-1)/n used in the forging bound
    for n in range(2, 60):
        assert count_irreducibles(n) >= 2 ** (n - 1) / n


def test_default_modulus_n46():
    assert poly_str(default_modulus(46)) == "x^46 + x + 1"
    assert is_irreducible(default_modulus(46))


def test_clmul_pmod_identities():
    for _ in range(50):
        a, b, m = (int.from_bytes(RNG.bytes(8), "little") | 1 for _ in range(3))
        assert pmod(clmul(a, m), m) == 0
        assert clmul(a, b) == clmul(b, a)


# ---------------------------------------------------------------- Berlekamp-Massey / f(Z)
def _extend(c, seed, total):
    seq = list(seed)
    while len(seq) < total:
        seq.append(sum(c[j] & seq[-j] for j in range(1, len(c))) & 1)
    return seq


def test_berlekamp_massey_recovers_recurrence():
    # irreducible characteristic polynomial + non-zero seed  =>  the minimal recurrence is exactly the generating one
    for L in (3, 7, 12):
        for P in list(irreducibles(L))[:6]:
            c = [1] + [(P >> (L - j)) & 1 for j in range(1, L + 1)]
            seed = [int(b) for b in RNG.integers(0, 2, size=L)]
            seed[0] = 1
            assert berlekamp_massey_gf2(_extend(c, seed, 4 * L)) == (c, L)


def test_berlekamp_massey_output_regenerates_sequence():
    # arbitrary (possibly reducible) recurrences: BM may find a SHORTER one, but it must reproduce the whole sequence
    for _ in range(100):
        L = int(RNG.integers(2, 13))
        c = [1] + [int(b) for b in RNG.integers(0, 2, size=L)]
        seed = [int(b) for b in RNG.integers(0, 2, size=L)]
        seq = _extend(c, seed, 4 * L)
        cf, Lf = berlekamp_massey_gf2(seq)
        assert Lf <= L
        assert all(seq[k] == (sum(cf[j] & seq[k - j] for j in range(1, Lf + 1)) & 1) for k in range(Lf, len(seq)))


def test_berlekamp_massey_degenerate_sequences():
    assert berlekamp_massey_gf2([0, 0, 0, 0]) == ([1], 0)
    c, L = berlekamp_massey_gf2([1, 0, 0, 0, 0, 0])
    assert characteristic_polynomial(c, L) == 0b10                       # minimal polynomial of the element 0 is x


def test_f_is_exactly_uniform_on_irreducibles_small_n():
    """Every irreducible polynomial of degree n has exactly n preimages (its n roots): f is exactly uniform."""
    for n in (5, 8, 10, 11):
        cnt, degenerate = Counter(), 0
        for z in range(1 << n):
            try:
                cnt[poly_from_key(int_to_bits(z, n))] += 1
            except DegenerateKeyError:
                degenerate += 1
        irr = set(irreducibles(n))
        assert set(cnt) == irr
        assert set(cnt.values()) == {n}
        assert degenerate == (1 << n) - n * count_irreducibles(n)


def test_f_uniform_for_any_public_modulus():
    n = 8
    others = [p for p in irreducibles(n)][:5]
    for mod in others:
        cnt = Counter(poly_from_key(int_to_bits(z, n), mod) for z in range(1 << n)
                      if not _degenerate(z, n, mod))
        assert set(cnt.values()) == {n} and len(cnt) == count_irreducibles(n)


def _degenerate(z, n, mod):
    try:
        poly_from_key(int_to_bits(z, n), mod)
        return False
    except DegenerateKeyError:
        return True


def test_f_paper_size_outputs_irreducible_and_deterministic():
    for _ in range(25):
        z = random_bits(46, RNG)
        try:
            p1 = f(z)
        except DegenerateKeyError:
            continue
        assert len(p1) == 46
        assert np.array_equal(p1, f(z.copy()))
        P = (1 << 46) | bits_to_int(p1)
        assert deg(P) == 46 and is_irreducible(P)


def test_standin_is_irreducible_but_not_uniform():
    n = 10
    cnt = Counter(standin_poly_from_key(int_to_bits(z, n)) for z in range(1 << n))
    assert all(is_irreducible(p) for p in cnt)
    assert len(set(cnt.values())) > 1                                     # preimage counts differ -> non-uniform


# ---------------------------------------------------------------- Appendix A: LFSR-Toeplitz hash
def _rand_irreducible_bits(n):
    while True:
        z = random_bits(n, RNG)
        try:
            return f(z)
        except DegenerateKeyError:
            continue


def test_toeplitz_structure_and_shape():
    n, m = 13, 60
    H = toeplitz_matrix(_rand_irreducible_bits(n), random_bits(n, RNG), m)
    assert H.shape == (n, m)
    assert np.array_equal(H[1:, 1:], H[:-1, :-1])                         # constant along diagonals


def test_columns_are_powers_of_companion_matrix():
    for n, m in [(4, 9), (8, 40), (13, 70), (46, 200)]:
        p, s = _rand_irreducible_bits(n), random_bits(n, RNG)
        assert np.array_equal(toeplitz_matrix(p, s, m), toeplitz_matrix_from_powers(p, s, m))


def test_lfsr_recurrence_eq_A6():
    n = 9
    p, s = _rand_irreducible_bits(n), random_bits(n, RNG)
    S = lfsr_sequence(p, s, n + 1)
    assert S[n] == (int(np.dot(p.astype(int), s.astype(int))) & 1)       # S_{n+1} = p . s
    W = companion_matrix(p).astype(int)
    s1 = (W @ s.astype(int)) % 2
    assert np.array_equal(s1[1:], s[:-1]) and s1[0] == S[n]               # shift down, prepend S_{n+1}


def test_hash_forms_agree_and_match_eq_A10():
    for n, m in [(5, 20), (8, 64), (13, 100), (46, 1000)]:
        p, s, M = _rand_irreducible_bits(n), random_bits(n, RNG), random_bits(m, RNG)
        h1, h2 = hash_matrix(p, s, M), hash_horner(p, s, M)
        W = companion_matrix(p).astype(int)
        MW = np.zeros((n, n), dtype=int)                                  # M_W(W) by Horner with explicit matrices
        for bit in M[::-1]:
            MW = (MW @ W + int(bit) * np.eye(n, dtype=int)) % 2
        h3 = (MW @ s.astype(int)) % 2
        assert np.array_equal(h1, h2) and np.array_equal(h1, h3)


def test_proposition1_multiples_of_p_hash_to_zero():
    """Eq. (A12): p(x) | M(x)  =>  h_{p,s}(M) = 0 for every initial state s."""
    for n, m in [(8, 64), (13, 100), (46, 1000)]:
        z = _rand_irreducible_bits(n)
        P = (1 << n) | bits_to_int(z)
        for _ in range(5):
            q = int.from_bytes(RNG.bytes((m - n + 7) // 8), "little") & ((1 << (m - n)) - 1) | 1
            M = polynomial_message(clmul(P, q), m)
            assert pmod(message_polynomial(M), P) == 0
            for _ in range(3):
                assert not hash_matrix(z, random_bits(n, RNG), M).any()


def test_non_multiples_give_a_bijection_in_s():
    """If p does not divide M(x), M_W(W) is invertible, so s -> h_{p,s}(M) is a bijection: each digest has probability 2^-n."""
    n, m = 8, 40
    p = _rand_irreducible_bits(n)
    P = (1 << n) | bits_to_int(p)
    while True:
        M = random_bits(m, RNG)
        if pmod(message_polynomial(M), P) != 0:
            break
    digests = {tuple(hash_horner(p, int_to_bits(s, n), M)) for s in range(1 << n)}
    assert len(digests) == 1 << n
    assert hash_horner(p, np.zeros(n, dtype=np.uint8), M).sum() == 0


def test_hash_is_linear():
    n, m = 13, 80
    p, s = _rand_irreducible_bits(n), random_bits(n, RNG)
    M1, M2 = random_bits(m, RNG), random_bits(m, RNG)
    assert np.array_equal(hash_matrix(p, s, M1 ^ M2), hash_matrix(p, s, M1) ^ hash_matrix(p, s, M2))


# ---------------------------------------------------------------- keys
def test_key_triples_satisfy_constraints():
    for _ in range(20):
        A, B, C = ideal_key_triples(46, RNG)
        assert satisfies_correlations(A, B, C)
        assert not satisfies_correlations(A, B, PartyKeys(C.X, C.Y ^ 1, C.Z))


def test_split_pairwise_keys_counts_and_constraints():
    ns, n_sig = 46, 5
    L = key_bits_per_signature(ns) * n_sig + 17
    k_ab, k_ac = random_bits(L, RNG), random_bits(L, RNG)
    triples = split_pairwise_keys(k_ab, k_ab.copy(), k_ac, k_ac.copy(), ns)
    assert n_sig - 1 <= len(triples) <= n_sig                              # at most a degenerate-Z block skipped
    assert all(satisfies_correlations(*t) for t in triples)


# ---------------------------------------------------------------- protocol (Sec. II.B)
def test_honest_run_accepts_both_hash_methods():
    for method in ("matrix", "horner"):
        for _ in range(10):
            A, B, C = ideal_key_triples(46, RNG)
            doc = random_bits(1000, RNG)
            out = run_messaging_stage(doc, A, B, C, hash_method=method)
            assert out.bob_accepts and out.charlie_accepts and out.valid


def test_document_encoding_roundtrip_and_signing():
    A, B, C = ideal_key_triples(46, RNG)
    doc = encode_document(b"pay Bob 100", 1000)
    assert decode_document(doc) == b"pay Bob 100"
    assert run_messaging_stage(doc, A, B, C).valid


def test_ordering_rule_is_enforced():
    A, B, C = ideal_key_triples(46, RNG)
    try:
        Charlie(C).reveal()
        assert False, "Charlie revealed her keys before recording Bob's packet"
    except ProtocolOrderError:
        pass
    ch = Charlie(C)
    try:
        ch.certify(True)
        assert False
    except ProtocolOrderError:
        pass
    try:
        Bob(B).verify({"X": C.X, "Y": C.Y, "Z": C.Z})
        assert False
    except ProtocolOrderError:
        pass


def test_tampering_with_doc_sig_or_Yb_is_rejected():
    A, B, C = ideal_key_triples(46, RNG)
    doc = random_bits(1000, RNG)

    def flip_doc(sig, d, y):
        d[int(RNG.integers(1000))] ^= 1
        return sig, d, y

    def flip_sig(sig, d, y):
        sig[int(RNG.integers(46))] ^= 1
        return sig, d, y

    def flip_y(sig, d, y):
        y[int(RNG.integers(46))] ^= 1
        return sig, d, y

    for tamper in (flip_doc, flip_sig, flip_y):
        for _ in range(20):
            assert not run_messaging_stage(doc, A, B, C, bob_tamper=tamper).charlie_accepts


def test_Yb_modification_is_equivalent_to_signature_modification():
    """Sec. III.A: any change of the forwarded Y_b can be absorbed into a change of the signature."""
    A, B, C = ideal_key_triples(46, RNG)
    doc = random_bits(1000, RNG)
    delta = random_bits(46, RNG)
    o1 = run_messaging_stage(doc, A, B, C, bob_tamper=lambda s, d, y: (s ^ delta, d, y))
    o2 = run_messaging_stage(doc, A, B, C, bob_tamper=lambda s, d, y: (s, d, y ^ delta))
    assert np.array_equal(o1.details["charlie"]["expected"], o2.details["charlie"]["expected"])
    assert o1.charlie_accepts == o2.charlie_accepts


def test_repudiation_bob_and_charlie_always_agree():
    """Sec. III.B: Bob and Charlie hold identical new keys, so they take the same decision on ANY {Sig, Doc}."""
    A, B, C = ideal_key_triples(46, RNG)
    doc = random_bits(1000, RNG)
    from diqds.part1_messaging import sign
    genuine, _ = sign(doc, A)
    candidates = [genuine.sig] + [random_bits(46, RNG) for _ in range(60)]
    for sig in candidates:
        bob, ch = Bob(B), Charlie(C)
        bob.receive(type(genuine)(sig, doc))
        ch.record(bob.forward_packet())
        reveal = ch.reveal()
        ok_b, info_b = bob.verify(reveal)
        ok_c, info_c = ch.certify(True)
        assert ok_b == ok_c
        assert np.array_equal(info_b["expected"], info_c["expected"])
        assert np.array_equal(info_b["actual"], info_c["actual"])
    assert candidates[0] is genuine.sig and run_messaging_stage(doc, A, B, C).valid


def test_robustness_key_mismatch_causes_abort():
    """Sec. III.C: an honest run fails only if the keys disagree (probability eps_corr + classical errors)."""
    doc = random_bits(1000, RNG)
    for which in "XYZ":
        for _ in range(10):
            A, B, C = ideal_key_triples(46, RNG)
            arr = getattr(C, which).copy()
            arr[int(RNG.integers(46))] ^= 1
            bad = PartyKeys(**{**dict(X=C.X, Y=C.Y, Z=C.Z), which: arr})
            assert not run_messaging_stage(doc, A, B, bad).valid


def test_random_forgery_is_rejected_at_paper_size():
    A, B, C = ideal_key_triples(46, RNG)
    doc = random_bits(1000, RNG)
    for _ in range(100):
        dM, t = random_forgery(46, 1000, RNG)
        out = run_messaging_stage(doc, A, B, C, bob_tamper=make_bob_tamper(dM, t), hash_method="horner")
        assert not out.charlie_accepts


def test_attacker_who_knows_pa_forges_perfectly_but_wrong_guess_fails():
    """Proposition 1: knowing p_a (i.e. Z_a) is enough to forge for certain; the ordering rule withholds it."""
    A, B, C = ideal_key_triples(46, RNG)
    doc = random_bits(1000, RNG)
    P_true = poly_from_key(A.Z)
    for _ in range(10):
        dM, t = multiple_of_polynomial(P_true, 1000, RNG)
        assert dM.any()
        out = run_messaging_stage(doc, A, B, C, bob_tamper=make_bob_tamper(dM, t))
        assert out.charlie_accepts and out.bob_accepts is None            # forged document accepted
        assert not np.array_equal(out.details["signed"].doc ^ dM, out.details["signed"].doc)
    _, B2, C2 = ideal_key_triples(46, RNG)
    A2, _, _ = ideal_key_triples(46, RNG)
    P_wrong = poly_from_key(A2.Z)
    dM, t = multiple_of_polynomial(P_wrong, 1000, RNG)
    assert not run_messaging_stage(doc, A, B, C, bob_tamper=make_bob_tamper(dM, t)).charlie_accepts


def test_optimal_forgery_probability_matches_theory_small_n():
    """Best attack of Sec. III.A at ns = 8, nm = 32: dM = product of k distinct irreducible polynomials, t = 0.
    P_succ = q + (1-q) 2^-ns with q = k / |Irr_ns|, and it must respect the paper's bound nm * 2^(-ns+1)."""
    ns, nm, trials = 8, 32, 6000
    k = max_guesses(ns, nm)
    polys = list(irreducibles(ns))[:k]
    q = k / count_irreducibles(ns)
    theory = q + (1 - q) * 2.0 ** -ns
    bound = eps_forgery(ns, nm, 0.0)
    dM, t = product_of_polynomials(polys, nm, ns)
    hits = 0
    for _ in range(trials):
        A, B, C = ideal_key_triples(ns, RNG)
        doc = random_bits(nm, RNG)
        hits += run_messaging_stage(doc, A, B, C, bob_tamper=make_bob_tamper(dM, t), hash_method="horner").charlie_accepts
    obs = hits / trials
    sigma = math.sqrt(theory * (1 - theory) / trials)
    assert abs(obs - theory) < 5 * sigma, (obs, theory, sigma)
    assert theory <= bound


# ---------------------------------------------------------------- security numbers (Eqs. 4-8, Table I)
def test_table1_signature_length_and_budget():
    b = budget(TABLE_I)
    assert b.ns == 46
    assert math.isclose(b.eps_for, 1000 * 2.0 ** -45 + 5e-11)
    assert math.isclose(b.eps_for, 7.842e-11, rel_tol=1e-3)
    assert math.isclose(b.eps_rep, 2e-11) and math.isclose(b.eps_rob, 2 * 2.0 ** -61 + 2e-11)
    assert b.eps == b.eps_for <= TABLE_I.eps
    assert b.key_bits_per_signature == 138
    assert eps_forgery(45, 1000, 5e-11) > 1e-10                           # ns = 45 would violate the target


def test_signature_length_depends_on_assumed_eps_sec():
    assert minimal_signature_length(1000, 1e-10, 5e-11, 2.0 ** -61, 1e-11) == 46
    assert minimal_signature_length(1000, 1e-10, 0.0, 2.0 ** -61, 1e-11) == 45


def test_signature_length_vs_message_length():
    f_ = lambda nm: minimal_signature_length(nm, 1e-10, 5e-11, 2.0 ** -61, 1e-11)
    assert (f_(10 ** 2), f_(10 ** 3), f_(10 ** 6)) == (42, 46, 56)
    r = signature_rate(1e9, 56) / signature_rate(1e9, 42)
    assert math.isclose(1 - r, 0.25, abs_tol=1e-9)                        # ~25 % change of r_sig, 1e2 -> 1e6 bits


def test_signature_rate_eq4():
    assert math.isclose(signature_rate(138, 46), 1.0)
    assert math.isclose(signature_rate(2.4e8, 46), 2.4e8 / 138)
    assert signature_rate(300, 46, floor=True) == 2.0
    assert math.isclose(security.eps_total(46, 1000, 5e-11, 2.0 ** -61, 1e-11), budget(TABLE_I).eps)


# ---------------------------------------------------------------- known-answer vectors
def test_kat_vectors_are_reproduced_from_stored_inputs():
    """The vectors are checked by recomputing outputs from the STORED inputs, so this is independent of the RNG stream
    (and of NumPy version); the same JSON can be used to check any other implementation, e.g. diqds.hashing."""
    k = build_kat()
    assert check_kat(k) == []
    assert k["security"]["ns"] == 46 and k["protocol"]["charlie_accepts"] and k["protocol"]["bob_accepts"]
    shipped = Path(__file__).resolve().parents[1] / "results" / "part1_kat.json"
    if shipped.exists():
        assert check_kat(json.loads(shipped.read_text())) == []


def test_kat_detects_a_changed_convention():
    k = build_kat()
    bad = json.loads(json.dumps(k))
    d = bad["toy_hash"]["digest"]
    bad["toy_hash"]["digest"] = ("1" if d[0] == "0" else "0") + d[1:]
    assert any("toy_hash" in e for e in check_kat(bad))
