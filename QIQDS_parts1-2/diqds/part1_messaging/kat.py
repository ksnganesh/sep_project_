"""Known-answer test (KAT) vectors for the messaging stage.

The vectors are plain strings of '0'/'1' so that ANY implementation (including the pre-existing
``diqds.hashing`` / ``diqds.messaging`` modules) can be checked bit for bit, whatever its API.

Conventions (see ``bits.py``): all strings are MSB-first as written in the paper:
    p = (p_{n-1} ... p_0)    s = (S_n ... S_1)    M = (M_0 ... M_{m-1})    digest = H M  (n bits)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .bits import bits_to_str, encode_document, str_to_bits
from .gf2poly import default_modulus, poly_str
from .keys import PartyKeys, ideal_key_triples
from .lfsr_toeplitz import hash_horner, hash_matrix, toeplitz_matrix
from .polygen import f
from .protocol import run_messaging_stage
from .security import TABLE_I, budget


def build_kat() -> dict:
    rng = np.random.default_rng(20260922)
    kat: dict = {
        "about": "Known-answer vectors for the messaging stage of Fan et al., PRA 114, 032617. "
                 "Strings are MSB-first: p=(p_{n-1}..p_0), s=(S_n..S_1), M=(M_0..M_{m-1}).",
    }

    # 1. hand-checkable toy example: p(x) = x^4 + x + 1
    p = str_to_bits("0011")
    s = str_to_bits("1000")
    M = str_to_bits("10110100")
    H = toeplitz_matrix(p, s, len(M))
    kat["toy_hash"] = {
        "p": bits_to_str(p), "s": bits_to_str(s), "M": bits_to_str(M),
        "H_rows": [bits_to_str(r) for r in H],
        "digest": bits_to_str(hash_matrix(p, s, M)),
    }

    # 2. polynomial generator f(Z)  (Shoup / Berlekamp-Massey, Ref. [32] App. B.1)
    kat["f"] = {"modulus_n46": poly_str(default_modulus(46)), "vectors": []}
    for n in (8, 46):
        for _ in range(3):
            while True:
                z = rng.integers(0, 2, size=n, dtype=np.uint8)
                try:
                    kat["f"]["vectors"].append({"n": n, "Z": bits_to_str(z), "p_a": bits_to_str(f(z))})
                    break
                except ValueError:
                    continue

    # 3. hash at paper size
    ns, nm = 46, 1000
    vec = []
    for _ in range(2):
        z = rng.integers(0, 2, size=ns, dtype=np.uint8)
        pa = f(z)
        s = rng.integers(0, 2, size=ns, dtype=np.uint8)
        M = rng.integers(0, 2, size=nm, dtype=np.uint8)
        d1, d2 = hash_matrix(pa, s, M), hash_horner(pa, s, M)
        assert np.array_equal(d1, d2)
        vec.append({"p": bits_to_str(pa), "s": bits_to_str(s), "M": bits_to_str(M), "digest": bits_to_str(d1)})
    kat["hash_paper_size"] = vec

    # 4. one full run of Fig. 2
    A, B, C = ideal_key_triples(ns, rng)
    doc = encode_document(b"KAT document: pay Bob 100 units", nm)
    out = run_messaging_stage(doc, A, B, C)
    kat["protocol"] = {
        "ns": ns, "nm": nm,
        "alice": {k: bits_to_str(getattr(A, k)) for k in "XYZ"},
        "bob": {k: bits_to_str(getattr(B, k)) for k in "XYZ"},
        "charlie": {k: bits_to_str(getattr(C, k)) for k in "XYZ"},
        "doc": bits_to_str(doc),
        "p_a": bits_to_str(out.details["alice"]["p_a"]),
        "digest": bits_to_str(out.details["alice"]["digest"]),
        "sig": bits_to_str(out.details["signed"].sig),
        "bob_accepts": bool(out.bob_accepts), "charlie_accepts": bool(out.charlie_accepts),
    }

    # 5. security bookkeeping (Table I)
    b = budget(TABLE_I)
    kat["security"] = {"ns": b.ns, "eps_for": b.eps_for, "eps_rep": b.eps_rep, "eps_rob": b.eps_rob,
                       "eps": b.eps, "key_bits_per_signature": b.key_bits_per_signature}
    return kat


def write_kat(path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_kat(), indent=1))
    return path


def check_kat(kat: dict) -> list:
    """Recompute every output of a KAT dictionary from its STORED inputs with this package.

    Returns a list of mismatch descriptions (empty = all vectors reproduced).  RNG-independent, so it is
    valid for a file produced on any machine; the same vectors can be fed to any other implementation."""
    errs = []

    t = kat["toy_hash"]
    p, s, M = (str_to_bits(t[k]) for k in ("p", "s", "M"))
    if [bits_to_str(r) for r in toeplitz_matrix(p, s, len(M))] != t["H_rows"]:
        errs.append("toy_hash: Toeplitz matrix rows differ")
    for name, fn in (("matrix", hash_matrix), ("horner", hash_horner)):
        if bits_to_str(fn(p, s, M)) != t["digest"]:
            errs.append(f"toy_hash: digest differs ({name})")

    if poly_str(default_modulus(46)) != kat["f"]["modulus_n46"]:
        errs.append("f: public modulus for n=46 differs")
    for v in kat["f"]["vectors"]:
        if bits_to_str(f(str_to_bits(v["Z"]))) != v["p_a"]:
            errs.append(f"f: p_a differs for Z={v['Z']}")

    for i, v in enumerate(kat["hash_paper_size"]):
        p, s, M = (str_to_bits(v[k]) for k in ("p", "s", "M"))
        for name, fn in (("matrix", hash_matrix), ("horner", hash_horner)):
            if bits_to_str(fn(p, s, M)) != v["digest"]:
                errs.append(f"hash_paper_size[{i}]: digest differs ({name})")

    pr = kat["protocol"]
    A, B, C = (PartyKeys(**{k: str_to_bits(pr[who][k]) for k in "XYZ"}) for who in ("alice", "bob", "charlie"))
    out = run_messaging_stage(str_to_bits(pr["doc"]), A, B, C)
    got = {"p_a": bits_to_str(out.details["alice"]["p_a"]), "digest": bits_to_str(out.details["alice"]["digest"]),
           "sig": bits_to_str(out.details["signed"].sig),
           "bob_accepts": bool(out.bob_accepts), "charlie_accepts": bool(out.charlie_accepts)}
    for k, v in got.items():
        if pr[k] != v:
            errs.append(f"protocol: {k} differs")

    b, sec = budget(TABLE_I), kat["security"]
    if b.ns != sec["ns"] or b.key_bits_per_signature != sec["key_bits_per_signature"]:
        errs.append("security: ns / key bits per signature differ")
    for k in ("eps_for", "eps_rep", "eps_rob", "eps"):
        if not math.isclose(getattr(b, k), sec[k], rel_tol=1e-12):
            errs.append(f"security: {k} differs")
    return errs
