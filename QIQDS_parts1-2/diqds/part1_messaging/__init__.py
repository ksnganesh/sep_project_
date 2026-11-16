"""Part 1 of the reproduction of Fan et al., PRA 114, 032617 (2026): the messaging stage.

Paper section -> module
    App. A  (LFSR-based Toeplitz hash)            lfsr_toeplitz.py
    II.B.1  p_a = f(Z_a)  (Ref. [32], App. B.1)   polygen.py (+ gf2poly.py)
    II.B    sign / verify / certify, Fig. 2       protocol.py, keys.py
    III.A   forging strategies                    attacks.py
    Eqs. 4-8, Table I                             security.py
"""
from .attacks import (make_bob_tamper, max_guesses, multiple_of_polynomial,  # noqa: F401
                      product_of_polynomials, random_forgery)
from .bits import (bits_to_str, decode_document, encode_document, int_to_bits,  # noqa: F401
                   random_bits, str_to_bits)
from .keys import (PartyKeys, ideal_key_triples, key_bits_per_signature,  # noqa: F401
                   satisfies_correlations, split_pairwise_keys)
from .lfsr_toeplitz import hash_digest, toeplitz_matrix  # noqa: F401
from .polygen import DegenerateKeyError, f, poly_from_key  # noqa: F401
from .protocol import (Bob, Charlie, Outcome, ProtocolOrderError,  # noqa: F401
                       run_messaging_stage, sign, verify_with_new_keys)
from .security import (TABLE_I, SecurityBudget, Table1, budget, eps_forgery,  # noqa: F401
                       eps_repudiation, eps_robustness, eps_total,
                       minimal_signature_length, signature_rate)
