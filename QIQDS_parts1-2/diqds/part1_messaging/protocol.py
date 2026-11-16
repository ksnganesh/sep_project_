"""Messaging stage, Sec. II.B and Fig. 2 of the paper.

1. Signing (Alice)        p_a = f(Z_a);  H = LFSR-Toeplitz(p_a, initial state X_a);
                          Dig = H . Doc;  Sig = Dig xor Y_a;   Alice -> Bob : {Sig, Doc}
2. Verification (Bob)     Bob -> Charlie : {Sig, Doc, Y_b}              (authenticated channel)
                          only AFTER recording it, Charlie -> Bob : {X_c, Y_c, Z_c}
                          K_Xb = X_c,  K_Yb = Y_b xor Y_c,  K_Zb = Z_c
                          Dig_expected = Sig xor K_Yb;  p_b = f(K_Zb);
                          Dig_actual = LFSR-Toeplitz(p_b, K_Xb) . Doc;  accept iff equal;  Bob -> Charlie : result
3. Certification (Charlie) after Bob's acceptance: K_Xc = X_c, K_Yc = Y_b xor Y_c, K_Zc = Z_c; same test.

The ordering in step 2 is what the forgery analysis (Sec. III.A) rests on: when Bob commits to
{Sig, Doc, Y_b} he knows nothing about X_a = X_c and Z_a = Z_c, i.e. nothing about the hash function.
``Charlie.reveal`` therefore refuses to run before the packet has been recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .bits import as_bits, bits_to_str
from .keys import PartyKeys
from .lfsr_toeplitz import hash_digest
from .polygen import DegenerateKeyError, f


class ProtocolOrderError(RuntimeError):
    """A step was attempted before the step it depends on."""


@dataclass
class SignedMessage:
    sig: np.ndarray
    doc: np.ndarray


@dataclass
class TranscriptLine:
    step: int
    src: str
    dst: str
    label: str
    content: str

    def __str__(self) -> str:
        return f"[{self.step}] {self.src:>7} -> {self.dst:<7} {self.label:<22} {self.content}"


@dataclass
class Outcome:
    bob_accepts: Optional[bool]          # None when Bob is the (dishonest) forger and his verdict is irrelevant
    charlie_accepts: bool
    transcript: List[TranscriptLine] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        """The signature is validated iff both recipients accept (Fig. 2 caption)."""
        return bool(self.charlie_accepts and (self.bob_accepts in (True, None)))


def _short(bits, k: int = 24) -> str:
    s = bits_to_str(bits)
    return s if len(s) <= k else f"{s[:k]}...({len(s)} b)"


# --------------------------------------------------------------------------
# Step 1: signing
# --------------------------------------------------------------------------
def sign(doc, keys: PartyKeys, hash_method: str = "matrix") -> Tuple[SignedMessage, dict]:
    doc = as_bits(doc)
    p_a = f(keys.Z)                                       # p_a = f(Z_a)
    dig = hash_digest(p_a, keys.X, doc, hash_method)       # Dig = H_{ns,nm} . Doc, initial state X_a
    sig = dig ^ keys.Y                                     # Sig = Dig xor Y_a   (one-time pad)
    return SignedMessage(sig=sig, doc=doc), {"p_a": p_a, "digest": dig}


# --------------------------------------------------------------------------
# Steps 2-3: the common test run by Bob and by Charlie
# --------------------------------------------------------------------------
def verify_with_new_keys(sig, doc, K_X, K_Y, K_Z, hash_method: str = "matrix") -> Tuple[bool, dict]:
    expected = as_bits(sig) ^ as_bits(K_Y)                 # Dig_expected = Sig xor K_Y
    try:
        p = f(K_Z)                                         # p = f(K_Z)
    except DegenerateKeyError:
        return False, {"expected": expected, "actual": None, "p": None}
    actual = hash_digest(p, K_X, doc, hash_method)         # Dig_actual
    return bool(np.array_equal(actual, expected)), {"expected": expected, "actual": actual, "p": p}


class Bob:
    name = "Bob"

    def __init__(self, keys: PartyKeys, hash_method: str = "matrix"):
        self.keys, self.hash_method = keys, hash_method
        self.received: Optional[SignedMessage] = None

    def receive(self, msg: SignedMessage) -> None:
        self.received = SignedMessage(msg.sig.copy(), msg.doc.copy())

    def forward_packet(self) -> dict:
        if self.received is None:
            raise ProtocolOrderError("Bob has nothing to forward")
        return {"sig": self.received.sig, "doc": self.received.doc, "Y_b": self.keys.Y}

    def verify(self, reveal: dict) -> Tuple[bool, dict]:
        if self.received is None:
            raise ProtocolOrderError("Bob has not received {Sig, Doc}")
        K_X, K_Y, K_Z = reveal["X"], self.keys.Y ^ reveal["Y"], reveal["Z"]   # K_Xb = X_c, K_Yb = Y_b^Y_c, K_Zb = Z_c
        return verify_with_new_keys(self.received.sig, self.received.doc, K_X, K_Y, K_Z, self.hash_method)


class Charlie:
    name = "Charlie"

    def __init__(self, keys: PartyKeys, hash_method: str = "matrix"):
        self.keys, self.hash_method = keys, hash_method
        self.packet: Optional[dict] = None
        self.revealed = False

    def record(self, packet: dict) -> None:
        self.packet = {k: as_bits(v) for k, v in packet.items()}

    def reveal(self) -> dict:
        if self.packet is None:
            raise ProtocolOrderError("Charlie must record Bob's packet before revealing {X_c, Y_c, Z_c}")
        self.revealed = True
        return {"X": self.keys.X.copy(), "Y": self.keys.Y.copy(), "Z": self.keys.Z.copy()}

    def certify(self, bob_accepted: bool) -> Tuple[bool, dict]:
        if self.packet is None or not self.revealed:
            raise ProtocolOrderError("Charlie certifies only after recording Bob's packet and sending his keys")
        if not bob_accepted:
            return False, {"reason": "Bob rejected; protocol aborted"}
        K_X, K_Y, K_Z = self.keys.X, self.packet["Y_b"] ^ self.keys.Y, self.keys.Z  # K_Xc = X_c, K_Yc = Y_b^Y_c, K_Zc = Z_c
        return verify_with_new_keys(self.packet["sig"], self.packet["doc"], K_X, K_Y, K_Z, self.hash_method)


# --------------------------------------------------------------------------
# Whole stage
# --------------------------------------------------------------------------
BobTamper = Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]


def run_messaging_stage(doc, alice: PartyKeys, bob: PartyKeys, charlie: PartyKeys, *,
                        bob_tamper: Optional[BobTamper] = None,
                        hash_method: str = "matrix") -> Outcome:
    """Run Fig. 2 end to end.

    ``bob_tamper(sig, doc, Y_b) -> (sig', doc', Y_b')`` models a dishonest Bob (the forger of
    Sec. III.A): he rewrites what he forwards to Charlie and then announces "accept".
    """
    t: List[TranscriptLine] = []
    B_, C_ = Bob(bob, hash_method), Charlie(charlie, hash_method)

    msg, rec = sign(doc, alice, hash_method)
    t.append(TranscriptLine(1, "Alice", "Bob", "{Sig, Doc}", f"Sig={_short(msg.sig)}"))
    B_.receive(msg)

    packet = B_.forward_packet()
    if bob_tamper is not None:
        s2, d2, y2 = bob_tamper(packet["sig"].copy(), packet["doc"].copy(), packet["Y_b"].copy())
        packet = {"sig": as_bits(s2), "doc": as_bits(d2), "Y_b": as_bits(y2)}
    t.append(TranscriptLine(2, "Bob", "Charlie", "{Sig, Doc, Y_b}", f"Y_b={_short(packet['Y_b'])}"
                            + ("   [TAMPERED by Bob]" if bob_tamper else "")))
    C_.record(packet)

    reveal = C_.reveal()
    t.append(TranscriptLine(3, "Charlie", "Bob", "{X_c, Y_c, Z_c}", f"X_c={_short(reveal['X'])}"))

    if bob_tamper is None:
        bob_ok, bob_info = B_.verify(reveal)
    else:
        bob_ok, bob_info = None, {}
    t.append(TranscriptLine(4, "Bob", "Charlie", "accept / reject",
                            "accept" if bob_ok in (True, None) else "REJECT (abort)"))

    ch_ok, ch_info = C_.certify(bob_accepted=bob_ok in (True, None))
    t.append(TranscriptLine(5, "Charlie", "-", "certification", "accept" if ch_ok else "REJECT"))

    return Outcome(bob_accepts=bob_ok, charlie_accepts=ch_ok, transcript=t,
                   details={"alice": rec, "bob": bob_info, "charlie": ch_info, "signed": msg})
