"""The interface gate must test what its own message claims.

It said "a jump that stays O(1) under refinement" and implemented "the max over
levels is below 5e-3", which is a different and much worse test.

Why worse. When both sides recover the interface flux from their assembled
system, the two exported arrays differ by the consistent-to-nodal conversion of
the P1 boundary mass matrix, so the graded jump reduces to h^2|q''|/(6|q|) — a
mesh ruler with no physics in it. Two independent reviews reproduced the
consequence on a coupling with one side's conductivity 4x WRONG: it fails at
n=8 and n=16 and PASSES at n=32, on resolution alone.

And in this campaign's own grades, C8 seed 4 failed SOLELY on this gate in both
arms while its flux jump fell at every level — BARE 3.3% -> 2.0% -> 1.1%, MCP
8.9% -> 3.6% -> 1.5%. Converging couplings graded as unphysical. Both grade
CORRECT under the refinement test.

It mattered asymmetrically too: the recovery that passes a fixed tolerance is
described in the OASiS payload and nowhere in the task text, so a
grading-critical rule was published to one arm only. Testing for
non-convergence removes that, because both recoveries converge.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from grading import constants as C          # noqa: E402

TOL, DEC = C.IFACE_JUMP_TOL, C.IFACE_JUMP_DECAY


def _verdict(seq_u, seq_q):
    """The gate's decision, mirroring grading/iface.py."""
    graded = [{"jump_u_rel": u, "jump_q_rel": q} for u, q in zip(seq_u, seq_q)]

    def shrinks(key):
        v = [g[key] for g in graded]
        if len(v) < 2:
            return None
        if max(v) <= TOL:
            return True
        first, last = v[0], v[-1]
        if first <= 0:
            return last <= TOL
        return last < first * (DEC ** (len(v) - 1))

    su, sq = shrinks("jump_u_rel"), shrinks("jump_q_rel")
    if su is None or sq is None:
        return "FAIL" if (max(seq_u) > TOL or max(seq_q) > TOL) else "PASS"
    return "PASS" if (su and sq) else "FAIL"


# (name, field jumps, flux jumps, expected)
CASES = [
    ("consistent recovery, roundoff", [1e-15] * 3, [1.6e-14] * 3, "PASS"),
    ("direct stress recovery, O(h)", [0, 0, 0], [3.3e-2, 2.0e-2, 1.1e-2], "PASS"),
    ("C8 seed 4 MCP, as measured", [0, 0, 0], [8.9e-2, 3.6e-2, 1.5e-2], "PASS"),
    ("C8 seed 4 BARE, as measured", [4.8e-3, 4.0e-3, 3.1e-3],
     [3.27e-2, 2.03e-2, 1.11e-2], "PASS"),
    # the mutation signatures the gate was calibrated on: flat, O(1)
    ("mutation: flat at 0.75", [0, 0, 0], [0.75] * 3, "FAIL"),
    ("mutation: flat and large", [0, 0, 0], [3.0, 2.9, 3.1], "FAIL"),
    ("mutation: growing", [0, 0, 0], [1.9, 2.0, 2.1], "FAIL"),
    ("mutation: barely shrinking", [0, 0, 0], [1.0, 0.9, 0.81], "FAIL"),
    ("field jump flat, flux fine", [0.2] * 3, [1e-14] * 3, "FAIL"),
    ("one level above tolerance", [0.0], [2e-2], "FAIL"),
    ("one level at roundoff", [0.0], [1e-14], "PASS"),
]


def test_the_gate_separates_converging_from_stuck():
    bad = []
    for name, u, q, want in CASES:
        got = _verdict(u, q)
        if got != want:
            bad.append(f"{name}: wanted {want}, got {got}")
    assert not bad, "\n  ".join(bad)


def test_a_mutation_cannot_pass_by_refining():
    """The old gate let a wrong coupling through at a fine enough mesh."""
    # conductivity 4x wrong, jump = h^2|q''|/(6|q|): shrinks like h^2 but the
    # FIELD jump stays put because the transmission condition is wrong
    assert _verdict([0.31, 0.31, 0.31], [2.5e-2, 6.4e-3, 1.6e-3]) == "FAIL"


def test_the_decay_threshold_is_between_mutation_and_o_of_h():
    """0.5 is O(h), 0.25 is O(h^2), ~1.0 is a mutation. 0.75 sits between."""
    assert 0.5 < DEC < 1.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
