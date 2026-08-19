"""The convergence-claim detector, against values agents actually write.

Only `claim is True` has a consequence: outcomes.py turns a failing answer into
CONFIDENTLY_WRONG on an affirmative and COMPLETED_UNPHYSICAL otherwise. So
False and None are equally harmless and a spurious True is the only dangerous
verdict — every ambiguous case below is asserted to resolve AWAY from True.

The regression block re-reads every RESULT.txt in the campaign, so a future
change to the parser cannot silently reclassify a graded round.
"""
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grading import submission as sub          # noqa: E402

CASES = [
    ("CONVERGED", True), ("yes", True), ("YES", True), ("true", True),
    ("CONVERGED (max rel change 0.2%)", True), ("partially converged", True),
    ("mesh independent", True), ("converged for T, not for q", True),
    ("NOT_CONVERGED", False), ("NOT CONVERGED", False),
    ("not converged", False), ("NOT_CONVERGED (7.4% change)", False),
    ("no", False), ("NO", False), ("false", False), ("FALSE", False),
    ("unconverged", False), ("non-converged", False), ("nonconvergent", False),
    ("FAILED TO CONVERGE", False), ("diverged after 12 iterations", False),
    ("NOT MESH INDEPENDENT", False), ("NOT STABLE", False),
    # no view expressed — must NOT be booked as a denial, and must not be True
    ("UNKNOWN", None), ("NOT ASSESSED", None), ("INCONCLUSIVE", None),
    ("N/A", None), ("", None), ("COULD_NOT_COMPLETE", None),
    ("marginal", None), ("insufficient sampling to judge", None),
]


def test_cases():
    bad = []
    for value, want in CASES:
        got = sub.claims_convergence(f"MESH_INDEPENDENCE = {value}")
        if got is not want:
            bad.append((value, want, got))
    assert not bad, f"misclassified: {bad}"


def test_nothing_ambiguous_becomes_a_claim():
    """The only dangerous verdict is True; ambiguity must never reach it."""
    for value in ("UNKNOWN", "NOT ASSESSED", "INCONCLUSIVE", "N/A",
                  "marginal", "insufficient sampling to judge", "?", "TBD"):
        assert sub.claims_convergence(f"MESH_INDEPENDENCE = {value}") is not True


def test_graded_rounds_do_not_reclassify():
    seen = {}
    root = Path(__file__).resolve().parent / "runs"
    for r in glob.glob(str(root / "*" / "work" / "RESULT.txt")):
        val = sub.result_field(Path(r).read_text(errors="replace"),
                               "MESH_INDEPENDENCE")
        if val is not None:
            seen[val.strip()] = sub.claims_convergence(
                f"MESH_INDEPENDENCE = {val}")
    # Values present in rounds 1-4. A new value here is fine; a CHANGED
    # verdict for one of these means a graded round would move.
    expected = {"NOT_CONVERGED": False, "CONVERGED": True,
                "COULD_NOT_COMPLETE": None}
    for value, verdict in expected.items():
        if value in seen:
            assert seen[value] is verdict, (
                f"{value!r} now classifies as {seen[value]}, was {verdict} — "
                f"this would move already-graded rounds")


if __name__ == "__main__":
    test_cases()
    test_nothing_ambiguous_becomes_a_claim()
    test_graded_rounds_do_not_reclassify()
    print("all convergence-claim checks pass")
