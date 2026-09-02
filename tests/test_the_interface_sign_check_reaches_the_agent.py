"""The interface sign check must be callable BY THE AGENT, and must fire.

It lived only inside the grader. That is the eighth time in this repo that a
mechanism existed, was instrumented, and did not reach the case it was built
for -- and the ninth was this test's own subject: the first version of the tool
unpacked read_interface_csv wrongly and abstained on every side with
"TypeError: type tuple doesn't define __round__", which looks exactly like an
honest NOT_ASSESSED.

So this test asserts three things, in order of what actually goes wrong:
  1. the tool is registered, so an agent can call it at all;
  2. it FIRES on the submission it was written for;
  3. it stays QUIET on the one that is correct.

Fixtures are two real submissions, both graded with the answers sealed:
  C2_27b_MCP_seed303  COMPLETED_UNPHYSICAL, order 1.2434. Both prescribed
      codes genuinely ran, the coupling genuinely iterated over three levels,
      the interface field matched to 0.000e+00 -- and one side reported its
      flux with the INWARD normal, giving an implied coefficient of -250.8
      where +200 was right, and a relative flux jump that GREW under
      refinement: 8.139e-01, 9.066e-01, 9.530e-01.
  the 4C+Kratos reference  CORRECT, order 1.9796. Implied coefficients +0.98
      to +1.30 on the k=1 side and +200.4 to +206.7 on the k=200 side, which
      is the check recovering both conductivities from the submission alone.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WRONG = ROOT / "campaign3_blind" / "runs" / "C2_27b_MCP_seed303" / "work"
RIGHT = Path("/tmp/claude-1001/-home-alexander-4C/"
             "b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/c2_real/"
             "submission")


def _tool():
    from tools.consolidated import register_consolidated_tools

    class Cap:
        def __init__(self):
            self.fns = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.fns[fn.__name__] = fn
                return fn
            return deco

    cap = Cap()
    register_consolidated_tools(cap)
    return cap.fns


def _coefficients(root: Path):
    fns = _tool()
    fn = fns["verify_interface_flux"]
    ifs = sorted(glob.glob(str(root / "**" / "interface_level*_[AB].csv"),
                           recursive=True))
    sol = sorted(glob.glob(str(root / "**" / "solution_level*_[AB].csv"),
                           recursive=True))
    raw = fn(",".join(ifs), ",".join(sol), 0)
    rep = json.loads(raw[:raw.rindex("}") + 1])
    out = []
    for side in rep["per_side"]:
        for comp in side.get("per_component") or []:
            c = comp.get("implied_coefficient")
            if isinstance(c, (int, float)):
                out.append((side["level"], side["side"], c))
    return rep, out


def test_the_agent_can_call_it_at_all():
    """A check only the grader can run cannot change what an agent submits."""
    assert "verify_interface_flux" in _tool(), (
        "verify_interface_flux is not registered as an MCP tool, so the "
        "interface sign check is invisible to the arm under test — which is "
        "the state it was in while it decided a coupled round")


@pytest.mark.skipif(not WRONG.is_dir(), reason="seed303 run tree absent")
def test_it_fires_on_the_run_it_was_written_for():
    rep, coeffs = _coefficients(WRONG)
    assert coeffs, (
        f"the check assessed NOTHING on a complete three-level submission; "
        f"an abstention here is indistinguishable from a pass. per_side: "
        f"{json.dumps(rep['per_side'])[:400]}")
    negative = [c for c in coeffs if c[2] < 0]
    assert negative, (
        f"the inverted normal at level 3 side B was not detected; implied "
        f"coefficients were {coeffs}")


@pytest.mark.skipif(not RIGHT.is_dir(), reason="reference submission absent")
def test_it_stays_quiet_on_the_correct_submission():
    rep, coeffs = _coefficients(RIGHT)
    assert len(coeffs) >= 6, (
        f"only {len(coeffs)} of six sides were assessed on the CORRECT "
        f"submission, so the check is mostly abstaining: {coeffs}")
    negative = [c for c in coeffs if c[2] < 0]
    assert not negative, (
        f"the check accuses a submission graded CORRECT at order 1.9796 of an "
        f"inverted normal: {negative}")
    # and it must recover both conductivities, which is the property that
    # makes it usable with the answers sealed
    a = [c[2] for c in coeffs if c[1] == "A"]
    b = [c[2] for c in coeffs if c[1] == "B"]
    assert a and all(0.5 < v < 2.0 for v in a), f"k_A not recovered: {a}"
    assert b and all(100 < v < 400 for v in b), f"k_B not recovered: {b}"
