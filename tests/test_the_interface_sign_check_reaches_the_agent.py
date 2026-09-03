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


def test_the_finding_arrives_on_the_SUBMISSION_WRITE_not_only_on_request():
    """The tenth instance of the theme, and the reason this test exists.

    verify_interface_flux was registered, worked, and was described in the
    coupling must-read with the numbers from the round it decided. In the very
    next round it was called by ZERO of six runs. The auto-audit on submit
    reached FIVE of those six. Round 7 had already measured the same thing for
    the audit tool itself: 1 of 51 agents called it voluntarily.

    So the check is wired into the audit that fires when RESULT.txt is
    written, and this test goes through that write rather than through the
    tool, because the write is what every submitter does.
    """
    import shutil
    import tempfile

    if not WRONG.is_dir():
        pytest.skip("seed303 run tree absent")
    sys.path.insert(0, str(ROOT))
    from langgraph_eval.agent import _read_write_tools_for

    tmp = Path(tempfile.mkdtemp())
    try:
        for f in WRONG.rglob("interface_level*_[AB].csv"):
            shutil.copy(f, tmp / f.name)
        for f in WRONG.rglob("solution_level*_[AB].csv"):
            shutil.copy(f, tmp / f.name)
        assert list(tmp.glob("interface_level*")), "fixture copy found nothing"
        tools = _read_write_tools_for(tmp, audit_on_submit=True)
        wf = [t for t in tools if t.name == "write_file"][0]
        out = wf.invoke({"path": "RESULT.txt",
                         "content": "LEVELS = 3\nMESH_INDEPENDENCE = CONVERGED\n"})
        assert "AUTO-AUDIT" in out, f"no audit ran on the submission write: {out[:300]}"
        assert "WRONG SIGN" in out.upper(), (
            "the submission carries a flux computed with the INWARD normal "
            f"and the write did not say so:\n{out[:800]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ═══════════ both sides on the same convention, without a derivative ════════
#
# C2_27b_MCP_seed1301 is the furthest any OASiS run reached on the coupled
# cell: both codes really ran, the interface iteration converged to 4.4e-07,
# the temperature matched across the seam to 1.3e-13, and the graded order was
# 1.9367. It reported BOTH sides' flux with the same sign, so it graded
# COMPLETED_UNPHYSICAL / INTERFACE_NOT_SATISFIED. It had worked this out
# itself -- its report says "Both sides report negative fluxes of similar
# magnitude (~0.8), giving qn_A + qn_B = -1.6" -- and read it as physics to
# repair rather than a sign on a value being written out.
#
# The pre-existing branch needed recover_normal_derivative, which on the
# Neumann side gave implied k = None, +63.6, -128.0 across the three levels,
# so only level 3 tripped `k < 0`. The ratio test below needs no derivative,
# no material coefficient and no mesh, and separates by three orders:
#
#     that run                89.2   272.4   1036.3     (and GROWS: the
#     a CORRECT reference      0.03    0.00     0.00      denominator shrinks)

def _sign_findings(work):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from tools.result_audit import interface_sign_findings
    return [f for f in interface_sign_findings(work)
            if f.get("sequence") == "interface flux convention"]


def _write_pair(w, lvl, qa_sign, qb_sign):
    """Two interface files whose fluxes differ only in the sign of side B."""
    ys = [0.25 + (i + 0.5) * 0.5 / 44 for i in range(44)]
    for side, sgn in (("A", qa_sign), ("B", qb_sign)):
        rows = ["x, y, u, qn"]
        for i, y in enumerate(ys):
            u = -2.9e-03 - i * 1e-06
            q = sgn * (0.65 + i * 1e-04) + (1e-05 if side == 'B' else 0.0)
            rows.append(f"0.625, {y!r}, {u!r}, {q!r}")
        (w / f"interface_level{lvl}_{side}.csv").write_text("\n".join(rows))


def test_the_same_convention_is_named_with_the_ratio(tmp_path):
    for lvl in (1, 2, 3):
        _write_pair(tmp_path, lvl, -1.0, -1.0)      # both negative
    got = _sign_findings(tmp_path)
    assert got, "silent on two sides reporting the same sign"
    text = got[0]["finding"]
    assert "BOTH SIDES REPORTED THEIR FLUX WITH THE SAME SIGN" in text
    assert "|sum|/|difference|" in text
    # it must say the fix is a sign on the output, not a repair of the solve
    assert "not a defect in your solve" in text
    assert "leave the temperature column alone" in text
    # and how to confirm the fix
    assert "must SHRINK from level to level" in text


def test_opposite_signs_are_left_alone(tmp_path):
    for lvl in (1, 2, 3):
        _write_pair(tmp_path, lvl, +1.0, -1.0)      # the prescribed convention
    assert _sign_findings(tmp_path) == []


def test_it_fires_on_the_real_run_at_every_level_not_just_one():
    w = ROOT / "campaign3_blind/runs/C2_27b_MCP_seed1301/work"
    if not (w / "interface_level1_A.csv").exists():
        import pytest as _p
        _p.skip("seed1301 run data absent")
    got = _sign_findings(w)
    assert got, "silent on the run this check was built for"
    assert len(got[0]["values"]) == 3, (
        "the derivative-based branch named level 3 alone; the ratio test must "
        f"name all three, got {got[0]['values']}")
    assert min(got[0]["values"]) > 4.0


def test_it_is_silent_on_a_submission_that_grades_correct():
    ref = Path("/tmp/claude-1001/-home-alexander-4C/"
               "b1c8e459-ec06-467a-bad7-474c8459f0f3/scratchpad/c2_real/"
               "submission")
    if not ref.exists():
        import pytest as _p
        _p.skip("reference submission not on this machine")
    assert _sign_findings(ref) == []
