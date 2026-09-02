"""A finding delivered at 98% of the run changes nothing. Fire it earlier.

MEASURED, from file mtimes over the six C2 OASiS runs of rounds 4 and 5:

    seed301  RESULT.txt at 98% of the file-activity span,  40s of activity after
    seed302  93%,  115s after
    seed303  68%,  357s after
    seed401  98%,   26s after
    seed402  99%,    8s after
    seed403  99%,   16s after

seed303 is the only one that wrote its submission with real time left, and it
is the only one of the six that reached a gradeable convergence order with
BOTH prescribed codes proven. seed301 received SEVEN findings at the moment of
submission -- three too-short residual histories, an identical-history-across-
levels, two near-zero fields -- and had forty seconds.

The submission audit is not wrong and is not removed. It is simply the last
possible moment to be told. So the two coupled checks now also fire on the
ARTEFACT write, each running only the check its own file makes possible:

    residual_level<k>.csv         length, and identity across levels
    interface_level<k>_<side>.csv the flux SIGN, once both sides and the
                                  matching solution file exist

This is the eleventh time in this repo that a mechanism reached the agent and
still did not reach the case it was built for -- here by arriving too late
rather than by not arriving. The test therefore asserts on the REPLY THE AGENT
GETS from a write, with two real submissions as fixtures.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

RUNS = ROOT / "campaign3_blind" / "runs"
WRONG_SIGN = RUNS / "C2_27b_MCP_seed303" / "work"
BAD_HISTORY = RUNS / "C2_27b_MCP_seed301" / "work"
GOOD = Path("/tmp/claude-1001/-home-alexander-4C/"
            "b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/c2_real/submission")

_PATTERNS = ("interface_level*_[AB].csv", "solution_level*_[AB].csv",
             "residual_level*.csv")


def _reply_when_writing(src: Path, last: str) -> str:
    """Stage every artefact EXCEPT `last`, then write `last` and return the
    reply — which is what the agent actually sees."""
    from langgraph_eval.agent import _read_write_tools_for

    tmp = Path(tempfile.mkdtemp())
    try:
        for pat in _PATTERNS:
            for f in src.rglob(pat):
                if f.name != last:
                    shutil.copy(f, tmp / f.name)
        target = next((f for f in src.rglob(last)), None)
        if target is None:
            pytest.skip(f"fixture {last} absent under {src}")
        wf = [t for t in _read_write_tools_for(tmp, audit_on_submit=True)
              if t.name == "write_file"][0]
        return wf.invoke({"path": last, "content": target.read_text()})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.skipif(not WRONG_SIGN.is_dir(), reason="seed303 tree absent")
def test_the_sign_finding_arrives_when_the_interface_file_is_written():
    out = _reply_when_writing(WRONG_SIGN, "interface_level3_B.csv")
    assert "WRONG SIGN" in out.upper(), (
        "writing the last interface file of a submission whose level-3 side B "
        "flux carries the INWARD normal produced no warning; the agent would "
        "next learn this at submission, with seconds left:\n" + out[:600])
    assert "-250.8" in out or "250.8" in out, (
        "the finding must carry the measured implied coefficient, so the "
        f"agent can see WHICH side and by how much: {out[:400]}")


@pytest.mark.skipif(not BAD_HISTORY.is_dir(), reason="seed301 tree absent")
def test_the_history_finding_arrives_when_the_residual_file_is_written():
    out = _reply_when_writing(BAD_HISTORY, "residual_level3.csv")
    assert "TOO SHORT" in out.upper(), (
        f"a 2-row residual history drew no warning on write: {out[:600]}")
    assert "IDENTICAL" in out.upper(), (
        "three bit-identical histories across three different meshes drew no "
        f"warning on write: {out[:600]}")


@pytest.mark.skipif(not GOOD.is_dir(), reason="reference submission absent")
@pytest.mark.parametrize("last", ["interface_level3_B.csv",
                                  "residual_level3.csv"])
def test_a_correct_submission_is_not_nagged(last):
    """The cost side. A check that fires on the correct case teaches the agent
    to ignore it, which is worse than not having it."""
    out = _reply_when_writing(GOOD, last)
    body = out.split("\n", 1)[1] if "\n" in out else ""
    assert not body.strip(), (
        f"writing {last} of a submission graded CORRECT at order 1.9796 "
        f"produced a warning:\n{body[:600]}")


def test_the_bare_arm_gets_none_of_this():
    """The hook is OASiS's capability; the control arm must be untouched, or
    the measured uplift includes a mechanism the bare arm was also given."""
    from langgraph_eval.agent import _read_write_tools_for

    if not BAD_HISTORY.is_dir():
        pytest.skip("seed301 tree absent")
    tmp = Path(tempfile.mkdtemp())
    try:
        for pat in _PATTERNS:
            for f in BAD_HISTORY.rglob(pat):
                shutil.copy(f, tmp / f.name)
        wf = [t for t in _read_write_tools_for(tmp, audit_on_submit=False)
              if t.name == "write_file"][0]
        out = wf.invoke({"path": "residual_level3.csv", "content": "1,1.0\n"})
        body = out.split("\n", 1)[1] if "\n" in out else ""
        assert not body.strip(), f"the bare arm was audited: {body[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
