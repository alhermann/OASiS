"""A give-up filed on top of finished work is caught when it is written.

Measured twice, and the second time is why this is structural rather than
better wording. Ten coupled runs drove a coupling to convergence, hit a
flux-balance finding, and declared COULD_NOT_COMPLETE with a median 69% of
their budget unspent. The `couple` tool's reply was then rewritten to say that
NOT VERIFIED and NOT A RESULT are different verdicts and to write the
deliverables first. In the very next probe, three of six runs did it again —
C8 stating in its own words that the coupling converged in about 7 iterations
before giving up on the balance check.

So the check reads the agent's OWN files at the moment it writes the give-up.
It supplies no knowledge, no method and no numbers, and it fires without the
agent choosing to invoke anything.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "langgraph_eval"))
sys.path.insert(0, str(REPO / "src"))


def _fn():
    import agent as A
    return A._work_on_disk_contradicting_a_give_up


def _converged_run(tmp: Path, *, levels=3, resid=True):
    """The shape C7/C8/C9 were in when they filed COULD_NOT_COMPLETE."""
    for k in range(1, levels + 1):
        for side in ("A", "B"):
            (tmp / f"solution_level{k}_{side}.csv").write_text(
                "x,y,ux,uy\n0.5,0.5,1.0,2.0\n")
            (tmp / f"interface_level{k}_{side}.csv").write_text(
                "x,y,ux,uy,tx,ty\n0.5,0.625,1.0,2.0,3.0,4.0\n")
        if resid:
            (tmp / f"residual_level{k}.csv").write_text(
                "iteration,interface_residual\n1,1e-1\n2,1e-3\n3,5.7e-7\n")


def test_it_fires_on_a_give_up_over_a_converged_run(tmp_path):
    _converged_run(tmp_path)
    out = _fn()(tmp_path)
    assert out, "silent on a give-up filed over three converged levels"
    assert "6 solution_level" in out, out
    assert "6 interface_level" in out, out
    assert "5.7e-07" in out or "5.7e-7" in out, out
    # it must name the remedy, not merely scold
    assert "COULD_NOT_COMPLETE is graded as nothing" in out
    assert "different verdicts" in out


def test_it_is_silent_when_the_give_up_is_honest(tmp_path):
    """Nothing on disk means nothing was finished — the give-up is the truth
    and must not be argued with."""
    (tmp_path / "notes.txt").write_text("tried and failed\n")
    (tmp_path / "participant_A.py").write_text("# a script, not a result\n")
    assert _fn()(tmp_path) == ""


def test_partial_work_still_fires(tmp_path):
    """One level solved is still a submission worth more than nothing."""
    (tmp_path / "solution_level1_A.csv").write_text("x,y,ux,uy\n0,0,1,1\n")
    out = _fn()(tmp_path)
    assert out and "1 solution_level" in out


def test_a_residual_history_alone_fires(tmp_path):
    """The coupling ran even if the fields were never written out."""
    (tmp_path / "residual_level1.csv").write_text(
        "iteration,interface_residual\n1,1e-2\n2,4e-9\n")
    out = _fn()(tmp_path)
    assert out and "2 iterations" in out


def test_it_supplies_no_physics(tmp_path):
    """It reports what is on disk. It must not teach anything."""
    _converged_run(tmp_path)
    out = _fn()(tmp_path).lower()
    for leak in ("dirichlet", "neumann", "reaction", "b_vol", "sigma",
                 "order 2", "manufactured", "lambda", "traction ="):
        assert leak not in out, f"the notice leaks method: {leak!r}"


def test_only_the_oasis_arm_gets_it():
    """The bare arm must be untouched — this is an OASiS gate."""
    src = (REPO / "langgraph_eval" / "agent.py").read_text()
    assert "audit_on_submit=True" in src
    i = src.index("_work_on_disk_contradicting_a_give_up(workdir)")
    # the call sits inside the audit_on_submit branch, which only
    # build_mcp_agent sets
    head = src[:i]
    assert "if audit_on_submit and p.name ==" in head[-2000:], (
        "the give-up check is not inside the OASiS-only branch")


def test_a_placeholder_only_export_does_not_fire(tmp_path):
    """Crying wolf is how a gate gets ignored.

    A participant that wrote only the iteration-1 fallback has solved nothing,
    so an exports.json of zeros must not count as work on disk.
    """
    import json
    (tmp_path / "exports.json").write_text(json.dumps({
        "values": [0.0] * 9, "normal_fluxes": [0.0] * 9,
        "coordinates": [[0.6, i / 9] for i in range(9)]}))
    assert _fn()(tmp_path) == ""


def test_it_fires_on_every_real_give_up_of_the_seed13_probe():
    """The four runs this gate exists for, on their own files.

    C7, C8 and C9 converged and gave up on a flux-balance finding; C10 ran out
    of clock. All four had real numbers on disk (|q| up to 0.98) and filed
    COULD_NOT_COMPLETE. C8 is the one that shows why exports.json has to count:
    it wrote no CSV at all and still had both halves of level 1 exchanged.
    """
    base = REPO / "campaign3_blind" / "runs"
    seen = 0
    for cell in ("C7", "C8", "C9", "C10"):
        d = base / f"{cell}_27b_MCP_seed13" / "work"
        if not d.is_dir():
            continue
        res = d / "RESULT.txt"
        if not res.is_file() or "COULD_NOT_COMPLETE" not in res.read_text(
                errors="replace").upper():
            continue
        seen += 1
        assert _fn()(d), f"{cell} filed a give-up over real work and was missed"
    if seen == 0:
        import pytest
        pytest.skip("seed-13 probe runs not present in this checkout")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
