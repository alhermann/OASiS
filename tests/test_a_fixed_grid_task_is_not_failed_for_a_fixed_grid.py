"""The DSMC cells prescribe a fixed grid, and the grader demanded refinement.

`ndof_growth` requires the reported count to grow like 2**dim per level, which
is right wherever a mesh is halved. The SPARTA cells prescribe the opposite, in
as many words:

    "DISCRETISATION AND REFINEMENT: THE GRID IS FIXED AND THE PARTICLE COUNT
     REFINES. ... do NOT refine the grid instead, and do NOT change dt."

and the same task adds that "particle count plays the role the degree-of-freedom
count plays elsewhere". Their EXECUTION LOG clause still asks for an NDOF line,
so the growth rule ran on them anyway: an agent reporting the grid cell count it
was told to hold fixed was charged MESH_SEQUENCE_NOT_PRESCRIBED for obeying the
task, while one reporting particles passed only because quadrupling happens to
land inside the 2.2-7.2 band at dim = 2.

Which quantity the line carries cannot be told from the number alone, so on
these cells the observation is recorded and not charged. Everywhere else the
rule keeps its teeth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign3_blind"))
sys.path.insert(0, str(ROOT / "src"))

from grading.evidence2 import ndof_growth  # noqa: E402

FIXED_GRID = ("DISCRETISATION AND REFINEMENT: THE GRID IS FIXED AND THE "
              "PARTICLE COUNT REFINES. run_level<k>.log NDOF")
REFINING = ("MESH SEQUENCE: uniform meshes with h = 1/8, 1/16, 1/32. "
            "run_level<k>.log NDOF")


def test_the_underlying_rule_still_flags_a_flat_sequence():
    """The repair is at the call site; the rule itself must not be blunted."""
    _, viol, _, _ = ndof_growth({"-": {1: 610, 2: 610, 3: 610}}, [1, 2, 3], 2, True)
    assert viol, "a flat NDOF sequence must still be a violation in general"


def test_the_sparta_task_really_does_prescribe_a_fixed_grid():
    t = (ROOT / "campaign3_blind" / "problems" / "SP1" / "task.txt")
    if not t.is_file():
        pytest.skip("SP1 not present")
    text = t.read_text()
    assert "GRID IS FIXED" in text.upper()
    assert "do NOT refine the grid" in text


def test_the_guard_condition_matches_that_wording():
    """The gate keys on the task's own words, not on a cell name."""
    src = (ROOT / "campaign3_blind" / "grading" / "evidence2.py").read_text()
    assert '"GRID IS FIXED" in (task_txt or "").upper()' in src, (
        "the exemption must be driven by what the task says, so a new "
        "fixed-grid cell is covered without being listed anywhere"
    )


def test_an_ordinary_refining_task_is_untouched():
    src = (ROOT / "campaign3_blind" / "grading" / "evidence2.py").read_text()
    i = src.index('"GRID IS FIXED" in (task_txt or "").upper()')
    window = src[i:i + 900]
    assert "violations = []" in window
    assert "NDOF GROWTH NOT CHARGED" in window, (
        "the observation must still be recorded, or a real fixed-grid defect "
        "becomes invisible instead of uncharged"
    )
    assert "GRID IS FIXED" not in REFINING.upper(), (
        "sanity: an ordinary refining task must not match the guard"
    )
