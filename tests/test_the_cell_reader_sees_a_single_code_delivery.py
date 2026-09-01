"""The development loop's reading tool must see what the AGENT ended up with.

`campaign3_blind/cell_read.py` is how every cell is root-caused: it answers
"where did the run stop and what was it trying to do", which is the question a
score cannot answer. It counted the per-level files with ONE glob,

    w.rglob("*level*_[AB].csv")

which matches only the COUPLED per-side shape. 23 of the 47 cells ask for
`solution_level<k>.csv` with no side suffix, so for every single-code run the
counter was structurally zero and the tool printed

    level files: 0
    wrote level files 0/1

under a run with all four files sitting in `work/`. NG1 MCP seed 96 is the
measured instance: four `solution_level*.csv` on disk, read as none.

This is the FIFTH time a mechanism existed, was instrumented, and did not reach
the case it was built for — and the most dangerous of the five, because it does
not fail loudly. It answers, confidently, with a zero.

So these tests do not ask what the code does. They build the run directory an
agent would leave behind, in both shapes, and ask what the reader says about it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign3_blind"))

cell_read = pytest.importorskip("cell_read")


def _run_dir(tmp: Path, cell: str, arm: str, seed: int, files) -> Path:
    """The directory an agent leaves behind, and nothing else."""
    d = tmp / "runs" / f"{cell}_27b_{arm}_seed{seed}"
    w = d / "work"
    w.mkdir(parents=True)
    (d / "ledger.json").write_text('{"tool_calls": 7, "wall_s": 111}')
    (w / "RESULT.txt").write_text("LEVELS = 4\n")
    (w / "trajectory_live.txt").write_text("TOOL_CALL run_bash\n")
    for name in files:
        (w / name).write_text("x,y,u\n0.5,0.5,1.0\n")
    return d


SINGLE = ["solution_level1.csv", "solution_level2.csv",
          "solution_level3.csv", "solution_level4.csv",
          "run_level1.log", "run_level2.log",
          "run_level3.log", "run_level4.log"]

COUPLED = ["solution_level1_A.csv", "solution_level1_B.csv",
           "solution_level2_A.csv", "solution_level2_B.csv",
           "interface_level1_A.csv", "interface_level1_B.csv",
           "residual_level1.csv", "residual_level2.csv",
           "run_level1_A.log", "run_level1_B.log"]


def test_a_single_code_delivery_is_counted(tmp_path, monkeypatch):
    """Four solution files on disk must read as four, not as zero."""
    monkeypatch.setattr(cell_read, "D", ROOT / "campaign3_blind")
    d = _run_dir(tmp_path, "NG1", "MCP", 96, SINGLE)
    r = cell_read.read_one(d)
    assert r["on_disk"]["solution.csv"] == [
        "solution_level1.csv", "solution_level2.csv",
        "solution_level3.csv", "solution_level4.csv"], (
        "the single-code deliverable was not seen. This is the exact defect: a "
        "per-side glob applied to a task that asks for no side.")
    assert len(r["on_disk"]["run.log"]) == 4
    assert r["level_files"] == 4, (
        "level_files is what the aggregate line prints; it must follow the "
        "solution files, not the coupled naming")


def test_a_coupled_delivery_is_still_counted(tmp_path, monkeypatch):
    """The shape that used to work must keep working."""
    monkeypatch.setattr(cell_read, "D", ROOT / "campaign3_blind")
    d = _run_dir(tmp_path, "C8", "BARE", 4, COUPLED)
    r = cell_read.read_one(d)
    assert len(r["on_disk"]["solution.csv"]) == 4
    assert len(r["on_disk"]["interface.csv"]) == 2
    assert len(r["on_disk"]["residual.csv"]) == 2


def test_an_empty_workspace_reads_as_empty(tmp_path, monkeypatch):
    """A zero must still be reachable — for a run that really delivered none."""
    monkeypatch.setattr(cell_read, "D", ROOT / "campaign3_blind")
    d = _run_dir(tmp_path, "NG1", "MCP", 96, [])
    r = cell_read.read_one(d)
    assert r["on_disk"]["solution.csv"] == []
    assert r["level_files"] == 0


def test_every_real_cell_has_its_deliverables_understood():
    """A renamed deliverable must break a test, not go quietly uncounted.

    The tool now reads the demanded names out of the task text. If a task is
    reworded so the parser no longer recognises its per-level files, the count
    silently returns to zero — which is how the original defect survived. This
    test is the tripwire.
    """
    probs = ROOT / "campaign3_blind" / "problems"
    if not probs.is_dir():
        pytest.skip("problems/ not present")
    cells = sorted(p.name for p in probs.iterdir() if p.is_dir())
    assert len(cells) >= 40, f"only {len(cells)} cells found"
    blind, unparsed = [], []
    for c in cells:
        if not cell_read.demanded(c):
            blind.append(c)
        if cell_read._unparsed_templates(c):
            unparsed.append((c, cell_read._unparsed_templates(c)))
    assert not blind, (
        f"the reader cannot name the deliverable of {blind} — it would report "
        f"a confident zero for every run of those cells")
    assert not unparsed, (
        f"these tasks name a per-level file the parser does not recognise, so "
        f"it would be reported as missing: {unparsed}")


def test_an_unreadable_template_is_announced_not_zeroed(tmp_path, monkeypatch):
    """The failure mode itself must be visible.

    A tool that prints 0 when it means "I do not understand the question" is
    what cost us weeks of misread cells. Given a task naming a shape the parser
    does not model, it must appear in `unparsed` so the reader can say so.
    """
    fake = tmp_path / "campaign"
    (fake / "problems" / "ZZ9").mkdir(parents=True)
    (fake / "problems" / "ZZ9" / "task.txt").write_text(
        "REQUIRED OUTPUT: write field_level<k>_<subdomain>.dat per level\n")
    monkeypatch.setattr(cell_read, "D", fake)
    assert cell_read._unparsed_templates("ZZ9") == [
        "field_level<k>_<subdomain>.dat"], (
        "an unrecognised per-level artefact must be surfaced by name")


def test_the_reader_prints_the_delivery_for_a_single_code_run(tmp_path,
                                                             monkeypatch,
                                                             capsys):
    """What the OPERATOR ends up seeing, not what the function returns.

    The defect was in the printed line, and a return value can be right while
    the print is wrong. So this drives `main()` and reads the terminal.
    """
    monkeypatch.setattr(cell_read, "D", ROOT / "campaign3_blind")
    _run_dir(tmp_path, "NG1", "MCP", 96, SINGLE)
    monkeypatch.setattr(cell_read, "runs_for",
                        lambda cell, arm, seeds: [
                            (96, tmp_path / "runs" / "NG1_27b_MCP_seed96")])
    monkeypatch.setattr(sys, "argv", ["cell_read.py", "NG1", "MCP", "96"])
    assert cell_read.main() == 0
    out = capsys.readouterr().out
    assert "solution.csv" in out
    assert "solution.csv    : 0" not in out, (
        "the operator would read a complete delivery as no delivery")
    assert "wrote solution.csv" in out and "wrote solution.csv           0/1" \
        not in out
