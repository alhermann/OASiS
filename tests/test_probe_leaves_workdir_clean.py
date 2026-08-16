"""The interface-sensitivity probe must not leave a perturbed solve on disk.

`probe_interface_sensitivity` re-runs every participant TWICE — once on its
own saved imports, once on imports nudged by `delta` — and used to restore
only imports.json and exports.json. Every OTHER file the participant writes
was therefore left holding the output of the PERTURBED run: solution CSVs,
VTU dumps, logs, RESULT files. Anything downstream that reads those files
reads a solve of a deliberately wrong problem, off by a relative 1e-3, with
nothing in the output saying so.

`probe` defaults to True, so this was the default behaviour.

Without the fix this test fails with the values shifted by exactly the nudge:
  before: 0,2.000000000000e+00 ...
  after:  0,2.002000000000e+00 ...
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from core.coupling_driver import (Participant,  # noqa: E402
                                  probe_interface_sensitivity)

SOLVER = textwrap.dedent("""
    import json, pathlib
    p = pathlib.Path(".")
    imp = {}
    f = p / "imports.json"
    if f.exists():
        try:
            imp = json.loads(f.read_text() or "{}")
        except Exception:
            imp = {}
    vals = [1.0, 2.0, 3.0]
    for d in imp.values():
        if isinstance(d, dict) and d.get("values"):
            vals = [2.0 * v for v in d["values"]]
    # THE SIDE FILE: what a real participant writes and a grader later reads
    (p / "solution_level1.csv").write_text(
        "\\n".join(f"{i},{v:.12e}" for i, v in enumerate(vals)) + "\\n")
    json.dump({"field_name": "T", "n_points": len(vals),
               "coordinates": [[0.0, float(i)] for i in range(len(vals))],
               "values": vals, "normal_fluxes": [0.5] * len(vals)},
              open("exports.json", "w"))
""")


def _build(tmp: Path) -> None:
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "solver.py").write_text(SOLVER)
    json.dump({"partner": {"field_name": "T", "n_points": 3,
                           "coordinates": [[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]],
                           "values": [1.0, 2.0, 3.0],
                           "normal_fluxes": [0.5, 0.5, 0.5]}},
              open(tmp / "imports.json", "w"))
    subprocess.run([sys.executable, "solver.py"], cwd=tmp, check=True)


def test_probe_leaves_side_files_untouched(tmp_path):
    tmp = tmp_path / "case"
    _build(tmp)
    before = (tmp / "solution_level1.csv").read_text()

    p = Participant(name="A", command=[sys.executable, "solver.py"],
                    work_dir=tmp, imports_from=["partner"])
    res = probe_interface_sensitivity(
        [p], last_imports={"A": (tmp / "imports.json").read_text()})

    after = (tmp / "solution_level1.csv").read_text()
    assert before == after, (
        "the sensitivity probe left a PERTURBED solve on disk:\n"
        f"  before: {before.strip()}\n  after:  {after.strip()}")
    # and it must still do its job
    assert res["A"]["S"] is not None, "the probe stopped measuring sensitivity"
    assert res["A"]["S"] == pytest.approx(1.0, abs=0.05)


def test_probe_removes_files_the_perturbed_run_created(tmp_path):
    """A stray output from a perturbed solve is as misleading as a changed one."""
    tmp = tmp_path / "case"
    _build(tmp)
    (tmp / "solver.py").write_text(
        SOLVER + "\n(p / 'stray_from_this_run.txt').write_text('x')\n")
    subprocess.run([sys.executable, "solver.py"], cwd=tmp, check=True)
    (tmp / "stray_from_this_run.txt").unlink()   # not present in the snapshot

    p = Participant(name="A", command=[sys.executable, "solver.py"],
                    work_dir=tmp, imports_from=["partner"])
    probe_interface_sensitivity(
        [p], last_imports={"A": (tmp / "imports.json").read_text()})

    assert not (tmp / "stray_from_this_run.txt").exists(), (
        "the probe left behind a file its perturbed re-run created")
