"""Real C2 development regression: 4C/Kratos must converge at second order."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from benchmarks.coupling_pairs.fourc_kratos_mms.run_pair import (
    FOURC, PYTHON, REPO, SPEC, SPEC_PATH, run_all,
)

sys.path.insert(0, str(REPO / "campaign3_blind"))


def _kratos_usable() -> bool:
    import subprocess

    try:
        result = subprocess.run(
            [PYTHON, "-c", "import KratosMultiphysics"],
            capture_output=True, timeout=120)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


pytestmark = [
    pytest.mark.skipif(not Path(FOURC).is_file(), reason="4C binary unavailable"),
    pytest.mark.skipif(not _kratos_usable(), reason="Kratos unavailable"),
]


def test_c2_real_pair_converges_and_writes_gradeable_evidence(tmp_path):
    report = run_all(tmp_path)

    for side in ("A", "B"):
        assert min(report["orders"][side]) > 1.8, report
        assert report["errors"][side][-1] < report["errors"][side][0] / 12
    assert report["max_relative_change"] < 0.05

    interface_jumps = []
    for level in range(1, 4):
        detail = report["levels"][level - 1]["report"]
        assert detail["coupling"]["residual"] < 1e-6
        assert len(detail["coupling"]["finite_history"]) >= 3
        interface_jumps.append(detail["interface"]["balance_relative"])
        for side in ("A", "B"):
            solution = np.genfromtxt(
                tmp_path / f"solution_level{level}_{side}.csv",
                delimiter=",", names=True)
            interface = np.genfromtxt(
                tmp_path / f"interface_level{level}_{side}.csv",
                delimiter=",", names=True)
            assert len(solution) == 1936
            assert len(interface) == 44
            assert np.isfinite(solution["u"]).all()
        for side, signature in (("A", "processor 0 finished normally"),
                                ("B", "ResidualBasedLinearStrategy")):
            assert signature in (
                tmp_path / f"run_level{level}_{side}.log").read_text()

    # Same public gate as grading/iface.py: h halves, so a real O(h) or O(h^2)
    # recovery must shrink clearly and finish below the 10% ceiling.
    assert interface_jumps[-1] < interface_jumps[0] * 0.75 ** 2
    assert interface_jumps[-1] <= 0.10

    result = (tmp_path / "RESULT.txt").read_text()
    assert "LEVELS = 3" in result
    assert "MESH_INDEPENDENCE = CONVERGED" in result
    assert json.loads((tmp_path / "level_3/A/field.json").read_text())
    assert shutil.which(PYTHON) or Path(PYTHON).is_file()

    from grading.evidence2 import assess_execution
    from grading.iface import interface_legs, interface_phase
    from tools.result_audit import audit

    task = SPEC_PATH.with_name("task.txt").read_text()
    public_audit = audit(str(tmp_path))
    evidence = assess_execution(
        tmp_path, SPEC["codes"], True, task, SPEC["mesh_N"], SPEC["dim"],
        1e-6, report["levels"][-1]["iterations"])
    interface = interface_phase(
        tmp_path, SPEC, SPEC, SPEC["dim"], 1, SPEC["mesh_N"],
        interface_legs(SPEC, SPEC, SPEC["dim"]))

    assert public_audit["clean"], public_audit
    assert evidence["fatal"] is None, evidence
    assert evidence["verdict"] == "PROVEN", evidence
    assert interface["verdict"] == "INTERFACE_SATISFIED", interface
    assert not interface["malformed"], interface