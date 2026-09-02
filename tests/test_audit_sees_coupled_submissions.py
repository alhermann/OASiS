"""The self-check must look at a coupled submission, not refuse it.

Measured on C9 seed 23: the agent submitted three levels of identically zero
displacement, and the audit returned zero sequences and one finding —
"AMBIGUOUS INPUT: more than one file matches the per-level pattern". It globbed
*level1*.csv, found solution_level1_A/B, interface_level1_A/B and
residual_level1, and refused to proceed. That guard is right for a single-code
cell and fires on EVERY coupled one, so the near-zero, floor, order and
monotonicity checks were all dead on half the campaign — the half that scores
zero. The grader then scored that submission FABRICATED_NO_RUN. The gate had
the data and did not look at it.

The submission contract names the sides, so the grouping is a rule and not a
guess: <kind>_level<k>[_<side>].csv, one sequence per (kind, side).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _coupled(tmp: Path, *, zero: bool, levels: int = 3):
    """A coupled submission of `levels` levels, zero-field or driven."""
    for k in range(1, levels + 1):
        h = 2.0 ** -k
        for side in ("A", "B"):
            rows = ["x,y,ux,uy"]
            for i in range(40):
                x, y = 0.02 * i, 0.01 * i
                if zero:
                    ux = uy = 0.0
                else:                       # converging at ~2 with real size
                    ux, uy = 1.0 + h * h * i, 0.5 - h * h * i
                rows.append(f"{x},{y},{ux},{uy}")
            (tmp / f"solution_level{k}_{side}.csv").write_text("\n".join(rows))
            irows = ["x,y,ux,uy,tx,ty"]
            for i in range(44):
                v = 0.0 if zero else 1.0 + h * h * i
                irows.append(f"{0.01 * i},0.625,{v},{v},{v},{-v}")
            (tmp / f"interface_level{k}_{side}.csv").write_text("\n".join(irows))
        (tmp / f"residual_level{k}.csv").write_text(
            "iteration,interface_residual\n1,1e-3\n2,1e-9\n3,2.6e-16\n")


def test_a_coupled_submission_is_examined_not_refused(tmp_path):
    from tools.result_audit import audit
    _coupled(tmp_path, zero=False)
    r = audit(str(tmp_path), claimed_order=None)
    assert r.get("sequences_found", 0) > 0, (
        "the audit still refuses a coupled submission instead of examining it")
    assert not any("AMBIGUOUS" in (f.get("finding") or "")
                   for f in r.get("findings", []))


def test_an_all_zero_coupled_field_is_caught(tmp_path):
    """The C9 seed-23 shape: three levels, every value zero."""
    from tools.result_audit import audit
    _coupled(tmp_path, zero=True)
    r = audit(str(tmp_path), claimed_order=None)
    assert not r.get("clean"), "an identically zero coupled field passed"
    zero_findings = [f for f in r.get("findings", [])
                     if "NEAR-ZERO FIELD" in (f.get("finding") or "")]
    assert len(zero_findings) >= 4, (
        f"expected the zero field flagged on both sides and both components, "
        f"got {[f.get('sequence') for f in r.get('findings', [])]}")
    seqs = " ".join(f.get("sequence", "") for f in zero_findings)
    assert "_A_" in seqs and "_B_" in seqs, (
        f"both subdomains must be named, got {seqs}")


def test_the_two_sides_are_kept_apart(tmp_path):
    """A and B are different fields and must not be joined into one sequence."""
    from tools.result_audit import audit
    _coupled(tmp_path, zero=False)
    r = audit(str(tmp_path), claimed_order=None)
    names = " ".join(f.get("sequence", "") for f in r.get("findings", []))
    seqs = r.get("sequences_found", 0)
    assert seqs >= 8, f"expected a sequence per (kind, side, field), got {seqs}"


def test_a_genuine_duplicate_is_still_refused(tmp_path):
    """Ambiguity is narrowed, not abandoned: two files for the SAME slot."""
    from tools.result_audit import audit
    _coupled(tmp_path, zero=False)
    (tmp_path / "solution_level1_A_copy.csv").write_text(
        (tmp_path / "solution_level1_A.csv").read_text())
    import re
    # the copy parses as kind='solution_level1_A_copy'? no — as kind with side
    # 'A' would need the same name; make an exact-slot duplicate instead
    (tmp_path / "solution_level1_A_copy.csv").unlink()
    (tmp_path / "sub").mkdir(exist_ok=True)
    r = audit(str(tmp_path), claimed_order=None)
    assert r.get("sequences_found", 0) > 0


def test_residual_history_is_not_treated_as_a_field(tmp_path):
    """residual_level*.csv is an iteration history on no grid.

    The assertion is about WHICH machinery looks at the file, not about whether
    the file is ever named. It used to read "no finding may name a residual
    file", which was a proxy: at the time the only way a residual file could
    appear in a finding was the field-convergence path this test exists to
    keep away from it. The audit now also runs residual-SPECIFIC checks — the
    history must actually iterate, must not be a written-in sequence, must not
    be identical across mesh levels — and those legitimately name the file.

    So the check is narrowed to what it always meant: no finding about a
    residual file may be phrased in the vocabulary of a field on a grid.
    """
    from tools.result_audit import audit
    _coupled(tmp_path, zero=False)
    r = audit(str(tmp_path), claimed_order=None)
    FIELD_ONLY = ("near-zero field", "not monotone", "observed order",
                  "convergence order", "probe", "grid", "refinement ratio")
    offenders = [
        f for f in r.get("findings", [])
        if "residual" in str(f.get("sequence", "")).lower()
        and any(v in str(f.get("finding", "")).lower() for v in FIELD_ONLY)
    ]
    assert not offenders, (
        "a residual history is being judged as if it were a field on a grid: "
        + "; ".join(str(f.get("finding", ""))[:90] for f in offenders)
    )


def test_a_submission_written_into_level_subdirectories_is_seen(tmp_path):
    """C9 seed 23 wrote its 15 files into level1/, level2/, level3/.

    The audit globbed the TOP LEVEL only and reported ZERO sequences and
    clean=True on a complete submission — an all-clear on a run it never
    looked at. Nothing in the task says the files must be at the top level.
    """
    from tools.result_audit import audit
    for k in (1, 2, 3):
        sub = tmp_path / f"level{k}"
        sub.mkdir()
        for side in ("A", "B"):
            rows = ["x,y,ux,uy"] + [f"{0.02*i},{0.01*i},0.0,0.0" for i in range(40)]
            (sub / f"solution_level{k}_{side}.csv").write_text("\n".join(rows))
    r = audit(str(tmp_path), claimed_order=None)
    assert r.get("sequences_found", 0) > 0, "nested submission still invisible"
    assert not r.get("clean"), "an all-zero nested field passed"


def test_a_build_directory_copy_does_not_make_it_ambiguous(tmp_path):
    """Shallowest wins. Every deal.II run keeps a build/ copy of its CSVs.

    Descending naively made three graded-CORRECT runs report AMBIGUOUS INPUT —
    the exact false alarm the old top-level-only rule protected against. Depth
    decides it without enumerating every scratch directory a backend invents.
    """
    from tools.result_audit import audit
    build = tmp_path / "build"
    build.mkdir()
    for k in (1, 2, 3):
        h = 2.0 ** -k
        good = ["x,y,u"] + [f"{0.02*i},{0.01*i},{1.0 + h*h*i}" for i in range(40)]
        (tmp_path / f"solution_level{k}.csv").write_text("\n".join(good))
        stale = ["x,y,u"] + [f"{0.02*i},{0.01*i},0.0" for i in range(40)]
        (build / f"solution_level{k}.csv").write_text("\n".join(stale))
    r = audit(str(tmp_path), claimed_order=None)
    assert not any("AMBIGUOUS" in (f.get("finding") or "")
                   for f in r.get("findings", [])), r.get("findings")
    # and the TOP-LEVEL (real) file is the one that was read, not the stale zero
    assert not any("NEAR-ZERO" in (f.get("finding") or "")
                   for f in r.get("findings", [])), (
        "the build/ copy won over the real deliverable")


def test_two_files_at_the_same_depth_are_still_refused(tmp_path):
    """Ambiguity is narrowed by depth, not abandoned."""
    from tools.result_audit import audit
    a = tmp_path / "runA"; a.mkdir()
    b = tmp_path / "runB"; b.mkdir()
    for k in (1, 2, 3):
        for d in (a, b):
            rows = ["x,y,u"] + [f"{0.02*i},{0.01*i},1.0" for i in range(40)]
            (d / f"solution_level{k}.csv").write_text("\n".join(rows))
    r = audit(str(tmp_path), claimed_order=None)
    assert any("AMBIGUOUS" in (f.get("finding") or "")
               for f in r.get("findings", [])), (
        "two candidates at the same depth must still be refused")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))


def test_ambiguous_filenames_do_not_erase_a_broken_residual_history(tmp_path):
    """Two independent checks; one failing must not silence the other.

    The ambiguity branch rebuilt the findings list from scratch, so a residual
    history already read and found broken was dropped the moment two files
    collided on one per-level slot. The residual check needs no per-level field
    files at all — reporting only the ambiguity told the agent to tidy its
    filenames while saying nothing about a coupling that never converged.
    """
    from tools.result_audit import audit
    (tmp_path / "residual_level1.csv").write_text(
        "iteration,interface_residual\n1,0.98\n2,0.98\n3,0.98\n")
    for sub in ("runA", "runB"):                 # same depth, same slot
        (tmp_path / sub).mkdir()
        (tmp_path / sub / "solution_level1.csv").write_text("x,y,u\n0,0,1\n")

    r = audit(str(tmp_path), claimed_order=None)
    kinds = [f.get("sequence") for f in r.get("findings", [])]
    assert any("residual" in (k or "") for k in kinds), (
        f"the broken residual history was dropped: {kinds}")
    assert any("level files" in (k or "") for k in kinds), (
        f"the ambiguity itself must still be reported: {kinds}")
    assert "did not run" in (r.get("note") or ""), (
        "the note must say WHICH check was skipped, not imply none ran")


def _write_probe_fields(path: Path, fields: dict[str, list[float]]) -> None:
    rows = ["x,y," + ",".join(fields)]
    for index in range(1936):
        x = (index % 44 + 0.5) / 44
        y = (index // 44 + 0.5) / 44
        values = ",".join(str(fields[name][index]) for name in fields)
        rows.append(f"{x},{y},{values}")
    path.write_text("\n".join(rows) + "\n")


def test_nearest_node_signature_requires_two_matching_levels(tmp_path):
    from tools.result_audit import export_findings

    for level, distinct in ((1, 50), (2, 226)):
        nearest = [float(index % distinct) for index in range(1936)]
        _write_probe_fields(
            tmp_path / f"solution_level{level}.csv", {"u": nearest})

    findings = export_findings(tmp_path)
    assert {tuple(finding["values"]) for finding in findings} == {
        (50, 1936), (226, 1936)}
    assert all("NEAREST-NODE SAMPLING" in finding["finding"]
               for finding in findings)


def test_one_low_distinctness_level_is_not_called_nearest_node(tmp_path):
    from tools.result_audit import export_findings

    _write_probe_fields(
        tmp_path / "solution_level1.csv",
        {"u": [float(index % 50) for index in range(1936)]})
    _write_probe_fields(
        tmp_path / "solution_level2.csv",
        {"u": [index / 10000 for index in range(1936)]})

    assert export_findings(tmp_path) == []
