"""The just-in-time "nothing is written down yet" notice.

Five rounds of measurement say the runs that fail most are not the ones that
fail to solve — they solve and end with nothing a reader can find. Two static
attempts (a rule in one backend's table, then the same rule on every knowledge
payload) did not move it, because both are read before there is anything to
write. This notice is stateful and fires at the moment output exists and no
summary does.

A notice that cannot stay silent is nagging, and a notice that cannot fire is
decoration, so both directions are asserted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tools.consolidated import _unsaved_work_notice          # noqa: E402


def test_fires_when_output_exists_but_nothing_is_written(tmp_path):
    (tmp_path / "solution.vtu").write_text("<VTKFile/>")
    n = _unsaved_work_notice(str(tmp_path), ["solution.vtu"])
    assert n is not None
    assert "not a result" in n


def test_silent_once_a_summary_exists(tmp_path):
    (tmp_path / "solution.vtu").write_text("<VTKFile/>")
    (tmp_path / "RESULT.txt").write_text("ORDER = 1.99")
    assert _unsaved_work_notice(str(tmp_path), ["solution.vtu"]) is None


def test_silent_for_any_summary_name(tmp_path):
    """It must not hardcode this campaign's filename — the server is a product."""
    for name in ("answer.md", "orders.csv", "summary.json", "report.txt"):
        d = tmp_path / name.replace(".", "_")
        d.mkdir()
        (d / "solution.vtu").write_text("x")
        (d / name).write_text("content")
        assert _unsaved_work_notice(str(d), ["solution.vtu"]) is None, name


def test_silent_when_the_run_produced_nothing(tmp_path):
    """A failed solve has no numbers to write; nagging there is noise."""
    assert _unsaved_work_notice(str(tmp_path), []) is None


def test_coupling_handshake_files_do_not_count_as_a_summary(tmp_path):
    """imports/exports are the driver's plumbing, not the agent's answer."""
    (tmp_path / "solution.vtu").write_text("x")
    (tmp_path / "exports.json").write_text("{}")
    (tmp_path / "imports.json").write_text("{}")
    assert _unsaved_work_notice(str(tmp_path), ["solution.vtu"]) is not None


def test_trajectory_logs_do_not_count_as_a_summary(tmp_path):
    """The harness writes these; they are not the agent's deliverable."""
    (tmp_path / "solution.vtu").write_text("x")
    (tmp_path / "trajectory.txt").write_text("TOOL_CALL ...")
    (tmp_path / "trajectory_live.txt").write_text("TOOL_CALL ...")
    assert _unsaved_work_notice(str(tmp_path), ["solution.vtu"]) is not None


def test_finds_a_summary_in_a_subdirectory(tmp_path):
    """Agents organise work into subfolders; the grader searches recursively."""
    (tmp_path / "solution.vtu").write_text("x")
    sub = tmp_path / "level1"
    sub.mkdir()
    (sub / "RESULT.txt").write_text("ORDER = 2.0")
    assert _unsaved_work_notice(str(tmp_path), ["solution.vtu"]) is None


def test_missing_directory_is_not_an_error(tmp_path):
    assert _unsaved_work_notice(str(tmp_path / "nope"), ["a.vtu"]) is None


if __name__ == "__main__":
    import tempfile
    ok = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        ok += 1
        print(f"  pass  {name}")
    print(f"  {ok} checks passed")
