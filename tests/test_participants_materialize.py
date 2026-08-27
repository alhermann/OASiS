"""Asking for coupling knowledge PUTS THE PARTICIPANT ON DISK.

Structural, not offered. Measured twice at 27B: a capability the agent must
choose to invoke is not invoked — audit_results 1 of 51 runs, the
materialize_participant tool 0 of 6, while the same runs hand-wrote
participants (259 of 260 scripts in the campaign were hand-written, and they
repeated the traction-sign bug the shipped headers document).

So the delivery rides the call agents demonstrably DO make.
"""
import asyncio
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _sandbox(d):
    """A cell's sandbox, exactly as langgraph_eval/agent.py sets one up.

    The destination is NOT the process's cwd: the MCP server runs with
    cwd=<repo>/src, so a delivery keyed to cwd would land in the OASiS source
    tree and never reach the agent. It is keyed to the coupling directory the
    driver itself resolves, which the harness points into the cell.
    """
    d = Path(d)
    old = {k: os.environ.get(k) for k in ("OASIS_COUPLING_DIR", "OASIS_WORK_DIR")}
    os.environ["OASIS_COUPLING_DIR"] = str(d / "coupling")
    os.environ["OASIS_WORK_DIR"] = str(d)
    try:
        yield d / "coupling"
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _knowledge():
    from core.registry import load_all_backends
    load_all_backends()

    class FakeMCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    import tools.consolidated as C
    m = FakeMCP()
    for n in dir(C):
        if n.startswith("register") and callable(getattr(C, n)):
            try:
                getattr(C, n)(m)
            except Exception:                      # noqa: BLE001
                pass
    return m.tools["knowledge"]


def _call(fn, **kw):
    r = fn(**kw)
    return str(asyncio.run(r) if asyncio.iscoroutine(r) else r)


def test_asking_for_a_backend_writes_its_participant():
    fn = _knowledge()
    for solver in ("fenics", "ngsolve", "skfem", "dune", "febio", "kratos",
                   "fourc", "dealii"):
        with tempfile.TemporaryDirectory() as d:
            with _sandbox(d) as cdir:
                out = _call(fn, topic="coupling", solver=solver)
                on_disk = sorted(p.name for p in cdir.glob("*.py"))
            assert on_disk, f"{solver}: nothing written to {cdir}"
            assert "NOW ON DISK" in out, f"{solver}: payload does not say so"
            assert any(solver in n.lower() for n in on_disk), (
                f"{solver}: wrote {on_disk}, none matching the backend")
            # the agent must be given a path its own file tools accept
            assert "./coupling/participant_" in out, (
                f"{solver}: banner does not name a sandbox-relative path:\n"
                + "\n".join(l for l in out.splitlines() if "participant_" in l)[:400])


def test_an_edited_file_is_never_overwritten():
    """The agent's own work outranks ours."""
    fn = _knowledge()
    with tempfile.TemporaryDirectory() as d:
        with _sandbox(d) as cdir:
            cdir.mkdir(parents=True, exist_ok=True)
            mine = cdir / "participant_fenics.py"
            mine.write_text("# MY EDITS\n")
            out = _call(fn, topic="coupling", solver="fenics")
        assert mine.read_text() == "# MY EDITS\n", "overwrote the agent's file"
        assert "untouched" in out


def test_no_solver_writes_nothing():
    fn = _knowledge()
    with tempfile.TemporaryDirectory() as d:
        with _sandbox(d) as cdir:
            _call(fn, topic="coupling")
        assert not list(cdir.glob("*.py")) if cdir.is_dir() else True, (
            "wrote participants without being asked for a backend")


def test_nothing_is_written_outside_the_sandbox(tmp_path, monkeypatch):
    """The repo is not a scratch directory.

    Keyed to cwd, this delivery littered the repository root with 24 participant
    copies the first time the suite ran from there.
    """
    fn = _knowledge()
    before = sorted(p.name for p in Path.cwd().glob("participant_*.py"))
    with _sandbox(tmp_path):
        _call(fn, topic="coupling", solver="fenics")
    after = sorted(p.name for p in Path.cwd().glob("participant_*.py"))
    assert before == after, f"wrote into the process cwd: {set(after) - set(before)}"


if __name__ == "__main__":
    test_asking_for_a_backend_writes_its_participant()
    print("  pass  every backend's participant lands on disk")
    test_an_edited_file_is_never_overwritten()
    print("  pass  an existing file is never overwritten")
    test_no_solver_writes_nothing()
    print("  pass  no solver named -> nothing written")
