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
from pathlib import Path

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
            cwd = os.getcwd()
            try:
                os.chdir(d)
                out = _call(fn, topic="coupling", solver=solver)
                on_disk = sorted(p.name for p in Path(d).glob("*.py"))
            finally:
                os.chdir(cwd)
            assert on_disk, f"{solver}: nothing written to the work dir"
            assert "NOW ON DISK" in out, f"{solver}: payload does not say so"
            assert any(solver in n.lower() for n in on_disk), (
                f"{solver}: wrote {on_disk}, none matching the backend")


def test_an_edited_file_is_never_overwritten():
    """The agent's own work outranks ours."""
    fn = _knowledge()
    with tempfile.TemporaryDirectory() as d:
        mine = Path(d) / "participant_fenics.py"
        mine.write_text("# MY EDITS\n")
        cwd = os.getcwd()
        try:
            os.chdir(d)
            out = _call(fn, topic="coupling", solver="fenics")
        finally:
            os.chdir(cwd)
        assert mine.read_text() == "# MY EDITS\n", "overwrote the agent's file"
        assert "untouched" in out


def test_no_solver_writes_nothing():
    fn = _knowledge()
    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        try:
            os.chdir(d)
            _call(fn, topic="coupling")
            assert not list(Path(d).glob("*.py")), (
                "wrote participants without being asked for a backend")
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    test_asking_for_a_backend_writes_its_participant()
    print("  pass  every backend's participant lands on disk")
    test_an_edited_file_is_never_overwritten()
    print("  pass  an existing file is never overwritten")
    test_no_solver_writes_nothing()
    print("  pass  no solver named -> nothing written")
