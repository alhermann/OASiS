"""A timed-out solve must die, and so must every child it spawned.

Before this test's fix, wait_for() abandoned a timed-out solver without
terminating it: the process ran on at 100% of a core (one was found 3.2
CPU-hours later, its MPI orted daemon beside it), invisibly taxing whatever
ran next. All nine backends shared the defect; only the preCICE path killed
its process group.

The test runs a REAL backend's run() on a script that spawns a child and
sleeps far past the timeout, then asserts both the solver and its child are
gone. It would have failed on every backend before the fix.
"""
import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Pids are written to FILES, not stdout: on the timeout path communicate()
# is interrupted, so stdout.log is never written and a stdout-based check
# silently SKIPS — the first version of this test passed vacuously that way.
SCRIPT = """\
import os, subprocess, sys, time, pathlib
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
pathlib.Path("pids.txt").write_text(f"{os.getpid()} {child.pid}")
time.sleep(600)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_timeout_kills_the_whole_group(tmp_path):
    from core.registry import load_all_backends, get_backend
    load_all_backends()
    b = get_backend("skfem")          # pure-python: runs anywhere
    job = asyncio.run(b.run(SCRIPT, tmp_path, np=1, timeout=3))
    assert job.status == "failed" and "imed out" in (job.error or ""), (
        job.status, job.error)
    time.sleep(1.0)
    pids_file = tmp_path / "pids.txt"
    assert pids_file.is_file(), (
        "the solver never started or never wrote pids.txt — the test cannot "
        "verify anything and must FAIL rather than pass vacuously")
    solver_pid, child_pid = (int(x) for x in pids_file.read_text().split())
    assert not _alive(solver_pid), f"solver {solver_pid} survived its timeout"
    assert not _alive(child_pid), (
        f"child {child_pid} survived — the process group was not killed")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        test_timeout_kills_the_whole_group(Path(d))
    print("timeout kills the whole solver group: PASS")
