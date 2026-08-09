"""A backend skipped in its entirety must not be written as a result.

WHY THIS EXISTS
---------------
`scripts/run_tier2_fixtures.py --write-results` persisted a snapshot in which
~130 Kratos fixtures read `skipped`. Kratos was not missing: `KRATOS_PYTHON`
was unset, so the runner probed the repo venv, whose Kratos wheel fails on this
host with `GLIBC_2.32 not found`. Skips went 15 to 145 and 149 previously
PASSING rows would have been downgraded to a non-result.

It was caught by diffing the new file against the committed one before
committing — by hand, not by any check. Nothing in the tool objected, and a
reader of the snapshot cannot tell "skipped because that backend is not
installed here" from "skipped because the caller forgot an environment
variable". Under `scan_results/`, a reader takes the file for a measurement.

This is the same shape as an auditor that examines nothing and prints a pass,
and as an import audit that records a library as unavailable because it looked
for an interpreter that does not exist. An instrument that could not look must
not file a negative.

WHAT IS CHECKED
---------------
`backends_with_no_verdict` is the predicate the guard uses. It is exercised
directly rather than through an hour-long whole-tree run, because a guard
nobody has watched fire is not a guard.

The override (`--allow-unrun`) is deliberately kept: a host that genuinely
lacks a backend must still be able to record a snapshot, and when it does, the
file names those backends in `backends_not_run_here` so the two cases stay
distinguishable forever after.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "scripts" / "run_tier2_fixtures.py"


def _runner():
    """Load the runner as a real module.

    It must be registered in sys.modules under the name the spec carries
    BEFORE exec_module: the file defines a @dataclass, and dataclasses resolve
    their own module by name at decoration time, so an unregistered module
    fails with `'NoneType' object has no attribute '__dict__'`. The same
    registration trick is used where the runner's matcher is shared with
    mutate_tier2_fixtures.
    """
    import sys

    name = "run_tier2_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNER)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _row(backend, status):
    return {"backend": backend, "status": status}


def test_a_backend_with_every_fixture_skipped_is_reported() -> None:
    r = _runner()
    results = {
        "kratos::dem::0": _row("kratos", "skipped"),
        "kratos::dem::1": _row("kratos", "skipped"),
        "fourc::fsi::0": _row("fourc", "passed"),
    }
    assert r.backends_with_no_verdict(results) == ["kratos"]


def test_a_backend_with_one_real_verdict_is_not_reported() -> None:
    """One fixture that actually ran means the backend was reachable. The
    guard is about a run that could not look at a backend AT ALL, not about
    individual skips, which are ordinary and expected."""
    r = _runner()
    results = {
        "kratos::dem::0": _row("kratos", "skipped"),
        "kratos::dem::1": _row("kratos", "passed"),
    }
    assert r.backends_with_no_verdict(results) == []


def test_a_failing_backend_is_not_reported() -> None:
    """FEBio fails 51 fixtures here because this build has no pardiso. That is
    a MEASUREMENT — the fixtures ran and did not pass — and must not be
    confused with never having been asked."""
    r = _runner()
    results = {f"febio::biphasic::{i}": _row("febio", "failed") for i in range(5)}
    assert r.backends_with_no_verdict(results) == []


def test_every_backend_blind_is_all_reported() -> None:
    r = _runner()
    results = {
        "kratos::dem::0": _row("kratos", "skipped"),
        "dune::poisson::0": _row("dune", "skipped"),
        "fenics::heat::0": _row("fenics", "passed"),
    }
    assert r.backends_with_no_verdict(results) == ["dune", "kratos"]


def test_the_key_is_used_when_a_row_carries_no_backend() -> None:
    """Older rows predate the `backend` field. Falling back to the key keeps
    the guard working on a snapshot written before it existed."""
    r = _runner()
    results = {"sparta::chemistry::3": {"status": "skipped"}}
    assert r.backends_with_no_verdict(results) == ["sparta"]
