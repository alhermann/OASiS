"""The grader's own evaluation set, and the proof the task asked for the same one.

THE GRADER DEFINES THE EVALUATION SET, NOT THE AGENT. The task prescribes a
fixed, cell-centred probe grid independent of the agent's mesh, and a
submission is rejected unless it reports exactly those points. This blocks
norm dilution with known-zero boundary nodes, node clustering in a shrinking
boundary layer, dropping bad nodes as NaN, and submitting fewer levels than
prescribed — all four scored CORRECT before this rule existed.

Two hard-won specifics live here:

* **The grid the task states must BE the grid graded.** For some time the
  single-code tasks asked for 1024 points at (i+0.5)/32 while the grader
  demanded 1936 at /44; every submission would have graded INVALID on arrival
  and the campaign would have produced a table of zeros that looked like a
  finding. `task_grid_agreement` re-derives the grid from the task text at
  grading time and refuses to grade a cell whose sheet and marking scheme
  disagree. The probe count itself is imported from `constants.PROBE_M` and
  duplicated nowhere.

* **A subdomain is not always a rectangle.** D5's subdomain A is the unit
  square minus subdomain B minus a notch; its task states the 1331 points that
  remain. Exclusions come from the PUBLIC spec (they are printed in the task,
  so reading them costs no secrecy) and remove every point lying STRICTLY
  inside an excluded box — a point on the box boundary belongs to the
  subdomain and stays.
"""
from __future__ import annotations

import re

from . import constants as C
from .loading import GraderConfigError


# ── grid construction ─────────────────────────────────────────────────────
def assert_probe_grid_incommensurate(dim: int, mesh_N) -> None:
    """A probe count that divides or is divided by a mesh level aliases: the
    RMS becomes a one-point-per-cell sample of an error field that oscillates
    cell to cell, and the observed order is biased (measured: -0.27)."""
    M = C.PROBE_M[dim]
    bad = [n for n in (mesh_N or []) if M % n == 0 or n % M == 0]
    if bad:
        raise GraderConfigError(
            f"probe count {M} is commensurate with mesh level(s) {bad}; "
            f"choose a probe count sharing no factor with any mesh level")


def probe_grid(dim: int, bounds=None, exclude=None) -> list[tuple]:
    """Cell-centred grid, last index varying fastest, never on a mesh node.

    `exclude` is a list of axis-aligned boxes, each [(lo, hi), ...] per axis;
    a point is removed only if it lies STRICTLY inside a box.
    """
    M = C.PROBE_M[dim]
    b = bounds or [(0.0, 1.0)] * dim
    axes = [[lo + (i + 0.5) * (hi - lo) / M for i in range(M)] for lo, hi in b]
    pts = [()]
    for ax in axes:
        pts = [p + (v,) for p in pts for v in ax]
    if not exclude:
        return pts
    boxes = [[tuple(axis) for axis in box] for box in exclude]
    for box in boxes:
        if len(box) != dim:
            raise GraderConfigError(
                f"probe exclusion box has {len(box)} axes but the problem "
                f"is {dim}D")

    def inside(p, box):
        return all(lo < c < hi for c, (lo, hi) in zip(p, box))

    return [p for p in pts if not any(inside(p, box) for box in boxes)]


def matches_probe_grid(pts, expected) -> tuple[bool, str]:
    if len(pts) != len(expected):
        return False, f"expected {len(expected)} probe points, got {len(pts)}"
    for i, (got, want) in enumerate(zip(pts, expected)):
        if len(got) != len(want):
            return False, f"row {i}: wrong coordinate dimension"
        for a, b in zip(got, want):
            if abs(a - b) > C.PROBE_TOL:
                return False, (f"row {i}: point {got} is not the prescribed "
                               f"probe point {want}")
    return True, "ok"


# ── where each (sub)domain lives ──────────────────────────────────────────
def subdomain_bounds(key: dict, side: str, dim: int) -> list[tuple]:
    """Coupled: from the key the problem was built with. A grader that cannot
    say where the subdomain lives must stop, not assume the unit square —
    assuming it once built the probe grid over the wrong region and reported a
    number that looked like a result."""
    field = "extent_a" if side == "A" else "extent_b"
    extent = key.get(field)
    if not extent:
        raise GraderConfigError(
            f"key for {key.get('id', '?')} carries no {field!r}: cannot place "
            f"probe points without the subdomain bounds the problem was built "
            f"with. Rebuild the problem set.")
    if len(extent) != dim:
        raise GraderConfigError(
            f"{field} has {len(extent)} axes but the problem is {dim}D")
    return [tuple(axis) for axis in extent]


def single_bounds(key: dict, dim: int) -> list[tuple]:
    """Single-code: the key's extent if it carries one, else the unit domain.
    The unit default is a CONSTRAINT, not a coincidence (DESIGN.md §3): a
    single-code instance on any other domain must carry its extent in the key."""
    extent = key.get("extent")
    if extent:
        if len(extent) != dim:
            raise GraderConfigError(
                f"extent has {len(extent)} axes but the problem is {dim}D")
        return [tuple(axis) for axis in extent]
    return [(0.0, 1.0)] * dim


def probe_exclusions(spec: dict, side: str):
    """Excluded boxes for a subdomain's grid, from the PUBLIC spec. `side` is
    'A'/'B' for coupled cells or '-' for single-code (field `probe_exclude`)."""
    if side in ("A", "B"):
        return spec.get(f"probe_{side.lower()}_exclude")
    return spec.get("probe_exclude")


# ── the task text and the grader must state the same grid ─────────────────
_SINGLE_LINE = re.compile(
    r"PROBE POINTS: the (?P<count>\d+) points given by x = \(i_x\+0\.5\)/(?P<m>\d+)")
_COUPLED_LINE = re.compile(
    r"PROBE POINTS, subdomain (?P<side>[AB]): the (?P<count>\d+) points"
    r"(?P<rest>[^\n]*)")
_REMAIN = re.compile(r"(\d+) points remain")


def task_grid_agreement(problem_id: str, task_txt: str, spec: dict,
                        key: dict) -> None:
    """Refuse to grade a cell whose task text and grading grid disagree.

    The stated count is compared against the very grid this grader builds —
    bounds, exclusions and all — so `PROBE_M`, the extents and the exclusions
    have exactly one authority each. For a non-rectangular subdomain the
    operative number is the '<n> points remain' the task states after its
    exclusion sentence.

    Raises GraderConfigError: a disagreement here means every submission would
    be graded INVALID for following its own instructions, which is a defect of
    the CELL and must never be booked as an outcome.
    """
    dim = spec["dim"]
    coupled = spec.get("kind") == "coupled" or "codes" in spec
    states_3d = "(i_z+0.5)" in task_txt
    if states_3d != (dim == 3):
        raise GraderConfigError(
            f"{problem_id}: the task text states a {'3' if states_3d else '2'}D "
            f"probe grid but the spec says dim = {dim}; sheet and marking "
            f"scheme disagree")

    if coupled:
        sides = {}
        for m in _COUPLED_LINE.finditer(task_txt):
            remain = _REMAIN.search(m.group("rest"))
            sides[m.group("side")] = int(
                remain.group(1) if remain else m.group("count"))
        if set(sides) != {"A", "B"}:
            raise GraderConfigError(
                f"{problem_id}: coupled task text states probe grids for "
                f"side(s) {sorted(sides) or 'none'}; both A and B are "
                f"required, so the sheet cannot be checked against the grader")
        for side, stated in sides.items():
            built = len(probe_grid(dim, subdomain_bounds(key, side, dim),
                                   probe_exclusions(spec, side)))
            if built != stated:
                raise GraderConfigError(
                    f"{problem_id} subdomain {side}: task states {stated} "
                    f"probe points, the grader builds {built}. A submission "
                    f"following the task would be rejected for it — the cell "
                    f"is broken, fix the sheet or the spec, and do not grade "
                    f"until they agree.")
        return

    m = _SINGLE_LINE.search(task_txt)
    if not m:
        raise GraderConfigError(
            f"{problem_id}: the task text states no recognisable probe grid "
            f"('PROBE POINTS: the N points given by x = (i_x+0.5)/M'); the "
            f"sheet cannot be checked against the grader, so the cell must "
            f"not be graded")
    stated_m, stated_count = int(m.group("m")), int(m.group("count"))
    built = len(probe_grid(dim, single_bounds(key, dim),
                           probe_exclusions(spec, "-")))
    want_m = C.PROBE_M[dim]
    if stated_m != want_m or stated_count != built:
        raise GraderConfigError(
            f"{problem_id}: task asks for {stated_count} points at "
            f"(i+0.5)/{stated_m}; the grader accepts only {built} at "
            f"(i+0.5)/{want_m}. PROBE_M in grading/constants.py is the single "
            f"source of truth; the task text must move with it.")
