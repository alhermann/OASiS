"""Shared plumbing for the path-walk participants.

WHAT A PATH WALK IS FOR. A coupled task whose intended execution path has never
run measures the path, not the agent: a tool bug in it reads as agent failure and
is charged to the arm under test. So every instance gets one throwaway, non-blind
run through the arrangement it prescribes, the result is discarded, and the fact
that it ran is recorded in path_readiness.json. ``run_blind.py`` refuses any
problem not recorded true there.

WHAT THESE SCRIPTS ARE NOT. They are not the answer to the task and they are not
shipped as participants for the agent to copy. They exist so the operator can
establish that the arrangement is servable at all, in the interpreters this
machine actually has, before a paid run is charged against it.

CONTRACT (the same one data/coupling_participants/* follow, so the walk exercises
the driver the campaign uses): run in the work_dir with no arguments, read
imports.json (written every iteration, ``{}`` on iteration 1), write exports.json
LAST. Everything problem-specific is read from cfg.json rather than edited into
the script, because eight backends x two roles x three physics families is far
too many hand-edited copies to keep consistent.

DEPENDENCIES. numpy only. The source term arrives as a sympy-printed string and
is evaluated with ``eval`` against a numpy namespace, because sympy is not
importable in every one of the six interpreters involved.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

# sympy prints ** for powers and bare names for the elementary functions. The
# manufactured sources here are polynomials except for the transient instance,
# which carries exp.
_NS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp,
    "log": np.log, "sqrt": np.sqrt, "pi": math.pi, "E": math.e,
    "Abs": np.abs, "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
}


def load_cfg(path="cfg.json") -> dict:
    return json.loads(Path(path).read_text())


def make_fun(expr, dim: int, extra=()):
    """A callable f(x, y[, z][, t]) from a sympy-printed string.

    Works on numpy arrays and on floats. ``extra`` names further scalar
    arguments (``t`` for the transient instance) appended after the coordinates.
    """
    names = ["x", "y", "z"][:dim] + list(extra)
    code = compile(f"lambda {', '.join(names)}: ({expr})", "<source>", "eval")
    return eval(code, dict(_NS))


def make_vec_fun(exprs, dim: int, extra=()):
    fns = [make_fun(e, dim, extra) for e in exprs]

    def f(*a):
        return [fn(*a) for fn in fns]
    return f


# ── the interface ─────────────────────────────────────────────────────
def read_imports(partner: str):
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(partner) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, here, fallback, ncomp: int, free_axes):
    """Map the partner's interface samples onto THIS participant's points.

    The driver moves opaque numbers on coordinates and does no interpolation, so
    non-matching interface discretisations are the participant's problem. Three
    cases, in order:

      * the two point sets AGREE to 1e-9 -- take the values straight across.
        This is the normal case in the walk, where both sides use the same
        transverse mesh at each level, and it is exact;
      * one free coordinate -- 1-D linear interpolation along it;
      * two free coordinates (the 3-D instance, whose interface is a PLANE) --
        bilinear interpolation on the partner's structured (u, v) grid. The 1-D
        interpolation every shipped participant uses cannot do this, and the
        failure is silent: np.interp on a scrambled coordinate returns numbers of
        the right length and the coupling still converges.

    Returns an (n, ncomp) array; ``here`` is (n, dim).
    """
    fb = np.atleast_1d(np.asarray(fallback, float)).ravel()
    if fb.size == 1 and ncomp > 1:
        fb = np.repeat(fb, ncomp)
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(here), 1))
    src = np.asarray(imp["coordinates"], float)
    val = np.asarray(imp.get(key) or [], float)
    if val.ndim == 1:
        val = val.reshape(-1, 1)
    if val.shape[0] != src.shape[0] or val.shape[1] != ncomp:
        return np.tile(fb, (len(here), 1))

    a = np.asarray(here, float)[:, free_axes]
    b = src[:, free_axes]
    if a.shape == b.shape and np.allclose(a, b, atol=1e-9):
        return val
    if len(free_axes) == 1:
        o = np.argsort(b[:, 0])
        return np.column_stack([np.interp(a[:, 0], b[o, 0], val[o, c])
                                for c in range(ncomp)])
    return _bilinear(b, val, a, ncomp)


def _bilinear(b, val, a, ncomp):
    """Bilinear interpolation from a structured (u, v) point cloud."""
    us = np.unique(np.round(b[:, 0], 12))
    vs = np.unique(np.round(b[:, 1], 12))
    if us.size * vs.size != b.shape[0]:
        idx = [int(np.argmin(np.sum((b - p) ** 2, axis=1))) for p in a]
        return val[idx]
    iu = np.searchsorted(us, np.round(b[:, 0], 12))
    iv = np.searchsorted(vs, np.round(b[:, 1], 12))
    grid = np.zeros((us.size, vs.size, ncomp))
    grid[iu, iv, :] = val
    out = np.zeros((len(a), ncomp))
    for n, (u, v) in enumerate(a):
        i = int(np.clip(np.searchsorted(us, u) - 1, 0, us.size - 2))
        j = int(np.clip(np.searchsorted(vs, v) - 1, 0, vs.size - 2))
        du = (u - us[i]) / (us[i + 1] - us[i]) if us.size > 1 else 0.0
        dv = (v - vs[j]) / (vs[j + 1] - vs[j]) if vs.size > 1 else 0.0
        du, dv = float(np.clip(du, 0, 1)), float(np.clip(dv, 0, 1))
        out[n] = ((1 - du) * (1 - dv) * grid[i, j]
                  + du * (1 - dv) * grid[i + 1, j]
                  + (1 - du) * dv * grid[i, j + 1]
                  + du * dv * grid[i + 1, j + 1])
    return out


def write_exports(coords, values, fluxes, field_name="field"):
    """exports.json LAST: the driver takes its existence as proof of success."""
    v = np.atleast_2d(np.asarray(values, float))
    q = np.atleast_2d(np.asarray(fluxes, float))
    if v.shape[0] == 1 and len(coords) != 1:
        v = v.T
    if q.shape[0] == 1 and len(coords) != 1:
        q = q.T
    Path("exports.json").write_text(json.dumps({
        "field_name": field_name,
        "n_points": int(len(coords)),
        "coordinates": [[float(c) for c in p] for p in coords],
        "values": [[float(a) for a in row] for row in v],
        "normal_fluxes": [[float(a) for a in row] for row in q],
    }, indent=2))


def write_nodes(path, coords, values):
    """The participant's own nodal field. The walk measures the error against
    the manufactured solution here rather than at the graded probe grid: the
    nodes need no interpolation, so the number reported is the solver's, not an
    interpolator's."""
    v = np.atleast_2d(np.asarray(values, float))
    if v.shape[0] != len(coords):
        v = v.T
    with open(path, "w") as fh:
        dim = len(coords[0])
        fh.write(",".join(["x", "y", "z"][:dim]
                          + [f"u{i}" for i in range(v.shape[1])]) + "\n")
        for p, row in zip(coords, v):
            fh.write(",".join(f"{c:.17g}" for c in list(p) + list(row)) + "\n")


def write_log(cfg, ndof, extra=""):
    """The canonical execution-evidence line.

    blind_eval.evidence needs a line carrying a number the solver computed, and
    it knew a different phrasing per code -- NGSolve's `ndof` matched, dolfinx's
    did not -- so an honest quiet run graded FABRICATED_NO_RUN purely by print
    phrasing. `NDOF = <n>` is the canonical form every task now asks for.
    """
    name = f"run_level{cfg['level']}_{cfg['sidename']}.log"
    Path(name).write_text(
        f"code = {cfg.get('code', '?')}\n"
        f"side = {cfg['sidename']} ({cfg['side']})\n"
        f"NDOF = {int(ndof)}\n" + (extra + "\n" if extra else ""))
    return name


# ── structured meshes, shared by every participant that builds its own ──
def grid_points(extent, n):
    """Tensor-product node coordinates, last axis varying fastest."""
    axes = [np.linspace(lo, hi, k + 1) for (lo, hi), k in zip(extent, n)]
    return axes


def iface_mask(pts, axis, xi, tol=1e-9):
    return np.abs(np.asarray(pts)[:, axis] - xi) < tol


def outward_sign(extent, axis, xi):
    """+1 if the interface is this subdomain's upper face along ``axis``."""
    lo, hi = extent[axis]
    return 1.0 if abs(hi - xi) < abs(lo - xi) else -1.0
