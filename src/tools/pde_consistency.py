"""Does the submitted field actually satisfy the equation the task stated?

WHY THIS EXISTS. Measured over 464 single-code runs, among
submissions with a complete level set the SELF-convergence order — computed
from the agent's own numbers, no reference — has a median of 1.96 (bare) and
1.99 (OASiS). The discretisations converge cleanly. So a graded order near zero
is almost never the finite element method failing to converge; it is a field
converging beautifully TO THE WRONG FUNCTION (7% of all runs), or a field whose
overall size is wrong by orders of magnitude (11-30% of submissions).

A refinement study cannot see either one, and neither can the agent's own
verdict: MESH_INDEPENDENCE = NOT_CONVERGED catches about three quarters of the
wrong runs but also fires on HALF the correct ones, so on its own it is close to
uninformative.

WHAT THIS CHECKS, AND WHY IT IS LEGAL UNDER BLIND GRADING. For any smooth test
function v that vanishes on the boundary, a field u solving

    L u = -div(K grad u) = f          with u prescribed on the boundary

satisfies the weak identity

    integral of u * (L* v)  ==  integral of f * v

where L* is the adjoint, and L* = L for symmetric constant K. Every ingredient
is PUBLIC: the operator and the source come from the task text, and the values
come from the agent's own submission. No exact solution is used, none is
revealed, and nothing here can be run backwards to obtain one — a single
scalar identity per test function cannot reconstruct a field.

MEASURED SEPARATION, by execution on 26 anisotropic-Poisson runs, whose probe grid
is a 44x44 midpoint rule so the quadrature is exact to O(h^2):

    solves the stated PDE   2.32e-02 -> 5.27e-03 -> 1.23e-03 -> 2.56e-04
    does not                6.25e+00 -> 6.37e+00 -> 6.40e+00 -> 6.40e+00
    does not (other cause)  3.13e+00 -> 3.01e+00 -> 2.96e+00 -> 2.93e+00

Four orders of magnitude apart at the finest level, and the failing ones are
flat, which is the signature: a wrong operator, a wrong source, or a source
evaluated in the wrong coordinate frame leaves a residual that refinement
cannot remove.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class LevelResult:
    level: int
    n_points: int
    residual: float
    detail: str = ""


@dataclass
class ConsistencyResult:
    levels: list = field(default_factory=list)
    rate: float | None = None
    verdict: str = "UNKNOWN"
    explanation: str = ""

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "explanation": self.explanation,
            "observed_rate": self.rate,
            "levels": [{"level": r.level, "points": r.n_points,
                        "relative_weak_residual": r.residual,
                        "detail": r.detail} for r in self.levels],
        }


def _detect_midpoint_grid(coords: list) -> tuple:
    """Is this a tensor grid of cell midpoints, and what is the cell volume?

    The quadrature below is a midpoint rule, which is second-order accurate on
    exactly this arrangement and meaningless on a scatter. Returning the weight
    ONLY when the arrangement is right is what keeps a number from being
    reported that the method cannot support.
    """
    axes = []
    for d in range(len(coords[0])):
        vals = sorted({round(p[d], 12) for p in coords})
        if len(vals) < 2:
            return None, "an axis carries a single distinct coordinate"
        steps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
        h = sum(steps) / len(steps)
        if h <= 0 or max(abs(s - h) for s in steps) > 1e-9 * max(h, 1e-30):
            return None, "the points are not uniformly spaced along every axis"
        axes.append((vals, h))
    if len(coords) != math.prod(len(v) for v, _ in axes):
        return None, "the points do not fill a full tensor grid"
    weight = 1.0
    for _, h in axes:
        weight *= h
    return weight, ""


def _adjoint_of_v(kind: str, coeff, pts, box):
    """L* v and v at the given points, for v that vanishes on the box.

    v is a product of half-period sines over the box, so it is smooth and zero
    on every face. For -div(K grad .) with constant symmetric K the operator is
    self-adjoint, and

        L* v = - sum_ij K_ij d2v/dx_i dx_j
    """
    import numpy as np
    dim = len(box)
    xs = [np.asarray([p[d] for p in pts], dtype=float) for d in range(dim)]
    Ls = [float(hi - lo) for lo, hi in box]
    args = [math.pi * (xs[d] - box[d][0]) / Ls[d] for d in range(dim)]
    sin = [np.sin(a) for a in args]
    cos = [np.cos(a) for a in args]
    v = np.ones_like(xs[0])
    for s in sin:
        v = v * s
    # second derivatives of the product of sines
    def d2(i, j):
        out = np.ones_like(xs[0])
        for d in range(dim):
            if d == i and d == j:
                out = out * (-(math.pi / Ls[d]) ** 2) * sin[d]
            elif d == i or d == j:
                out = out * (math.pi / Ls[d]) * cos[d]
            else:
                out = out * sin[d]
        return out
    K = _as_tensor(coeff, dim)
    Lv = np.zeros_like(xs[0])
    for i in range(dim):
        for j in range(dim):
            if K[i][j] != 0.0:
                Lv = Lv - K[i][j] * d2(i, j)
    return v, Lv


def _as_tensor(coeff, dim: int) -> list:
    if isinstance(coeff, (int, float)):
        return [[float(coeff) if i == j else 0.0 for j in range(dim)]
                for i in range(dim)]
    rows = [[float(c) for c in row] for row in coeff]
    if len(rows) != dim or any(len(r) != dim for r in rows):
        raise ValueError(f"coefficient tensor is not {dim}x{dim}")
    for i in range(dim):
        for j in range(i + 1, dim):
            if abs(rows[i][j] - rows[j][i]) > 1e-12 * max(1.0, abs(rows[i][j])):
                raise ValueError(
                    "the coefficient tensor is not symmetric; this check "
                    "assumes a self-adjoint operator, so a non-symmetric K "
                    "needs the true adjoint and is refused rather than "
                    "answered wrongly")
    return rows


def _eval_source(expr: str, pts, dim: int):
    """Evaluate the task's source expression at the points, in PYTHON syntax."""
    import numpy as np
    names = ["x", "y", "z"][:dim]
    env = {n: np.asarray([p[d] for p in pts], dtype=float)
           for d, n in enumerate(names)}
    env.update({"np": np, "pi": math.pi, "sin": np.sin, "cos": np.cos,
                "exp": np.exp, "log": np.log, "sqrt": np.sqrt,
                "tan": np.tan, "tanh": np.tanh, "sinh": np.sinh,
                "cosh": np.cosh, "abs": np.abs, "Abs": np.abs})
    out = eval(expr, {"__builtins__": {}}, env)              # noqa: S307
    return np.asarray(out, dtype=float) * np.ones_like(env[names[0]])


def check_levels(levels: dict, source_expr: str, coefficient,
                 box: list) -> ConsistencyResult:
    """levels maps a level number to a list of (coords..., u) rows."""
    import numpy as np
    res = ConsistencyResult()
    for lvl in sorted(levels):
        rows = levels[lvl]
        if not rows:
            res.levels.append(LevelResult(lvl, 0, float("nan"),
                                          "no rows"))
            continue
        dim = len(rows[0]) - 1
        pts = [r[:dim] for r in rows]
        u = np.asarray([r[dim] for r in rows], dtype=float)
        if not np.all(np.isfinite(u)):
            res.levels.append(LevelResult(
                lvl, len(rows), float("nan"),
                "the submitted field carries a non-finite value"))
            continue
        weight, why = _detect_midpoint_grid(pts)
        if weight is None:
            res.levels.append(LevelResult(lvl, len(rows), float("nan"), why))
            continue
        v, Lv = _adjoint_of_v("poisson", coefficient, pts, box)
        f = _eval_source(source_expr, pts, dim)
        lhs = float(np.sum(u * Lv) * weight)
        rhs = float(np.sum(f * v) * weight)
        denom = max(abs(rhs), 1e-300)
        res.levels.append(LevelResult(lvl, len(rows), abs(lhs - rhs) / denom,
                                      f"lhs={lhs:.6e} rhs={rhs:.6e}"))
    good = [r for r in res.levels if math.isfinite(r.residual)]
    if len(good) < 2:
        res.verdict = "NOT_APPLICABLE"
        res.explanation = (
            "fewer than two levels could be checked. This test needs the "
            "prescribed probe points, which form a uniform tensor grid of cell "
            "midpoints; on a scatter the midpoint quadrature has no accuracy "
            "and no number is reported rather than a misleading one."
            + (" " + good[0].detail if good else "")
            + " " + "; ".join(r.detail for r in res.levels if r.detail)[:300])
        return res
    first, last = good[0].residual, good[-1].residual
    if last > 0 and first > 0 and len(good) >= 2:
        span = len(good) - 1
        res.rate = math.log(first / last) / (span * math.log(2.0)) if last else None
    if last < first / 3.0:
        res.verdict = "CONSISTENT"
        res.explanation = (
            f"the weak residual falls {first:.3e} -> {last:.3e} across "
            f"{len(good)} levels (rate {res.rate:.2f} per refinement). Your "
            f"field satisfies the equation you were given, so a remaining "
            f"error is discretisation, not a wrong model.")
    else:
        res.verdict = "INCONSISTENT"
        res.explanation = (
            f"the weak residual is FLAT: {first:.3e} -> {last:.3e} over "
            f"{len(good)} levels. Refinement cannot remove it, so the field is "
            f"converging to something that is not the solution of the stated "
            f"problem. Look at the source term first — a sign, a missing term, "
            f"or an expression evaluated in element-local instead of global "
            f"coordinates — then the coefficient, then which boundary carries "
            f"which condition. A run that reports this honestly scores above "
            f"one that submits it as converged.")
    return res
