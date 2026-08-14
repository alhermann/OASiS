"""True error per FIELD, the order fit, and every numeric check on them.

Per-field grading is the repair for a measured defect: the pooled, unweighted
RMS over all components let a large field swamp a small one (temperature over
displacement by many orders of magnitude on the thermo-mechanical cells), so a
submission whose displacement never converged still graded CORRECT off the
temperature alone. v2 grades EVERY component against its own reference RMS —
computed from the sealed exact solution on the very probe grid the errors use,
so numerator and denominator live in the same metric — and requires every
field to pass. Per-field orders are reported.

The absolute checks run BEFORE the rate is consulted (an order is a ratio, and
a ratio can be manufactured without solving anything):

* plausibility band on the fitted order, from the key — pre-registered in
  DESIGN.md §5 item 6 and previously dead code (unpacked, never read);
* magnitude of the COARSEST error against the reference RMS (MAX_COARSE_REL);
* NEW: magnitude of the FINEST error (MAX_FINEST_REL) — without it, an order
  at the band floor over 3 levels lets a 54% finest-level error grade CORRECT
  (derivation of the bound in constants.py).
"""
from __future__ import annotations

import math

import sympy as sp

from . import constants as C

X, Y, Z = sp.symbols("x y z", real=True)
SYMS = {"x": X, "y": Y, "z": Z}


def parse_exprs(src, ncomp_hint=None):
    exprs = src if isinstance(src, list) else [src]
    return [sp.sympify(e, locals=SYMS) for e in exprs]


def field_errors(pts, vals, exprs, coords):
    """Per-component (error RMS, reference RMS) over the probe grid, plus the
    pooled sum-over-components RMS the previous grader reported.

    Returns (err[c], ref[c], pooled, ok, why); the reference is the exact
    component's own RMS on the SAME points, so the per-field relative error is
    dimensionless and comparable across fields of any scale.
    """
    fns = [sp.lambdify([SYMS[c] for c in coords], e, "math") for e in exprs]
    nc = len(fns)
    acc_e = [0.0] * nc
    acc_r = [0.0] * nc
    n = 0
    for p, v in zip(pts, vals):
        if len(v) != nc:
            return None, None, None, False, (
                f"row carries {len(v)} components, the key defines {nc}")
        try:
            ex = [float(fn(*p)) for fn in fns]
        except Exception as e:
            return None, None, None, False, f"exact solution not evaluable: {e}"
        if not all(math.isfinite(a) for a in ex):
            return None, None, None, False, "exact solution non-finite at a probe"
        for c in range(nc):
            acc_e[c] += (ex[c] - float(v[c])) ** 2
            acc_r[c] += ex[c] ** 2
        n += 1
    if not n:
        return None, None, None, False, "no rows"
    err = [math.sqrt(a / n) for a in acc_e]
    ref = [math.sqrt(a / n) for a in acc_r]
    pooled = math.sqrt(sum(acc_e) / n)
    return err, ref, pooled, True, "ok"


def fit_order(errs):
    """Order from successive halvings, with monotonicity and fit quality."""
    if len(errs) < 2 or any(e <= 0 for e in errs):
        return None, 0.0, False
    monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    ratios = [math.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    order = sum(ratios) / len(ratios)
    le = [math.log(e) for e in errs]
    n = len(le)
    xs = list(range(n))
    mx, me = sum(xs) / n, sum(le) / n
    sxx = sum((a - mx) ** 2 for a in xs)
    slope = (sum((a - mx) * (b - me) for a, b in zip(xs, le)) / sxx) if sxx else 0.0
    ss_tot = sum((b - me) ** 2 for b in le)
    ss_res = sum((b - (me + slope * (a - mx))) ** 2 for a, b in zip(xs, le))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return order, r2, monotone


def check_field(name: str, errs: list, refs: list, theo: float, tol: float,
                band) -> dict:
    """Every numeric check for one field. Returns the field's record with
    `reasons` — the checks its NUMBERS failed — and `superconvergent`."""
    rec = {"field": name, "errors": list(errs), "reasons": [],
           "superconvergent": False}
    order, r2, monotone = fit_order(errs)
    rec.update(order=order, r2=(round(r2, 4) if order is not None else None),
               monotone=monotone)
    if order is None:
        rec["reasons"].append(f"ORDER_NOT_FITTABLE({name})")
        return rec
    if not monotone or r2 < C.MIN_R2:
        rec["reasons"].append(f"DECAY_NOT_CLEAN({name})")

    lo, hi = float(band[0]), float(band[1])
    if not (lo <= order <= hi):
        rec["reasons"].append(f"IMPLAUSIBLE_RATE({name})")

    ref = max(refs)
    if ref <= C.ZERO_REF_FLOOR:
        rec["magnitude_checked"] = False
        rec["note"] = (f"magnitude NOT CHECKED for {name}: the exact "
                       f"component's own RMS is zero on the probe grid, so a "
                       f"relative bound is undefined")
    else:
        rec["magnitude_checked"] = True
        rec["rel_coarse"] = errs[0] / ref
        rec["rel_finest"] = errs[-1] / ref
        if rec["rel_coarse"] > C.MAX_COARSE_REL:
            rec["reasons"].append(f"IMPLAUSIBLE_MAGNITUDE_COARSE({name})")
        if rec["rel_finest"] > C.MAX_FINEST_REL:
            rec["reasons"].append(f"IMPLAUSIBLE_MAGNITUDE_FINEST({name})")

    if not rec["reasons"]:
        if abs(order - theo) <= tol:
            pass
        elif order > theo + tol:
            rec["superconvergent"] = True
        else:
            rec["reasons"].append(f"UNDER_CONVERGED({name})")
    return rec
