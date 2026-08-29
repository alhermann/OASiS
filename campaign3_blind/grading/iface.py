"""The interface phase — the quantities that move when the coupling breaks.

Measured (DESIGN.md Amendment 2 §2): every interface-mechanism mutation of a
partitioned coupling still self-converges at order ~1.85, indistinguishable
from the correct run, while the two-sided flux jump goes from roundoff (~1e-14)
to O(1). So for a coupled cell the jump is not a diagnostic to report — it is a
GATE, and this module is where it closes.

Repairs carried by this rebuild:

* **Per-leg normal axis.** The previous phase hardcoded axis 0 and applied the
  single `iface_graded_band` to every tangential coordinate. On D5 — whose
  interface is a bent polyline with one leg normal to x and one normal to y —
  that rejects every leg-2 point of a CORRECT submission as "outside the
  graded band". The interface is now a list of LEGS, each with its own normal
  axis, position and graded band, from structured spec data
  (`iface_legs` | `interface_axis`+`interface_value` | extents that touch);
  a coupled spec from which no leg can be derived is a hard error, never a
  guess.

* **The graded band is honoured per leg, in both directions.** Points at the
  interface ENDS sit on Dirichlet-Neumann corners where the recovered flux
  gets WORSE under refinement (measured 2.11x -> 2.51x over a 4x refinement):
  grading them fails a correct submission. Points OUTSIDE the band are refused
  just as firmly: excluding the ends must not become a way for the agent to
  choose where it is measured.

* **NOT CHECKED is not PASSED.** An interface submission that is missing,
  empty, without flux columns, or carrying identically-zero fluxes balances
  trivially and proves nothing; each of those cases is surfaced as an explicit
  finding on the cell (the same discipline as
  `core.quality_checks.check_interface_balance`, whose silent-pass siblings
  were closed on 2026-08-11), and none of them lets the cell grade CORRECT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import constants as C
from .loading import GraderConfigError, interface_mod

IFACE_FILE = re.compile(r"interface_level(\d+)_([ABab])\.csv$")
_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Leg:
    axis: int          # coordinate the leg is a level set of
    value: float       # the level-set value
    band: tuple | None  # graded band on the tangential coordinate(s)


def _num(v) -> float:
    """Spec numbers arrive as floats or as fraction strings ('5/8')."""
    if isinstance(v, (int, float)):
        return float(v)
    return float(Fraction(str(v).strip()))


def interface_legs(spec: dict, key: dict, dim: int) -> list[Leg]:
    """The structured description of where the interface is graded.

    Priority: explicit `iface_legs` in the public spec; then the C-series
    `interface_axis`/`interface_value` pair; then a single leg derived from
    key extents that touch on exactly one axis. Anything else refuses — the
    grader must never guess geometry (that is how probe grids end up built
    over the wrong region).
    """
    legs = spec.get("iface_legs")
    if legs:
        out = []
        for i, leg in enumerate(legs):
            try:
                band = leg.get("band")
                out.append(Leg(int(leg["axis"]), _num(leg["value"]),
                               tuple(float(b) for b in band) if band else None))
            except (KeyError, TypeError, ValueError) as ex:
                raise GraderConfigError(
                    f"iface_legs[{i}] is malformed ({ex}); each leg needs "
                    f"'axis', 'value' and optionally 'band'")
        return out

    band = spec.get("iface_graded_band")
    band = tuple(float(b) for b in band) if band else None

    axis_name = spec.get("interface_axis")
    if axis_name is not None and spec.get("interface_value") is not None:
        axis = _AXIS.get(str(axis_name).strip().lower())
        if axis is None or axis >= dim:
            raise GraderConfigError(
                f"interface_axis {axis_name!r} is not a coordinate of this "
                f"{dim}D problem. A bent interface (e.g. a polyline) cannot "
                f"be described by one axis: record `iface_legs` in the public "
                f"spec — a list of {{axis, value, band}} per leg, matching "
                f"the INTERFACE PROBE POINTS the task text states.")
        return [Leg(axis, _num(spec["interface_value"]), band)]

    ea, eb = key.get("extent_a"), key.get("extent_b")
    if ea and eb:
        touching = [i for i in range(dim)
                    if abs(ea[i][1] - eb[i][0]) < 1e-9
                    or abs(eb[i][1] - ea[i][0]) < 1e-9]
        if len(touching) == 1:
            i = touching[0]
            value = ea[i][1] if abs(ea[i][1] - eb[i][0]) < 1e-9 else ea[i][0]
            return [Leg(i, float(value), band)]

    raise GraderConfigError(
        "no structured interface description: the spec carries neither "
        "`iface_legs` nor `interface_axis`/`interface_value`, and the key "
        "extents do not touch on exactly one axis. A bent or non-adjacent "
        "interface must be described in spec_public.json (public: the task "
        "text already prints it); the grader will not guess where the "
        "coupling is graded.")


def iface_column_counts(spec: dict, dim: int, ncomp: int) -> tuple[int, int]:
    """(n values, n fluxes) per interface row, from the public `iface_header`
    (e.g. 'x, y, T, ux, uy, qn, tx, ty'). Without a header the legacy layout
    (ncomp values then ncomp fluxes) is assumed."""
    header = spec.get("iface_header")
    if not header:
        return ncomp, ncomp
    names = [c.strip() for c in str(header).split(",") if c.strip()]
    nflux = len(names) - dim - ncomp
    if nflux < 1:
        raise GraderConfigError(
            f"iface_header {header!r} has {len(names)} columns, which cannot "
            f"hold {dim} coordinates + {ncomp} components + a flux")
    return ncomp, nflux


def _assign_to_leg(pts, legs):
    """point index -> leg index by |p[axis] - value| <= PROBE_TOL, or None."""
    out = []
    for p in pts:
        hit = None
        for li, leg in enumerate(legs):
            if abs(p[leg.axis] - leg.value) <= C.PROBE_TOL:
                hit = li
                break
        out.append(hit)
    return out


def interface_phase(work: Path, spec: dict, key: dict, dim: int,
                    ncomp: int, mesh_N, legs: list | None = None) -> dict:
    """Grade the interface submission. Returns
        verdict   INTERFACE_SATISFIED | INTERFACE_NOT_SATISFIED | NOT_CHECKED
        malformed list of contract violations (missing/empty/short files,
                  stray points) — these make the SUBMISSION malformed
        reasons   machine-readable gate reasons when not satisfied
        findings  explicit NOT-CHECKED / refusal findings (never silent)
        per_level worst jumps per graded level

    `legs` is normally validated up front by the orchestrator (a spec from
    which no leg can be derived is a CELL defect and must halt grading before
    any submission is read); it is derived here only when called standalone.
    """
    IF = interface_mod()
    legs = legs if legs is not None else interface_legs(spec, key, dim)
    nval, nflux = iface_column_counts(spec, dim, ncomp)

    out = {"verdict": "NOT_CHECKED", "malformed": [], "reasons": [],
           "findings": [], "per_level": [], "legs": [
               {"axis": leg.axis, "value": leg.value, "band": leg.band}
               for leg in legs]}

    lvls: dict[int, dict[str, Path]] = {}
    # RECURSIVE, like submission._submission_candidates. This was a
    # non-recursive glob while the solution-file discovery next door used
    # rglob — the exact asymmetry submission.py's docstring records as fixed
    # for solutions and never fixed here. Measured consequence: of the 5
    # coupled cells booked "no interface_level files", 4 HAD submitted them
    # (into results/, output/, level1/), and one of those four satisfies the
    # interface gate. Agents organise their output into subdirectories; the
    # grader must find it where the rest of the grader already looks.
    for f in sorted(work.rglob("interface_level*.csv")):
        m = IFACE_FILE.match(f.name)
        if m:
            lvls.setdefault(int(m.group(1)), {})[m.group(2).upper()] = f

    nlevels = len(mesh_N or [])
    if not lvls:
        out["malformed"].append(
            "no interface_level<k>_<side>.csv files: the task requires them, "
            "and the interface flux is the only reference-free quantity that "
            "can detect convergence to the wrong transmission condition")
        out["findings"].append(
            "Interface NOT CHECKED: nothing was submitted for it. This is "
            "not a passing check — nothing was compared.")
        return out
    missing_levels = [k for k in range(1, nlevels + 1)
                      if set(lvls.get(k, {})) != {"A", "B"}]
    if missing_levels:
        out["malformed"].append(
            f"interface files missing or one-sided for level(s) "
            f"{missing_levels}: every prescribed level needs both "
            f"interface_level<k>_A.csv and _B.csv")

    zero_flux_levels, graded = [], []
    for lvl in sorted(lvls):
        sides = lvls[lvl]
        if set(sides) != {"A", "B"}:
            continue
        parsed = {}
        for s, p in sorted(sides.items()):
            got, why = IF.read_interface_csv(p, dim, nval, nflux)
            if got is None:
                out["malformed"].append(f"{p.name}: {why}")
                break
            parsed[s] = got
        if len(parsed) != 2:
            continue

        # the grader owns the interface evaluation set: every point must lie
        # ON a declared leg and INSIDE that leg's graded band
        stray = []
        assignment = {}
        for s, (pts, _v, _q) in parsed.items():
            hits = _assign_to_leg(pts, legs)
            assignment[s] = hits
            for p, hit in zip(pts, hits):
                if hit is None:
                    stray.append((s, p, "on no declared interface leg"))
                    continue
                leg = legs[hit]
                if leg.band is None:
                    continue
                lo, hi = leg.band[0] - 1e-9, leg.band[1] + 1e-9
                tang = [c for i, c in enumerate(p) if i != leg.axis]
                if any(not (lo <= c <= hi) for c in tang):
                    stray.append((s, p, f"outside the graded band {leg.band}"))
        if any(leg.band is None for leg in legs):
            out["findings"].append(
                "IFACE BAND NOT RECORDED for at least one leg: the interface "
                "ends (Dirichlet-Neumann corners) are not being excluded for "
                "this cell; record iface_graded_band / iface_legs bands in "
                "the public spec")
        if stray:
            out["malformed"].append(
                f"level {lvl}: {len(stray)} interface point(s) refused, e.g. "
                f"{stray[0][1]} ({stray[0][2]}). The interface ends are "
                f"corners of the split problem and are deliberately not "
                f"graded; points elsewhere than the prescribed probes would "
                f"let the agent choose where it is measured.")
            continue

        # all-zero flux balances trivially and proves nothing — the signature
        # of a transfer that never ran, a wrong field name, or an unfilled
        # buffer. NOT CHECKED, and said so.
        qa = [x for _p, _v, q in [parsed["A"]] for row in q for x in row]
        qb = [x for _p, _v, q in [parsed["B"]] for row in q for x in row]
        if qa and qb and not any(abs(x) > 0 for x in qa + qb):
            zero_flux_levels.append(lvl)
            out["findings"].append(
                f"Interface flux balance NOT CHECKED at level {lvl}: both "
                f"sides exported fluxes that are identically zero. That "
                f"balances trivially and says nothing about the coupling.")
            continue

        # per-leg two-sided jumps at matched points with opposite outward
        # normals: continuity means u_A - u_B = 0 and q_A + q_B = 0
        worst_u = worst_q = 0.0
        leg_ok = True
        for li in range(len(legs)):
            sub = {}
            for s, (pts, vals, flux) in parsed.items():
                keep = [i for i, hit in enumerate(assignment[s]) if hit == li]
                sub[s] = ([pts[i] for i in keep], [vals[i] for i in keep],
                          [flux[i] for i in keep])
            if not sub["A"][0] and not sub["B"][0]:
                continue
            d, why = IF.two_sided_jumps(sub["A"], sub["B"])
            if d is None:
                out["malformed"].append(f"level {lvl}, leg {li + 1}: {why}")
                leg_ok = False
                break
            worst_u = max(worst_u, d["jump_u_rel"])
            worst_q = max(worst_q, d["jump_q_rel"])
        if not leg_ok:
            continue
        graded.append({"level": lvl, "jump_u_rel": worst_u,
                       "jump_q_rel": worst_q})

    out["per_level"] = graded
    if not graded:
        if zero_flux_levels:
            out["reasons"].append("INTERFACE_FLUX_ALL_ZERO")
            out["verdict"] = "NOT_CHECKED"
        return out

    if zero_flux_levels:
        out["reasons"].append("INTERFACE_FLUX_ALL_ZERO")
    # THE GATE MUST TEST WHAT ITS OWN MESSAGE CLAIMS: a jump that STAYS O(1)
    # under refinement. It took the MAX over levels against a fixed tolerance,
    # which is a different test and a much worse one.
    #
    # Why it is worse. When both sides recover the interface flux from their
    # assembled system, the two exported arrays differ by the consistent-to-
    # nodal conversion of the P1 boundary mass matrix, so the graded jump is
    # h^2 |q''| / (6|q|) — a MESH RULER with no physics in it. Two independent
    # reviews demonstrated the consequence: a coupling with one side's
    # conductivity 4x WRONG fails at n=8 and n=16 and PASSES at n=32, purely
    # on resolution. And a correct coupling whose tractions are evaluated by
    # direct stress sampling — which is what the task text describes, "the
    # traction evaluated from that subdomain's own solution" — is O(h) at the
    # interface and fails a fixed 5e-3 while converging perfectly well.
    #
    # Measured in this campaign's own grades: 3 coupled cells failed SOLELY on
    # this gate, and 2 of the 3 had a flux jump decreasing at every level
    # (C8 seed 4, BARE: 3.3% -> 2.0% -> 1.1%; MCP: 8.9% -> 3.6% -> 1.5%).
    # Those are converging couplings graded as unphysical.
    #
    # It also mattered asymmetrically: the consistent recovery that passes a
    # fixed tolerance is described in the OASiS payload and nowhere in the task
    # text, so a grading-critical rule was published to one arm. Testing for
    # non-convergence instead removes that, because both recoveries converge.
    #
    # THE REPLACEMENT. Fail when the jump does not shrink under refinement,
    # which is precisely the mutation signature the gate was calibrated on
    # (mutations sit at 0.75-3.0 and flat; a correct run is at roundoff or
    # decreasing). A single level cannot show a trend, so there the fixed
    # tolerance still applies — it is all the evidence there is.
    worst_u = max(g["jump_u_rel"] for g in graded)
    worst_q = max(g["jump_q_rel"] for g in graded)

    def _shrinks(key):
        """Does the jump fall by a clear factor across the sequence?"""
        v = [g[key] for g in graded]
        if len(v) < 2:
            return None                      # no trend available
        if max(v) <= C.IFACE_JUMP_TOL:
            return True                      # at roundoff/tolerance throughout
        first, last = v[0], v[-1]
        if first <= 0:
            return last <= C.IFACE_JUMP_TOL
        # h halves per level, so a genuine O(h) recovery falls ~2x per level.
        # Require a clear overall fall, not a per-step monotone chain, so that
        # one noisy level does not condemn a converging sequence — AND a floor
        # on where it lands, because decay alone admitted a 49% mismatch.
        return (last < first * (C.IFACE_JUMP_DECAY ** (len(v) - 1))
                and last <= C.IFACE_JUMP_CEILING)

    su, sq = _shrinks("jump_u_rel"), _shrinks("jump_q_rel")
    if su is None or sq is None:
        bad = worst_u > C.IFACE_JUMP_TOL or worst_q > C.IFACE_JUMP_TOL
        why = (f"only {len(graded)} graded level(s), so no refinement trend "
               f"exists and the fixed tolerance {C.IFACE_JUMP_TOL:g} is all "
               f"the evidence there is")
    else:
        bad = not (su and sq)
        _u = ", ".join("%.3e" % g["jump_u_rel"] for g in graded)
        _q = ", ".join("%.3e" % g["jump_q_rel"] for g in graded)
        why = (f"the jump does not shrink under refinement "
               f"(field [{_u}], flux [{_q}])")
    # A BIT-EXACT ZERO FLUX JUMP IS NOT A PERFECT COUPLING. IT IS ONE FIELD.
    #
    # Two independently solved subdomains cannot agree on the interface flux to
    # the last bit. Exactly 0.0 means both sides were evaluated from the same
    # solution — the signature of a monolithic solve reported as a partitioned
    # one. C7_27b_BARE_seed2 scored 0.0 for field AND flux at all three levels
    # and graded CORRECT on that basis; its own notes say the coupling was
    # "simulated based on the monolithic solution".
    #
    # Only the FLUX carries this meaning. A zero FIELD jump is ordinary and must
    # not be flagged: Dirichlet-Neumann sets one side's interface displacement
    # to the other's, so field agreement can be exact by construction. Measured
    # across the 60 graded interface levels in the tree: jump_u_rel is exactly
    # zero 12 times, jump_q_rel exactly 3 times — and those 3 are this one
    # fabricated run.
    exact_zero_flux = [g["level"] for g in graded if g["jump_q_rel"] == 0.0]
    if exact_zero_flux and not bad:
        out["verdict"] = "INTERFACE_NOT_SATISFIED"
        out["reasons"].append("INTERFACE_NOT_SATISFIED")
        out["findings"].append(
            f"the relative FLUX jump is exactly 0.0 at level(s) "
            f"{exact_zero_flux}. Two subdomains solved separately cannot agree "
            f"on the interface flux to the last bit; a bit-exact zero means "
            f"both sides were read off one field, so there is no evidence of "
            f"two solves. This is not a tolerance being met — it is the "
            f"quantity being absent.")
        return out

    if bad:
        out["verdict"] = "INTERFACE_NOT_SATISFIED"
        out["reasons"].append("INTERFACE_NOT_SATISFIED")
        out["findings"].append(
            f"the two subdomains disagree at the interface: max relative "
            f"field jump {worst_u:.3e}, max relative FLUX jump {worst_q:.3e}. "
            f"{why}. A jump that stays O(1) under refinement means the scheme "
            f"converged to a fixed point of the wrong transmission condition "
            f"— which the observed order cannot see, because convergence to a "
            f"wrong answer is still convergence.")
    elif zero_flux_levels:
        out["verdict"] = "NOT_CHECKED"
    else:
        out["verdict"] = "INTERFACE_SATISFIED"
    return out
