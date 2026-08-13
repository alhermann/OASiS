#!/usr/bin/env python
"""Blind-evaluation grader, v2 — the rebuild from DESIGN.md and three days of
patches, as ONE coherent piece.

Runs OUTSIDE the agent, after the fact, and is the only component that opens a
sealed key. The central principle is unchanged:

    THE GRADER DEFINES THE EVALUATION SET, NOT THE AGENT.

What is different from v1 is not the principle but the discipline: every check
that history added as a patch is a named check here, with a test in
`tests/test_grader_v2_fires.py` that constructs the defective input and proves
the check FIRES. A gate nobody has watched fire is not a gate.

The checks, in the order they run (integrity strictly before any comparison
against truth):

  0. the CELL itself is coherent — spec_public.json exists (never a 2-D
     default), key and spec agree, and the probe grid the task text states IS
     the grid graded (else GraderConfigError: a broken cell halts grading and
     is never booked against the agent);
  1. HONEST_INCOMPLETE — `COULD_NOT_COMPLETE` at the start of a line, with or
     without the explanation the task asks for, and no solution files;
  2. FAILED — no usable solution files;
  3. FABRICATED_NO_RUN — no structured execution evidence for a named code
     (canonical `NDOF = <n>` contract line or per-code solver signatures,
     from this checkout's blind_eval.evidence), or no converging
     partitioned-residual history for a coupled cell, or run logs the task
     prescribes are missing;
  4. MALFORMED_SUBMISSION — the contract was not followed: wrong probe grid
     (exclusions included), unparsable or non-finite values, wrong level
     count, missing subdomain files, refused interface points, or an NDOF
     sequence that is not the prescribed halving;
  5. the numbers — per-FIELD true error against the sealed solution, order
     fit with monotonicity and R², the pre-registered plausibility band,
     magnitude bounds on the coarsest AND finest levels, and for coupled
     cells the per-leg two-sided interface jumps as a gate;
  6. the label — CORRECT / CORRECT_SUPERCONVERGENT when every field passes;
     otherwise the numbers say the answer is wrong and the agent's claim
     decides only the adverb: CONFIDENTLY_WRONG on any affirmative
     convergence claim, COMPLETED_UNPHYSICAL without one.

Cells graded band-only (grade 3) or against a monolithic reference (grade 2)
take their own paths with their own verdict vocabularies and their own result
TYPES, so they cannot be pooled with grade-1 results even by accident;
`grading.outcomes.aggregate` refuses a mixed list outright.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ── the grading package, pinned to this checkout ──────────────────────────
# Loaded by absolute file path under a unique name: nothing another worktree
# or an installed package puts on sys.path can shadow any part of the grader.
_PKG = "c3grading"
if _PKG not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _PKG, HERE / "grading" / "__init__.py",
        submodule_search_locations=[str(HERE / "grading")])
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_PKG] = _mod
    _spec.loader.exec_module(_mod)

from c3grading import constants as C                              # noqa: E402
from c3grading import checks, evidence2, iface, outcomes, probes  # noqa: E402
from c3grading import loading, submission as sub                  # noqa: E402
from c3grading.loading import GraderConfigError                   # noqa: E402

# Re-exported so the task-text agreement tests (ported from v1) and the
# two-phase wrapper address one authority. These ARE constants.* — not copies.
PROBE_M = C.PROBE_M
PROBE_TOL = C.PROBE_TOL
MIN_R2 = C.MIN_R2
MAX_COARSE_REL = C.MAX_COARSE_REL
MAX_FINEST_REL = C.MAX_FINEST_REL
probe_grid = probes.probe_grid
matches_probe_grid = probes.matches_probe_grid
subdomain_bounds = probes.subdomain_bounds


def probe_exclusions(problem_id: str, side: str, problems_dir=None):
    """Excluded probe boxes from the PUBLIC spec (compat signature)."""
    try:
        spec = loading.load_spec(problem_id, problems_dir)
    except GraderConfigError:
        return None
    return probes.probe_exclusions(spec, side)


# ── the grader ────────────────────────────────────────────────────────────
def grade_run(run_dir: Path, problem_id: str, *, passphrase: str | None = None,
              key: dict | None = None, problems_dir=None,
              keys_dir=None) -> dict:
    run_dir = Path(run_dir)
    work = run_dir / "work"

    # 0. the cell itself, before any submission is touched
    spec = loading.load_spec(problem_id, problems_dir)
    task_txt = loading.load_task(problem_id, problems_dir)
    key = key if key is not None else loading.load_key(
        problem_id, keys_dir, passphrase)
    loading.crosscheck_spec_key(problem_id, spec, key)
    grade, grade_notes = loading.evidence_grade(problem_id, key, spec)

    dim = spec["dim"]
    kind = key.get("kind", spec.get("kind", "single"))
    coupled = kind == "coupled"
    codes = key.get("codes") or spec.get("codes") or \
        [key.get("code") or spec.get("code")]
    if not codes or codes == [None]:
        raise GraderConfigError(f"{problem_id}: neither key nor spec names "
                                f"the code(s) under test")
    mesh_N = key.get("mesh_N") or spec.get("mesh_N") or []
    iface_tol = loading.parse_rel_tol(spec.get("interface_tol"))

    result_txt = sub.result_text(run_dir, work)
    honest = sub.could_not_complete(result_txt)
    claim = sub.claims_convergence(result_txt)

    mode = key.get("grading")
    if mode in ("band-only", "reference"):
        needs_coupling_history = coupled and "residual_level" in task_txt
        evid = evidence2.assess_execution(
            work, codes, needs_coupling_history, task_txt, mesh_N, dim,
            iface_tol, sub.claimed_iterations(result_txt))
        fn = (outcomes.grade_band_only if mode == "band-only"
              else outcomes.grade_reference)
        return fn(problem_id, key, spec, result_txt, evid, honest, grade,
                  grade_notes).to_dict()
    if mode not in (None, "order", "exact"):
        raise GraderConfigError(
            f"{problem_id}: unknown grading mode {mode!r} in the key")

    # ── grade-1/order path ────────────────────────────────────────────
    for field_name in ("theoretical_order", "tol", "band"):
        if key.get(field_name) is None:
            raise GraderConfigError(
                f"{problem_id}: the key carries no {field_name!r}; an "
                f"order-graded cell cannot be graded without it")
    probes.task_grid_agreement(problem_id, task_txt, spec, key)
    probes.assert_probe_grid_incommensurate(dim, mesh_N)
    # a coupled cell whose interface cannot be placed is broken as a CELL:
    # validate the structured interface description before any submission is
    # read, so the defect halts grading instead of becoming an outcome
    iface_legs = iface.interface_legs(spec, key, dim) if coupled else None

    res = outcomes.OrderResult(
        problem=problem_id, kind=kind, evidence_grade=grade, outcome="",
        theoretical_order=key["theoretical_order"], tol=key["tol"],
        band=list(key["band"]), notes=list(grade_notes),
        agent_claim={
            "MESH_INDEPENDENCE": sub.result_field(result_txt,
                                                  "MESH_INDEPENDENCE"),
            "claims_convergence": claim,
            "MAX_REL_CHANGE": sub.result_field(result_txt, "MAX_REL_CHANGE"),
            **({"INTERFACE_RESIDUAL": sub.result_field(result_txt,
                                                       "INTERFACE_RESIDUAL"),
                "COUPLING_ITERATIONS": sub.result_field(
                    result_txt, "COUPLING_ITERATIONS")} if coupled else {})})

    def finish(outcome, *reasons, note=None):
        res.outcome = outcome
        res.reasons.extend(reasons)
        if note:
            res.notes.append(note)
        return res.to_dict()

    # 1./2. honesty before anything else
    levels, discovery_problems = sub.discover_levels(work, coupled)
    if honest and not levels:
        return finish("HONEST_INCOMPLETE", "COULD_NOT_COMPLETE_DECLARED")
    if not levels:
        return finish("FAILED", "NO_SOLUTION_FILES",
                      note="no usable solution files")

    # 3. execution evidence, before any comparison against truth
    evid = evidence2.assess_execution(
        work, codes, coupled, task_txt, mesh_N, dim, iface_tol,
        sub.claimed_iterations(result_txt))
    res.evidence = evid
    if evid["fatal"]:
        return finish(evid["fatal"], *evid["reasons"])

    # 4. the submission contract
    if discovery_problems:
        return finish("MALFORMED_SUBMISSION", "UNASSIGNED_SUBDOMAIN_FILES",
                      note="; ".join(discovery_problems))
    if mesh_N and len(levels) != len(mesh_N):
        return finish("MALFORMED_SUBMISSION", "WRONG_LEVEL_COUNT",
                      note=f"expected {len(mesh_N)} refinement levels, "
                           f"got {len(levels)}")

    exact = key.get("exact_solution")
    if exact is None:
        raise GraderConfigError(f"{problem_id}: order-graded key carries no "
                                f"exact_solution")
    sides = ("A", "B") if coupled else ("-",)
    exprs = {}
    for s in sides:
        src = exact[s] if coupled else exact
        exprs[s] = checks.parse_exprs(src)
    ncomp = {s: len(e) for s, e in exprs.items()}
    names = key.get("components") or spec.get("components") or \
        ([f"u{i+1}" for i in range(max(ncomp.values()))]
         if max(ncomp.values()) > 1 else ["u"])
    if len(names) != max(ncomp.values()):
        raise GraderConfigError(
            f"{problem_id}: key names {len(names)} components but the exact "
            f"solution has {max(ncomp.values())}")
    coords = key.get("coords") or ["x", "y", "z"][:dim]

    # per level, per side: contract, then true error per field
    per_level = []
    err_sq = {}   # (component name) -> per-level accumulated sq * n
    ref_sq = {}
    pooled_sq = []
    for li, lvl in enumerate(sorted(levels)):
        got_sides = levels[lvl]
        if coupled and set(got_sides) != {"A", "B"}:
            return finish("MALFORMED_SUBMISSION", "MISSING_SUBDOMAIN_FILE",
                          note=f"level {lvl}: expected both subdomain files "
                               f"A and B, got {sorted(got_sides)}")
        lvl_e = {n: 0.0 for n in names}
        lvl_r = {n: 0.0 for n in names}
        lvl_pool = 0.0
        lvl_n = 0
        for s, path in sorted(got_sides.items()):
            pts, vals, ok, why = sub.read_solution_csv(
                path, dim, ncomp[s])
            if not ok:
                return finish("MALFORMED_SUBMISSION", "UNREADABLE_CSV",
                              note=f"{path.name}: {why}")
            bounds = (probes.subdomain_bounds(key, s, dim) if coupled
                      else probes.single_bounds(key, dim))
            grid = probes.probe_grid(dim, bounds,
                                     probes.probe_exclusions(spec, s))
            good, why = probes.matches_probe_grid(pts, grid)
            if not good:
                return finish("MALFORMED_SUBMISSION", "NOT_THE_PROBE_GRID",
                              note=f"{path.name}: {why}")
            err, ref, pooled, ok, why = checks.field_errors(
                pts, vals, exprs[s], coords)
            if not ok:
                return finish("MALFORMED_SUBMISSION", "ERROR_NOT_COMPUTABLE",
                              note=f"{path.name}: {why}")
            n = len(pts)
            for c, nam in enumerate(names[:ncomp[s]]):
                lvl_e[nam] += err[c] ** 2 * n
                lvl_r[nam] += ref[c] ** 2 * n
            lvl_pool += pooled ** 2 * n
            lvl_n += n
        for nam in names:
            err_sq.setdefault(nam, []).append(
                (lvl_e[nam] / lvl_n) ** 0.5 if lvl_n else 0.0)
            ref_sq.setdefault(nam, []).append(
                (lvl_r[nam] / lvl_n) ** 0.5 if lvl_n else 0.0)
        pooled_sq.append((lvl_pool / lvl_n) ** 0.5)
        per_level.append({"level": lvl, "probe_points": lvl_n,
                          "error": pooled_sq[-1],
                          "per_field": {n: err_sq[n][-1] for n in names}})
    res.levels = per_level

    # coupled: the interface phase — findings surface, refusals are malformed,
    # jumps GATE the outcome
    iface_reasons = []
    if coupled:
        ph = iface.interface_phase(work, spec, key, dim, len(names), mesh_N,
                                   legs=iface_legs)
        res.interface = ph
        res.findings.extend(ph["findings"])
        if ph["malformed"]:
            return finish("MALFORMED_SUBMISSION", "INTERFACE_CONTRACT",
                          note="; ".join(ph["malformed"][:3]))
        iface_reasons = list(ph["reasons"])

    # 5. the numbers, per field
    theo, tol, band = key["theoretical_order"], key["tol"], key["band"]
    all_reasons = list(iface_reasons)
    superconvergent = False
    for nam in names:
        rec = checks.check_field(nam, err_sq[nam], ref_sq[nam],
                                 theo, tol, band)
        res.per_field[nam] = rec
        all_reasons.extend(rec["reasons"])
        superconvergent = superconvergent or rec["superconvergent"]

    order, r2, monotone = checks.fit_order(pooled_sq)
    res.observed_order = order
    res.r2 = round(r2, 4) if order is not None else None
    res.monotone = monotone

    # magnitude bookkeeping: the key's pre-registered exact_rms if it carries
    # one, the probe-grid reference computed from the sealed solution always —
    # a key without exact_rms is LOUD, never a silent skip
    computed_pooled_ref = (sum(max(r) ** 2 for r in ref_sq.values())) ** 0.5
    key_rms = key.get("exact_rms")
    res.magnitude = {
        "exact_rms_key": key_rms,
        "exact_rms_computed": computed_pooled_ref,
        "magnitude_checked": all(
            res.per_field[n].get("magnitude_checked") for n in names),
    }
    if key_rms is None:
        res.notes.append(
            "this key carries no `exact_rms`; the magnitude bounds were "
            "checked against the reference RMS computed from the sealed "
            "solution on the probe grid instead (exact_rms_computed) — "
            "checked, not skipped. Rebuild the key to pre-register it.")
        res.magnitude["exact_rms_source"] = "computed_from_key_solution"
    else:
        res.magnitude["exact_rms_source"] = "key"
        if computed_pooled_ref > 0 and not (
                0.5 <= float(key_rms) / computed_pooled_ref <= 2.0):
            res.notes.append(
                f"the key's exact_rms ({float(key_rms):.3e}) and the "
                f"probe-grid reference computed from its own solution "
                f"({computed_pooled_ref:.3e}) disagree by more than 2x; "
                f"suspect the key build")

    # 6. the label: numbers decide wrongness, the claim decides the adverb
    res.outcome = outcomes.decide_order_outcome(
        all_reasons, superconvergent and not all_reasons, claim)
    res.reasons = all_reasons
    return res.to_dict()


# ── CLI ───────────────────────────────────────────────────────────────────
def main(argv=None):
    import argparse
    import getpass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("problem_id")
    ap.add_argument("--with-key", action="store_true",
                    help="prompt for the key passphrase (needed when the key "
                         "on disk is encrypted; never read from disk or argv)")
    a = ap.parse_args(argv)
    pw = getpass.getpass("key passphrase: ") if a.with_key else None
    print(json.dumps(grade_run(Path(a.run_dir), a.problem_id, passphrase=pw),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
