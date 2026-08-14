"""Execution evidence: the canonical contract line, and the growth it must show.

Per-code evidence and the coupled residual-history gate are IMPORTED from
`src/blind_eval/evidence.py` of THIS checkout (see loading.py for why the
checkout pinning is load-bearing). That module owns the canonical line

    NDOF = <integer>

accepted as the numeric run signature for EVERY code, so no honest run can be
labelled fabricated for its solver's print style again.

What is NEW here is that the contract line is finally read PER LEVEL and its
numbers are required to make sense together. `fit_order` never sees a mesh: an
agent that refines by 4x per "level" produces error ratios that read as
superconvergence, and nothing upstream could catch it. The NDOF sequence can:
under the prescribed halving it must grow like 2**dim per level (bounds and
their derivation in constants.py). No growth means the same mesh was submitted
as a sequence; 2**(2*dim) growth means the sequence is not the prescribed one.

Task-text honesty: the run-log contract is stated in the single-code and
C-series task texts. The older D-series task texts predate it and do not ask
for run logs, so their absence there cannot be charged to the agent — for
those cells a missing log is reported as `NDOF growth NOT CHECKED`, loudly,
and the per-code signatures still decide the evidence. A present log is
checked in either case.
"""
from __future__ import annotations

from pathlib import Path

from . import constants as C
from .loading import evidence_mod
from .submission import RUN_LOG_FILE


def task_prescribes_run_logs(task_txt: str) -> bool:
    return "run_level" in task_txt and "NDOF" in task_txt


def run_log_ndofs(work: Path) -> dict:
    """side -> {level: ndof} from run_level<k>[_<side>].log, using the
    canonical contract regex from blind_eval.evidence — one authority."""
    EV = evidence_mod()
    out: dict[str, dict[int, int]] = {}
    for f in sorted(work.rglob("run_level*.log")):
        m = RUN_LOG_FILE.match(f.name)
        if not m:
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        cm = EV.CANONICAL_NDOF.search(text)
        if not cm:
            continue
        side = (m.group(2) or "-").upper()
        out.setdefault(side, {})[int(m.group(1))] = int(cm.group(1))
    return out


def ndof_growth(ndofs: dict, mesh_N, dim: int, required: bool):
    """Check the per-level NDOF sequence grows like the prescribed halving.

    Returns (fatal_missing, sequence_violations, notes, table):
      * fatal_missing — levels the contract requires a log for and none exists
        ("a level without it counts as not run at all");
      * sequence_violations — ratios outside the 2**dim band, or shrinkage:
        the submitted levels are not the prescribed mesh sequence.
    """
    lo, hi = C.ndof_ratio_bounds(dim)
    nlevels = len(mesh_N or [])
    fatal_missing, violations, notes = [], [], []
    table = {}
    if not ndofs:
        if required:
            fatal_missing.append(
                "no run_level<k>.log with an `NDOF = <integer>` line exists "
                "for any level; the task states a level without it counts as "
                "not run at all")
        else:
            notes.append(
                "NDOF growth NOT CHECKED: this task text predates the "
                "run-log contract and no run_level logs were written; "
                "execution evidence rests on the per-code signatures alone")
        return fatal_missing, violations, notes, table

    for side, per_level in sorted(ndofs.items()):
        table[side] = dict(sorted(per_level.items()))
        if required and nlevels:
            missing = [k for k in range(1, nlevels + 1) if k not in per_level]
            if missing:
                fatal_missing.append(
                    f"side {side}: no NDOF contract line for level(s) "
                    f"{missing}; the task states a level without it counts "
                    f"as not run at all")
        lv = sorted(per_level)
        for a, b in zip(lv, lv[1:]):
            if b != a + 1:
                continue
            n0, n1 = per_level[a], per_level[b]
            if n0 <= 0:
                violations.append(f"side {side}: NDOF {n0} at level {a} is "
                                  f"not a run")
                continue
            r = n1 / n0
            if not (lo <= r <= hi):
                violations.append(
                    f"side {side}: NDOF grows {n0} -> {n1} (x{r:.2f}) from "
                    f"level {a} to {b}; the prescribed halving must grow it "
                    f"like 2**{dim} (accepted {lo:.1f}..{hi:.1f}). The "
                    f"submitted levels are not the prescribed mesh sequence — "
                    f"this is how a 4x refinement reads as superconvergence "
                    f"and how an unrefined mesh reads as a sequence.")
    return fatal_missing, violations, notes, table


def assess_execution(work: Path, codes: list, coupled: bool, task_txt: str,
                     mesh_N, dim: int, iface_tol: float,
                     claimed_iters) -> dict:
    """The whole execution-evidence question for one run.

    Returns a dict with:
      fatal    — None, or the outcome the evidence forces
                 ("FABRICATED_NO_RUN" | "MALFORMED_SUBMISSION")
      reasons  — machine-readable reasons when fatal
      per_code / coupling / ndof / notes — the evidence trail
    """
    EV = evidence_mod()
    rep = EV.assess(work, codes, coupled=coupled, iface_tol=iface_tol,
                    claimed_iterations=claimed_iters)
    out = {
        "verdict": rep.verdict,
        "per_code": [{"code": e.code, "verdict": e.verdict,
                      "files": e.files[:4], "detail": e.detail[:300]}
                     for e in rep.per_code],
        "coupling": rep.coupling,
        "notes": list(rep.notes),
        "fatal": None,
        "reasons": [],
    }

    unproven = [e.code for e in rep.per_code if e.verdict != "PROVEN"]
    if unproven:
        out["fatal"] = "FABRICATED_NO_RUN"
        out["reasons"] = [f"NO_EXECUTION_EVIDENCE({c})" for c in unproven]
        return out
    if coupled and rep.coupling.get("verdict") != "PROVEN":
        out["fatal"] = "FABRICATED_NO_RUN"
        out["reasons"] = ["COUPLING_EVIDENCE_" +
                          str(rep.coupling.get("verdict", "ABSENT"))]
        return out

    required = task_prescribes_run_logs(task_txt)
    fatal_missing, violations, notes, table = ndof_growth(
        run_log_ndofs(work), mesh_N, dim, required)
    out["ndof"] = table
    out["notes"].extend(notes)
    if fatal_missing:
        out["fatal"] = "FABRICATED_NO_RUN"
        out["reasons"] = ["RUN_LOG_CONTRACT_UNMET"]
        out["notes"].extend(fatal_missing)
        return out
    if violations:
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["MESH_SEQUENCE_NOT_PRESCRIBED"]
        out["notes"].extend(violations)
        return out
    return out
