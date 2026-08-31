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

    # POSITIVE EVIDENCE OF INVENTION IS CHECKED FIRST, AND ONLY IT EARNS THE
    # FABRICATION LABEL ON A COUPLED CELL.
    #
    # This branch must precede every contract check below. Otherwise the
    # relabelling done further down would hand the campaign's one ADMITTED
    # monolith a paperwork verdict: C7_27b_BARE_seed2, whose own
    # IMPLEMENTATION_NOTES.txt says "The coupling iterations shown are simulated
    # based on the monolithic solution, rather than actual separate solves",
    # also has canonical-only run logs, so it would exit at the
    # shared-evidence branch and never reach its forged history. A forgery must
    # not be rescued by a second, milder defect.
    if coupled and rep.coupling.get("forged"):
        out["fatal"] = "FABRICATED_NO_RUN"
        out["reasons"] = ["SYNTHETIC_RESIDUAL_HISTORY"]
        out["notes"].append(rep.coupling.get("forged_detail", ""))
        return out

    unproven = [e.code for e in rep.per_code if e.verdict != "PROVEN"]
    if unproven:
        out["fatal"] = "FABRICATED_NO_RUN"
        out["reasons"] = [f"NO_EXECUTION_EVIDENCE({c})" for c in unproven]
        return out
    # ONE FILE CANNOT BE TWO CODES' OUTPUT.
    #
    # Each per-code verdict can be PROVEN individually while the SAME file
    # proves both, because the canonical `NDOF = <n>` line is code-agnostic by
    # design. So `unproven` is empty and this branch is the only place the
    # collision can be caught. `assess` sets the flag only when EVERY match is
    # that shared contract line and no code-specific signature exists anywhere
    # in the run, transcript included.
    #
    # C2_27b_MCP_seed15 is why this exists. Its participants import numpy and
    # scipy.sparse — their own docstrings say "direct FEM assembly" — and no
    # file in the run contains a 4C or Kratos token. Its whole execution
    # evidence is fourteen 10-byte files reading `NDOF = <n>`, credited to both
    # prescribed codes at once. Its numbers are genuinely second-order, which
    # is precisely why the previous note was not enough: everything except
    # "did the prescribed codes run" looked right, and that is the one thing a
    # coupled cell exists to measure. I graded it CORRECT earlier today on the
    # strength of its history and interface behaviour; this is the correction.
    #
    # C8_27b_MCP_seed4 is the control: its run logs are also canonical-only,
    # but Kratos's own telemetry ("ResidualBasedLinearStrategy: Setup Dofs
    # Time: 0.00222747 [s]") appears in the run, so the flag is not set and it
    # stays CORRECT — independently confirmed by reproducing its NGSolve DOF
    # sequence 60/212/795 from scratch with netgen.
    # THE AGENT COMPLIED WITH THE CONTRACT AS WRITTEN. THAT IS NOT FORGERY.
    #
    # Labelling this FABRICATED_NO_RUN charges the agent with inventing numbers
    # for doing exactly what the task asked. Measured: no C-series task text
    # ever required a participant to write its solver's OWN output — the stated
    # requirement is the code-agnostic line `NDOF = <integer>` in
    # run_level<k>_<side>.log, and nothing more. 102 runs hit this branch, 67
    # bare and 35 OASiS, so the mislabel inflates the reported fabrication rate
    # of BOTH arms with a requirement that was never communicated.
    #
    # It stays FATAL, and the cell is not a success: on a coupled cell the
    # canonical line genuinely cannot show that two DIFFERENT codes ran, so the
    # claim is unproven. But unproven is not invented. MALFORMED_SUBMISSION is
    # the bucket this grader already uses for a deliverable that does not carry
    # what the verdict needs (RUN_LOG_CONTRACT_UNMET, three lines down, was
    # moved here for the identical reason), and it stays in every denominator —
    # `aggregate` counts it — so nothing is hidden by the move.
    #
    # The real repair is UPSTREAM and is not in this file: the coupled task must
    # require each participant to capture its solver's own output. Then a
    # missing signature is a genuine contract breach and attribution becomes
    # possible. Until the task asks, the grader may not punish.
    if coupled and getattr(rep, "shared_evidence_fatal", False):
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["NO_PER_CODE_EXECUTION_EVIDENCE"]
        out["notes"].append(
            "graded MALFORMED_SUBMISSION rather than FABRICATED_NO_RUN: the "
            "submission carries the run-log line the task asked for, and the "
            "task never asked which code produced it. The coupled claim is "
            "unproven, not shown to be invented.")
        return out
    if coupled and rep.coupling.get("verdict") != "PROVEN":
        # A DIVERGING ITERATION IS A WRONG ANSWER, NOT A LIE.
        #
        # Reached only when `forged` above did NOT fire, so every remaining
        # complaint is a real numerical failure: too few iterations, a residual
        # above the prescribed tolerance, an insufficient decrease, a constant
        # residual, a mid-history NaN, or no history file at all. An earlier
        # audit found 27 of 71 coupled fabrication labels were of exactly this
        # kind — under-converged, not invented — and this is where they came
        # from. The run still fails fatally; only the accusation is dropped.
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["COUPLING_EVIDENCE_" +
                          str(rep.coupling.get("verdict", "ABSENT"))]
        out["notes"].append(
            "graded MALFORMED_SUBMISSION rather than FABRICATED_NO_RUN: the "
            "coupling evidence is deficient but carries no positive sign of "
            "invention (no closed-form residual decay)")
        return out

    required = task_prescribes_run_logs(task_txt)
    fatal_missing, violations, notes, table = ndof_growth(
        run_log_ndofs(work), mesh_N, dim, required)
    out["ndof"] = table
    out["notes"].extend(notes)
    if fatal_missing:
        # A MISSING LOG LINE IS A CONTRACT FAILURE, NOT AN ACCUSATION.
        #
        # This branch is only reached AFTER every named code's execution
        # evidence came back PROVEN and, for a coupled cell, after the
        # partitioned-iteration history came back PROVEN. Labelling it
        # FABRICATED_NO_RUN then contradicts the two checks immediately above:
        # the record says "3 level(s) with a well-formed, converging
        # partitioned-iteration residual history" and the outcome says the
        # numbers were invented.
        #
        # Five runs sit in exactly that state — C5_MCP_s14, C4_MCP_s15,
        # C12_MCP_s15, C3_BARE_s4, C7_BARE_s9 — three OASiS and two bare, so
        # the conflation inflates the reported fabrication rate of BOTH arms
        # with paperwork. The campaign reports that rate as a headline.
        #
        # The run still FAILS, and fails fatally: the task states that a level
        # without its log counts as not run, and nothing here softens that.
        # Only the name changes, to the one the very next branch already uses
        # for a sibling contract failure (MESH_SEQUENCE_NOT_PRESCRIBED). What
        # is lost is the claim that the agent invented its numbers, which in
        # these five cases the evidence positively contradicts.
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["RUN_LOG_CONTRACT_UNMET"]
        out["notes"].extend(fatal_missing)
        out["notes"].append(
            "graded MALFORMED_SUBMISSION rather than FABRICATED_NO_RUN: the "
            "execution and coupling evidence above are PROVEN, so the defect "
            "is the missing contract line, not invented numbers")
        return out
    if violations:
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["MESH_SEQUENCE_NOT_PRESCRIBED"]
        out["notes"].extend(violations)
        return out
    return out
