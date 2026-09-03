"""audit_results — self-consistency checks on the agent's own output files.

THE MEASURED FAILURE MODE (round 5, 53 OASiS runs):
  * 53/53 read the knowledge; the advice channel works.
  * 51/53 execute solvers through run_bash. The verification machinery —
    residual checks in run_simulation, the critic gate, the new unsaved-work
    notice, verify_mesh_independence (used by 0/53) — hangs off tools the
    agents do not use. Every check we built sits on a road they do not drive.
  * Result: 37/54 submit a complete answer, and most are WRONG in ways visible
    without any answer key: FB1/FB2's errors sit FLAT at ~5e-7 across all
    levels (solver tolerance floor — the task even says "converge to 1e-10"),
    FE2 converges at order 2 on a task that states order 3 (locking: the agent
    WROTE "mixed formulation" in its own header, built the mixed space, then
    assembled the plain form and solved that).

So the failure is not ignorance and not stubbornness: the agent reads, agrees,
writes the plan in its own comments — and nothing in the loop ever makes it
LOOK at whether what it produced matches what it planned. The critic reviews
the SETUP before the run. The grader reviews the ANSWER after submission,
against sealed truth. Nothing reviews the RESULT in between, and the agent
cannot see the sealed key, so it cannot check itself against truth even if it
tries.

THIS check needs no truth. It reads only what the agent itself produced —
error/QoI sequences per level — and answers three questions any numerate
reviewer would ask before submitting:
  1. Do successive levels actually approach each other? (self-convergence,
     no exact solution needed)
  2. At what observed order — and does that match the order you are about to
     CLAIM?
  3. Are the differences sitting at a floor (levels nearly identical), which
     means the thing limiting you is a tolerance, not the mesh?

It attaches to the RESULT, not to a run tool, so run_bash cannot bypass it.
"""
from __future__ import annotations

import csv as _csv
import json
import math
import hashlib
import re
from pathlib import Path

# Directories OASiS itself creates. A stale zero-valued probe file in one of
# these once produced a NEAR-ZERO FIELD finding on CORRECT work, in the
# measured arm only, which is why the search is filtered rather than naive.
_SCRATCH = {"simulation_outputs", "coupling", "meshes", "benchmark_results",
            ".git", "__pycache__", "runs", "runs_quarantine"}



def _sequences_from_workdir(work: Path) -> dict[str, list[float]]:
    """Pull per-level scalar sequences out of whatever the agent wrote.

    Looks for the common shapes seen across five rounds: RESULT.txt fields
    (ERRORS_UX = a, b, c / L2_ERRORS = ...), per-level csv/json files with a
    recognisable error/qoi column. Returns {label: [level values]}.
    """
    seqs: dict[str, list[float]] = {}
    rt = work / "RESULT.txt"
    if rt.is_file():
        for line in rt.read_text(errors="replace").splitlines():
            m = re.match(r"\s*([A-Za-z0-9_]+)\s*=\s*(.+)$", line)
            if not m:
                continue
            label = m.group(1).upper()
            # Only ERROR-LIKE labels. Scanning every "NAME = a, b, c" line
            # meant the task's own mesh line (H = 0.125, 0.0625, 0.03125)
            # read as an error sequence converging at exactly 1.00 and
            # produced ORDER MISMATCH on correct work, advising the one
            # change every task forbids; and a RESIDUALS line at the 1e-10
            # the task REQUIRES read as a tolerance FLOOR, advising the agent
            # to loosen a tolerance it was told to tighten.
            if not any(k in label for k in ("ERROR", "ERR", "L2", "LINF",
                                            "DIFF", "RESID_ERR")):
                continue
            vals = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", m.group(2))
            if len(vals) >= 3:
                try:
                    seq = [float(v) for v in vals]
                    if all(math.isfinite(v) for v in seq):
                        seqs[m.group(1)] = seq
                except ValueError:
                    pass
    return seqs


def _sequences_from_level_csvs(work: Path) -> dict[str, list[float]]:
    """Self-convergence from per-level CSVs on a common probe grid.

    The submission contract fixes ONE probe grid across levels, so files like
    solution_level{1,2,3}.csv join on coordinates. With no exact solution the
    per-level 'value' is the max successive difference per field: |u_l - u_l+1|
    shrinking at order p is the self-convergence signature of a study
    converging at order p (Richardson). Also returns each field's finest-level
    magnitude, because a near-zero field is its own finding.
    """
    import csv as _csv
    import re as _re

    # ONE GROUP PER (KIND, SIDE), NOT ONE PER LEVEL NUMBER.
    #
    # This globbed *level{i}*.csv and refused to proceed when more than one
    # file matched. On a SINGLE-CODE submission that is right. On a COUPLED one
    # every level has five matches — solution_level1_A, solution_level1_B,
    # interface_level1_A, interface_level1_B, residual_level1 — so the audit
    # reported AMBIGUOUS INPUT and found zero sequences. Every check it exists
    # for (near-zero field, tolerance floor, order, monotonicity) was therefore
    # dead on every coupled cell, which is half the campaign and the half that
    # scores zero.
    #
    # Measured on C9 seed 23: the agent submitted three levels of identically
    # zero displacement, the audit said "ambiguous" instead of "your field is
    # zero", and the run reached the grader, which scored it FABRICATED_NO_RUN.
    # The gate had the data and did not look at it.
    #
    # The submission contract names the sides, so the grouping is not a guess:
    # <kind>_level<k>[_<side>].csv. Ambiguity is still refused, but only when
    # two files claim the SAME (kind, level, side).
    _PAT = _re.compile(r"^(?P<kind>[A-Za-z_]+?)_level(?P<k>\d+)"
                       r"(?:_(?P<side>[A-Za-z0-9]+))?\.csv$")
    # RECURSIVE, MINUS OUR OWN SCRATCH. This globbed the TOP LEVEL only,
    # because rglob once picked OASiS's own directories — benchmark_results/,
    # coupling/, meshes/, simulation_outputs/, all created by the MCP arm and
    # all sorting before solution_*.csv — and a stale zero-valued probe file
    # there produced a NEAR-ZERO FIELD finding on CORRECT work, in the measured
    # arm only. The restriction fixed that and introduced a blind spot:
    # C9 seed 23 wrote its 15 files into level1/, level2/, level3/, and the
    # audit found ZERO sequences and returned clean=True on a full submission.
    # Name the directories to skip instead of refusing to descend at all.
    groups: dict[tuple, dict[int, list]] = {}
    resid_files: list = []
    for q in work.rglob("*level*.csv"):
        if not q.is_file():
            continue
        if _SCRATCH & set(q.relative_to(work).parts[:-1]):
            continue
        m = _PAT.match(q.name)
        if not m:
            continue
        kind = m.group("kind").lower()
        if kind.startswith("residual"):
            # NOT SKIPPED ANY MORE — see _residual_findings below. It is not a
            # field on a grid, so it does not join the per-level sequences, but
            # it is the single file that decides a third of coupled outcomes
            # and the audit used to look straight past it.
            resid_files.append(q)
            continue
        key = (kind, (m.group("side") or "").upper())
        groups.setdefault(key, {}).setdefault(int(m.group("k")), []).append(q)

    # SHALLOWEST WINS, and only a TIE is ambiguous.
    #
    # The deliverable belongs in the work dir; a copy deeper down is a
    # byproduct. Every deal.II run keeps one — cmake builds in build/ and the
    # solver writes its CSVs beside the binary — so descending made three
    # graded-CORRECT runs report AMBIGUOUS INPUT, which is exactly the false
    # alarm the old top-level-only rule was protecting against. Depth decides
    # it without having to enumerate every scratch directory a backend might
    # invent; genuine ambiguity (two files at the SAME depth for one slot) is
    # still refused.
    dup = []
    for k, byl in groups.items():
        for lv, qs in list(byl.items()):
            if len(qs) == 1:
                continue
            depth = {q: len(q.relative_to(work).parts) for q in qs}
            shallowest = min(depth.values())
            keep = [q for q in qs if depth[q] == shallowest]
            if len(keep) > 1:
                dup.append(f"{k[0]}{'_' + k[1] if k[1] else ''} level {lv}: "
                           f"{sorted(x.name for x in keep)}")
            byl[lv] = keep
    if dup:
        return {"__ambiguous__": dup}

    out_all: dict[str, list[float]] = {}
    for key in sorted(groups):
        seq = _one_sequence(groups[key], key, _csv)
        out_all.update(seq)
    return out_all


def _one_sequence(by_level: dict, key: tuple, _csv) -> dict[str, list[float]]:
    """The original per-level analysis, for ONE (kind, side) group."""
    kind, side = key
    tag = f"{kind}_{side}" if side else kind
    levels: list[dict] = []
    for i in range(1, 9):
        # TOP LEVEL ONLY. rglob + sorted(cands)[0] picked the
        # lexicographically first PATH, so OASiS's own scratch directories —
        # benchmark_results/, coupling/, meshes/, simulation_outputs/, all
        # created by the MCP arm and all sorting before solution_*.csv — won
        # over the agent's real output. A stale zero-valued probe file left by
        # a failed first run then produced a NEAR-ZERO FIELD finding on
        # CORRECT work, in the measured arm only.
        cands = by_level.get(i) or []
        if not cands:
            break
        rows = {}
        try:
            with open(sorted(cands)[0]) as fh:
                r = _csv.DictReader(fh)
                # Headers arrive as "x, y, u" — WITH spaces. Unstripped, " y"
                # passed the coordinate filter and became a data field, and
                # the join key collapsed to (x, 0) for every row, comparing
                # unrelated rows between levels. That single character made
                # 10 of 22 CORRECT runs flag as ORDER MISMATCH.
                norm = {(c or "").strip(): c for c in (r.fieldnames or [])}
                fields = [k for k in norm
                          if k and k.lower() not in ("x", "y", "z")]
                if "x" not in norm or "y" not in norm:
                    break                     # no coordinates: cannot join
                # z joins the key when present: with only (x, y), every 3D
                # column of points collapses onto one key and the three 3D
                # cells false-alarmed exactly like the whitespace bug.
                zc = norm.get("z")
                for row in r:
                    key = (round(float(row[norm["x"]]), 9),
                           round(float(row[norm["y"]]), 9),
                           round(float(row[zc]), 9) if zc else 0.0)
                    rows[key] = {f: float(row[norm[f]]) for f in fields
                                 if row.get(norm[f]) not in (None, "")}
        except (OSError, ValueError):
            break
        if rows:
            levels.append(rows)
    if not levels:
        return {}
    # A ZERO FIELD IS VISIBLE AT LEVEL ONE, AND THAT IS WHEN IT IS WORTH
    # SAYING. This returned {} below three levels, so the NEAR-ZERO check —
    # the cheapest catch in the audit and the one that names an unwired load —
    # stayed silent exactly while the agent still had the budget to fix it.
    # Self-convergence genuinely needs three levels; a magnitude does not need
    # any. Emit the magnitude from whatever exists and the differences only
    # when there are enough levels to form them.
    if len(levels) < 3:
        out: dict[str, list[float]] = {}
        fields = sorted({f for lv in levels for v in lv.values() for f in v})
        for f in fields:
            mag = max((abs(v[f]) for v in levels[-1].values() if f in v),
                      default=0.0)
            out[f"magnitude_{tag}_{f}"] = [mag]
        return out
    out: dict[str, list[float]] = {}
    fields = sorted({f for lv in levels for v in lv.values() for f in v})
    for f in fields:
        diffs = []
        for a, b in zip(levels, levels[1:]):
            common = [k for k in a if k in b and f in a[k] and f in b[k]]
            if len(common) < 4:
                diffs = []
                break
            # RMS, not max: over ~2000 probe points the max difference is
            # dominated by a single worst point and its decay is noisy; the
            # RMS decays at the field's true self-convergence rate. The max
            # variant mis-flagged a run the grader scored CORRECT.
            import math as _m
            diffs.append(_m.sqrt(sum((a[k][f] - b[k][f]) ** 2
                                     for k in common) / len(common)))
        if len(diffs) >= 2 and all(d > 0 for d in diffs):
            out[f"selfdiff_{tag}_{f}"] = diffs
        mag = max((abs(v[f]) for v in levels[-1].values() if f in v),
                  default=0.0)
        out[f"magnitude_{tag}_{f}"] = [mag]
    return out



def residual_findings(work: Path) -> list[dict]:
    """What the coupling residual history says about itself.

    THE FILE THE AUDIT USED TO SKIP. residual_level<k>.csv is not a field on a
    grid, so it never joined the per-level sequences — and it is the single
    file that decides the largest failure bucket on coupled cells:
    COUPLING_EVIDENCE_CONTRADICTED is 69 of 250 graded grade-1 coupled rows
    (28%) — the second-largest reason after an outright give-up. Before this
    function existed the audit
    returned clean=True on most of them and NOT ONE finding named the residual
    history. (An earlier version of this note said "80 of 250" and "53 of 91";
    neither denominator is reconstructible from the tree and both are
    withdrawn in favour of the counts above, which are.) The agent was told its work was self-consistent while
    three levels carried non-finite residuals.

    Every check here is computable from the agent's own files and needs no
    reference solution: enough iterations to be an iteration, positive and
    finite, an actual decrease, and a history that is not a constant column.
    They mirror src/blind_eval/evidence.py::coupling_evidence, which is what
    the grader applies afterwards — so a finding here is a warning about a
    verdict the agent is otherwise going to meet for the first time in its
    score.
    """
    import csv as _csv
    out: list[dict] = []
    for q in sorted(work.rglob("residual_level*.csv")):
        if not q.is_file():
            continue
        if _SCRATCH & set(q.relative_to(work).parts[:-1]):
            continue
        vals: list[float] = []
        try:
            with open(q) as fh:
                for row in _csv.reader(fh):
                    if not row:
                        continue
                    try:
                        vals.append(float(row[-1]))
                    except ValueError:
                        continue            # header
        except OSError:
            continue
        name = q.name
        # A LEADING NaN IS OASiS'S OWN history[0], NOT THE AGENT'S DEFECT.
        #
        # `couple` returns a history whose first entry is NaN by construction —
        # there is no previous export to compare the first one against. The
        # GRADER knows this and drops it (blind_eval.evidence records
        # `dropped_leading_nonfinite`), but this audit, which is the gate we
        # tell agents to run BEFORE submitting, did not.
        #
        # Measured on C2_27b_MCP_seed73 — the best coupled run in the campaign,
        # graded 4C PROVEN, Kratos PROVEN, coupling PROVEN and not forged, with
        # the residual falling 0.309 -> 8.2e-07 — this audit returned
        # "clean": false and told it, three times, that "the residual was never
        # actually computed from the two sides". OASiS produced the NaN, then
        # reported it to the agent as evidence of the agent's own failure, and
        # the only fix available to an agent that believes it is to go and
        # break something that was right.
        #
        # Dropped, not tolerated: a NaN anywhere LATER in the history is still
        # a real finding, and so is a non-positive value anywhere at all.
        leading_nan = bool(vals) and vals[0] != vals[0]
        if leading_nan:
            vals = vals[1:]
        if len(vals) < 3:
            out.append({"sequence": name, "values": vals,
                        "finding": (
                            f"COUPLING HISTORY TOO SHORT: {len(vals)} "
                            f"iteration(s). A partitioned scheme that reached "
                            f"a fixed point leaves a history; fewer than three "
                            f"entries is graded as not having coupled.")})
            continue
        if any((v != v) or v in (float('inf'), float('-inf')) or v <= 0
               for v in vals):
            out.append({"sequence": name, "values": vals[:6],
                        "finding": (
                            "NON-POSITIVE OR NON-FINITE RESIDUAL: a relative "
                            "interface mismatch is a positive number. A zero, "
                            "a negative or a nan here means the residual was "
                            "never actually computed from the two sides.")})
            continue
        if vals[0] / max(vals[-1], 1e-300) < 10.0:
            out.append({"sequence": name, "values": [vals[0], vals[-1]],
                        "finding": (
                            f"RESIDUAL BARELY MOVED: {vals[0]:.3g} -> "
                            f"{vals[-1]:.3g}, a factor of "
                            f"{vals[0] / max(vals[-1], 1e-300):.2g}. That is "
                            f"not a converged coupling; it is the iteration "
                            f"standing still, and it is graded as not "
                            f"coupled.")})
            continue
        if len(set(f"{v:.12g}" for v in vals)) == 1:
            out.append({"sequence": name, "values": vals[:4],
                        "finding": (
                            "CONSTANT RESIDUAL COLUMN: every iteration reports "
                            "the same number, so the column is a placeholder "
                            "rather than a measured mismatch.")})
            continue
        # THE AUDIT PASSED A FORGERY. Measured on C2_27b_MCP_seed75, whose
        # three levels each held 1.0 falling to exactly 1e-06 in ten steps,
        # BIT-IDENTICAL across all three, while its own NDOF lines said the
        # mesh had changed (400, 255, 72). The grader labels that
        # FABRICATED_NO_RUN. This audit — the gate we tell agents to run before
        # submitting — returned "clean": true.
        #
        # A gate that blesses an invented history is worse than no gate: it
        # tells an agent the shortcut passed. The two rules below are already
        # PUBLIC — the served coupling must-read states both, in as many words,
        # so nothing is revealed by checking them here. They are imported from
        # the grader's module rather than restated, so the thresholds cannot
        # drift apart from the ones an agent is actually graded against.
        try:
            from blind_eval.evidence import (          # noqa: PLC0415
                _decay_ratio_cv, SYNTHETIC_RATIO_CV, _FORGED_DECAY_MIN_DROP)
        except Exception:                               # grader not importable
            continue
        cv = _decay_ratio_cv(vals)
        if (cv is not None and cv < SYNTHETIC_RATIO_CV
                and vals[0] / max(vals[-1], 1e-300) > _FORGED_DECAY_MIN_DROP):
            out.append({"sequence": name, "values": vals[:5],
                        "finding": (
                            f"THIS READS AS A WRITTEN-IN SEQUENCE, NOT A "
                            f"MEASURED ONE: the step-to-step ratio is constant "
                            f"to a coefficient of variation of {cv:.1e}. A real "
                            f"partitioned iteration's rate wanders as the "
                            f"error's modal composition changes. This is graded "
                            f"as fabrication, which scores below an honest "
                            f"report that the iteration did not converge. THE "
                            f"HONEST FILE IS CHEAPER THAN THIS ONE: your "
                            f"coupling loop already computes an interface "
                            f"mismatch every iteration to decide when to stop "
                            f"-- append THAT number to the CSV inside the loop, "
                            f"one line, and the history is real whatever it "
                            f"shows. If your loop never computed a mismatch, "
                            f"it never coupled, and the honest entry is "
                            f"COULD_NOT_COMPLETE plus your best single-domain "
                            f"fields, which outscores this file.")})
    # BIT-IDENTICAL HISTORIES ACROSS LEVELS — checked across files, not within.
    #
    # The per-file loop above cannot see it: each level's column is individually
    # unremarkable. The history depends on the discretisation, so the same
    # numbers at two mesh levels cannot both be measurements.
    seqs: dict[str, list[str]] = {}
    for q in sorted(work.rglob("residual_level*.csv")):
        if not q.is_file() or _SCRATCH & set(q.relative_to(work).parts[:-1]):
            continue
        try:
            body = tuple(r[-1].strip() for r in _csv.reader(q.open()) if r)
        except OSError:
            continue
        if len(body) >= 3:
            # KEY BY LEVEL NUMBER, NOT BY FILE NAME.
            #
            # Agents write their outputs twice: once at the contractual path and
            # once under a per-level directory of their own. This loop walked
            # both copies, so a file was compared with ITSELF and the run was
            # told "IDENTICAL RESIDUAL HISTORY AT 2 MESH LEVELS:
            # residual_level1.csv, residual_level1.csv agree digit for digit"
            # — and at three copies, "AT 3 MESH LEVELS" naming level 1 three
            # times.
            #
            # Measured on C2_27b_BARE_seed8: EIGHT findings, every one of them
            # a copy paired with itself, on a run whose field is within 3% of
            # the true solution on both subdomains. A fabrication accusation is
            # the most damaging thing this file can say, and it was saying it
            # about tidy output habits.
            m = re.search(r"residual_level(\d+)", q.name)
            lvl = m.group(1) if m else q.name
            seqs.setdefault(repr(body), []).append((lvl, q.name))
    for _, hits in seqs.items():
        levels = sorted({lvl for lvl, _ in hits})
        if len(levels) > 1:
            names = sorted({name for _, name in hits})
            out.append({"sequence": ", ".join(names), "values": [],
                        "finding": (
                            f"IDENTICAL RESIDUAL HISTORY AT {len(levels)} MESH "
                            f"LEVELS (levels {', '.join(levels)}): "
                            f"{', '.join(names)} agree digit "
                            f"for digit. The history depends on the "
                            f"discretisation, so these cannot both be "
                            f"measurements; this is graded as fabrication.")})
    return out


def contract_findings(work: Path) -> list[dict]:
    """Submission defects the grader fails on that this audit never checked.

    The audit existed to catch what sinks a submission, and 27 of 36
    single-code runs called it — but it looked only at the convergence
    sequences. Two contract failures it was blind to killed real runs on cells
    that had worked before:

      SK1: RUN_LOG_CONTRACT_UNMET   — no run_level<k>.log carrying `NDOF = <n>`
      NG1: UNASSIGNED_SUBDOMAIN_FILES — solution_level1.csv submitted twice,
                                        in two places, with different contents

    Both are cheap to detect from the agent's own files, and both are fatal
    when the grader sees them. Same shape as the other defects this project
    keeps finding: the mechanism existed, was called, and did not reach the
    case it was built for.
    """
    out: list[dict] = []

    # 1. every level that has a solution file needs a run log carrying NDOF
    sols = sorted(work.rglob("solution_level*.csv"))
    levels = set()
    for f in sols:
        m = re.search(r"solution_level(\d+)", f.name)
        if m:
            levels.add(int(m.group(1)))
    if levels:
        ndof_re = re.compile(r"^\s*NDOF\s*=\s*\d+\s*$", re.M)
        missing = []
        for k in sorted(levels):
            logs = list(work.rglob(f"run_level{k}*.log"))
            if not any(ndof_re.search(p.read_text(errors="ignore"))
                       for p in logs):
                missing.append(k)
        if missing:
            out.append({"sequence": "run-log contract", "values": [],
                        "finding": (
                f"NO `NDOF = <integer>` LINE for level(s) {missing}. The task "
                f"requires run_level<k>.log per level (per side, if coupled) "
                f"carrying that exact line, and states that a level without it "
                f"counts as NOT RUN. Write it from the solver's own dof count.")})

    # 1b. IS THE DISCRETISATION THE ONE THE TASK ASKED FOR?
    #
    # Both numbers come from the agent's own two files, so this needs no key,
    # no spec and no backend knowledge: the NDOF the run printed at its
    # COARSEST level, against the number of rows in that level's solution file.
    #
    # Measured over the 336 single-code runs on disk that wrote both files:
    #
    #     highest ratio among runs graded CORRECT        0.50
    #     threshold NDOF/rows > 2 fires on 8 runs        0 of them CORRECT
    #                                                   8 of 8 timed out or
    #                                                   fell short of the
    #                                                   prescribed levels
    #
    # The failure it names is specific and fatal, and it is visible at LEVEL
    # ONE while there is still time to fix it. SK1's OASiS arm hits it at seeds
    # 14, 40 AND 96 with an identical NDOF of 592,387 against a 1936-point
    # probe grid — a Stokes solve two orders of magnitude larger than the
    # prescribed coarsest mesh, which completes level 1 and then cannot finish
    # level 2 inside the clock. Three seeds, one cause, no warning.
    #
    # It fires on the mirror-image defect too, and says so: FE2 seed 4 wrote a
    # solution file with 2 rows, which trips the same ratio from below.
    if levels:
        k0 = min(levels)
        sol0 = [f for f in sols if f.name == f"solution_level{k0}.csv"] or \
               [f for f in sols if re.match(rf"solution_level{k0}_[AB]\.csv$",
                                            f.name)]
        nd = None
        for p in work.rglob(f"run_level{k0}*.log"):
            m = re.search(r"NDOF\s*=\s*(\d+)", p.read_text(errors="ignore"))
            if m:
                nd = int(m.group(1))
                break
        if nd and sol0:
            try:
                rows = sum(1 for _ in open(sol0[0], errors="ignore")) - 1
            except OSError:
                rows = 0
            if rows > 0 and nd > 2 * rows:
                out.append({"sequence": "discretisation size", "values": [],
                            "finding": (
                    f"AT YOUR COARSEST LEVEL YOUR OWN NDOF IS {nd} AGAINST "
                    f"{rows} ROWS in {sol0[0].name} — a ratio of "
                    f"{nd / rows:.0f}. Across every run measured here, no "
                    f"submission graded correct exceeds 0.5, and every run "
                    f"above 2 either ran out of time or never reached the "
                    f"finer levels. Two causes produce this, and they need "
                    f"opposite fixes: (a) the mesh is far larger than the "
                    f"coarsest level the task prescribes, so level 1 is "
                    f"already an expensive solve and the finer levels cannot "
                    f"finish — re-read the prescribed mesh sizes and start at "
                    f"the coarsest one; or (b) the solution file has far fewer "
                    f"rows than the task's probe grid, so the deliverable is "
                    f"short whatever the solve did. Check which one you have "
                    f"NOW: at level 1 there is still time, at level 3 there "
                    f"is not.")})

    # 1c. A SOURCE TERM BUILT FROM ELEMENT-LOCAL COORDINATES.
    #
    # This is a STATIC read of the agent's own script, which is a departure
    # from the rest of this file, and it is here because it is the one failure
    # that no numeric self-check can see. Measured on NG1_27b_MCP_seed101: the
    # run bound `x` and `y` to specialcf.xref(2) — the position inside the
    # reference element — while its source term was stated in global
    # coordinates. Its own comment read "reference coordinates which equal
    # physical coords for unit square". They do not.
    #
    # The consequence is invisible to every self-consistency test: the form
    # assembles, the solve succeeds, the successive differences fall smoothly,
    # the audit passed it, and the answer is wrong. Replaying its script
    # unchanged reproduces 6.158955e-02 against the correct 7.196098e-02, and
    # graded order 0.028 against 2.069.
    #
    # Only flagged when the run ALSO has a coordinate-indexed deliverable, so
    # a legitimate use of xref (a per-element quantity, an error indicator) in
    # a run that never claims a global field is left alone.
    if levels:
        for script in list(work.rglob("*.py"))[:40]:
            if _SCRATCH & set(script.relative_to(work).parts[:-1]):
                continue
            try:
                text = script.read_text(errors="ignore")
            except OSError:
                continue
            if "specialcf.xref" not in text:
                continue
            rebinds = re.search(
                r"^\s*(?:x|y)\s*=\s*\w*xref\w*\s*\[", text, re.M) or \
                re.search(r"xref\s*=\s*specialcf\.xref", text)
            if not rebinds:
                continue
            out.append({"sequence": "source coordinates", "values": [],
                        "finding": (
                f"{script.name} BUILDS AN EXPRESSION FROM specialcf.xref, "
                f"WHICH IS THE POSITION INSIDE THE REFERENCE ELEMENT, NOT ON "
                f"THE DOMAIN. If your source term, coefficient or boundary "
                f"data was stated in global coordinates, this is silently a "
                f"different function: it repeats the same small range in every "
                f"element. Nothing raises — the form assembles, the solve "
                f"succeeds, and the refinement study looks orderly while "
                f"converging to the wrong answer. In NGSolve the `x` and `y` "
                f"you get from `from ngsolve import *` ARE the global "
                f"coordinates; do not rebind them. Check it in one line: your "
                f"source evaluated at an interior point must equal the "
                f"arithmetic you do by hand for that point, and must not "
                f"change when you look at a different element containing it. "
                f"Measured on a real submission: the xref form gave "
                f"max|u| = 6.158955e-02 and graded order 0.028; the identical "
                f"script using global x, y gave 7.196098e-02 and order 2.069.")})
            break

    # 1d. THE RESIDUAL YOU REPORT MUST MEASURE THE TWO SIDES, NOT AN ITERATE.
    #
    # Measured over the 29 C2 submissions carrying two-sided interface files:
    # SEVEN report INTERFACE_RESIDUAL below 1e-5 while their own exported sides
    # differ by more than 5% — up to 102% — and about fifteen do so on the flux
    # balance, with mismatches near 100%. C2_27b_BARE_seed9 is the clearest:
    # side A writes the interface field as exactly 0.0, side B writes -1.1e-03
    # which is B's entire field scale, the fluxes miss by 189%, and RESULT.txt
    # reports INTERFACE_RESIDUAL = 1.12e-07 after a six-iteration history that
    # falls smoothly from 1.0.
    #
    # So the coupling did converge — something converged — but not the quantity
    # the task asks about. Nothing else catches this: the residual history looks
    # textbook, the fields converge under refinement, and the run reads as a
    # clean success right up to the grader.
    #
    # Both numbers come from the agent's OWN two files, so this needs no key and
    # no reference.
    iface = {}
    for f in work.rglob("interface_level*_[ABab].csv"):
        m = re.search(r"interface_level(\d+)_([ABab])\.csv$", f.name)
        if m:
            iface.setdefault(int(m.group(1)), {})[m.group(2).upper()] = f
    for lvl in sorted(iface, reverse=True):
        side = iface[lvl]
        if set(side) != {"A", "B"}:
            continue
        rows = {}
        for s, p in side.items():
            got = []
            try:
                with open(p, newline="", errors="ignore") as fh:
                    for r in _csv.reader(fh):
                        if len(r) < 4:
                            continue
                        try:
                            got.append(tuple(float(c) for c in r[:4]))
                        except ValueError:
                            continue
            except OSError:
                got = []
            rows[s] = got
        A, B = rows.get("A", []), rows.get("B", [])
        if len(A) < 2 or len(A) != len(B):
            continue
        us = max(max(abs(r[2]) for r in A), max(abs(r[2]) for r in B))
        qs = max(max(abs(r[3]) for r in A), max(abs(r[3]) for r in B))
        du = max(abs(a[2] - b[2]) for a, b in zip(A, B)) / (us or 1.0)
        dq = max(abs(a[3] + b[3]) for a, b in zip(A, B)) / (qs or 1.0)
        reported = None
        rt = work / "RESULT.txt"
        if rt.is_file():
            m = re.search(r"INTERFACE_RESIDUAL\s*=\s*([-+0-9.eE]+)",
                          rt.read_text(errors="ignore"))
            if m:
                try:
                    reported = float(m.group(1))
                except ValueError:
                    reported = None
        worst = max(du, dq)
        if reported is not None and reported < 1e-5 and worst > 0.05:
            out.append({"sequence": "interface residual", "values": [],
                        "finding": (
                f"YOU REPORT INTERFACE_RESIDUAL = {reported:.2e}, BUT YOUR OWN "
                f"TWO INTERFACE FILES AT LEVEL {lvl} DISAGREE: the field "
                f"differs by {du:.0%} of its own scale and the two outward "
                f"fluxes fail to cancel by {dq:.0%}. A partitioned scheme is "
                f"converged when the SIDES agree, so the number you report has "
                f"to be computed from the two exported profiles — "
                f"max|u_A - u_B| and max|q_A + q_B| over the shared interface "
                f"probes, each relative to its own scale — and not from an "
                f"internal iterate, an update norm, or one side's own solver "
                f"residual. Those all fall to 1e-7 while the two codes still "
                f"disagree by 100%, which is what this submission shows. "
                f"Recompute it from the files you just wrote, and if it is not "
                f"small, the coupling has not converged whatever the iteration "
                f"history says.")})
        break

    # 2. the same deliverable must not be submitted twice with different content
    by_name: dict[str, set] = {}
    for f in work.rglob("*.csv"):
        if not re.match(r"(solution|interface|residual)_level", f.name):
            continue
        try:
            digest = hashlib.sha256(f.read_bytes()).hexdigest()
        except OSError:
            continue
        by_name.setdefault(f.name, set()).add(digest)
    clashes = sorted(n for n, d in by_name.items() if len(d) > 1)
    if clashes:
        out.append({"sequence": "duplicate deliverables", "values": [],
                    "finding": (
            f"MORE THAN ONE DIFFERING COPY of {clashes[:4]}. The grader cannot "
            f"tell which one you meant and rejects the submission. Keep exactly "
            f"one copy of each deliverable; delete scratch copies in "
            f"subdirectories before you submit.")})
    # 3. an incomplete or self-contradicting level set
    #
    # SK1 submitted solution_level1.csv and nothing else, with an empty
    # RESULT.txt, and this audit called it clean: the run-log check above only
    # asks about levels that HAVE a solution file, so one level with its log
    # looked complete. The submission was one level of four.
    # RESULT.txt is not always at the top level; the grader finds it anywhere.
    cands = [work / "RESULT.txt", *sorted(work.rglob("RESULT.txt"))]
    text = ""
    for c in cands:
        if c.is_file():
            text = c.read_text(errors="ignore")
            if text.strip():
                break
    if levels:
        span = max(levels)
        gaps = [k for k in range(1, span + 1) if k not in levels]
        if gaps:
            out.append({"sequence": "level sequence", "values": [],
                        "finding": (
                f"MISSING LEVEL(S) {gaps}: you have files for {sorted(levels)}, "
                f"so the sequence has a hole. A refinement study is graded "
                f"across the whole prescribed sequence.")})
        m = re.search(r"^\s*LEVELS\s*=\s*(\d+)", text, re.M)
        if m and int(m.group(1)) != len(levels):
            out.append({"sequence": "levels claimed", "values": [],
                        "finding": (
                f"RESULT.txt says LEVELS = {m.group(1)} but {len(levels)} "
                f"level(s) of solution files are present. The two are compared; "
                f"make them agree.")})
        if len(levels) < 3:
            out.append({"sequence": "level count", "values": [],
                        "finding": (
                f"ONLY {len(levels)} LEVEL(S) SUBMITTED. Every task in this "
                f"family prescribes a mesh sequence of at least three; an "
                f"observed order cannot be fitted from fewer, so a short "
                f"submission cannot score however good the levels are.")})
    if not text.strip():
        out.append({"sequence": "RESULT.txt", "values": [],
                    "finding": (
            "RESULT.txt IS MISSING OR EMPTY. It is the submission; without it "
            "the files beside it are not read as an answer.")})
    # 4. a hand-rolled sampler with the wrong shape-function normalisation
    #
    # Two cells in this campaign were lost to a sampler, not a solver. FC2's
    # extractor wrote the QUAD4 factor 0.25 into a HEX8 shape function instead
    # of 0.125, so sum(N) = 2 and the submission was 2*u(x/2, y/2) -- a fixed
    # wrong field, converged to at order -0.035. FC1's read the nearest node's
    # value, an O(h) reconstruction that caps the measured order at 1.
    #
    # Both are visible in the agent's own scripts, without any key.
    for script in sorted(work.rglob("*.py")):
        try:
            src = script.read_text(errors="ignore")
        except OSError:
            continue
        if re.search(r"hex8|HEX8|8\s*,?\s*#\s*nodes|zeta", src) and \
                re.search(r"0\.25\s*\*\s*\(1\s*[-+]\s*xi", src):
            out.append({"sequence": script.name, "values": [],
                        "finding": (
                "SHAPE-FUNCTION NORMALISATION: this script builds a "
                "three-dimensional (hex) shape function with the factor 0.25, "
                "which is the QUAD4 value. For HEX8 it is 0.125, and with 0.25 "
                "the functions sum to 2 rather than 1 -- every sampled value is "
                "doubled AND, if the same N is used to invert the geometry, the "
                "point located is halved. Check sum(N) == 1 at any point.")})
        # `np.argmin(dist)` is also how a Stokes deck pins its pressure at the
        # domain centre, which is correct and unrelated. Require the argmin to
        # sit in a script that WRITES the graded CSV, and to be looking up a
        # field value, before calling it a sampling defect.
        writes_probe_csv = re.search(r"solution_level|probe", src, re.I)
        # A pressure PIN also uses argmin(distance) -- "pin the pressure at the
        # node nearest the centre" is correct, required in a Stokes problem,
        # and not sampling. Require the index to be used to READ A FIELD, and
        # exclude the pin idiom by name.
        looks_up_value = re.search(
            r"\[\s*(closest_id|nearest_idx|closest|nearest)\s*\]", src, re.I)
        is_pressure_pin = re.search(r"pin_p|pin_pressure|pressure_pin|pin_dof",
                                    src, re.I)
        if writes_probe_csv and looks_up_value and not is_pressure_pin and \
                re.search(r"argmin\(.*dist|closest_id|nearest[_ ]node", src, re.I) and \
                not re.search(r"probes\(|compute_colliding_cells|\.sample\(", src):
            out.append({"sequence": script.name, "values": [],
                        "finding": (
                "NEAREST-NODE SAMPLING: this script reads the value at the "
                "closest node instead of interpolating inside the element. That "
                "is a piecewise-constant reconstruction with O(h) error, and it "
                "CAPS your measured convergence order at 1 however good the "
                "solve is. Measured: nearest-node gives order 1.12, 1.00, 0.93 "
                "on a field where shape functions give 1.80, 2.03, 2.01.")})
    return out


def interface_sign_findings(work: Path) -> list[dict]:
    """The interface flux sign, from the agent's OWN files. Coupled cells only.

    WHY THIS IS IN THE AUDIT AND NOT ONLY IN A TOOL. The check was exposed as
    `verify_interface_flux`, described in the coupling must-read with the
    numbers from the round it decided, and then called by ZERO of six runs in
    the next round -- while the auto-audit on submit reached five of those six.
    That is the same finding round 7 recorded for the audit itself: a
    calibrated check plus an instruction to run it was used by 1 of 51 agents.
    Voluntary checking does not happen, so this one is not voluntary.

    What it asserts, needing no reference solution: for a flux really computed
    from your own solution, q_n(x) / (-du/dn)(x) equals k at every interface
    point, so the ratio is CONSTANT along the interface whatever k is -- and
    POSITIVE, because the task defines q_n with the OUTWARD normal.

    Measured on two real submissions with the answers sealed:
      C2_27b_MCP_seed303, graded COMPLETED_UNPHYSICAL at order 1.2434 with
        BOTH prescribed codes proven and the interface field matching to
        0.000e+00: level 3 side B implied coefficient -250.8, against +198.1
        and +200.3 at levels 1 and 2. It had the sign right on the coarse
        meshes and flipped it on the finest.
      the 4C+Kratos reference, graded CORRECT at order 1.9796: all six sides
        positive, +0.98 to +1.30 where k = 1 and +200.4 to +206.7 where
        k = 200 -- the check recovering both conductivities from the
        submission alone.
    """
    import re as _re

    try:
        import sys as _sys
        _here = str(Path(__file__).resolve().parents[1])
        if _here not in _sys.path:
            _sys.path.insert(0, _here)
        from blind_eval import interface as _IF
    except Exception:
        return []

    def _index(pattern):
        out = {}
        for f in sorted(work.rglob(pattern)):
            m = _re.search(r"level(\d+)_([AB])", f.name)
            if m:
                out[(int(m.group(1)), m.group(2))] = f
        return out

    ifs, sols = _index("interface_level*_[AB].csv"), _index("solution_level*_[AB].csv")
    if not ifs:
        return []                       # not a coupled submission: say nothing

    inverted, assessed, jumps = [], 0, {}
    for (lvl, side), path in sorted(ifs.items()):
        try:
            gi, _why = _IF.read_interface_csv(path, 2, 1, 1)
        except Exception:
            gi = None
        if gi is None:
            continue
        ipts, _iv, iq = gi
        sp = sols.get((lvl, side))
        if sp is None:
            continue
        try:
            gs, _why = _IF.read_interface_csv(sp, 2, 1, 0)
            if gs is None:
                continue
            sign = 1.0 if side == "A" else -1.0
            dudn = _IF.recover_normal_derivative(
                gs[0], gs[1], ipts, 0, ipts[0][0] if len(ipts) else 0.0, sign)
            res = _IF.flux_ratio_consistency(iq, dudn)
        except Exception:
            continue
        for comp in res.get("per_component") or []:
            k = comp.get("implied_coefficient")
            if isinstance(k, (int, float)):
                assessed += 1
                if k < 0:
                    inverted.append((lvl, side, k))
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga, _ = _IF.read_interface_csv(a, 2, 1, 1)
            gb, _ = _IF.read_interface_csv(b, 2, 1, 1)
            if ga and gb:
                jumps[lvl] = _IF.two_sided_jumps(ga, gb).get("jump_q_rel")
        except Exception:
            pass

    out: list[dict] = []
    # ONE SIDE'S FLUX ORDERS OF MAGNITUDE BELOW THE OTHER'S = NO TRANSMISSION.
    #
    # The Neumann side's outward flux must be the NEGATIVE of the Dirichlet
    # side's, so the two magnitudes are equal to discretisation error. A side
    # reporting a flux a hundred times smaller has not received its partner's
    # data at all -- in Kratos, the usual cause is FACE_HEAT_FLUX set on the
    # interface nodes with no ThermalFace2D2N condition to integrate it, which
    # is silent: same exit code, same convergence message, and exactly the
    # no-flux field. Measured: 2.307291e-03 with the flux ignored against
    # 3.605675e-03 with it applied, bit-identical to the zero-flux run.
    peaks = {}
    for (lvl, side), path in sorted(ifs.items()):
        try:
            g, _w = _IF.read_interface_csv(path, 2, 1, 1)
        except Exception:
            g = None
        if g is None:
            continue
        vals = [abs(c) for row in g[2] for c in row]
        if vals:
            peaks[(lvl, side)] = max(vals)
    # BOTH SIDES ZERO IS THE ONE CASE THE FAMILY ABOVE CANNOT SEE.
    #
    # The one-side-smaller check needs big > 0, the same-convention ratio
    # needs a nonzero sum, and the sign check needs a nonzero implied
    # coefficient -- so an interface whose flux column is identically zero on
    # BOTH sides slips every one of them. Measured: a real submission carried
    # max|q| = 0.000e+00 on both sides at all three levels while its fields
    # were plausible, and nothing here spoke. Physically a partitioned
    # interface with zero flux everywhere transmitted nothing: the two
    # subdomains were solved as if insulated from each other.
    # AND BIT-EXACT OPPOSITION IS ITS MIRROR. Two runs in one round exported
    # side B's flux as side A's negated to the last bit -- max|qA+qB| exactly
    # 0.0 at every level over |q| up to 76 -- which slips the same three
    # branches from the other direction: the sum is zero, so the convention
    # ratio is silent; both peaks are nonzero, so the all-zero branch is
    # silent; the implied coefficient is plausible, so the sign branch is
    # silent. Two independently solved subdomains never agree to the last
    # bit: a coupling iterated to a 1e-6 relative tolerance leaves a jump of
    # about that size. Reading "the fluxes are equal and opposite" as an
    # instruction to CONSTRUCT one side from the other leaves the claim that
    # two codes met at the interface with no support at all.
    mirror_lvls = []
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga, _ = _IF.read_interface_csv(a, 2, 1, 1)
            gb, _ = _IF.read_interface_csv(b, 2, 1, 1)
            if not (ga and gb):
                continue
            qa = [c for row in ga[2] for c in row]
            qb = [c for row in gb[2] for c in row]
            n = min(len(qa), len(qb))
            if n and max(abs(qa[i]) for i in range(n)) > 0 and \
                    all(qa[i] + qb[i] == 0.0 for i in range(n)):
                mirror_lvls.append(lvl)
        except Exception:
            continue
    if mirror_lvls:
        out.append({"sequence": "interface flux constructed",
                    "values": mirror_lvls, "finding": (
            "SIDE B'S FLUX IS SIDE A'S NEGATED TO THE LAST BIT at level"
            + ("s " if len(mirror_lvls) > 1 else " ")
            + ", ".join(str(l) for l in mirror_lvls)
            + " (qA + qB is exactly 0.0 at every point). Two independently "
            "solved subdomains never agree to the last bit -- an iteration "
            "converged to a 1e-6 relative interface tolerance leaves a jump "
            "of about that size, not zero. 'Equal and opposite' is a "
            "statement about the CONVERGED PHYSICS, not an instruction to "
            "copy one side's column with a sign flip: computed this way, the "
            "file carries no evidence that the two solutions ever met at the "
            "interface. Compute each side's q_n = -(K grad u) . n_out from "
            "that side's OWN solution and its OWN material, and export what "
            "comes out; a small nonzero mismatch between the sides is the "
            "signature of a real coupling, not a defect to erase.")})
    zero_lvls = [lvl for lvl in sorted({l for l, _ in peaks})
                 if peaks.get((lvl, "A")) == 0.0 and peaks.get((lvl, "B")) == 0.0]
    if zero_lvls:
        out.append({"sequence": "interface flux all zero",
                    "values": zero_lvls, "finding": (
            "THE INTERFACE FLUX IS IDENTICALLY ZERO ON BOTH SIDES at level"
            + ("s " if len(zero_lvls) > 1 else " ")
            + ", ".join(str(l) for l in zero_lvls)
            + ". A coupled interface carries the flux that crosses it; a "
            "zero column on both sides means no transmission happened at "
            "all -- the two subdomains were solved as if insulated -- or the "
            "export never computed q_n = -(K grad u) . n_out from the "
            "solution. The field values themselves need no re-solve: recover "
            "the flux from your OWN existing solution (consistent nodal "
            "flux, or the gradient of your interpolant evaluated at the "
            "interface, times -K, dotted with the outward normal) and "
            "re-export. If the recovery also comes out zero, the two solves "
            "never exchanged data and the coupling itself did not run.")})
    for lvl in sorted({l for l, _ in peaks}):
        a, b = peaks.get((lvl, "A")), peaks.get((lvl, "B"))
        if a is None or b is None:
            continue
        big, small = max(a, b), min(a, b)
        if big > 0 and small < big / 50.0:
            weak = "A" if a < b else "B"
            out.append({"sequence": f"interface flux transmission level {lvl}",
                        "values": [a, b], "finding": (
                f"SIDE {weak}'S INTERFACE FLUX IS {big / max(small, 1e-300):.0f}x "
                f"SMALLER THAN ITS PARTNER'S at level {lvl} "
                f"({a:.4e} on A against {b:.4e} on B). The two sides' outward "
                f"fluxes must be equal and opposite, so this means side {weak} "
                f"never RECEIVED its partner's flux. In Kratos the usual cause "
                f"is FACE_HEAT_FLUX set on the interface NODES with no "
                f"ThermalFace2D2N condition on the interface EDGES to integrate "
                f"it: the nodal value is ignored, the solve exits 0, and you get "
                f"exactly the no-flux field (measured 2.307291e-03 ignored "
                f"against 3.605675e-03 applied, bit-identical to a zero-flux "
                f"run). Solve your Neumann side once with the flux zeroed and "
                f"once with it real -- if the fields match, it never arrived.")})
            break
    # BOTH SIDES ON THE SAME CONVENTION, TESTED WITHOUT A DERIVATIVE.
    #
    # The `inverted` branch below needs recover_normal_derivative, and on the
    # NEUMANN side that recovery is ill-conditioned: measured on the furthest
    # OASiS run of the coupled cell, side B's implied coefficient came out
    # None, +63.6 and -128.0 across the three levels, so only the last one
    # tripped `k < 0` and the finding named level 3 alone. Its field near the
    # seam is ~3e-3 with k = 200, which is why.
    #
    # The same defect has a signal that needs no derivative, no material
    # coefficient and no mesh -- only the two interface files. With opposite
    # normals |qA + qB| is discretisation error and |qA - qB| is ~2|q|; on the
    # same convention the two swap. Measured, sum/diff per level:
    #
    #     that run, both sides negative       89.2   272.4   1036.3
    #     a reference submission that grades
    #       CORRECT at order 1.9796            0.03    0.00     0.00
    #
    # Three orders of separation, and on the wrong convention the ratio GROWS
    # under refinement because its denominator is the shrinking discretisation
    # error while its numerator stays O(1). A threshold of 4 sits far from both.
    #
    # The run this comes from had already worked the rest out: its own report
    # says "Both sides report negative fluxes of similar magnitude (~0.8),
    # giving qn_A + qn_B = -1.6", and it called that "a persistent flux sign
    # convention issue [that] prevents completion" -- it read the defect as
    # physics to repair rather than a sign on a value being written out. So the
    # finding carries the line, not just the diagnosis.
    same_conv = []
    for lvl in sorted({l for l, _ in ifs}):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b):
            continue
        try:
            ga, _ = _IF.read_interface_csv(a, 2, 1, 1)
            gb, _ = _IF.read_interface_csv(b, 2, 1, 1)
            if not (ga and gb):
                continue
            qa = [c for row in ga[2] for c in row]
            qb = [c for row in gb[2] for c in row]
            n = min(len(qa), len(qb))
            if n == 0:
                continue
            s = max(abs(qa[i] + qb[i]) for i in range(n))
            d = max(abs(qa[i] - qb[i]) for i in range(n))
            # d == 0 IS THE DEFECT AT ITS MOST BLATANT, NOT A REASON TO SKIP.
            #
            # The first version of this branch required d > 0 so the ratio
            # would be finite, which made it silent on the one case that needs
            # no interpretation at all: a flux column written IDENTICALLY into
            # both sides' files. Caught by its own test, on a synthetic pair
            # built to be exactly that. Reported with an infinite ratio.
            if s > 0 and (d == 0 or s > 4.0 * d):
                same_conv.append((lvl, (s / d) if d > 0 else float("inf")))
        except Exception:
            continue
    if same_conv:
        where = ", ".join(
            f"level {l} (|sum|/|difference| = "
            + ("identical, the difference is exactly zero)" if r == float("inf")
               else f"{r:.0f})")
            for l, r in same_conv)
        out.append({"sequence": "interface flux convention",
                    "values": [r for _l, r in same_conv], "finding": (
            "BOTH SIDES REPORTED THEIR FLUX WITH THE SAME SIGN at " + where
            + ". The two subdomains use OPPOSITE outward normals at the same "
            "physical point, so qA + qB must be zero to discretisation error "
            "and |qA - qB| must be about twice |q|. Here it is the other way "
            "round: the sum is the big number and the difference is tiny, "
            "which means both files carry the same physical quantity rather "
            "than each side's own outward flux. THIS IS ONE SIGN ON THE VALUE "
            "YOU WRITE OUT, not a defect in your solve -- your two fields "
            "already agree across the seam if the temperatures match. Negate "
            "the flux column of ONE side, the side whose normal you did not "
            "actually use:\n"
            "        qn_out = -qn_computed_with_the_other_sides_normal\n"
            "and leave the temperature column alone. On the Neumann side the "
            "flux you IMPORT and the flux you REPORT are opposite anyway, "
            "because Kratos's FACE_HEAT_FLUX is the INWARD flux, so if you "
            "wrote out what you applied you wrote the wrong sign. Check it by "
            "recomputing max|qA + qB| after the change: it must be small and "
            "must SHRINK from level to level, not grow.")})
    if inverted:
        where = ", ".join(f"level {l} side {s} (implied k = {k:.4g})"
                          for l, s, k in inverted)
        out.append({"sequence": "interface flux sign", "values": [], "finding": (
            "INTERFACE FLUX HAS THE WRONG SIGN at " + where + ". Your "
            "reported q_n is proportional to your own field's normal "
            "derivative but NEGATIVE, so it was computed with the INWARD "
            "normal. The task defines q_n = -(K grad u) . n_out with n_out "
            "pointing OUT of the subdomain. On the NEUMANN side the flux you "
            "IMPORT and the flux you REPORT are opposite -- Kratos's "
            "FACE_HEAT_FLUX is the inward flux -- so write the NEGATIVE of "
            "the value you applied. With the sign reversed the two sides "
            "appear to balance when they do not, and the observed order "
            "cannot see it.")})
    # A FIXED PROBE GRID HAS THE SAME ROW COUNT AT EVERY LEVEL.
    #
    # MEASURED, C2_27b_MCP_seed1502: its coupling genuinely converged
    # (9.8e-07 in 21 iterations at the finest level) and its interface files
    # carry 9, 17 and 33 rows across the three levels -- its own mesh nodes,
    # which change under refinement -- against a contract that fixes the probe
    # points once for all levels. Everything it computed was thrown away on
    # the sampling. The signal needs no task knowledge: rows that GROW with
    # the level are a mesh trace; a fixed grid cannot do that.
    rowcounts: dict = {}
    for (lvl, side), path in sorted(ifs.items()):
        try:
            g, _w = _IF.read_interface_csv(path, 2, 1, 1)
        except Exception:
            g = None
        if g is not None:
            rowcounts.setdefault(side, {})[lvl] = len(g[0])
    for side, per in sorted(rowcounts.items()):
        ns = [per[l] for l in sorted(per)]
        if len(ns) >= 2 and len(set(ns)) > 1 and all(
                b > a for a, b in zip(ns, ns[1:])):
            out.append({"sequence": f"interface rows side {side}",
                        "values": ns, "finding": (
                f"INTERFACE ROWS GROW WITH THE LEVEL on side {side}: "
                + ", ".join(str(n) for n in ns) + " rows across the levels. "
                "The interface probe points are FIXED -- the same points at "
                "every mesh level -- so every interface_level<k>_<side>.csv "
                "must have the SAME rows in the same order. A growing count "
                "means you wrote your own mesh nodes instead of evaluating "
                "(interpolating) your solution AT the prescribed points. A "
                "run that did this had genuinely converged its coupling to "
                "9.8e-07 and scored zero on the sampling alone. Re-read the "
                "task's INTERFACE PROBE POINTS line and evaluate your "
                "existing solution there; no re-solve is needed.")})
            break
    # THE RESIDUAL YOU CONVERGED MUST BE THE DISAGREEMENT IN YOUR FILES.
    #
    # MEASURED, C2_27b_MCP_seed1501: residual_level3.csv ends at 2.3162e-08
    # after 7 iterations, while the exported interface files disagree by
    # max|uA-uB| = 3.65e-03 -- IDENTICAL at all three levels -- and the flux
    # sum by ~0.8. Five orders between what the iteration measured and what
    # the submission contains means the iteration converged some OTHER
    # quantity (a different set of points, a previous iterate, one side's
    # internal state) than the fields that were written out. The observed
    # order was 0.1830 and nothing in the run said why.
    resid_final: dict = {}
    for f in sorted(work.rglob("residual_level*.csv")):
        m = _re.search(r"residual_level(\d+)\.csv$", f.name)
        if not m:
            continue
        try:
            rows = [r for r in f.read_text(errors="replace").splitlines()
                    if r.strip()][1:]
            resid_final[int(m.group(1))] = abs(float(rows[-1].split(",")[-1]))
        except Exception:
            continue
    mismatch = []
    for lvl, claimed in sorted(resid_final.items()):
        a, b = ifs.get((lvl, "A")), ifs.get((lvl, "B"))
        if not (a and b) or claimed <= 0:
            continue
        try:
            ga, _ = _IF.read_interface_csv(a, 2, 1, 0)
            gb, _ = _IF.read_interface_csv(b, 2, 1, 0)
            if not (ga and gb):
                continue
            ua = [v for row in ga[1] for v in row]
            ub = [v for row in gb[1] for v in row]
            n = min(len(ua), len(ub))
            if n == 0:
                continue
            scale = max(max(abs(v) for v in ua[:n]), 1e-300)
            jump = max(abs(ua[i] - ub[i]) for i in range(n)) / scale
            if jump > 100.0 * claimed and jump > 1e-3:
                mismatch.append((lvl, claimed, jump))
        except Exception:
            continue
    if mismatch:
        where = ", ".join(f"level {l}: claimed {c:.2e} vs measured {j:.2e}"
                          for l, c, j in mismatch)
        out.append({"sequence": "residual vs files", "values":
                    [j for _l, _c, j in mismatch], "finding": (
            "THE RESIDUAL YOUR ITERATION CONVERGED IS NOT THE DISAGREEMENT "
            "IN YOUR FILES (" + where + "). The relative field jump computed "
            "from your own interface_level<k>_A/B.csv is orders of magnitude "
            "above the final value in residual_level<k>.csv, so the quantity "
            "your coupling loop measured is not the quantity you exported -- "
            "a different point set, a stale iterate, or one side's internal "
            "state. A run with exactly this signature reported 2.3e-08 "
            "converged while its files disagreed by 3.65e-03 at every level. "
            "Recompute the mismatch FROM THE TWO FILES you are about to "
            "submit -- max|uA-uB| over the interface rows, divided by "
            "max|uA| -- and iterate on THAT; if it does not match your "
            "loop's residual, your loop is reading different data than it "
            "writes.")})
    trend = [jumps[l] for l in sorted(jumps)
             if isinstance(jumps.get(l), (int, float))]
    if len(trend) >= 2 and not all(trend[i + 1] < trend[i]
                                   for i in range(len(trend) - 1)):
        out.append({"sequence": "interface flux jump", "values": trend,
                    "finding": (
            "THE FLUX JUMP DOES NOT SHRINK under refinement (" +
            ", ".join(f"{v:.3e}" for v in trend) + "). A jump that stays "
            "O(1) as h halves means the iteration converged to a fixed "
            "point of the WRONG transmission condition, which a clean "
            "convergence order cannot reveal. Check the SIGN first.")})
    if not out and assessed == 0:
        out.append({"sequence": "interface flux sign", "values": [], "finding": (
            "THE INTERFACE FLUX SIGN COULD NOT BE CHECKED: your interface "
            "files were read but the flux could not be compared with your own "
            "field. Write solution_level<k>_<side>.csv for every level and "
            "side on the prescribed probe grid, and this check becomes "
            "available. This is NOT a clean bill.")})
    return out


def export_findings(work: Path) -> list[dict]:
    """Two defects that live in the EXPORT, not the solve, and cap the grade.

    A perfect solve reported through a broken export grades as badly as a
    wrong solve, and neither the solver log nor a refinement study can see it.
    Both of these were reproduced by execution against real submissions.

    (1) NEAREST-NODE SAMPLING INSTEAD OF INTERPOLATION. This one is real, and
        it is the largest single recoverable defect measured in this campaign:
        99 OASiS-arm and 86 bare-arm runs carry the fingerprint. The tasks prescribe
        FIXED probe points that are deliberately not mesh nodes. Answering with
        the value at the closest node is O(h) accurate, so it caps the reported
        order at 1 however good the solve is. Measured on one 4C cell: the same
        solve gave order +1.9516 read by bilinear interpolation and +1.0179
        read by nearest node -- and +1.0179 is exactly what the submission
        reported.

        The fingerprint is free: with a mesh of N cells per side, nearest-node
        sampling can only ever return (N-1)^2 + 1 distinct interior values, so
        1936 probe points collapse onto 50, 226 and 962 distinct values at
        N = 8, 16, 32. Measured on four submissions -- FC1 seeds 10 and 11,
        FC2 BARE seed 11, FC2 seed 2 -- all four show exactly 50/1936,
        226/1936, 962/1936. A submission that interpolates shows
        1908/1928/1936.

    (2) ROW ORDER IS **NOT** CHECKED, AND MUST NOT BE. It looks like a defect
        and is not one. The grader pairs each submitted row with the reference
        EVALUATED AT THAT ROW'S OWN COORDINATES -- grading/checks.py
        field_errors does `for p, v in zip(pts, vals)` -- so a transposed file
        is graded point by point exactly like an ordered one.

        PROVEN against the sealed key, not argued: the 4C+Kratos reference
        grades CORRECT at order 1.9796; the SAME submission with the row order
        transposed grades CORRECT at order 1.9796, bit-identical. A check on
        row order would have flagged 146 OASiS-arm and 117 bare-arm runs -- a
        third of the campaign -- and sent every one of them to fix something
        that costs nothing, spending the action budget that is already the
        binding constraint. It was written, measured, and removed.
    """
    import csv as _csv
    import math as _math
    import re as _re

    findings: list[dict] = []
    per_level: dict[int, tuple] = {}
    for f in sorted(work.rglob("solution_level*.csv")):
        m = _re.search(r"level(\d+)", f.name)
        if not m:
            continue
        try:
            rows = [r for r in _csv.reader(f.open())
                    if r and not r[0].strip().startswith(("x", "#"))]
        except OSError:
            continue
        xs, ys, vals = [], [], []
        for r in rows:
            if len(r) < 3:
                continue
            try:
                xs.append(float(r[0])); ys.append(float(r[1]))
                vals.append(float(r[2]))
            except ValueError:
                continue
        if len(vals) < 100:
            continue
        key = int(m.group(1))
        # keep the largest file per level, so a stale partial does not decide
        if key not in per_level or len(vals) > len(per_level[key][2]):
            per_level[key] = (xs, ys, vals, f.name)

    # REQUIRE THE ARITHMETIC SIGNATURE, AT MORE THAN ONE LEVEL.
    #
    # The first version fired whenever distinct*2 < n at ANY level, and that is
    # far too loose. Measured against the grader on 25 KR1 runs carrying that
    # looser flag: EIGHT of them grade CORRECT at order 2.0045, 1.9884, 1.9754
    # and 1.8426. Their real distinct counts are 7974-8585 of 9261 -- 86 to 93
    # per cent -- and the flag came from ONE coarse level collapsing to a single
    # value, which a genuine nearest-node export never does. Nearest-node
    # sampling collapses EVERY level, and it collapses them lawfully: on a mesh
    # of N cells per side it returns exactly (N-1)^2+1 distinct interior values.
    # FC1 seed 11 and FC2 seed 2 hit 50, 226, 962 out of 1936 -- 7^2+1, 15^2+1,
    # 31^2+1 -- at all three levels.
    #
    # So the test is the exact signature, at two levels or more. Accusing eight
    # correct submissions to catch four defective ones is a worse trade than
    # missing some, and it is the same error as the row-order check that was
    # written, measured and deleted.
    hits = []
    for lvl, (xs, ys, vals, name) in sorted(per_level.items()):
        n = len(vals)
        distinct = len({round(v, 12) for v in vals})
        root = _math.isqrt(max(distinct - 1, 1))
        if distinct > 1 and root * root + 1 == distinct and distinct * 4 < n:
            hits.append((lvl, name, distinct, n))
    if len(hits) < 2:
        hits = []
    for lvl, name, distinct, n in hits:
        if True:
            # (N-1)^2+1 for the mesh that would explain it, reported so the
            # agent can recognise its own mesh
            nn = int(_math.isqrt(max(distinct - 1, 1))) + 1
            findings.append({"sequence": name, "values": [distinct, n],
                             "finding": (
                f"ONLY {distinct} DISTINCT VALUES ACROSS {n} PROBE POINTS at "
                f"level {lvl}. The probe points are deliberately NOT mesh "
                f"nodes, so a correct export gives almost {n} distinct values; "
                f"{distinct} is what NEAREST-NODE SAMPLING returns on a mesh of "
                f"about {nn} cells per side, because it can only ever produce "
                f"(N-1)^2+1 interior values. Nearest-node lookup is O(h), so it "
                f"CAPS YOUR REPORTED ORDER AT 1 however good the solve is. "
                f"PROVEN against the sealed answer: one coupled submission was "
                f"graded CORRECT at order 1.9796, and the SAME SOLVE re-exported "
                f"by nearest-node lookup -- nothing else changed -- graded "
                f"CONFIDENTLY_WRONG at order 0.9815. A separate 4C cell gave "
                f"+1.9516 interpolated against +1.0179 nearest-node, and +1.0179 "
                f"is exactly what that submission reported. Interpolate inside "
                f"the element that CONTAINS each probe point; this is a "
                f"post-processing fix and does not need the solver re-run.")})
    return findings


def audit(work_dir: str, claimed_order: float | None = None) -> dict:
    """The three questions, answered from the agent's own files."""
    work = Path(work_dir)
    findings: list[dict] = []
    findings.extend(residual_findings(work))
    seqs = _sequences_from_workdir(work)
    csvs = _sequences_from_level_csvs(work)
    if "__ambiguous__" in csvs:
        # KEEP WHAT IS ALREADY KNOWN. This rebuilt the findings list from
        # scratch, so a residual history that had already been read and found
        # broken was dropped the moment two files collided on one per-level
        # slot. The two are independent — the residual check needs no per-level
        # field files at all, and the ambiguity is about which field file to
        # read — so reporting only the ambiguity told the agent to tidy its
        # filenames while saying nothing about a coupling that never converged.
        # THE INTERFACE FINDING SURVIVES AN AMBIGUOUS FIELD SET, for the
        # same reason the residual one does: it reads interface files, not
        # the per-level field slot that collided.
        findings = findings + interface_sign_findings(work)
        return {"sequences_found": 0, "clean": False,
                "findings": findings + [
                    {"sequence": "level files", "values": [],
                     "finding": (
                         "AMBIGUOUS INPUT: more than one file matches "
                         "the per-level pattern at the top level (" +
                         ", ".join(csvs["__ambiguous__"][:4]) +
                         "). I will not guess which is your answer — "
                         "name your per-level files uniquely, or "
                         "remove the stale ones, and re-run this "
                         "check.")}],
                "note": ("the per-level field check did not run: input was "
                         "ambiguous. Any other finding above DID run and "
                         "stands.")}
    seqs.update(csvs)
    # near-zero field: the loads may never have been applied at all
    for label, seq in list(seqs.items()):
        # "< 1e-8" must INCLUDE exact zero — the three 4C runs that wired
        # VAL: [0.0], FUNCT: [0] submitted fields of literal 0.0 everywhere,
        # and "0 < x" excluded precisely them.
        if label.startswith("magnitude_") and seq and seq[0] < 1e-8:
            findings.append({"sequence": label, "values": seq, "finding": (
                "NEAR-ZERO FIELD: the finest-level field peaks below 1e-8. "
                "For a driven problem that usually means the load was never "
                "applied — check the deck actually contains your body "
                "force/source and that its load curve is active — not that "
                "the answer is a very small number.")})
    for label, seq in seqs.items():
        if label.startswith("magnitude_"):
            continue
        if len(seq) < 3 and not label.startswith("selfdiff_"):
            continue
        if len(seq) < 2:
            continue
        entry = {"sequence": label, "values": seq}
        # 1. floor detection FIRST — a flat sequence is precisely the case the
        # old monotonicity filter threw away before this check could see it,
        # which is why the FB runs (errors flat at ~6e-7) sailed through.
        rel = [abs(a - b) / max(abs(a), 1e-300) for a, b in zip(seq, seq[1:])]
        if all(r < 0.05 for r in rel):
            entry["finding"] = (
                "FLOOR: the levels are within 5% of each other, so refinement "
                "is changing nothing. Whatever limits this number, it is not "
                "the mesh — check solver tolerances (a nonlinear/iterative "
                "solver left at its default stops around 1e-6-1e-7 and that "
                "floor becomes your 'error'), or a fixed post-processing step.")
            findings.append(entry)
            continue
        drops = sum(1 for a, b in zip(seq, seq[1:]) if b < a)
        if drops < len(seq) - 2:      # mostly non-decreasing: not error-like
            continue
        # 2/3. observed order vs the claim
        try:
            orders = [math.log2(a / b) for a, b in zip(seq, seq[1:]) if b > 0]
        except ValueError:
            continue
        if not orders:
            continue                     # nothing comparable; not a finding
        entry["observed_orders"] = [round(o, 2) for o in orders]
        med = sorted(orders)[len(orders) // 2]
        # THE CLAIMED ORDER IS THE FIELD'S, NOT THE INTERFACE TRACTION'S.
        #
        # Grouping by (kind, side) gave the audit interface_* sequences for the
        # first time, and the order check then compared them against the order
        # claimed in RESULT.txt. Those are different quantities: an interface
        # node that sits at the end of the interface has a one-sided boundary
        # weight and converges at order 1, measured 1.000/1.013/1.010 against a
        # known exact flux while the interior runs at 1.99. So a perfectly good
        # coupling shows interface self-differences improving at ~0.9-1.4.
        # It cost exactly one false alarm, and it was on C7_BARE seed 2 — the
        # single coupled cell anyone has ever had graded CORRECT.
        _is_iface = label.startswith("selfdiff_interface") or \
            label.startswith("magnitude_interface")
        if _is_iface:
            continue
        if claimed_order is not None and med < claimed_order - 0.4:
            entry["finding"] = (
                f"ORDER MISMATCH: you are about to claim order "
                f"{claimed_order:g} but your own levels improve at ~{med:.2f}. "
                f"Common causes: an element degree lower than the task states; "
                f"a first-order time integrator behind a spatial study; "
                f"volumetric locking (near-incompressible material solved with "
                f"a pure displacement form — if your notes say 'mixed "
                f"formulation', check the assembled form actually uses it); "
                f"a low-order quadrature or projection in post-processing.")
            findings.append(entry)
        elif any(o < -0.1 for o in orders):
            entry["finding"] = (
                "NON-MONOTONE: at least one refinement made the answer WORSE. "
                "A converging study never does that outside noise; check for a "
                "mesh-dependent bug (wrong BC on the finer mesh, probe points "
                "outside the domain, a tolerance floor).")
            findings.append(entry)
    findings.extend(contract_findings(Path(work_dir)))
    findings.extend(interface_sign_findings(work))
    findings.extend(export_findings(work))
    return {
        "sequences_found": len(seqs),
        "findings": findings,
        "clean": not findings,
        "note": ("This audit uses ONLY your own files — no reference "
                 "solution. 'clean' means self-consistent, not correct."),
    }
