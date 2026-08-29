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

import json
import math
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
    return out


def audit(work_dir: str, claimed_order: float | None = None) -> dict:
    """The three questions, answered from the agent's own files."""
    work = Path(work_dir)
    findings: list[dict] = []
    findings.extend(residual_findings(work))
    seqs = _sequences_from_workdir(work)
    csvs = _sequences_from_level_csvs(work)
    if "__ambiguous__" in csvs:
        return {"sequences_found": 0, "clean": False,
                "findings": [{"sequence": "level files", "values": [],
                              "finding": (
                                  "AMBIGUOUS INPUT: more than one file matches "
                                  "the per-level pattern at the top level (" +
                                  ", ".join(csvs["__ambiguous__"][:4]) +
                                  "). I will not guess which is your answer — "
                                  "name your per-level files uniquely, or "
                                  "remove the stale ones, and re-run this "
                                  "check.")}],
                "note": "audit did not run: input was ambiguous"}
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
    return {
        "sequences_found": len(seqs),
        "findings": findings,
        "clean": not findings,
        "note": ("This audit uses ONLY your own files — no reference "
                 "solution. 'clean' means self-consistent, not correct."),
    }
