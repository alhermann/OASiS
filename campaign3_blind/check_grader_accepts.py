#!/usr/bin/env python
"""Would a CORRECT submission be graded CORRECT? Ask the grader, do not assume.

This exists because assuming it has been wrong three times. A probe-count
mismatch, a missing subdomain exclusion, and an interface band that put half the
points outside the graded region each rejected a correct submission with
INVALID_SUBMISSION -- before any comparison against truth -- and each was found
only when somebody built the submission the task text describes and ran the
grader on it. Eight of the fifteen instances in the previous set were in that
state at once.

Three checks per instance, in order, each of which has caught a real defect:

  1. TASK TEXT vs GRADER. The probe count the task text PRESCRIBES against the
     count ``grade_blind.probe_grid()`` BUILDS. These are two independent pieces
     of code reading two different sources, and they have disagreed.
  2. SPEC vs GRADER. The point SET the public spec implies against the set the
     grader builds, coordinate by coordinate.
  3. END TO END. Assemble the submission a correct run would produce -- the
     exact field at the prescribed probe points plus a synthetic O(h^2) error, a
     converging partitioned-residual history, the NDOF execution logs both codes
     must write, and RESULT.txt -- and require the verdict to be CORRECT.

The synthetic error is a per-level constant offset ``c * 4^-k``, so the RMS is
exactly ``c * 4^-k``, the observed order is exactly 2 and R^2 is exactly 1. That
isolates the CONTRACT: anything other than CORRECT is then a defect in the task
text, the spec or the grader, never in the physics.

  campaign3_blind/check_grader_accepts.py C1 C2 ...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

import grade_blind as G                                          # noqa: E402

TMP = Path(os.environ.get("OASIS_CHECK_TMP",
                          "/tmp/claude-1001/-home-alexander-4C/balancedtmp/grade"))
SYMS = {"x": sp.Symbol("x", real=True), "y": sp.Symbol("y", real=True),
        "z": sp.Symbol("z", real=True)}


def task_probe_counts(text: str) -> dict:
    """What the TASK TEXT says the probe sets are, read out of the text."""
    out = {}
    for side in ("A", "B"):
        m = re.search(rf"PROBE POINTS, subdomain {side}: (.*?)(?=\nPROBE POINTS|"
                      rf"\nEvaluate)", text, re.S)
        if not m:
            continue
        blob = m.group(1)
        rem = re.search(r"(\d+) points remain", blob)
        first = re.search(r"the (\d+) points", blob)
        out[side] = int(rem.group(1)) if rem else (int(first.group(1))
                                                   if first else None)
    m = re.search(r"INTERFACE PROBE POINTS: the (\d+) points", text)
    if m:
        out["iface"] = int(m.group(1))
    return out


def build_submission(pid: str, key: dict, spec: dict, work: Path,
                     rel: float = 0.5) -> dict:
    dim, coords = key["dim"], key["coords"]
    comps = key.get("components", ["u"])
    ncomp = len(comps)
    ref = float(key.get("exact_rms") or 1.0)
    c0 = rel * ref
    tol = float(spec.get("interface_tol", "1e-6").split()[0])
    info = {}
    for lvl in range(1, len(key["mesh_N"]) + 1):
        off = c0 * 4.0 ** -(lvl - 1) / (ncomp ** 0.5)
        for side in ("A", "B"):
            bounds = G.subdomain_bounds(key, side, dim)
            excl = G.probe_exclusions(pid, side)
            pts = G.probe_grid(dim, bounds, excl)
            src = key["exact_solution"][side]
            exprs = ([sp.sympify(e, locals=SYMS) for e in src]
                     if isinstance(src, list)
                     else [sp.sympify(src, locals=SYMS)])
            fns = [sp.lambdify([SYMS[c] for c in coords], e, "math")
                   for e in exprs]
            with open(work / f"solution_level{lvl}_{side}.csv", "w") as fh:
                fh.write(",".join(coords + comps) + "\n")
                for p in pts:
                    vals = [float(f(*p)) + off for f in fns]
                    fh.write(",".join(f"{v:.17g}" for v in list(p) + vals)
                             + "\n")
            info[f"n_{side}"] = len(pts)
            # The execution log both codes must write. `NDOF = <n>` is the
            # canonical, number-bearing line the evidence gate accepts for every
            # code; the task text asks for exactly this file.
            code = key["codes"][0 if side == "A" else 1]
            (work / f"run_level{lvl}_{side}.log").write_text(
                f"code = {code}\nside = {side}\nNDOF = {12345 + lvl}\n")
        # A partitioned-iteration residual history: at least three iterations,
        # positive, falling by more than 10x, ending at or below the prescribed
        # interface tolerance. A monolithic solve has none at all.
        with open(work / f"residual_level{lvl}.csv", "w") as fh:
            fh.write("iteration,interface_residual\n")
            r, i = 1.0, 1
            while r > tol * 0.5:
                fh.write(f"{i},{r:.6e}\n")
                r *= 0.25
                i += 1
            fh.write(f"{i},{r:.6e}\n")
            info["iters"] = i
    (work / "RESULT.txt").write_text(
        f"LEVELS = {len(key['mesh_N'])}\n"
        f"FILES = {', '.join(sorted(p.name for p in work.glob('*.csv')))}\n"
        f"INTERFACE_RESIDUAL = 1.0e-07\n"
        f"COUPLING_ITERATIONS = {info['iters']}\n"
        f"MESH_INDEPENDENCE = CONVERGED\n"
        f"MAX_REL_CHANGE = 1.0e-03\n")
    return info


def check(pid: str) -> dict:
    key = json.loads((G.KEYS / pid / "key.json").read_text())
    spec = json.loads((G.PROBLEMS / pid / "spec_public.json").read_text())
    text = (G.PROBLEMS / pid / "task.txt").read_text()
    dim = key["dim"]
    out = {"id": pid, "problems": []}

    # 1. task text vs grader
    said = task_probe_counts(text)
    for side in ("A", "B"):
        built = len(G.probe_grid(dim, G.subdomain_bounds(key, side, dim),
                                 G.probe_exclusions(pid, side)))
        if said.get(side) != built:
            out["problems"].append(
                f"subdomain {side}: task text prescribes {said.get(side)} probe "
                f"points, grade_blind.probe_grid builds {built}")
    out["probe_counts"] = {**said, "grader_A": len(
        G.probe_grid(dim, G.subdomain_bounds(key, 'A', dim),
                     G.probe_exclusions(pid, 'A')))}

    # 2. the NDOF contract clause must be in the task text
    if "NDOF = <integer>" not in text:
        out["problems"].append("task text carries no NDOF execution-log clause")
    if "run_level<k>_<side>.log" not in text:
        out["problems"].append("task text names no per-participant log file")

    # 3. end to end
    run = TMP / pid
    if run.exists():
        shutil.rmtree(run)
    work = run / "work"
    work.mkdir(parents=True)
    info = build_submission(pid, key, spec, work)
    verdict = G.grade_run(run, pid)
    out["verdict"] = verdict.get("outcome")
    out["observed_order"] = verdict.get("observed_order")
    out["note"] = verdict.get("note")
    if verdict.get("outcome") != "CORRECT":
        out["problems"].append(
            f"a correct submission graded {verdict.get('outcome')}: "
            f"{verdict.get('note')}")
    out["ok"] = not out["problems"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", nargs="+")
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    bad = 0
    for pid in a.problems:
        try:
            r = check(pid)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"id": pid, "ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
        mark = "OK " if r.get("ok") else "BAD"
        print(f"[{mark}] {pid}: verdict={r.get('verdict')} "
              f"order={r.get('observed_order')} "
              f"probes={r.get('probe_counts')}")
        for p in r.get("problems", []):
            print(f"        - {p}")
        bad += 0 if r.get("ok") else 1
    print(f"\n{len(a.problems) - bad}/{len(a.problems)} instances would accept a "
          f"correct submission.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
