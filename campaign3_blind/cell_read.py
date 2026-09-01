"""Read one cell's runs the way the development loop needs them read.

Not a grader. The grader answers "was the answer right", offline, against a
sealed key. This answers the question that comes FIRST and that a score cannot
answer: where did the run stop, and what was it trying to do when it stopped.

For each seed it reports how far the run got along the path the task requires
(did it reach the coupling tool, did it write per-level files, did it write an
answer at all), what the harness did to it, and the last thing the agent said
before it stopped — which in practice is the single most informative line in
the whole trajectory.

usage: cell_read.py C1 [ARM] [seeds...]
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

D = Path(__file__).resolve().parent


def runs_for(cell, arm, seeds):
    out = []
    for s in seeds:
        d = D / "runs" / f"{cell}_27b_{arm}_seed{s}"
        if d.is_dir():
            out.append((s, d))
    return out


# WHICH FILES DID THE TASK ASK FOR? Ask the task, never a hardcoded pattern.
#
# This used to be one glob, `*level*_[AB].csv`, which matches ONLY the coupled
# per-side shape. Every single-code task asks for `solution_level<k>.csv` with
# no side suffix, so for 23 of the 47 cells the counter was structurally zero
# and printed a confident "level files: 0" under a run that had all four files
# on disk. NG1 MCP seed 96 wrote solution_level1..4.csv and read as 0/1.
#
# That is the fifth time a mechanism existed, was instrumented, and did not
# reach the case it was built for — and the worst-behaved of the five, because
# it does not fail loudly, it answers wrongly and confidently, to the tool I
# root-cause every cell with. So this reads the DEMANDED names out of the task
# text, and if the task demands a template it cannot parse it says so in
# capitals instead of reporting a zero.
# The side token appears both as the template `_<side>` and, where a task
# spells one level out, concretely as `_A` / `_B`. C13 and C14 write
# `run_level1_A.log` literally, and the first version of this regex accepted
# only the template — so the guard below caught my own parser on its first
# sweep, which is the whole reason it prints instead of returning zero.
_TEMPLATE = re.compile(
    r"\b([a-z_]+)_level(?:<k>|\d+)(_<side>|_[AB])?\.(csv|log)\b")


def demanded(cell: str) -> dict:
    """{artefact stem: glob} for every per-level file this task asks for."""
    task = D / "problems" / cell / "task.txt"
    if not task.is_file():
        task = D / "problems_dev3" / cell / "task.txt"
    if not task.is_file():
        return {}
    text = task.read_text(errors="replace")
    out = {}
    for stem, side, ext in _TEMPLATE.findall(text):
        # <k> and a literal level number are the same demand: one file per level
        glob = f"{stem}_level*{'_[AB]' if side else ''}.{ext}"
        out.setdefault(f"{stem}.{ext}", glob)
    return out


def _unparsed_templates(cell: str) -> list:
    """Per-level names in the task text that `_TEMPLATE` did not recognise.

    The bug this file is repairing was silent. A tool that reports "0" when it
    means "I do not understand the question" can mislead for weeks, so any
    unrecognised per-level artefact name is surfaced instead of dropped.
    """
    task = D / "problems" / cell / "task.txt"
    if not task.is_file():
        task = D / "problems_dev3" / cell / "task.txt"
    if not task.is_file():
        return []
    text = task.read_text(errors="replace")
    seen = {m.group(0) for m in _TEMPLATE.finditer(text)}
    all_names = set(re.findall(r"\b[A-Za-z_]+_level[0-9A-Za-z_<>]*\.[a-z]+\b", text))
    return sorted(all_names - seen)


def read_one(d: Path) -> dict:
    led = d / "ledger.json"
    L = json.loads(led.read_text()) if led.is_file() else {}
    w = d / "work"
    res = w / "RESULT.txt"
    txt = res.read_text(errors="replace") if res.is_file() else ""
    traj = w / "trajectory_live.txt"
    t = traj.read_text(errors="replace") if traj.is_file() else ""

    tools = Counter(re.findall(r"TOOL_CALL (\w+)", t))
    # the path a coupled task must walk
    parts = list((w / "coupling").glob("participant_*.py")) if (w / "coupling").is_dir() else []
    cell = d.name.split("_")[0]
    want = demanded(cell)
    on_disk = {name: sorted(p.name for p in w.rglob(g))
               for name, g in want.items()}
    # "the deliverable" is the solution field; the rest is supporting evidence
    sol_key = next((k for k in on_disk if k.startswith("solution")), None)
    lvl = on_disk.get(sol_key, []) if sol_key else []
    # the last thing the agent SAID (not a tool result) before it stopped
    said = ""
    for line in reversed(t.splitlines()):
        if not line.startswith(("TOOL_CALL", "TOOL_RESULT", "[harness]")):
            said = line.strip()
            if len(said) > 15:
                break
    return {
        "err": L.get("error"), "calls": L.get("tool_calls"),
        "cont": L.get("continuations"), "wall": L.get("wall_s"),
        "tok_in": L.get("tokens_in"), "tools": tools,
        "delivered": len(parts), "level_files": len(lvl),
        "want": want, "on_disk": on_disk, "unparsed": _unparsed_templates(cell),
        "answered": bool(txt.strip()),
        "gave_up": "COULD_NOT_COMPLETE" in txt,
        "answer_head": " ".join(txt.split())[:110],
        "said": said[:150],
    }


def main():
    cell = sys.argv[1] if len(sys.argv) > 1 else "C1"
    arm = sys.argv[2] if len(sys.argv) > 2 else "MCP"
    seeds = [int(x) for x in sys.argv[3:]] or [21, 22, 23]
    rr = runs_for(cell, arm, seeds)
    if not rr:
        print(f"no runs for {cell} {arm} seeds {seeds}")
        return 1

    print(f"=== {cell} {arm} ===")
    for s, d in rr:
        r = read_one(d)
        state = ("gave up" if r["gave_up"] else
                 "ANSWERED" if r["answered"] else "NO ANSWER FILE")
        print(f"\n seed {s}: {state}   calls={r['calls']} cont={r['cont']} "
              f"{int(r['wall'] or 0)}s  in={r['tok_in']}")
        if r["err"]:
            print(f"   error      : {str(r['err'])[:100]}")
        if r["delivered"]:
            print(f"   delivered  : {r['delivered']} participant files on disk")
        if not r["want"]:
            print("   files      : THE TASK TEXT NAMES NO PER-LEVEL FILE — "
                  "cannot tell whether this run delivered")
        for name, glob in sorted(r["want"].items()):
            found = r["on_disk"].get(name, [])
            shown = ", ".join(found[:6]) + (" ..." if len(found) > 6 else "")
            print(f"   {name:<16}: {len(found)}  {shown}"
                  if found else
                  f"   {name:<16}: 0  NONE (asked for {glob})")
        for u in r["unparsed"]:
            print(f"   !! UNRECOGNISED DELIVERABLE TEMPLATE IN THE TASK: {u}")
        top = ", ".join(f"{k}={v}" for k, v in r["tools"].most_common(14))
        print(f"   tools      : {top}")
        if r["answered"]:
            print(f"   answer     : {r['answer_head']}")
        print(f"   last said  : {r['said']}")

    print("\n--- across seeds ---")
    agg = [read_one(d) for _, d in rr]
    print(f"  answered {sum(a['answered'] for a in agg)}/{len(agg)}   "
          f"gave up {sum(a['gave_up'] for a in agg)}/{len(agg)}   "
          f"no answer {sum(not a['answered'] for a in agg)}/{len(agg)}")
    reached = sum(bool(a["tools"].get("couple")
                       or a["tools"].get("coupled_solve")) for a in agg)
    print(f"  reached couple    {reached}/{len(agg)}")
    want = agg[0]["want"] if agg else {}
    for name in sorted(want):
        n = sum(bool(a["on_disk"].get(name)) for a in agg)
        print(f"  wrote {name:<22} {n}/{len(agg)}")
    if not want:
        print("  wrote (deliverable) UNKNOWN — the task named no per-level file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
