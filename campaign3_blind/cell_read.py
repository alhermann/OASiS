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
    lvl = sorted(p.name for p in w.rglob("*level*_[AB].csv"))
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
        print(f"   delivered  : {r['delivered']} participant files on disk")
        print(f"   level files: {r['level_files']}")
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
    print(f"  wrote level files {sum(a['level_files'] > 0 for a in agg)}/{len(agg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
