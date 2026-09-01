"""Which runs of a round were served a DIFFERENT build from the others?

WHY THIS EXISTS. On 2026-09-01 three commits to `src/` landed while round 9 was
in flight. Each `run_blind.py` invocation is a fresh process that imports the
OASiS tools from the working tree, so runs that started after a commit were
served knowledge the earlier runs never saw: 16 of the round's 64 MCP runs
finished before 14:58:23 and got the pre-change core, the rest got rule 7 (the
halvings-not-cells refinement rule) and the mesh-size audit finding.

That makes the round's OASiS number a MIXTURE of two builds. The bare arm is
unaffected — it calls no OASiS tool — so the damage is one-sided, which is the
worse kind: it moves the uplift without moving the control.

The rule this exists to enforce is simple and was broken by inattention, not by
disagreement: **do not commit to src/ while a round is in flight.** When it
happens anyway, the drift must be COUNTABLE rather than invisible, in the same
spirit as `per_code_attribution: UNPROVEN` — a number you can see and correct is
worth more than a caveat in prose.

    python build_drift.py 96 97

prints, per seed, which runs predate the newest agent-facing commit and are
therefore due a re-run before the round's OASiS figure is quoted.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parent

# What an AGENT is actually served. `campaign3_blind/grading/` and the reading
# tools are excluded on purpose: the grader runs offline, after the fact, so
# changing it does not change what any run received.
AGENT_FACING = ["src/tools", "src/backends", "src/core", "langgraph_eval"]


def newest_agent_facing_commit() -> tuple[int, str]:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ct %h %s", "--"] + AGENT_FACING,
        cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    ts, rest = out.split(" ", 1)
    return int(ts), rest


def main(seeds) -> int:
    cut, desc = newest_agent_facing_commit()
    import datetime as _dt
    print(f"newest agent-facing commit: {desc}")
    print(f"                            "
          f"{_dt.datetime.fromtimestamp(cut):%Y-%m-%d %H:%M:%S}\n")
    total_stale = 0
    for seed in seeds:
        stale, fresh, unfinished = [], [], 0
        for d in sorted((D / "runs").glob(f"*_seed{seed}")):
            led = d / "ledger.json"
            if not led.is_file():
                unfinished += 1
                continue
            (stale if led.stat().st_mtime < cut else fresh).append(d.name)
        mcp_stale = [n for n in stale if "_MCP_" in n]
        print(f"seed {seed}: {len(fresh)} on the current build, "
              f"{len(stale)} on an earlier one, {unfinished} unfinished")
        if mcp_stale:
            total_stale += len(mcp_stale)
            print(f"   OASiS-arm runs to re-run before quoting this round "
                  f"({len(mcp_stale)}):")
            for n in mcp_stale:
                print(f"     {n}")
        bare_stale = len(stale) - len(mcp_stale)
        if bare_stale:
            print(f"   {bare_stale} bare-arm run(s) also predate it; those need "
                  f"no re-run, the bare arm calls no OASiS tool")
    if total_stale:
        print(f"\n{total_stale} OASiS-arm run(s) were served a different build "
              f"from the rest of the round. Re-run exactly those, then the "
              f"round is a single-build measurement again.")
    else:
        print("\nno drift: every run in these seeds saw the current build")
    return 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]] or [96, 97]))
