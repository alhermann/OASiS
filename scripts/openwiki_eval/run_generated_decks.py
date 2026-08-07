#!/usr/bin/env python3
"""Feed the wiki's own worked example decks to 4C and record what happens.

Every other check in this directory asks whether a statement matches the
grammar dump. This one asks the only question a user cares about: if an agent
copies what the documentation says is a working deck, does the binary accept
it?

A deck is taken verbatim from a fenced ```yaml block. Blocks that are visibly
fragments (no PROBLEM TYPE section) are reported as fragments rather than run,
because failing a snippet that never claimed to be complete would measure
nothing.

Outcomes:
  RAN        exit 0
  REJECTED   4C refused it -- the first error line is kept, since that names
             the key or section the documentation got wrong
  FRAGMENT   not a whole deck by its own shape
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

FENCE = re.compile(r"```yaml\n(.*?)```", re.S)


# The X server on this host emits "Invalid MIT-MAGIC-COOKIE-1 key" on every
# run. It matches "Invalid" and would be reported as 4C's complaint.
NOISE = ("MIT-MAGIC-COOKIE", "--------", "Xlib", "libGL")


def first_error(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    # 4C names the offending key under "The following data remains unused",
    # which is the line that says WHAT the documentation got wrong.
    for i, ln in enumerate(lines):
        if "remains unused" in ln or "could not be matched" in ln:
            block = [x.strip() for x in lines[i:i + 4] if x.strip()]
            return " | ".join(block)[:300]
    for line in lines:
        s = line.strip()
        if not s or any(n in s for n in NOISE):
            continue
        if any(w in s for w in ("ERROR", "Error", "error", "throw",
                                "Exception", "Unknown", "Invalid",
                                "does not", "not a", "no such", "unknown")):
            return s[:300]
    tail = [ln for ln in text.splitlines() if ln.strip()]
    return tail[-1][:300] if tail else ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wiki", type=Path)
    ap.add_argument("--binary", default="/home/alexander/4C/build/4C")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ("/opt/4C-dependencies/lib:"
                              + env.get("LD_LIBRARY_PATH", ""))

    results = []
    with tempfile.TemporaryDirectory(prefix="owdeck-") as td:
        tmp = Path(td)
        for md in sorted(args.wiki.rglob("*.md")):
            text = md.read_text(errors="replace")
            for n, block in enumerate(FENCE.findall(text), 1):
                rel = f"{md.relative_to(args.wiki)}#{n}"
                if "PROBLEM TYPE" not in block:
                    results.append({"deck": rel, "outcome": "FRAGMENT",
                                    "detail": "no PROBLEM TYPE section"})
                    continue
                f = tmp / f"deck_{len(results)}.4C.yaml"
                f.write_text(block)
                out = tmp / f"out_{len(results)}"
                try:
                    # Without stdbuf, MPI_ABORT tears the process down before
                    # 4C's own diagnostic is flushed and all you get back is
                    # "errorcode 1" -- which says a deck failed but not which
                    # key the documentation got wrong.
                    p = subprocess.run(
                        ["stdbuf", "-o0", "-e0", args.binary, str(f), str(out)],
                        capture_output=True, text=True,
                        timeout=args.timeout, env=env, cwd=td)
                    combined = (p.stdout or "") + (p.stderr or "")
                    if p.returncode == 0:
                        results.append({"deck": rel, "outcome": "RAN",
                                        "detail": ""})
                    else:
                        results.append({"deck": rel, "outcome": "REJECTED",
                                        "rc": p.returncode,
                                        "detail": first_error(combined)})
                except subprocess.TimeoutExpired:
                    results.append({"deck": rel, "outcome": "TIMEOUT",
                                    "detail": f"> {args.timeout}s"})

    counts: dict[str, int] = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    print("yaml blocks found:", len(results))
    for k, v in sorted(counts.items()):
        print(f"  {k:9s} {v}")
    print()
    for r in results:
        if r["outcome"] not in ("FRAGMENT",):
            print(f"  {r['outcome']:9s} {r['deck']}")
            if r["detail"]:
                print(f"             {r['detail']}")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
