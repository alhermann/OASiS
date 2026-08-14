#!/usr/bin/env python
"""Record a completed path walk in path_readiness.json.

``run_blind.py`` refuses any problem whose id is not recorded ``true`` under
``path_verified``, and it treats an id the file has never heard of as UNREADY --
which is the whole point of the control for a NEW problem, whose path has by
definition never run. This is the only thing that should silence it, and it must
be driven by an actual walk, so it takes the walk's own summary file rather than
a hand-typed verdict.

  record_walk.py C8 C9 --summary /tmp/.../walk_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SECTION = "walk_2026_08_13_balanced"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", nargs="+")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    summ = json.loads(Path(a.summary).read_text())
    p = HERE / "path_readiness.json"
    d = json.loads(p.read_text())
    d.setdefault("path_verified", {})
    d.setdefault(SECTION, {}).setdefault("per_instance", {})

    for pid in a.problems:
        levels = summ.get(pid) or []
        ok = bool(levels) and all(l.get("converged") for l in levels)
        d["path_verified"][pid] = ok
        d[SECTION]["per_instance"][pid] = {
            "converged_every_level": ok,
            "levels": [{k: l.get(k) for k in
                        ("level", "N", "converged", "iterations", "residual",
                         "wall_s", "errors")} for l in levels],
            "note": a.note or None,
        }
        print(f"{pid}: path_verified={ok} ({len(levels)} level(s))")
    p.write_text(json.dumps(d, indent=1))
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
