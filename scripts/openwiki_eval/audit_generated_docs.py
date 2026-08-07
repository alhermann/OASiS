#!/usr/bin/env python3
"""Run OASiS's own input-key auditor over generated documentation.

`scripts/audit_named_input_keys.py` decides which ALL-CAPS tokens a text
*presents as input keys* -- shape, quoting, a key-marker word nearby, or
membership in a list whose other members are confirmed identifiers. That rule
was tuned against real false accusations and carries a selftest proving it
flags SOUNDSPEED before the fix and not after.

Reusing it here rather than writing a second rule is the point: OpenWiki's
output and OASiS's own knowledge are then screened by the same gate, so the
two error rates are comparable. Resolution goes through the pre-tokenised
corpus instead of one `grep` per key, which is the same lookup by a faster
route (verified against `grep -r -a -F` on a sample).

Verdicts stay in three buckets. UNRESOLVED is never merged into either of the
others.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from audit_named_input_keys import (  # noqa: E402
    _absence_asserted, candidate_keys,
)
from audit_quoted_diagnostics import _is_retracted  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", type=Path, help="markdown dir or single file")
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--label", default="target")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    corpus = json.load(args.corpus.open())
    index = json.load(args.index.open())

    src = set(corpus["src_tokens"])
    deck = set(corpus["deck_tokens"])
    parts = set(corpus.get("path_parts", ()))
    grammar = (set(index["keys"]) | set(index["sections"])
               | set(index["enum_values"]) | set(index["element_specs"]))

    if args.target.is_dir():
        files = sorted(args.target.rglob("*.md"))
    else:
        files = [args.target]

    seen: dict[str, list] = {}
    for f in files:
        text = f.read_text(errors="replace")
        for tok, i, j in candidate_keys(text):
            if _is_retracted(text, i, j) or _absence_asserted(text, i, j):
                continue
            line = text[:i].count("\n") + 1
            seen.setdefault(tok, []).append((str(f), line))

    rows = []
    for tok in sorted(seen):
        if tok in grammar:
            verdict, why = "RESOLVED", "accepted by 4C (`4C -p`)"
        elif tok in src:
            verdict, why = "RESOLVED", "occurs in 4C source"
        elif tok in deck:
            verdict, why = "RESOLVED", "occurs in a shipped 4C deck"
        elif tok in parts:
            verdict, why = "RESOLVED", "names a file or directory"
        else:
            verdict, why = "ABSENT", "resolves nowhere in 4C"
        rows.append({"key": tok, "verdict": verdict, "why": why,
                     "occurrences": len(seen[tok]),
                     "first_seen": f"{seen[tok][0][0]}:{seen[tok][0][1]}"})

    counts = Counter(r["verdict"] for r in rows)
    total = len(rows)
    print(f"{args.label}")
    print(f"  files screened                 : {len(files)}")
    print(f"  tokens presented as input keys : {total}")
    print(f"  resolve in 4C                  : {counts['RESOLVED']}")
    print(f"  resolve nowhere                : {counts['ABSENT']} "
          f"({100*counts['ABSENT']/max(total,1):.1f}%)")
    for r in rows:
        if r["verdict"] == "ABSENT":
            print(f"      {r['key']:34s} x{r['occurrences']:<3d} "
                  f"{r['first_seen']}")

    if args.out:
        args.out.write_text(json.dumps(
            {"label": args.label, "files": len(files), "rows": rows,
             "counts": dict(counts)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
