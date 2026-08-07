#!/usr/bin/env python3
"""Prove the `4C -p` index resolves the decks 4C itself ships.

An index is only usable as evidence of *absence* if it first demonstrates
near-total *presence* on real input. This walks every `*.4C.yaml` under a
directory, extracts (section, key) pairs, and reports what the index fails to
resolve.

The known-and-expected non-resolvers are:

  * TITLE                     -- free text, named in the dump's metadata as
                                 `description_section_name`, not a spec
  * FUNCT<n>                  -- numbered section family
  * SOLVER <n>, MATERIALS...  -- numbered / list-shaped section families
  * legacy free-form sections -- NODE COORDS, element blocks, topologies:
                                 whitespace-delimited lines, no key/value spec

Anything else that fails to resolve is a hole in the index and must be
understood before the index is used to call a claim false.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

NUMBERED = re.compile(r"^(.*?)\s*\d+$")


def section_family(name: str) -> str:
    """`SOLVER 1` and `FUNCT12` both collapse onto their family name."""
    m = NUMBERED.match(name)
    if m and m.group(1):
        return m.group(1).strip()
    return name


def collect_pairs(node, section, out, depth=0):
    """Record every (section, key) the deck actually uses."""
    if depth > 6:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str):
                out.add((section, k))
            collect_pairs(v, section, out, depth + 1)
    elif isinstance(node, list):
        for item in node:
            collect_pairs(item, section, out, depth + 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    ap.add_argument("decks", type=Path)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()

    idx = json.load(args.index.open())
    sections = set(idx["sections"])
    section_families = {section_family(s) for s in sections}
    legacy = set(idx["legacy_string_sections"])
    all_keys = set(idx["keys"])
    element_specs = set(idx["element_specs"])

    # section-scoped key lookup: "SEC/a/b/KEY" -> (SEC, KEY)
    scoped = set()
    for p in idx["paths"]:
        parts = p.split("/")
        if len(parts) >= 2:
            scoped.add((parts[0], parts[-1]))
    # material names are themselves keys nested under MATERIALS
    for p in idx["paths"]:
        parts = p.split("/")
        for part in parts[1:]:
            scoped.add((parts[0], part))

    files = sorted(args.decks.glob("*.4C.yaml"))
    if args.limit:
        files = files[: args.limit]

    n_files = 0
    parse_fail = []
    sec_ok = Counter()
    sec_bad = Counter()
    key_ok = 0
    key_bad = Counter()

    for f in files:
        try:
            data = yaml.safe_load(f.read_text(errors="replace"))
        except Exception as exc:  # noqa: BLE001
            parse_fail.append((f.name, str(exc)[:100]))
            continue
        if not isinstance(data, dict):
            parse_fail.append((f.name, "top level is not a mapping"))
            continue
        n_files += 1

        for sname, sval in data.items():
            if not isinstance(sname, str):
                continue
            fam = section_family(sname)
            if sname in sections or fam in section_families or sname in legacy:
                sec_ok[sname] += 1
            else:
                sec_bad[sname] += 1
                continue

            pairs = set()
            collect_pairs(sval, sname, pairs)
            for _, k in pairs:
                if (sname, k) in scoped:
                    key_ok += 1
                elif (fam, k) in scoped:
                    key_ok += 1
                elif k in all_keys or k in element_specs:
                    key_ok += 1
                elif sname in legacy:
                    key_ok += 1
                else:
                    key_bad[(sname, k)] += 1

    total_sec = sum(sec_ok.values()) + sum(sec_bad.values())
    total_key = key_ok + sum(key_bad.values())

    print(f"decks parsed          : {n_files} / {len(files)}")
    if parse_fail:
        print(f"decks unparseable     : {len(parse_fail)}")
        for name, why in parse_fail[:5]:
            print(f"    {name}: {why}")
    print()
    print(f"section uses          : {total_sec}")
    print(f"  resolved            : {sum(sec_ok.values())} "
          f"({100 * sum(sec_ok.values()) / max(total_sec, 1):.3f}%)")
    print(f"  unresolved          : {sum(sec_bad.values())} "
          f"across {len(sec_bad)} distinct names")
    for name, n in sec_bad.most_common(25):
        print(f"      {n:6d}  {name}")
    print()
    print(f"key uses (in resolved sections): {total_key}")
    print(f"  resolved            : {key_ok} "
          f"({100 * key_ok / max(total_key, 1):.3f}%)")
    print(f"  unresolved          : {sum(key_bad.values())} "
          f"across {len(key_bad)} distinct (section, key) pairs")
    for (s, k), n in key_bad.most_common(30):
        print(f"      {n:6d}  {s} :: {k}")

    if args.json_out:
        args.json_out.write_text(json.dumps({
            "decks_parsed": n_files,
            "decks_total": len(files),
            "unparseable": parse_fail,
            "section_uses": total_sec,
            "section_resolved": sum(sec_ok.values()),
            "section_unresolved": {k: v for k, v in sec_bad.most_common()},
            "key_uses": total_key,
            "key_resolved": key_ok,
            "key_unresolved": {f"{s} :: {k}": v
                               for (s, k), v in key_bad.most_common()},
        }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
