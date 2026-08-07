#!/usr/bin/env python3
"""Build a queryable index of 4C's accepted input grammar from `4C -p`.

The binary is the authority on what 4C will parse. `4C -p` dumps the whole
accepted grammar as YAML; this flattens it into:

  sections : top-level input section names (e.g. "STRUCTURAL DYNAMIC")
  paths    : "SECTION/sub/NAME" for every named entry the grammar accepts
  keys     : the bare identifier set -- what an ALL-CAPS name in prose is
             checked against
  enums    : accepted enum *values*, kept separate from keys so that
             "is X a parameter" and "is X a legal value" stay distinct
             questions

The dump uses `$ref: "<n>"` pointers into a top-level `$references` block;
those are resolved here, with cycle protection.

"Absent from this index" only means "absent" once the index has been shown to
resolve the decks 4C itself ships -- which is what
validate_index_against_decks.py does. Until then a miss is unresolved, not
false.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

CONTAINER_TYPES = {"all_of", "one_of", "group", "list", "selection"}


class Index:
    def __init__(self, references):
        self.references = references or {}
        self.sections: set[str] = set()
        self.paths: set[str] = set()
        self.keys: set[str] = set()
        self.enums: set[str] = set()
        self.path_meta: dict[str, dict] = {}

    def resolve(self, node, seen):
        """Follow a $ref pointer into the dump's $references block."""
        while isinstance(node, dict) and "$ref" in node and len(node) == 1:
            ref = str(node["$ref"])
            if ref in seen:
                return None
            seen = seen | {ref}
            node = self.references.get(ref)
            if node is None:
                return None
        return node

    def walk(self, node, prefix, seen=frozenset()):
        node = self.resolve(node, seen)
        if isinstance(node, list):
            for item in node:
                self.walk(item, prefix, seen)
            return
        if not isinstance(node, dict):
            return

        name = node.get("name")
        here = prefix
        if isinstance(name, str) and name:
            here = f"{prefix}/{name}" if prefix else name
            self.paths.add(here)
            self.keys.add(name)
            meta = {
                "type": node.get("type"),
                "required": node.get("required"),
                "description": node.get("description"),
            }
            if "default" in node:
                meta["default"] = node["default"]
            if isinstance(node.get("choices"), list):
                vals = []
                for c in node["choices"]:
                    if isinstance(c, dict) and isinstance(c.get("name"), str):
                        vals.append(c["name"])
                    elif isinstance(c, str):
                        vals.append(c)
                if vals:
                    meta["values"] = vals
                    self.enums.update(vals)
            self.path_meta.setdefault(here, meta)

        for key in ("specs", "spec", "choices", "value_type", "entries"):
            if key in node:
                self.walk(node[key], here, seen)


def build(raw_path: Path) -> dict:
    data = yaml.safe_load(raw_path.read_text())
    idx = Index(data.get("$references"))

    top = data.get("sections", {})
    specs = top.get("specs", []) if isinstance(top, dict) else top
    for entry in specs:
        resolved = idx.resolve(entry, frozenset())
        if isinstance(resolved, dict) and isinstance(resolved.get("name"), str):
            idx.sections.add(resolved["name"])
        idx.walk(entry, "")

    legacy_string_sections = set()
    lss = data.get("legacy_string_sections")
    if isinstance(lss, dict):
        legacy_string_sections |= set(lss.keys())
    elif isinstance(lss, list):
        legacy_string_sections |= {x for x in lss if isinstance(x, str)}

    element_specs = {}
    for block in ("legacy_element_specs", "legacy_particle_specs"):
        b = data.get(block)
        if isinstance(b, dict):
            for ename, espec in b.items():
                cells = []
                if isinstance(espec, list):
                    for variant in espec:
                        if isinstance(variant, dict) and variant.get("cell_type"):
                            cells.append(variant["cell_type"])
                element_specs[ename] = sorted(set(cells))
                idx.walk(espec, f"[{block}]/{ename}")

    cell_types = set()
    ct = data.get("cell_types")
    if isinstance(ct, dict):
        cell_types |= set(ct.keys())
    elif isinstance(ct, list):
        cell_types |= {x for x in ct if isinstance(x, str)}

    return {
        "metadata": data.get("metadata", {}),
        "sections": sorted(idx.sections),
        "legacy_string_sections": sorted(legacy_string_sections),
        "element_specs": {k: v for k, v in sorted(element_specs.items())},
        "cell_types": sorted(cell_types),
        "paths": sorted(idx.paths),
        "keys": sorted(idx.keys),
        "enum_values": sorted(idx.enums),
        "path_meta": idx.path_meta,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", type=Path, help="output of `4C -p`")
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    index = build(args.raw)
    args.out.write_text(json.dumps(index, indent=1))

    print(f"commit       : {index['metadata'].get('commit_hash', '?')}")
    print(f"version      : {index['metadata'].get('version', '?')}")
    print(f"sections     : {len(index['sections'])}")
    print(f"legacy string sections: {len(index['legacy_string_sections'])}")
    print(f"element specs: {len(index['element_specs'])}")
    print(f"cell types   : {len(index['cell_types'])}")
    print(f"paths        : {len(index['paths'])}")
    print(f"keys         : {len(index['keys'])}")
    print(f"enum values  : {len(index['enum_values'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
