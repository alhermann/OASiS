#!/usr/bin/env python3
"""Build the token and path corpora a claim is checked against.

Doing one `grep -r` per claim over a 1.4M-line tree is too slow to check
hundreds of claims, so the corpus is tokenised once into sets:

  src_tokens   identifiers appearing anywhere under the source roots
  deck_tokens  identifiers appearing anywhere in the shipped input decks
  paths        every file path, relative to the repo root

Files are read as bytes and decoded latin-1, which is the in-Python equivalent
of `grep -a`: nothing is skipped for looking binary, so "not in the corpus"
cannot be an artefact of a file being passed over.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


def harvest(root: Path, subdirs, tokens: set, paths: set, cap_mb: int = 0):
    n_files = 0
    n_bytes = 0
    for sub in subdirs:
        base = root / sub
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else base.rglob("*")
        for f in candidates:
            if not f.is_file() or f.is_symlink():
                continue
            try:
                rel = f.relative_to(root)
            except ValueError:
                continue
            paths.add(str(rel))
            if cap_mb and f.stat().st_size > cap_mb * 1024 * 1024:
                continue
            try:
                blob = f.read_bytes().decode("latin-1")
            except OSError:
                continue
            n_files += 1
            n_bytes += len(blob)
            tokens.update(TOKEN.findall(blob))
    return n_files, n_bytes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path("/home/alexander/4C"))
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()

    src_tokens: set[str] = set()
    deck_tokens: set[str] = set()
    paths: set[str] = set()

    # Repo-root files (CMakeLists.txt, CONTRIBUTING.md, presets, CI config)
    # are part of the source the docs describe; leaving them out made the
    # checker call real CMake variables absent.
    roots = ["src", "unittests", "apps", "utilities", "doc", "cmake",
             "dependencies", "presets", "docker", ".github", "input_version"]
    roots += [p.name for p in args.repo.iterdir() if p.is_file()]

    src_files, src_bytes = harvest(args.repo, roots, src_tokens, paths)
    deck_files, deck_bytes = harvest(
        args.repo, ["tests"], deck_tokens, paths, cap_mb=64,
    )

    # Directory and file-stem names are identifiers too: a doc that says the
    # `fem` module lives under src/core is naming a directory, and the C++
    # tokeniser never emits "fem" on its own out of `4C_fem_general.hpp`.
    path_parts: set[str] = set()
    for p in paths:
        for part in p.split("/"):
            path_parts.add(part)
            path_parts.add(part.split(".")[0])
            for piece in re.split(r"[._-]", part):
                if len(piece) > 1:
                    path_parts.add(piece)

    out = {
        "repo": str(args.repo),
        "path_parts": sorted(path_parts),
        "src_files": src_files,
        "src_bytes": src_bytes,
        "src_tokens": sorted(src_tokens),
        "deck_files": deck_files,
        "deck_bytes": deck_bytes,
        "deck_tokens": sorted(deck_tokens),
        "paths": sorted(paths),
    }
    args.out.write_text(json.dumps(out))
    print(f"source files read : {src_files} ({src_bytes/1e6:.1f} MB)")
    print(f"source tokens     : {len(src_tokens)}")
    print(f"deck files read   : {deck_files} ({deck_bytes/1e6:.1f} MB)")
    print(f"deck tokens       : {len(deck_tokens)}")
    print(f"paths             : {len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
