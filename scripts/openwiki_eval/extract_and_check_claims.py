#!/usr/bin/env python3
"""Census every checkable identifier an OpenWiki wiki names, and check it.

This is the mechanical half of the fabrication measurement. It takes every
backticked span and every ALL-CAPS input-style name out of the generated
Markdown, decides what kind of thing each one claims to be, and resolves it
against evidence that cannot be argued with:

  file path       -> the path exists in the repo
  C++ symbol      -> the identifier occurs in the source corpus
  input section   -> the name is a section in `4C -p`
  input key       -> the name is an accepted key in `4C -p`

Three verdicts, deliberately kept apart:

  RESOLVED    the thing named exists
  ABSENT      the thing named exists nowhere in source, decks or grammar
  UNRESOLVED  the claim is not decidable by lookup (prose, generic words,
              a symbol built at runtime, a name whose kind is ambiguous)

UNRESOLVED is never folded into either of the other two. A rate computed by
counting unresolved things as correct is not a measurement.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

CODE_SPAN = re.compile(r"`([^`\n]{2,120})`")
FENCE = re.compile(r"```.*?```", re.S)
FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)

# An input-style name: ALL CAPS words, possibly space- or underscore-separated.
CAPS_NAME = re.compile(r"\b[A-Z][A-Z0-9_]{2,}(?:[ -][A-Z][A-Z0-9_]*)*\b")

PATHLIKE = re.compile(r"^[\w./-]+\.(cpp|hpp|h|H|py|cmake|txt|md|yaml|yml|json|rst)$")
DIRLIKE = re.compile(r"^/?(src|apps|tests|unittests|utilities|doc|cmake)(/[\w.-]+)*/?$")
CXX_SYMBOL = re.compile(r"^[A-Za-z_][\w:]*(\(\.\.\.\)|\(\)|\(int [\w ]+\))?$")

# Caps words that are English or format names, not 4C input names.
PROSE_CAPS = {
    "YAML", "JSON", "XML", "CSV", "VTK", "VTU", "HDF5", "MPI", "CPU", "GPU",
    "API", "CLI", "URL", "ADR", "FEM", "DEM", "SPH", "PD", "ALE", "FSI", "TSI",
    "XFEM", "DOF", "DOFS", "RHS", "LHS", "PDE", "ODE", "CFL", "AMG", "ILU",
    "LU", "QR", "SVD", "GMRES", "CG", "DG", "HDG", "TODO", "NOTE", "WARNING",
    "ERROR", "FATAL", "NOT", "AND", "OR", "ALL", "ANY", "ONLY", "MUST",
    "NEVER", "ALWAYS", "README", "LICENSE", "MIT", "CMAKE", "CMAKELISTS",
    "TRILINOS", "PETSC", "MUMPS", "UMFPACK", "SUPERLU", "BLAS", "LAPACK",
    "GMSH", "EXODUS", "OPENMP", "CUDA", "MPI", "IO", "OK", "YES", "NO",
    "III", "II", "IV", "3D", "2D", "1D", "SI", "EOS", "RVE", "AAA", "SMA",
    "CI", "PR", "OS", "RAM", "DAT", "NAN", "INF",
}

GENERIC_LOWER = {
    "run", "main", "setup", "read", "write", "create", "get", "set", "make",
    "type", "name", "id", "value", "true", "false", "null", "int", "double",
    "bool", "string", "vector", "map", "list", "size", "data", "file", "path",
}


def is_wiki_internal(tok: str, wiki: Path) -> bool:
    """True for links to the wiki's own pages, which are not code claims."""
    t = tok.strip().lstrip("./").lstrip("/")
    if t.startswith("openwiki/"):
        return True
    if t.endswith(".md") and (wiki / t).exists():
        return True
    if t.endswith(".md"):
        for cand in wiki.rglob(Path(t).name):
            if cand.is_file():
                return True
    return False


def classify(tok: str) -> str:
    t = tok.strip()
    if not t:
        return "skip"
    if t.startswith("--") or t.startswith("-"):
        return "cli_flag"
    if PATHLIKE.match(t) or DIRLIKE.match(t):
        return "path"
    if "::" in t:
        return "cxx"
    if t.lower() in GENERIC_LOWER:
        return "generic"
    if CAPS_NAME.fullmatch(t):
        if t in PROSE_CAPS:
            return "generic"
        return "input_name"
    if CXX_SYMBOL.match(t) and re.search(r"[a-z]", t):
        return "cxx"
    return "other"


class Checker:
    def __init__(self, corpus, index):
        self.src = set(corpus["src_tokens"])
        self.deck = set(corpus["deck_tokens"])
        self.paths = set(corpus["paths"])
        self.path_tails = set()
        for p in corpus["paths"]:
            parts = p.split("/")
            for i in range(len(parts)):
                self.path_tails.add("/".join(parts[i:]))
        self.dirs = set()
        for p in corpus["paths"]:
            parts = p.split("/")
            for i in range(1, len(parts)):
                self.dirs.add("/".join(parts[:i]))
        self.path_parts = set(corpus.get("path_parts", ()))
        self.sections = set(index["sections"])
        self.section_family = {re.sub(r"\s*\d+$", "", s).strip()
                               for s in index["sections"]}
        self.legacy = set(index["legacy_string_sections"])
        self.keys = set(index["keys"])
        self.enums = set(index["enum_values"])
        self.elements = set(index["element_specs"])

    def check(self, tok: str, kind: str):
        t = tok.strip().rstrip(".,;:")
        if kind == "path":
            p = t.lstrip("/").rstrip("/")
            if p in self.paths or p in self.path_tails:
                return "RESOLVED", "file exists"
            if p in self.dirs:
                return "RESOLVED", "directory exists"
            return "ABSENT", "no such path in the repo"
        if kind == "dirname":
            return ("RESOLVED", "directory or file-stem name in the repo") \
                if t in self.path_parts else \
                ("ABSENT", "no directory or file of that name")
        if kind == "cxx":
            base = re.sub(r"\(.*\)$", "", t)
            parts = [x for x in base.split("::") if x]
            if not parts:
                return "UNRESOLVED", "empty after normalisation"
            leaf = parts[-1]
            if all(p in self.src for p in parts):
                return "RESOLVED", "all name components occur in the source"
            if leaf in self.src:
                return "RESOLVED", "leaf identifier occurs in the source"
            if leaf in self.path_parts or base in self.path_parts:
                return "RESOLVED", "names a directory or file, not a symbol"
            # A template instantiation such as Foo<dim, spacedim> is a real
            # symbol if its head is; the angle-bracket payload is prose.
            head = re.sub(r"<.*", "", base).split("::")[-1]
            if head and head in self.src:
                return "RESOLVED", "template head occurs in the source"
            return "ABSENT", "identifier occurs nowhere in the source"
        if kind == "input_name":
            if t in self.sections or t in self.legacy:
                return "RESOLVED", "input section in `4C -p`"
            fam = re.sub(r"\s*\d+$", "", t).strip()
            fam = re.sub(r"<n>$", "", fam).strip()
            if fam in self.section_family:
                return "RESOLVED", "numbered input section family in `4C -p`"
            if t in self.keys:
                return "RESOLVED", "accepted input key in `4C -p`"
            if t in self.enums:
                return "RESOLVED", "accepted enum value in `4C -p`"
            if t in self.elements:
                return "RESOLVED", "element spec in `4C -p`"
            if t in self.src or t in self.deck:
                return "RESOLVED", "occurs in source or decks (non-grammar role)"
            if t in self.path_parts:
                return "RESOLVED", "names a directory or file in the repo"
            if " " in t:
                for w in t.split():
                    if w not in self.src and w not in self.deck:
                        return "ABSENT", "name resolves nowhere"
                return "UNRESOLVED", "words occur separately, phrase does not"
            return "ABSENT", "name resolves nowhere"
        return "UNRESOLVED", f"not mechanically checkable ({kind})"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wiki", type=Path)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    corpus = json.load(args.corpus.open())
    index = json.load(args.index.open())
    ck = Checker(corpus, index)

    claims = []
    for md in sorted(args.wiki.rglob("*.md")):
        text = md.read_text(errors="replace")
        body = FRONTMATTER.sub("", text)
        body_nofence = FENCE.sub(" ", body)
        rel = str(md.relative_to(args.wiki))

        seen = set()
        for m in CODE_SPAN.finditer(body_nofence):
            tok = m.group(1)
            kind = classify(tok)
            if kind in ("skip", "generic", "cli_flag"):
                continue
            # Links to the wiki's own pages are navigation, not claims about
            # the code. Scoring them as false paths would invent a defect.
            if is_wiki_internal(tok, args.wiki):
                continue
            key = (tok, kind)
            if key in seen:
                continue
            seen.add(key)
            line = body_nofence[: m.start()].count("\n") + 1
            verdict, why = ck.check(tok, kind)
            claims.append({"doc": rel, "line": line, "token": tok,
                           "kind": kind, "verdict": verdict, "why": why,
                           "source": "code-span"})

        # ALL-CAPS names in plain prose (outside code spans) claim to be
        # input sections/keys just as strongly as backticked ones.
        prose = CODE_SPAN.sub(" ", body_nofence)
        for m in CAPS_NAME.finditer(prose):
            tok = m.group(0)
            if tok in PROSE_CAPS or len(tok) < 4:
                continue
            key = (tok, "input_name")
            if key in seen:
                continue
            seen.add(key)
            line = prose[: m.start()].count("\n") + 1
            verdict, why = ck.check(tok, "input_name")
            claims.append({"doc": rel, "line": line, "token": tok,
                           "kind": "input_name", "verdict": verdict,
                           "why": why, "source": "prose"})

    by_kind = Counter((c["kind"], c["verdict"]) for c in claims)
    verdicts = Counter(c["verdict"] for c in claims)

    args.out.write_text(json.dumps({"claims": claims,
                                    "by_kind": {f"{k}|{v}": n
                                                for (k, v), n in by_kind.items()},
                                    "verdicts": dict(verdicts)}, indent=1))

    total = len(claims)
    print(f"distinct checkable identifiers: {total}")
    for v in ("RESOLVED", "ABSENT", "UNRESOLVED"):
        n = verdicts.get(v, 0)
        print(f"  {v:11s} {n:5d}  ({100*n/max(total,1):.1f}%)")
    print()
    print(f"{'kind':14s} {'RESOLVED':>9s} {'ABSENT':>7s} {'UNRESOLVED':>11s}")
    kinds = sorted({k for k, _ in by_kind})
    for k in kinds:
        print(f"{k:14s} {by_kind[(k,'RESOLVED')]:9d} "
              f"{by_kind[(k,'ABSENT')]:7d} {by_kind[(k,'UNRESOLVED')]:11d}")
    print()
    print("--- ABSENT ---")
    for c in claims:
        if c["verdict"] == "ABSENT":
            print(f"  {c['doc']}:{c['line']}  [{c['kind']}]  {c['token']}"
                  f"   <- {c['why']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
