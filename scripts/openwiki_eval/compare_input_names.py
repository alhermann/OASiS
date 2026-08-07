#!/usr/bin/env python3
"""Score two knowledge surfaces on the same question, the SOUNDSPEED question.

Both OpenWiki's generated wiki and OASiS's served knowledge are, for this
purpose, just text that names ALL-CAPS 4C input identifiers. The measurement
is identical for both: take every such name, look it up in the grammar the 4C
binary reports plus the source and deck corpora, and count.

  RESOLVED    the name is something 4C accepts or something the code contains
  ABSENT      the name resolves nowhere -- the SOUNDSPEED failure mode
  UNRESOLVED  the name is prose in caps, or a shorthand, not a literal claim

Running the same screen over both surfaces is the only way the comparison
means anything; scoring one by hand and the other by machine would not.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

CAPS_NAME = re.compile(r"\b[A-Z][A-Z0-9_]{3,}\b")

PROSE_CAPS = {
    "YAML", "JSON", "XML", "CSV", "HDF5", "VTK", "VTU", "VTP", "PVD", "EXODUS",
    "MPI", "OPENMP", "CUDA", "CPU", "GPU", "RAM", "API", "CLI", "URL", "HTTP",
    "TODO", "NOTE", "WARNING", "ERROR", "FATAL", "NONE", "TRUE", "FALSE",
    "NULL", "ONLY", "MUST", "NEVER", "ALWAYS", "WITH", "FROM", "THIS", "THAT",
    "WHEN", "WHAT", "THEN", "ELSE", "EACH", "BOTH", "SAME", "ALSO", "INTO",
    "OVER", "SUCH", "THAN", "THEY", "THEM", "WILL", "WOULD", "SHOULD",
    "README", "LICENSE", "TRILINOS", "PETSC", "MUMPS", "UMFPACK", "SUPERLU",
    "BLAS", "LAPACK", "GMSH", "PARAVIEW", "PYTHON", "CMAKE", "GCC", "CLANG",
    "XFEM", "XFLUID", "NURBS", "FSI", "TSI", "SSI", "SSTI", "DEM", "SPH",
    "PDE", "ODE", "CFL", "AMG", "ILU", "GMRES", "BICGSTAB", "NOX", "DOF",
    "DOFS", "RHS", "LHS", "SI", "EOS", "RVE", "ADR", "OASIS", "MCP", "LLM",
    "PRECICE", "FENICS", "FENICSX", "DOLFINX", "KRATOS", "NGSOLVE", "FEBIO",
    "DUNE", "SPARTA", "DEALII", "SKFEM", "FOURC", "CORRECTED", "RETRACTED",
    "UNVERIFIED", "VERIFIED", "PASS", "FAIL", "SETUP", "INPUT", "OUTPUT",
    "PHYSICS", "NUMERICAL", "SYNTAX", "UNITS", "MESH", "PERFORMANCE",
    "VALIDATION", "INTEGRATION",
}


def load_surface(path: Path) -> str:
    return path.read_text(errors="replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--surface", action="append", nargs=2,
                    metavar=("LABEL", "PATH"), required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    idx = json.load(args.index.open())
    corpus = json.load(args.corpus.open())

    sections = set(idx["sections"])
    keys = set(idx["keys"])
    enums = set(idx["enum_values"])
    elements = set(idx["element_specs"])
    src = set(corpus["src_tokens"])
    deck = set(corpus["deck_tokens"])
    parts = set(corpus.get("path_parts", ()))

    report = {}
    for label, path in args.surface:
        p = Path(path)
        if p.is_dir():
            text = "\n".join(f.read_text(errors="replace")
                             for f in sorted(p.rglob("*.md")))
        else:
            text = load_surface(p)

        names = Counter()
        for m in CAPS_NAME.finditer(text):
            tok = m.group(0)
            if tok in PROSE_CAPS:
                continue
            names[tok] += 1

        verdicts = {}
        for tok in names:
            if tok in sections or tok in keys or tok in enums or tok in elements:
                verdicts[tok] = "RESOLVED"
            elif tok in src or tok in deck or tok in parts:
                verdicts[tok] = "RESOLVED"
            else:
                verdicts[tok] = "ABSENT"

        counts = Counter(verdicts.values())
        distinct = len(names)
        absent = sorted(t for t, v in verdicts.items() if v == "ABSENT")
        report[label] = {
            "distinct_caps_names": distinct,
            "resolved": counts["RESOLVED"],
            "absent": counts["ABSENT"],
            "absent_names": absent,
            "occurrences": {t: names[t] for t in absent},
        }
        rate = 100 * counts["ABSENT"] / max(distinct, 1)
        print(f"{label}")
        print(f"  distinct ALL-CAPS 4C-style names : {distinct}")
        print(f"  resolve in grammar/source/decks   : {counts['RESOLVED']}")
        print(f"  resolve nowhere                   : {counts['ABSENT']}  "
              f"({rate:.1f}%)")
        for t in absent:
            print(f"      {t}  (x{names[t]})")
        print()

    if args.out:
        args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
