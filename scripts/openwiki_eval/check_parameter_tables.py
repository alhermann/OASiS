#!/usr/bin/env python3
"""Check generated parameter tables cell by cell against the 4C binary.

A documentation page that says

    | `TIMESTEP` | double | 0.1 | time step size |

makes four separate claims, and they can fail independently: the key may not
exist, the type may be wrong, the default may be wrong, the enum list may be
wrong. `4C -p` states all four, so each is decidable.

The section a row belongs to is taken from the nearest preceding heading that
names a real 4C section; without it, `TIMESTEP` is ambiguous across the ~40
sections that define one, and a claim that cannot be located is UNRESOLVED
rather than guessed at.

Counted separately and never merged:
  TRUE        the cell agrees with `4C -p`
  FALSE       the cell contradicts `4C -p`
  UNRESOLVED  the cell cannot be located or compared
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HEADING = re.compile(r"^#{1,6}\s+(.*)$")
TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
SEP_ROW = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
BACKTICK = re.compile(r"`([^`]+)`")

# Order matters. A table that carries both a Section and a Key column names
# the section first; taking the first matching header put the section name in
# the key slot and reported every row of that table as a wrong type.
KEY_COLS = ("key", "keys", "parameter", "parameters", "name", "field",
            "option", "setting", "section")
SECTION_COLS = {"section", "owning section", "parent section"}
TYPE_COLS = {"type", "value type", "datatype", "data type"}
DEFAULT_COLS = {"default", "defaults", "default value"}
VALUES_COLS = {"values", "accepted values", "options", "enum", "enum values",
               "allowed values", "choices"}
REQ_COLS = {"required", "required?", "requiredness", "mandatory"}

TYPE_SYNONYMS = {
    "double": {"double", "float", "real", "scalar"},
    "int": {"int", "integer"},
    "bool": {"bool", "boolean", "yes/no", "true/false"},
    "string": {"string", "str", "path", "text", "filename"},
    "enum": {"enum", "enumeration", "selection", "choice"},
    "vector": {"vector", "list", "array"},
    "group": {"group", "map", "section", "subsection"},
    "path": {"path", "string"},
    "all_of": {"group", "all_of"},
    "one_of": {"one_of", "selection", "choice"},
}


def norm_cell(c: str) -> str:
    """Unwrap a cell that is exactly one backticked span.

    Taking the first span out of "optional string matching `major.minor.patch`"
    threw away the word that carried the type and scored a correct description
    as wrong, so a cell with prose around the span is returned whole.
    """
    c = c.strip()
    m = BACKTICK.fullmatch(c) if hasattr(BACKTICK, "fullmatch") else None
    if m:
        return m.group(1).strip()
    spans = BACKTICK.findall(c)
    if len(spans) == 1 and c == f"`{spans[0]}`":
        return spans[0].strip()
    return c.strip().strip("*_ ")


def parse_tables(text: str, sections=frozenset()):
    """Yield (section_context, mentioned_sections, header, row) per table row.

    A heading is often a description ("Initial and boundary conditions
    subsection") rather than the literal section name, so the row also carries
    every real section name the document has mentioned so far, most recent
    first. Attributing a key to the wrong section makes a correct table look
    invented, which is the error this whole evaluation is trying not to make.
    """
    section = None
    mentioned: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        for name in BACKTICK.findall(line):
            n = name.strip()
            if n in sections:
                if n in mentioned:
                    mentioned.remove(n)
                mentioned.insert(0, n)
        h = HEADING.match(line)
        if h:
            title = h.group(1)
            found = BACKTICK.findall(title)
            cand = found[0] if found else title
            section = cand.strip()
        if TABLE_ROW.match(line) and i + 1 < len(lines) and SEP_ROW.match(lines[i + 1]):
            header = [c.strip().lower().strip("`*")
                      for c in line.strip().strip("|").split("|")]
            i += 2
            while i < len(lines) and TABLE_ROW.match(lines[i]):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                yield section, list(mentioned), header, cells
                i += 1
            continue
        i += 1


def col_index(header, names):
    """Index of the first header cell in `names`.

    When `names` is ordered (the key column), the order expresses preference:
    "Key" wins over "Section" no matter which comes first in the table.
    """
    if isinstance(names, tuple):
        for want in names:
            for idx, h in enumerate(header):
                if h == want:
                    return idx
        return None
    for idx, h in enumerate(header):
        if h in names:
            return idx
    return None


def default_matches(claimed: str, actual) -> str:
    """Return "match", "case" (differs only in capitalisation) or "mismatch".

    Case is broken out rather than scored, because whether 4C accepts a
    lowercased enum value is an empirical question about the parser, not
    something the dump answers. Merging it into either bucket would decide
    that question by assumption.
    """
    c = claimed.strip().strip("`").strip()
    if isinstance(actual, str) and c.strip('"\'') != actual and \
            c.strip('"\'').lower() == actual.lower():
        return "case"
    if c.lower() in {"empty string", "empty", "``", '""', "''", "(empty)"}:
        return "match" if actual == "" else "mismatch"
    if c.lower() in {"-", "—", "n/a", "(none)", ""} or \
            (c.lower() == "none" and not isinstance(actual, str)):
        return "match" if actual is None else "mismatch"
    if actual is None:
        return "mismatch"
    if isinstance(actual, bool):
        ok = c.lower() in ({"true", "yes", "on", "1"} if actual
                           else {"false", "no", "off", "0"})
        return "match" if ok else "mismatch"
    if isinstance(actual, (int, float)):
        try:
            ok = abs(float(c.replace(",", "")) - float(actual)) < 1e-12
        except ValueError:
            ok = c.strip() == str(actual)
        return "match" if ok else "mismatch"
    if isinstance(actual, (list, tuple)):
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", c)
        if len(nums) == len(actual):
            try:
                ok = all(abs(float(a) - float(b)) < 1e-12
                         for a, b in zip(nums, actual))
            except (TypeError, ValueError):
                ok = False
        else:
            ok = False
        return "match" if ok else "mismatch"
    # A numeric default written in another but equal notation (1e-06 vs 1E-6)
    # is the same default.
    try:
        if abs(float(c) - float(str(actual))) < 1e-15:
            return "match"
    except (TypeError, ValueError):
        pass
    return "match" if c.strip('"\'') == str(actual).strip('"\'') else "mismatch"


def type_matches(claimed: str, actual: str) -> bool:
    """Is a prose type description compatible with the dump's type?

    Generated tables write "optional list of paths" where the dump says
    "vector". Demanding string equality would score a correct description as
    a fabrication, so the claim is searched for any word that names the dump's
    type or one of its synonyms.
    """
    if not actual:
        return False
    c = claimed.strip().lower().strip("`")
    a = str(actual).strip().lower()
    words = set(re.findall(r"[a-z_]+", c))
    if a in words:
        return True
    if words & TYPE_SYNONYMS.get(a, set()):
        return True
    for canon, alts in TYPE_SYNONYMS.items():
        if a == canon and (words & alts):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wiki", type=Path)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    idx = json.load(args.index.open())
    pm = idx["path_meta"]
    sections = set(idx["sections"])

    # 4C section names contain slashes of their own -- "FLUID DYNAMIC/NONLINEAR
    # SOLVER TOLERANCES" is one top-level section, not a section and a
    # subsection. Splitting a path on "/" therefore has to match the longest
    # real section name first, or every key in a slash-named section gets
    # filed under the wrong owner and reported as absent from it.
    sections_by_len = sorted(sections, key=len, reverse=True)

    def split_path(path: str):
        for s in sections_by_len:
            if path == s:
                return s, s, 1
            if path.startswith(s + "/"):
                rest = path[len(s) + 1:]
                return s, rest.split("/")[-1], rest.count("/") + 2
        parts = path.split("/")
        return parts[0], parts[-1], len(parts)

    # section -> {key: meta}, taking the shallowest definition of each key
    by_section: dict[str, dict[str, dict]] = {}
    for path, meta in pm.items():
        sec, key, depth = split_path(path)
        slot = by_section.setdefault(sec, {})
        prev = slot.get(key)
        if prev is None or depth < prev["_depth"]:
            m = dict(meta)
            m["_depth"] = depth
            m["_path"] = path
            slot[key] = m

    results = []
    for md in sorted(args.wiki.rglob("*.md")):
        if md.name == "INSTRUCTIONS.md":
            continue
        text = md.read_text(errors="replace")
        rel = str(md.relative_to(args.wiki))
        for section, mentioned, header, cells in parse_tables(text, sections):
            ki = col_index(header, KEY_COLS)
            if ki is None or ki >= len(cells):
                continue
            key = norm_cell(cells[ki])
            if not key or not re.fullmatch(r"[A-Za-z_][\w /]*", key):
                continue

            # A Section column in the row itself beats the heading: the
            # subsection tables list several sections' keys under one heading.
            si = col_index(header, SECTION_COLS)
            row_section = (norm_cell(cells[si])
                           if si is not None and si < len(cells)
                           and si != ki else None)
            section = row_section or section

            sec = section if section in sections else None
            if sec is None and section:
                for s in sections:
                    if section.upper() == s or section.upper().startswith(s):
                        sec = s
                        break
            base = {"doc": rel, "section_claimed": section, "key": key}
            meta = None

            # Resolution order matters, and getting it wrong is silent. Some
            # names are BOTH a top-level section and a key inside another
            # section -- CONSTRAINT is a section and a key of PARTICLE
            # DYNAMIC/INITIAL AND BOUNDARY CONDITIONS -- so the context the
            # document supplies has to be exhausted before falling back to
            # reading the name as a section, or a correct row gets compared
            # against an unrelated group and reported as a wrong type.
            if sec and key in by_section.get(sec, {}):
                meta = by_section[sec][key]
            if meta is None:
                for s in mentioned:
                    if key in by_section.get(s, {}):
                        sec, meta = s, by_section[s][key]
                        break
            if meta is None and key in sections:
                results.append({**base, "claim": "section exists",
                                "verdict": "TRUE",
                                "why": "top-level section in `4C -p`"})
                sec = key
                raw = pm.get(key)
                if raw is None:
                    continue
                meta = dict(raw)
                meta["_path"] = key
            if meta is None:
                hits = [s for s, d in by_section.items() if key in d]
                if len(hits) == 1:
                    sec, meta = hits[0], by_section[hits[0]][key]

            if meta is None:
                if sec and sec in by_section:
                    results.append({**base, "claim": "key exists in section",
                                    "verdict": "FALSE",
                                    "why": f"`4C -p` has no key {key} anywhere "
                                           f"under {sec}"})
                else:
                    results.append({**base, "claim": "key exists",
                                    "verdict": "UNRESOLVED",
                                    "why": "cannot locate the owning section"})
                continue

            if meta.get("_path") != key:
                results.append({**base, "claim": "key exists in section",
                                "verdict": "TRUE", "why": meta["_path"]})

            ti = col_index(header, TYPE_COLS)
            if ti is not None and ti < len(cells) and cells[ti].strip():
                claimed = norm_cell(cells[ti])
                if claimed and claimed not in {"-", "—"}:
                    ok = type_matches(claimed, meta.get("type"))
                    results.append({**base, "claim": f"type = {claimed}",
                                    "verdict": "TRUE" if ok else "FALSE",
                                    "why": f"`4C -p` says {meta.get('type')}"})

            di = col_index(header, DEFAULT_COLS)
            if di is not None and di < len(cells) and cells[di].strip():
                claimed = norm_cell(cells[di])
                # A cell that promises a default and then declines to give one
                # ("source default", "see source") makes no claim, so there is
                # nothing to be right or wrong about.
                VACUOUS = {"source default", "see source", "varies",
                           "module default", "compiled default", "tbd"}
                # "source default `BE`" does name a value; the hedge in front
                # of it is not part of the claim.
                m_hedge = re.match(r"(?:source|module|compiled)\s+default\s+"
                                   r"`?([^`\s]+)`?$", claimed, re.I)
                if m_hedge:
                    claimed = m_hedge.group(1)
                if claimed and claimed not in {"-", "—"}:
                    if claimed.lower() in VACUOUS:
                        results.append({**base,
                                        "claim": f"default = {claimed}",
                                        "verdict": "UNRESOLVED",
                                        "why": "names no value"})
                    else:
                        if "default" in meta:
                            res = default_matches(claimed, meta["default"])
                        else:
                            res = ("match" if claimed.lower() in
                                   {"none", "n/a", "required", "no default"}
                                   else "mismatch")
                        # A wrong default is wrong. Whether it also breaks the
                        # deck depends on whether the value named is legal at
                        # all, and that is severity, recorded beside the
                        # verdict rather than in place of it.
                        vals = meta.get("values") or []
                        harmless = bool(vals) and claimed in vals
                        verdict = "TRUE" if res == "match" else "FALSE"
                        row = {**base, "claim": f"default = {claimed}",
                               "verdict": verdict,
                               "why": f"`4C -p` says "
                                      f"{meta.get('default', '<no default>')}"}
                        if verdict == "FALSE":
                            row["severity"] = ("accepted value, wrong default"
                                               if harmless
                                               else "value 4C would reject")
                        results.append(row)

            vi = col_index(header, VALUES_COLS)
            if vi is not None and vi < len(cells) and cells[vi].strip():
                cell = cells[vi]
                claimed_vals = [v.strip().strip("`\"'")
                                for v in re.split(r"[,/|]| or ", cell)
                                if v.strip().strip("`\"'")]
                actual_vals = meta.get("values")
                if actual_vals and claimed_vals:
                    lower = {v.lower(): v for v in actual_vals}
                    bad, casey = [], []
                    for v in claimed_vals:
                        if v in actual_vals or v in {"...", "etc."}:
                            continue
                        if v.lower() in lower:
                            casey.append(v)
                        else:
                            bad.append(v)
                    wrong = bad + casey
                    results.append({
                        **base,
                        "claim": f"accepted values include {claimed_vals}",
                        "verdict": "FALSE" if wrong else "TRUE",
                        "why": (f"4C does not accept {wrong}" if wrong
                                else "all listed values accepted")})
                elif claimed_vals and not actual_vals:
                    results.append({**base,
                                    "claim": f"accepted values {claimed_vals}",
                                    "verdict": "UNRESOLVED",
                                    "why": "`4C -p` lists no enum for this key"})

            ri = col_index(header, REQ_COLS)
            if ri is not None and ri < len(cells) and cells[ri].strip():
                claimed = norm_cell(cells[ri]).lower()
                actual = meta.get("required")
                if actual is not None and claimed:
                    says_req = claimed in {"yes", "y", "required", "true",
                                           "mandatory"}
                    says_opt = claimed in {"no", "n", "optional", "false"}
                    if says_req or says_opt:
                        ok = (says_req and actual) or (says_opt and not actual)
                        results.append({**base,
                                        "claim": f"required = {claimed}",
                                        "verdict": "TRUE" if ok else "FALSE",
                                        "why": f"`4C -p` says required={actual}"})

    counts = Counter(r["verdict"] for r in results)
    total = len(results)
    print(f"table cells checked : {total}")
    for v in ("TRUE", "FALSE", "UNRESOLVED"):
        print(f"  {v:11s} {counts[v]:5d}  ({100*counts[v]/max(total,1):.1f}%)")
    decidable = counts["TRUE"] + counts["FALSE"]
    print(f"\nfalse rate over decidable cells: "
          f"{counts['FALSE']}/{decidable} = "
          f"{100*counts['FALSE']/max(decidable,1):.1f}%")
    by_claim = Counter(r["claim"].split(" =")[0].split(" include")[0]
                       for r in results if r["verdict"] == "FALSE")
    if by_claim:
        print("false by claim type: " +
              ", ".join(f"{k} {v}" for k, v in by_claim.most_common()))
    sev = Counter(r.get("severity", "") for r in results
                  if r["verdict"] == "FALSE")
    print("\n--- FALSE ---")
    for r in results:
        if r["verdict"] == "FALSE":
            s = f"  [{r['severity']}]" if r.get("severity") else ""
            print(f"  {r['doc']} [{r['section_claimed']}] {r['key']}: "
                  f"{r['claim']}  <- {r['why']}{s}")
    if any(sev):
        print("\nseverity of the wrong defaults: " +
              ", ".join(f"{k or 'n/a'}: {v}" for k, v in sev.most_common()))

    if args.out:
        args.out.write_text(json.dumps(
            {"counts": dict(counts), "results": results}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
