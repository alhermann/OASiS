"""No served participant may be pre-configured to a campaign cell.

Paper section 3.1: served knowledge is a pattern, a calling convention or a
failure mechanism — never the answer to a benchmark problem. A template whose
defaults ARE a graded cell's geometry breaks that even when every number is
also printed in the task text, because it removes that cell's configuration
work and no other cell's. It is an assist keyed to the benchmark rather than
to the physics.

This is what the FSI pair did. Under a header reading

    # ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
    #    Replace ALL of them with your problem's geometry, material and BCs.

all twelve numbers were C14's: LX 1.0, HY 0.2, HS 0.05, MU 1.0, RHO 1.0,
U_MEAN 1.0, E 3.0e6, NU 0.3, fluid mesh 48x10, solid mesh 40x4. C14 was built
from this very fixture. Every other participant family already sits
deliberately off-campaign — the DD ones split at 0.6 and 0.55 where the cells
split at 0.625, with K=0.8 and E=1000 — so the convention existed and the FSI
pair was the one place it lapsed.

The test compares the served defaults against the campaign's own geometry
record rather than against numbers copied into this file, so it keeps working
if a cell is re-drawn.
"""

from __future__ import annotations

import ast
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTICIPANTS = REPO_ROOT / "data" / "coupling_participants"
BUILDER = REPO_ROOT / "campaign3_blind" / "build_offpool.py"

# How many of a cell's geometry values may coincide with one template's
# defaults before it stops being coincidence. Values like 0.3 (Poisson) and 1.0
# recur innocently, so a couple of hits mean nothing; most of them matching
# means the template is that cell.
MAX_SHARED_VALUES = 4


def _campaign_geometries() -> dict:
    """Every `<CELL>_GEOM = dict(...)` literal in the campaign builder."""
    if not BUILDER.is_file():
        return {}
    tree = ast.parse(BUILDER.read_text())
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.endswith("_GEOM"):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            if isinstance(node.value, ast.Call):
                try:
                    value = {kw.arg: ast.literal_eval(kw.value)
                             for kw in node.value.keywords}
                except ValueError:
                    continue
            else:
                continue
        if isinstance(value, dict):
            out[target.id] = {k: v for k, v in value.items()
                              if isinstance(v, (int, float))}
    return out


_ASSIGN = re.compile(r"^([A-Z][A-Z0-9_]*(?:\s*,\s*[A-Z][A-Z0-9_]*)*)\s*=\s*(.+?)"
                     r"(?:\s*#.*)?$", re.M)

# The campaign records a cell's geometry under its own names; a template names
# the same quantity its own way. Compare like with like — an earlier version of
# this test compared bare VALUES and reported six innocent files, because C14
# has four separate quantities equal to 1.0 (LX, MU, RHO, U_MEAN), so any file
# mentioning 1.0 anywhere scored four hits.
ALIASES = {
    "LX": ("LX",), "HY": ("HY", "Y0"), "HS": ("HS",),
    "MU": ("MU",), "RHO": ("RHO_F", "RHO_S", "RHO"),
    "U_MEAN": ("U_MEAN",), "E": ("E_MOD", "E"), "NU": ("NU",),
    "NXF": ("NX",), "NYF": ("NY",), "NXS": ("NXS",), "NYS": ("NYS",),
}


def _template_numbers(text: str) -> dict:
    """Top-level numeric constants a reader would edit, by name."""
    found = {}
    for lhs, rhs in _ASSIGN.findall(text):
        names = [n.strip() for n in lhs.split(",")]
        try:
            value = ast.literal_eval(rhs.strip())
        except (ValueError, SyntaxError):
            continue
        items = list(value) if isinstance(value, tuple) else [value]
        if len(names) != len(items):
            continue
        for name, item in zip(names, items):
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                found[name] = float(item)
    return found


def _shared_with_cell(served: dict, geom: dict) -> list:
    """Cell keys whose value the template sets under a corresponding name."""
    hits = []
    for key, value in geom.items():
        for alias in ALIASES.get(key, (key,)):
            if alias in served and served[alias] == float(value):
                hits.append(key)
                break
    return sorted(hits)


class TestServedParticipantsAreGeneric(unittest.TestCase):
    def test_no_participant_reproduces_a_cell_geometry(self):
        geoms = _campaign_geometries()
        self.assertTrue(
            geoms,
            "no <CELL>_GEOM record found in build_offpool.py — this test "
            "cannot see what the campaign prescribes, so it is not guarding "
            "anything. Point it at the current builder.",
        )
        offenders = []
        for path in sorted(PARTICIPANTS.glob("participant_*.py")):
            served = _template_numbers(path.read_text())
            for cell, geom in geoms.items():
                shared = _shared_with_cell(served, geom)
                if len(shared) > MAX_SHARED_VALUES:
                    offenders.append(
                        f"{path.name} carries {len(shared)} of {cell}'s "
                        f"{len(geom)} prescribed values {shared} — its "
                        f"defaults are that cell, not a placeholder"
                    )
        self.assertFalse(offenders, "\n  ".join(offenders))

    def test_the_placeholder_header_is_honest(self):
        """A block calling itself arbitrary must not be one specific problem."""
        geoms = _campaign_geometries()
        for path in sorted(PARTICIPANTS.glob("participant_fsi_*.py")):
            text = path.read_text()
            if "ARBITRARY PLACEHOLDER" not in text:
                continue
            served = _template_numbers(text)
            for cell, geom in geoms.items():
                shared = _shared_with_cell(served, geom)
                self.assertLessEqual(
                    len(shared), MAX_SHARED_VALUES,
                    f"{path.name} tells the agent every number is an arbitrary "
                    f"placeholder while {len(shared)} of them are {cell}'s "
                    f"actual prescription {shared}",
                )


if __name__ == "__main__":
    sys.exit(unittest.main())
