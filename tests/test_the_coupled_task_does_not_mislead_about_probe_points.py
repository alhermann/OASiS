"""The coupled task told the agent something that is not true.

Its interface-probe clause ended: "These are exactly the solution probe
coordinates lying in the graded band, so the same values appear in your
solution_level<k>.csv rows."

MEASURED on a drawn instance. The interface sits at x = 5/8, while subdomain A's
solution probes span x in [0.007102, 0.617898] and B's span [0.634943,
1.490057], because each set is the cell midpoints of its OWN subdomain. Neither
contains the interface coordinate. Only the transverse coordinates coincide.

An agent that believed the sentence would search its solution file for interface
rows, find none, and reasonably conclude its probe grid was wrong — and the
measured failure on this cell is precisely agents writing the wrong points into
the interface file.

The same sentence also named solution_level<k>.csv, the SINGLE-CODE filename,
inside a coupled task whose own output clause asks for
solution_level<k>_<side>.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign3_blind"))

from build_coupled_v2 import _straight_iface_probe  # noqa: E402


def test_the_false_subset_claim_is_gone():
    for dim in (2, 3):
        t = _straight_iface_probe(dim)
        assert "so the same values appear" not in t, (
            f"dim {dim}: the task still tells the agent the interface rows are "
            f"already in its solution file"
        )


def test_it_says_where_the_interface_coordinate_actually_sits():
    t = _straight_iface_probe(2)
    assert "lies BETWEEN" in t


def test_it_uses_the_coupled_filename():
    t = _straight_iface_probe(2)
    assert "solution_level<k>_<side>.csv" in t
    assert "solution_level<k>.csv" not in t.replace(
        "solution_level<k>_<side>.csv", "")


def test_the_probe_sets_really_do_exclude_the_interface():
    """The premise of the correction, re-measured rather than trusted."""
    xi = 0.625
    a = [(i + 0.5) * 0.625 / 44 for i in range(44)]
    b = [0.625 + (i + 0.5) * 0.875 / 44 for i in range(44)]
    assert not any(abs(x - xi) < 1e-12 for x in a)
    assert not any(abs(x - xi) < 1e-12 for x in b)
    assert max(a) < xi < min(b), (
        "the interface must sit strictly between the two probe columns for the "
        "corrected wording to be right"
    )


def test_the_interior_only_warning_survives():
    """The end points are excluded on purpose; that must still be said."""
    t = _straight_iface_probe(2)
    assert "INTERIOR of the interface only" in t
    assert "does not converge" in t
