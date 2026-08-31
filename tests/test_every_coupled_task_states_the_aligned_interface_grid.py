"""Read what the AGENT is told, for every coupled cell, and check the grids meet.

WHY THIS FILE EXISTS. The interface-probe text had FOUR copies:

    build_coupled_v2.iface_probe_rule       zero callers (dead; deleted)
    build_coupled_v2._straight_iface_probe  used by the v2 specs
    build_balanced.Geom.iface_probe         used by C1..C12   <- feeds C2
    build_balanced instance_C5's inline f-string  the bent interface

I fixed the first, found it dead, fixed the second, and then drew a fresh C2 and
read its task.txt: it still stated the OLD grid. Two more copies existed. Every
"fix" had been to a function the drawn task never touched.

That is the fifth time in this campaign a mechanism existed, was instrumented,
and did not reach the case it was built for. The lesson the campaign keeps
re-learning is the one this file applies: test what the AGENT ends up with, not
what a function returns.

WHAT MUST HOLD. The interface probe coordinates must be a SUBSET of the solution
probe coordinates. `flux_consistency` recomputes the interface flux from the
submitted FIELD and compares it with the flux the agent reported — the only
reference-free way to catch a fabricated flux. It needs three field samples in a
column AT an interface coordinate. Under the old independent spacing the two
sets could never share one: solution probes at (j+0.5)/44, i.e. odd multiples of
1/88, and interface probes at 0.25 + (i+0.5)/88 = (45+2i)/176 — even against odd
numerators in 1/176 units, so the intersection was empty for every C-series
cell.

These tests parse the rendered task text with a regex rather than calling the
generators, precisely so a fifth copy cannot pass them.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import campaign3_blind.build_balanced as B  # noqa: E402

# `(j+0.5)/44` — a solution-probe coordinate. The old text read
# `1/4 + (i+0.5)*1/2/44`, which this does not match.
ALIGNED = re.compile(r"\((?:j|k)\+0\.5\)/(\d+)")
OLD_STYLE = re.compile(r"\(i\+0\.5\)\s*\*\s*(?:1/2|\(1/4\))\s*/\s*\d+")


def _tasks(limit=6):
    out = []
    for fn in B.BUILDERS[:limit]:
        r = B.build_one(fn, 20260831)
        out.append((fn.__name__, r["task"]))
    return out


def _iface_line(task: str) -> str:
    for ln in task.splitlines():
        if ln.startswith("INTERFACE PROBE POINTS"):
            return ln
    return ""


class TestTheRenderedTaskText(unittest.TestCase):
    def test_every_coupled_task_states_an_interface_grid(self):
        for name, task in _tasks():
            self.assertTrue(
                _iface_line(task),
                f"{name}: the rendered task has no INTERFACE PROBE POINTS line")

    def test_the_interface_grid_is_expressed_on_the_solution_grid(self):
        """THE TEST THAT CAUGHT THE THIRD AND FOURTH COPIES."""
        for name, task in _tasks():
            line = _iface_line(task)
            self.assertRegex(
                line, ALIGNED,
                f"{name}: the interface points are not stated as solution-probe "
                f"coordinates (j+0.5)/M. A generator somewhere still emits its "
                f"own spacing, and the flux cross-check cannot run on this "
                f"cell. Line was: {line[:160]}")

    def test_no_task_still_uses_the_old_independent_spacing(self):
        for name, task in _tasks():
            line = _iface_line(task)
            self.assertNotRegex(
                line, OLD_STYLE,
                f"{name}: still emits the old band-relative spacing "
                f"`lo + (i+0.5)*w/M`, which shares no coordinate with the "
                f"solution grid. Line was: {line[:160]}")

    def test_the_stated_coordinates_really_lie_on_the_solution_grid(self):
        """Parse the M out of the task text and check the arithmetic, rather
        than trusting the wording."""
        import campaign3_blind.build_coupled_v2 as V
        for name, task in _tasks():
            line = _iface_line(task)
            Ms = {int(m) for m in ALIGNED.findall(line)}
            self.assertTrue(Ms, f"{name}: no denominator found in {line[:120]}")
            for M in Ms:
                self.assertEqual(
                    M, V.PROBE_M[2],
                    f"{name}: interface grid uses /{M} but the solution probe "
                    f"grid uses /{V.PROBE_M[2]}; the coordinates cannot "
                    f"coincide")

    def test_the_task_says_the_coordinates_are_shared(self):
        """An agent must be told WHY the same numbers appear twice, or it will
        assume one of the two lists is wrong."""
        for name, task in _tasks():
            self.assertIn(
                "solution probe coordinates", _iface_line(task),
                f"{name}: the task does not tell the agent these are the same "
                f"coordinates as its solution rows")


class TestTheAuthorityIsSingle(unittest.TestCase):
    def test_the_dead_generator_is_gone(self):
        import campaign3_blind.build_coupled_v2 as V
        self.assertFalse(
            hasattr(V, "iface_probe_rule"),
            "iface_probe_rule is back. It had zero callers and looked like the "
            "authority, which is how three copies drifted apart unnoticed.")

    def test_every_live_generator_routes_through_iface_probe_indices(self):
        import campaign3_blind.build_coupled_v2 as V
        src_b = Path(B.__file__).read_text()
        src_v = Path(V.__file__).read_text()
        self.assertIn("V.iface_probe_indices", src_b,
                      "build_balanced no longer derives its interface grid from "
                      "the shared authority")
        self.assertIn("iface_probe_indices(dim, _iface_band", src_v,
                      "_straight_iface_probe no longer derives its grid from "
                      "the shared authority")

    def test_a_band_too_narrow_for_three_points_raises(self):
        """The guard was written into iface_probe_rule, which had zero callers
        and was deleted — taking the guard with it. A band too narrow to hold
        three solution-probe coordinates would then quietly produce a one- or
        two-point interface statistic, which is not a statistic. It now lives in
        the authority."""
        import campaign3_blind.build_coupled_v2 as V
        narrow = (0.5, 0.5 + 1.0 / V.PROBE_M[2])      # at most one coordinate
        with self.assertRaises(RuntimeError):
            V.iface_probe_indices(2, narrow)

    def test_a_degenerate_band_raises_rather_than_returning_nothing(self):
        import campaign3_blind.build_coupled_v2 as V
        with self.assertRaises(RuntimeError):
            V.iface_probe_indices(3, (0.5, 0.5))

    def test_the_bands_actually_used_all_pass_the_guard(self):
        """The guard must not be so strict that a real cell cannot be drawn."""
        import campaign3_blind.build_coupled_v2 as V
        for dim, span in ((2, V._iface_band(0, 1)), (3, V._iface_band(0, 1)),
                          (2, (0.125, 0.375)), (2, (0.625, 0.875))):
            M, js = V.iface_probe_indices(dim, span)
            self.assertGreaterEqual(len(js), 3, f"dim={dim} span={span}")


if __name__ == "__main__":
    unittest.main()
