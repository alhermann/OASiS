"""Every per-code signature must match real solver output and reject narration.

WHAT WAS MEASURED. All nine backends were run on this machine and their output
captured. Of the previous table's 27 patterns, 24 NEVER matched the installed
version's real output, and all 3 that did also matched a hand-written log a
numpy script could produce. Zero were both real and non-fakeable. So every
`PROVEN` verdict in the campaign came from the code-agnostic `NDOF =` contract
line — which is exactly why coupled attribution collapsed: one file cannot be
two codes' output.

Concrete casualties found in the graded tree:

  * FEBio letter-spaces its banner as `N O R M A L   T E R M I N A T I O N`, so
    the pattern `normal termination` could not match, and ` Elapsed time :` is
    not `total elapsed time` either. 93 run directories hold a real FEBio log
    and the febio patterns matched none. `FB1_27b_MCP_seed5` was graded
    FABRICATED_NO_RUN with `Nr of equations : 196` in its own log.
  * `number of (nodes|elements)` was the 4C pattern, and 4C never prints it on
    the inline-mesh path. It is an ordinary English sentence an agent writes
    about its own mesh — the exact trap the table must avoid.
  * deal.II's `number of active cells` is a TUTORIAL PROGRAM print, not library
    output.

TWO FIXTURES, both committed so this test outlives the scratchpad:

  * `real_matched_lines.json` — the lines each code actually emitted, extracted
    from the captured runs. Every code must still match at least one.
  * `adversarial_agent_narration.log` — a hand-written log that narrates in each
    code's dialect (`Number of active cells: 1024`, `Nr of equations: 1089`,
    `DEAL:cg: Convergence after 47 steps`, `Mesh 0: Number of Nodes: 22`,
    `Fem::CG converged`, `NORMAL TERMINATION`, ...). NOTHING may match it.

The second fixture is the load-bearing one. If a future pattern is loosened into
plausible English, this test fails rather than the campaign silently accepting a
monolith. Measured now: 0 of 48 patterns match it.

A NOTE ON WHY re.MULTILINE IS ASSERTED HERE. code_evidence searched with
re.IGNORECASE alone, so every `^`-anchored pattern could only match at byte 0 of
a file. That made most of the rewritten table dead on arrival — found only
because an impact measurement came back as a suspiciously perfect 92 of 92.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import src.blind_eval.evidence as EV  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "measured_solver_output"
REAL = json.loads((FIXTURES / "real_matched_lines.json").read_text())
FAKE = (FIXTURES / "adversarial_agent_narration.log").read_text()

# The flags code_evidence itself uses. Kept in one place so the tests below
# cannot pass with flags the grader does not apply.
FLAGS = re.IGNORECASE | re.MULTILINE

# Codes with measured output. `fenicsx`/`fourc`/`dolfinx` are aliases.
MEASURED = ["fenics", "dealii", "4C", "ngsolve", "skfem", "kratos", "dune",
            "febio", "sparta"]


class TestPatternsMatchRealOutput(unittest.TestCase):
    def test_every_code_has_at_least_one_live_pattern(self):
        """A code whose patterns match nothing real grades every honest run of
        that code as unattributable."""
        dead = []
        for code in MEASURED:
            lines = REAL.get(code, [])
            self.assertTrue(lines, f"no measured output recorded for {code}")
            blob = "\n".join(lines)
            if not any(re.search(p, blob, FLAGS)
                       for p in EV.PER_CODE_SIGNATURES[code]):
                dead.append(code)
        self.assertEqual(dead, [],
                         f"these codes' patterns match none of their own "
                         f"measured output: {dead}")

    def test_no_pattern_in_the_table_is_dead_code(self):
        """Every individual pattern must match something real. A pattern that
        matches nothing is the defect this whole rewrite addressed — 24 of the
        previous 27 were in that state."""
        blob = "\n".join(l for ls in REAL.values() for l in ls)
        dead = []
        for code in MEASURED:
            for p in EV.PER_CODE_SIGNATURES[code]:
                if not re.search(p, blob, FLAGS):
                    dead.append((code, p[:60]))
        self.assertEqual(
            dead, [],
            "patterns matching no measured output (add a measured sample or "
            f"remove the pattern): {dead}")


class TestPatternsRejectNarration(unittest.TestCase):
    """The load-bearing half."""

    def test_nothing_matches_an_agent_narrating_in_each_dialect(self):
        leaks = []
        for code, pats in EV.PER_CODE_SIGNATURES.items():
            for p in pats:
                m = re.search(p, FAKE, FLAGS)
                if m:
                    leaks.append((code, p[:55], m.group(0)[:60]))
        self.assertEqual(
            leaks, [],
            "a pattern matched hand-written narration, so a numpy solver can "
            f"satisfy that code's execution evidence: {leaks}")

    def test_the_fixture_really_is_adversarial(self):
        """Guard the guard: if the fake were emptied or truncated, the test
        above would pass for the wrong reason."""
        self.assertGreater(len(FAKE), 1500,
                           "the adversarial fixture shrank; it can no longer "
                           "be evidence that patterns reject narration")
        for phrase in ("Number of active cells", "Nr of equations",
                       "NORMAL TERMINATION", "number of nodes",
                       "Number of Nodes", "Fem::CG", "DEAL:"):
            self.assertIn(phrase, FAKE,
                          f"the fake no longer narrates {phrase!r}, so it "
                          f"stopped testing that dialect")

    def test_the_old_patterns_would_fail_this_test(self):
        """Proof the test discriminates: the patterns that shipped before this
        rewrite DO match the narration fixture."""
        old = [r"number of (?:nodes|elements)\s*[:=]\s*(\d{2,})",
               r"number of active cells\s*[:=]\s*([\d,]{2,})",
               r"normal termination[^\n]{0,80}",
               r"total elapsed time\s*[:=]\s*([\d:\.]+)"]
        matched = [p for p in old if re.search(p, FAKE, FLAGS)]
        self.assertEqual(
            len(matched), len(old),
            "the pre-rewrite patterns no longer all match the narration "
            "fixture, so this test has stopped measuring the thing it was "
            f"built to measure. Matched {len(matched)} of {len(old)}.")


class TestTheGraderAppliesTheseFlags(unittest.TestCase):
    def test_code_evidence_searches_multiline(self):
        """An ^-anchored pattern must match on line 200 of a log, not only at
        byte 0. This was broken and silently killed most of the table."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)
            (w / "run_level1.log").write_text(
                "noise\n" * 200
                + "\tNr of equations ........................... : 196\n"
                + " N O R M A L   T E R M I N A T I O N\n")
            it = EV.code_evidence(w, "febio")
        specific = [m for m in it.matches if "canonical" not in m]
        self.assertTrue(
            specific,
            "a real FEBio line 200 lines into a log was not matched; "
            "code_evidence is not searching multiline")


class TestCanonicalMatchesAreLabelled(unittest.TestCase):
    """`only_canonical` decides coupled attribution by looking for the marker
    'canonical contract line' in every match. An unlabelled code-agnostic match
    therefore reads as ATTRIBUTION and disarms the rule.

    That happened: CANONICAL_SIGNATURES (`ndof[:=]<n>`, case-insensitive, so it
    also matches the contract line `NDOF = 54` itself) was concatenated onto
    each code's pattern list and recorded with no marker. Once the canonical
    check stopped short-circuiting the per-code scan, every compliant file
    matched it, `only_canonical` became False, and C2_27b_MCP_seed15 — numpy and
    scipy only, task naming 4C and Kratos — graded PROVEN again.
    """

    def test_a_code_agnostic_match_is_marked_as_such(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            w = Path(d)
            # lowercase `ndof =` only: no contract line, no code-specific output
            (w / "run_level1.log").write_text("ndof = 1089\n")
            it = EV.code_evidence(w, "dealii")
        self.assertEqual(it.verdict, "PROVEN",
                         "a quiet honest run lost its evidence")
        unlabelled = [m for m in it.matches if "canonical" not in m]
        self.assertEqual(
            unlabelled, [],
            "a code-agnostic match was recorded without the canonical marker, "
            f"so it will read as attribution: {unlabelled}")


if __name__ == "__main__":
    unittest.main()
