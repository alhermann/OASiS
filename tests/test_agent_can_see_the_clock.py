"""The agent is told how much wall-clock is left, on every command.

Measured on C9, 27B, seeds 22 and 23: both wrote COULD_NOT_COMPLETE blaming
the budget — "Time constraints (45 minutes) prevented completion",
"Insufficient time ... within the 45-minute budget" — after using 1093 s and
1110 s of 2700. They stopped at eighteen minutes believing they were out of
forty-five, and threw away 59% of the run each.

The prompt states the budget once and nothing updates it, so the agent has no
way to know. This is a harness fact, not a capability: BOTH arms get it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "langgraph_eval"))


def test_the_note_states_time_remaining():
    import agent as A
    A._DEADLINE = (time.time() + 1800.0, 2700.0)
    try:
        note = A._time_left_note()
        assert "clock:" in note
        assert "min left of 45" in note, note
        assert "29 min left" in note or "30 min left" in note, note
    finally:
        A._DEADLINE = None


def test_no_deadline_means_no_note():
    """Nothing is stamped outside a measured run."""
    import agent as A
    A._DEADLINE = None
    assert A._time_left_note() == ""


def test_a_spent_budget_says_so():
    import agent as A
    A._DEADLINE = (time.time() - 5.0, 2700.0)
    try:
        assert "budget spent" in A._time_left_note()
    finally:
        A._DEADLINE = None


def test_the_shell_tool_stamps_it(tmp_path):
    import agent as A
    A._DEADLINE = (time.time() + 600.0, 2700.0)
    try:
        run_bash = A._bash_tool_for(tmp_path)
        out = run_bash.invoke({"command": "echo hello"})
        assert "hello" in out
        assert "clock:" in out and "min left of 45" in out
    finally:
        A._DEADLINE = None


def test_both_arms_get_it():
    """A clock is not an OASiS capability.

    The bare arm builds its shell tool from the same factory, so the stamp is
    arm-neutral by construction. This pins that the factory is shared rather
    than duplicated per arm.
    """
    src = (REPO / "langgraph_eval" / "agent.py").read_text()
    assert src.count("def _bash_tool_for(") == 1
    assert "_time_left_note()" in src
    # and it must not be gated on the OASiS-only flag
    i = src.index("def _bash_tool_for(")
    j = src.index("def _kill_group(")
    assert "audit_on_submit" not in src[i:j], (
        "the clock is inside an OASiS-only branch; it must be arm-neutral")


def test_the_note_carries_no_domain_content():
    import agent as A
    A._DEADLINE = (time.time() + 900.0, 2700.0)
    try:
        low = A._time_left_note().lower()
        for leak in ("flux", "interface", "converge", "mesh", "solve",
                     "dirichlet", "neumann", "submit", "result"):
            assert leak not in low, f"the clock note leaks {leak!r}"
    finally:
        A._DEADLINE = None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
