"""OASiS must not tell the model under test that it is being tested.

The knowledge served to the OASiS arm carried, verbatim: "Measured over 256
runs of this campaign, agents stop VOLUNTARILY at a median of 46% of their wall
budget", "Three runs in this campaign wrote a complete manufactured source into
a deck", and the audit tool's own description said "measured over 40 graded
runs of a blind campaign". The bare arm has no knowledge tool, so ONLY the
treatment arm was told, in plain language, that it is one of many measured runs
in a study.

Worse, one line leaked the composition of the evaluation set: "One cell in this
campaign mandates the MORLEY element". That tells the agent such a cell exists.

The lessons those sentences carried are real and are kept. What is removed is
the provenance — an agent should read domain knowledge, not our lab notes.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# phrases that tell the reader it is inside a measured study, or describe the
# evaluation set
BANNED = re.compile(
    r"this campaign|blind campaign|graded runs of|this benchmark|"
    r"one cell in this|of the campaign|our campaign|the eval(uation)? set",
    re.I)


def _tools():
    from core.registry import load_all_backends
    load_all_backends()

    class M:
        def __init__(self):
            self.t = {}

        def tool(self, *a, **k):
            def d(f):
                self.t[f.__name__] = f
                return f
            return d

    import tools.consolidated as C
    m = M()
    for n in dir(C):
        if n.startswith("register") and callable(getattr(C, n)):
            try:
                getattr(C, n)(m)
            except Exception:                     # noqa: BLE001
                pass
    return m.t


def _call(fn, **kw):
    r = fn(**kw)
    return str(asyncio.run(r) if asyncio.iscoroutine(r) else r)


def test_no_served_payload_says_it_is_a_campaign():
    t = _tools()
    checks = [
        ("knowledge/physics", lambda: _call(t["knowledge"], topic="physics",
                                            solver="fenics", physics="poisson")),
        ("knowledge/coupling", lambda: _call(t["knowledge"], topic="coupling",
                                             solver="fenics")),
        ("knowledge/coupling/ngsolve", lambda: _call(t["knowledge"],
                                                     topic="coupling",
                                                     solver="ngsolve")),
        ("prepare_simulation", lambda: _call(t["prepare_simulation"],
                                             solver="fenics",
                                             physics="poisson")),
    ]
    bad = []
    for label, get in checks:
        hits = BANNED.findall(get())
        if hits:
            bad.append(f"{label}: {sorted({h if isinstance(h, str) else h[0] for h in hits})}")
    assert not bad, (
        "served text tells the model under test that it is in a study:\n  "
        + "\n  ".join(bad))


def test_tool_descriptions_do_not_say_it_either():
    """An MCP tool's docstring is surfaced to the model as its description."""
    t = _tools()
    bad = []
    for name, fn in t.items():
        doc = (fn.__doc__ or "")
        hits = BANNED.findall(doc)
        if hits:
            bad.append(f"{name}: {doc[:120]!r}")
    assert not bad, (
        "an MCP tool description tells the model it is in a study:\n  "
        + "\n  ".join(bad))


def test_the_lesson_survived_the_edit():
    """Removing provenance must not remove the knowledge."""
    t = _tools()
    text = _call(t["knowledge"], topic="physics", solver="fenics",
                 physics="poisson")
    # stopping early
    assert "wall budget" in text and "VOLUNTARILY" in text
    # the wiring rule
    assert "INERT UNTIL IT IS WIRED IN" in text
    # the prescribed-element rule, without naming a cell of the eval set
    assert "NONCONFORMING" in text and "Morley" in text


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
