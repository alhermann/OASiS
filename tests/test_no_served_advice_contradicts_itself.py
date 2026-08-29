"""Served text must not tell an agent to do something we elsewhere call impossible.

The DUNE not-installed message recommended `conda create -n ofa-dune -c
conda-forge dune-fem`, while src/backends/_setup.py says that command "cannot
succeed" and src/server.py says "conda-forge has NO dune-fem package". An agent
reads the message at exactly the moment DUNE has failed — the worst possible
place to spend its remaining budget on a command that cannot work.

Same defect class as promising participant sources "in the directory the
payload came from": a promise the install cannot keep reads as an instruction.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SRC = REPO / "src"


def test_no_source_file_recommends_conda_forge_dune_fem():
    """One claim, one direction. conda-forge does not ship dune-fem."""
    offenders = []
    pat = re.compile(r"conda\s+create[^\n]*conda-forge[^\n]*dune-fem")
    for f in SRC.rglob("*.py"):
        text = f.read_text(errors="replace")
        for m in pat.finditer(text):
            line_no = text[:m.start()].count("\n") + 1
            line = text.splitlines()[line_no - 1]
            # a passage may QUOTE the command in order to say it cannot work
            near = text[max(0, m.start() - 300):m.end() + 300].lower()
            if any(k in near for k in ("cannot succeed", "no dune-fem",
                                       "does not exist", "used to recommend",
                                       "impossible")):
                continue
            offenders.append(f"{f.relative_to(REPO)}:{line_no}: {line.strip()[:90]}")
    assert not offenders, (
        "served text recommends a conda-forge install of dune-fem, which "
        "other served text calls impossible:\n  " + "\n  ".join(offenders))


def test_the_dune_failure_hint_names_the_working_route():
    from backends.dune.backend import _DUNE_INSTALL_HINT as hint
    assert "pip install dune-fem" in hint
    assert "mpi4py" in hint, "the undeclared dependency must be named"
    assert "conda-forge has no dune-fem" in hint.lower()


def test_the_dune_hint_explains_the_failure_that_survives_import():
    """A stale JIT cache passes `import dune.fem` and dies at first use.

    That cost this campaign a repointed interpreter mid-round; an agent hitting
    it with no explanation would burn its budget on a working install.
    """
    from backends.dune.backend import _DUNE_INSTALL_HINT as hint
    low = hint.lower()
    assert "undefined symbol" in low
    assert "dune-py" in low or "cache" in low


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
