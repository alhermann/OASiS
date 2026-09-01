"""The live FEniCS mixed-elasticity knowledge must agree with its template."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.registry import get_backend, load_all_backends  # noqa: E402


def test_mixed_elasticity_knowledge_and_template_agree():
    load_all_backends()
    backend = get_backend("fenics")
    assert backend is not None
    knowledge = backend.get_knowledge("nearly_incompressible_elasticity")
    assert "p = -lambda*div(u)" in knowledge["weak_form"]
    assert "continuous vector Lagrange P2" in knowledge["function_space"]
    assert "continuous scalar Lagrange P1" in knowledge["function_space"]
    assert any("PARENT" in pitfall and "(W0, V0)" in pitfall
               for pitfall in knowledge["pitfalls"])

    script = backend.generate_input(
        "nearly_incompressible_elasticity", "2d", {})
    compile(script, "<fenics-mixed-elasticity>", "exec")
    assert "basix.ufl.mixed_element([P2, P1])" in script
    assert "(W0, V0), fdim, bottom_facets" in script
    assert "fem.dirichletbc(" in script and "W0)" in script