"""deal.II was the one backend that never said how to evaluate off-node.

MEASURED over 464 single-code runs: the largest single failure bucket is a
solve that SUCCEEDED and was then never read back at the required points — 60
runs, 12.9%. Every task states its probe points are "deliberately not mesh
nodes", so this is a required step.

Across the nine served backends, only deal.II's payload carried no way to do
it. FEniCSx serves bb_tree/compute_colliding_cells, NGSolve serves mesh(x,y),
scikit-fem serves probes, 4C serves its result files; deal.II served nothing,
and the runs that got it right used VectorTools::point_value from their own
prior knowledge.

The recipe is VERIFIED BY EXECUTION on this install, not quoted: a known
function interpolated onto FE_Q(1) and read back at 1936 non-nodal points gives
1.896963e-03 -> 4.810567e-04 -> 1.179388e-04, ratios 3.94 and 4.08, so the
evaluation is second order and does not cap the order a run can report.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PHYSICS = ["heat", "elasticity", "poisson", "advection_dg", "contact",
           "nonlinear_elasticity", "stokes", "not_a_physics"]


def _knowledge_tool():
    import os
    os.chdir(ROOT)
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools

    class _MCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **kw):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco
    mcp = _MCP()
    register_consolidated_tools(mcp)
    return mcp.tools["knowledge"]


def test_every_dealii_physics_carries_the_recipe():
    """Four return paths resolve this method; one branch is not enough."""
    from core.registry import load_all_backends, get_backend
    load_all_backends()
    b = get_backend("dealii")
    missing = [p for p in PHYSICS
               if "VectorTools::point_value" not in str(
                   (b.get_knowledge(p) or {}).get("probe_recipe", ""))]
    assert not missing, f"these deal.II physics reach the agent with no way to probe: {missing}"


def test_it_reaches_the_agent_through_the_real_tool():
    out = _knowledge_tool()(topic="physics", solver="dealii", physics="heat")
    assert "VectorTools::point_value" in out


def test_the_recipe_carries_its_measurement_not_just_a_claim():
    out = _knowledge_tool()(topic="physics", solver="dealii", physics="heat")
    for measured in ("1.896963e-03", "4.810567e-04", "1.179388e-04"):
        assert measured in out, (
            f"{measured} is missing; the recipe must carry the numbers it was "
            f"verified with, so a reader can re-run it"
        )


def test_it_warns_about_the_two_ways_it_goes_wrong():
    out = _knowledge_tool()(topic="physics", solver="dealii", physics="heat")
    assert "ExcPointNotAvailableHere" in out, (
        "a point a rounding error outside the mesh throws, and one exception "
        "can end a run"
    )
    assert "nearest vertex" in out, (
        "the alternative an agent reaches for caps the reported order at 1"
    )


def test_the_run_log_line_is_named_for_this_backend():
    """deal.II prints no mesh count at any verbosity; the agent must print it."""
    out = _knowledge_tool()(topic="physics", solver="dealii", physics="heat")
    assert "dof_handler.n_dofs()" in out
