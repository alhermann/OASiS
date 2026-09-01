"""A check that answers about an operator it does not implement is not weak.

It is a source of wrong answers pointed at the arm under test.

MEASURED, by handing real submissions to the unguarded tool:

    FC2 (linear elasticity, Lame lambda/mu)
        INCONSISTENT — "your field is converging to something that is not the
        solution of the stated problem", residual 1.197e+297
    SK2 (the biharmonic equation, lap(lap(u)) = f)
        CONSISTENT — "Your field satisfies the equation you were given",
        rate 2.09, residual 5.190e+293

The first tells an agent to discard work that may be right. The second BLESSES
a field on evidence that does not exist, which is worse. Of the campaign's 16
single-code cells, 13 lie outside the implemented form — elasticity, Stokes,
Navier-Stokes, biharmonic, transient heat, a nonlinear a(u), a variable a(x,y)
— and `verify_pde_consistency` is advertised in the universal core, which now
reaches 100% of knowledge calls. FE1, whose coefficient depends on u, was
observed calling it during round 9.

So the tool now requires the equation and refuses anything but the scalar
second-order diffusion form, and the underlying check refuses a residual too
large to be a discretisation's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tool():
    """The registered MCP tool, as an agent reaches it."""
    from mcp.server.fastmcp import FastMCP
    from tools import consolidated
    mcp = FastMCP("t")
    consolidated.register(mcp) if hasattr(consolidated, "register") else None
    for name in ("verify_pde_consistency",):
        fn = getattr(consolidated, name, None)
        if fn is not None:
            return fn
    pytest.skip("verify_pde_consistency is not reachable as a plain function")


def _levels(tmp: Path, n=8, scale=1.0):
    """u = scale * x*(1-x)*y*(1-y) on the prescribed midpoint grid."""
    files = []
    for lvl, m in enumerate((22, 44), start=1):
        p = tmp / f"solution_level{lvl}.csv"
        rows = ["x,y,u"]
        for i in range(m):
            for j in range(m):
                x = (i + 0.5) / m
                y = (j + 0.5) / m
                rows.append(f"{x},{y},{scale * x*(1-x)*y*(1-y)}")
        p.write_text("\n".join(rows))
        files.append(str(p))
    return ",".join(files)


def test_a_field_that_satisfies_the_identity_exactly_is_not_condemned():
    """Found by writing the test below, and worth its own name.

    The decay test asks whether the residual FALLS. It has no answer when the
    residual is already zero: an exact field gives 0.000e+00 at every level,
    `last < first/3` is false, and the verdict came out INCONSISTENT with the
    explanation "the weak residual is FLAT: 0.000e+00 -> 0.000e+00" — the one
    field that could not be more right, told it converges to the wrong
    solution.
    """
    import math

    from tools.pde_consistency import check_levels
    lv = {}
    for lvl, m in ((1, 22), (2, 44)):
        lv[lvl] = [[(i + 0.5) / m, (j + 0.5) / m,
                    math.sin(math.pi * (i + 0.5) / m)
                    * math.sin(math.pi * (j + 0.5) / m)]
                   for i in range(m) for j in range(m)]
    out = check_levels(lv, "2*pi**2*sin(pi*x)*sin(pi*y)", 1.0,
                       [(0.0, 1.0), (0.0, 1.0)])
    assert out.verdict == "CONSISTENT", out.explanation
    assert "round-off" in out.explanation
    # and it must not overclaim: round-off also describes an interpolated
    # exact field, which proves nothing about a solver having run
    assert "not that a solver ran" in out.explanation


def test_the_guard_exists_in_the_served_tool():
    """What the AGENT ends up with: the refusal, and what it points to."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    i = src.index("def verify_pde_consistency(")
    body = src[i:i + 6000]
    assert "equation: str = \"\"" in src[i:i + 400], (
        "the tool must take the equation; it cannot tell from the numbers "
        "alone whether its own operator applies")
    assert "REFUSED: pass `equation=`" in body
    assert "REFUSED: this check implements -div(K grad u) = f" in body
    for allowed in ("-div(kgradu)=f", "-lap(u)=f"):
        assert allowed in body, f"{allowed} must be accepted"
    # it must name the alternative rather than leaving the agent stuck
    assert body.count("audit_results(work_dir=") >= 2, (
        "a refusal that does not say what to use instead costs the run an "
        "action and gives it nothing")
    # the measured evidence must travel with the guard
    assert "2.09" in body and "biharmonic" in body


def test_an_absurd_residual_is_not_reported_as_a_verdict():
    from tools.pde_consistency import check_levels
    # a field of enormous values against a small source: the identity's two
    # sides are not comparable, which is exactly the elasticity/biharmonic case
    lv = {}
    for lvl, m in ((1, 22), (2, 44)):
        rows = []
        for i in range(m):
            for j in range(m):
                x, y = (i + 0.5) / m, (j + 0.5) / m
                rows.append([x, y, 1e150 * x * y])
        lv[lvl] = rows
    out = check_levels(lv, "1.0", 1.0, [(0.0, 1.0), (0.0, 1.0)])
    assert out.verdict == "NOT_APPLICABLE", (
        f"a residual outside anything a discretisation produces must not "
        f"become a CONSISTENT or INCONSISTENT verdict; got {out.verdict}")
    assert "REFUSED" in " ".join(
        r.detail for r in out.levels), out.explanation


def test_a_genuine_scalar_diffusion_case_still_passes():
    """The guard must not close the door on the three cells it does serve."""
    from tools.pde_consistency import check_levels
    import math
    # u = sin(pi x) sin(pi y) solves -lap u = 2 pi^2 sin(pi x) sin(pi y)
    lv = {}
    for lvl, m in ((1, 22), (2, 44)):
        rows = []
        for i in range(m):
            for j in range(m):
                x, y = (i + 0.5) / m, (j + 0.5) / m
                rows.append([x, y, math.sin(math.pi * x) * math.sin(math.pi * y)])
        lv[lvl] = rows
    out = check_levels(lv, "2*pi**2*sin(pi*x)*sin(pi*y)", 1.0,
                       [(0.0, 1.0), (0.0, 1.0)])
    assert out.verdict == "CONSISTENT", out.explanation
