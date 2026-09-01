"""The DUNE SIPG route must contain the operator it advertises."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.registry import get_backend, load_all_backends  # noqa: E402


def test_dune_sipg_capability_is_registered_and_complete():
    load_all_backends()
    backend = get_backend("dune")
    assert backend is not None
    names = {physics.name for physics in backend.supported_physics()}
    assert "dg_advection_diffusion" in names

    script = backend.generate_input(
        "dg_advection_diffusion", "2d",
        {"nx": 3, "order": 2, "diffusion": 1, "bx": 2, "by": 1,
         "beta": 20})
    compile(script, "<dune-sipg-template>", "exec")
    for required in (
            "aluConformGrid", "dglagrange", "DuneCellDiameter",
            "avg(grad(u))", "jump(v, normal)",
            "beta * degree * degree / h_avg", "solver=\"gmres\"",
            "pointSample", "DUNE_SIPG_OK"):
        assert required in script
    assert "from dune.ufl import DirichletBC" not in script
    assert "DirichletBC(" not in script