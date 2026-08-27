"""The shipped participants must RECOVER an interface flux, not approximate it.

WHY THIS FILE EXISTS. The pair test next door runs both sides of every shipped
participant and passes — and it cannot see this defect at all. Its placeholder
problem has a zero source and equal outer temperatures, so the solution is the
1-D profile between them, the interface flux is ONE CONSTANT (18.46 along the
whole edge), and a P1 gradient is exact on a linear field. Every recovery,
right or wrong, agrees to machine precision there. A test whose fixture cannot
express the failure is not evidence about the failure.

So this one imposes a flux that VARIES along the interface, q(y) = 2 + 3 sin(4y),
hands it to the Neumann side as if a partner had exported it, and asks the
participant to give it back. No manufactured PDE solution and no coupling
iteration are needed: the participant applies the partner's number as
+int_Gamma g v ds, so the exported outward flux must come back as -g.

Measured for FEniCSx with the two candidate recoveries, interior interface
nodes, 8/16/32/64/128 uniform triangle meshes:

    projected gradient   max error stalls at 2.6 — order 0.00, it never
                         converges; 0.93 away from the ends, 0.50 in rms
    reaction vs b_vol    order 2.00 in max, away-from-ends and rms alike

The end nodes are reported apart from the interior throughout, because an
interface node that also sits on the outer Dirichlet boundary carries the outer
reaction too and is physically a different quantity.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PART_DIR = REPO / "data" / "coupling_participants"

Y0, Y1 = 0.0, 0.4
X0, X1 = 0.6, 1.1
IFACE_X = 0.6                      # the Neumann side's LEFT edge


def q_partner(y):
    """What the partner claims to be exporting, in ITS outward normal."""
    return 2.0 + 3.0 * np.sin(4.0 * np.asarray(y, float))


# A non-zero volume source, so that subtracting the WRONG vector is visible:
# with f == 0 the volume load is zero and a residual taken against the full
# load would still be near zero for the wrong reason.
SRC_LINE = "return np.zeros_like(x)"
SRC_REPL = "return -K * (6.0 * x * y**2 + 2.0 * x**3)"

BASE_EDITS = {
    "SIDE": '"neumann"', "PARTNER": '"left"',
    "X0, X1": f"{X0}, {X1}", "Y0, Y1": f"{Y0}, {Y1}",
    "IFACE_X": str(IFACE_X), "K": "1.5", "T_OUTER": "300.0",
    "T_INIT": "310.0", "Q_INIT": "0.0",
}


def _edit(text: str, edits: dict) -> str:
    for k, v in edits.items():
        pat = re.compile(rf"^({re.escape(k)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        assert pat.search(text), f"edit key {k!r} not found in the shipped script"
        text = pat.sub(lambda m: m.group(1) + v + m.group(3), text, count=1)
    return text


def _interpreter(backend: str):
    if backend == "fenics":
        from backends.fenics.backend import _find_fenics_python
        p = _find_fenics_python()
        return str(p) if p else None
    if backend in ("skfem", "ngsolve"):
        return sys.executable
    if backend == "dune":
        from backends.dune.backend import _find_dune_python
        return _find_dune_python()
    return None


def _available(name: str) -> bool:
    from core.registry import get_backend, load_all_backends
    load_all_backends()
    b = get_backend(name)
    return bool(b) and b.check_availability()[0].value == "available"


def _run_once(tmp: Path, backend: str, n: int):
    """One solve at mesh resolution n; returns (y, q_exported) or None."""
    src = PART_DIR / f"participant_{backend}.py"
    if not src.is_file():
        pytest.skip(f"no shipped participant for {backend}")
    text = _edit(src.read_text(), {**BASE_EDITS, "NX, NY": f"{n}, {n}"})
    assert SRC_LINE in text, f"{backend}: F_SRC placeholder not found"
    text = text.replace(SRC_LINE, SRC_REPL)

    wd = tmp / f"{backend}_{n}"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "participant.py").write_text(text)

    # A partner export dense enough that the participant's own interpolation of
    # it is far below the discretisation error being measured.
    ys = np.linspace(Y0, Y1, 2001)
    (wd / "imports.json").write_text(json.dumps({
        "left": {"field_name": "temperature",
                 "n_points": len(ys),
                 "coordinates": [[IFACE_X, float(y)] for y in ys],
                 "values": [310.0] * len(ys),
                 "normal_fluxes": [float(v) for v in q_partner(ys)]}}))

    r = subprocess.run([_interpreter(backend), "participant.py"], cwd=str(wd),
                       capture_output=True, text=True, timeout=900)
    ep = wd / "exports.json"
    assert ep.is_file(), (f"{backend} n={n} wrote no exports.json, "
                          f"rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    d = json.loads(ep.read_text())
    assert "normal_fluxes" in d, f"{backend}: exported no normal_fluxes"
    y = np.array([c[1] for c in d["coordinates"]], float)
    q = np.array(d["normal_fluxes"], float)
    o = np.argsort(y)
    return y[o], q[o]


BACKENDS = ["fenics", "skfem", "ngsolve", "dune"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_recovered_flux_converges_to_the_imposed_one(tmp_path, backend):
    """Order >= 1.8 on the interior, against the flux the partner sent.

    The participant applies the partner's g as +int g v ds, so its own outward
    flux is -g. Anything that merely APPROXIMATES the boundary gradient lands
    near this value on a coarse mesh and then stops improving, which is what
    the order check catches and a tolerance check would not.
    """
    if not _available(backend):
        pytest.skip(f"{backend} not available on this install")
    if not _interpreter(backend):
        pytest.skip(f"no interpreter resolved for {backend}")

    ns = (8, 16, 32)
    errs = []
    for n in ns:
        y, q = _run_once(tmp_path, backend, n)
        want = -q_partner(y)
        # Drop the two end nodes: they carry the outer reaction as well.
        inner = (y > Y0 + 1e-12) & (y < Y1 - 1e-12)
        assert inner.sum() >= 3, f"{backend}: interface too small to judge"
        errs.append(float(np.max(np.abs(q[inner] - want[inner]))))

    assert all(math.isfinite(e) for e in errs), f"{backend}: {errs}"
    orders = [math.log(errs[i - 1] / errs[i]) / math.log(2.0)
              for i in range(1, len(errs))]
    assert min(orders) >= 1.8, (
        f"{backend}: interface flux recovery converges at {orders} "
        f"(errors {errs}); the consistent reaction gives 2.0 and an L2-"
        f"projected boundary gradient gives ~0 in this norm")


def test_kratos_3d_measures_what_its_conditions_assembled(tmp_path):
    """The 3-D Kratos participant, whose Neumann side has no reaction to read.

    Kratos stores REACTION_FLUX only on FIXED dofs, so the Dirichlet branch's
    route is closed on the Neumann side. Echoing the imported array would be
    algebraically exact and useless as evidence — applied on the wrong facets
    or with the wrong sign it would read the same, and the two-sided balance
    check would still report roundoff. Summing each condition's own
    right-hand side is a MEASUREMENT of what entered the linear system.

    Measured here against q(y,z) = 2 + 3 sin(4y) cos(3z), interior nodes of the
    interface plane (the rim carries the outer reaction too):
        assembled conditions   5.16e-01, 1.77e-01, 4.77e-02  order 1.55, 1.89
        gradient averaging     1.27e+00, 8.37e-01, 3.95e-01  order 0.60, 1.08
    """
    if not _available("kratos"):
        pytest.skip("kratos not available on this install")
    import numpy as _np
    src = PART_DIR / "participant_kratos_3d.py"
    if not src.is_file():
        pytest.skip("no 3-D Kratos participant")

    def q_ex(y, z):
        return 2.0 + 3.0 * _np.sin(4.0 * y) * _np.cos(3.0 * z)

    errs = []
    for n in (4, 8, 16):
        wd = tmp_path / f"k{n}"
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "p.py").write_text(_edit(src.read_text(), {
            "SIDE": '"neumann"', "PARTNER": '"left"', "IFACE_AXIS": "0",
            "IFACE_POS": "0.5", "X0, X1": "0.5, 1.0", "Y0, Y1": "0.0, 1.0",
            "Z0, Z1": "0.0, 1.0", "NX, NY, NZ": f"{n}, {n}, {n}",
            "K": "1.5", "DIRICHLET_FACES": '("x1",)', "LIN_SOLVER": '"direct"'}))
        m = 41
        ys, zs = _np.meshgrid(_np.linspace(0, 1, m), _np.linspace(0, 1, m),
                              indexing="ij")
        ys, zs = ys.ravel(), zs.ravel()
        (wd / "imports.json").write_text(json.dumps({"left": {
            "field_name": "temperature", "n_points": len(ys),
            "coordinates": [[0.5, float(a), float(b)] for a, b in zip(ys, zs)],
            "values": [300.0] * len(ys),
            "normal_fluxes": [float(v) for v in q_ex(ys, zs)]}}))
        r = subprocess.run([sys.executable, "p.py"], cwd=str(wd),
                           capture_output=True, text=True, timeout=2400)
        ep = wd / "exports.json"
        assert ep.is_file(), (f"kratos_3d n={n} wrote no exports.json, "
                              f"rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        d = json.loads(ep.read_text())
        c = _np.array(d["coordinates"], float)
        q = _np.array(d["normal_fluxes"], float)
        yy, zz = c[:, 1], c[:, 2]
        rim = (_np.isclose(yy, 0.0) | _np.isclose(yy, 1.0) |
               _np.isclose(zz, 0.0) | _np.isclose(zz, 1.0))
        errs.append(float(_np.max(_np.abs(q[~rim] + q_ex(yy, zz)[~rim]))))

    orders = [math.log(errs[i - 1] / errs[i]) / math.log(2.0)
              for i in range(1, len(errs))]
    assert orders[-1] >= 1.5, (
        f"kratos_3d interface flux converges at {orders} (errors {errs}); "
        f"the gradient averaging this replaced gives ~1.0 and the assembled "
        f"condition measurement gives ~1.9")


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_participant_projects_the_boundary_gradient(backend):
    """The retired recovery must not come back by copy-paste.

    Cheap, and it covers the participants this host cannot execute.
    """
    hits = [p.name for p in PART_DIR.glob(f"participant_{backend}*.py")
            if "MUST NOT be used here" in p.read_text()]
    assert not hits, (
        f"{hits} still carry the retired 'the reaction formula MUST NOT be "
        f"used here' branch. It is true only when the residual is taken "
        f"against a load that already contains the interface term; against "
        f"the volume load alone those rows ARE the interface functional.")


def test_served_guidance_does_not_recommend_the_retired_recovery():
    """What we TELL agents must match what we SHIP them.

    The participants were fixed while the served FEniCSx guidance still read
    "The exported flux is an L2 projection of -K*S*grad(T)[0] onto the same CG1
    space" — a recommendation to do the thing every shipped participant had
    just stopped doing. Served text that contradicts the shipped code is the
    same defect class as a path we promise and do not have: the agent reads it
    as an instruction and spends its budget on it.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    from tools.coupling_knowledge import coupling_knowledge

    for solver in ("", "fenics", "skfem", "ngsolve", "dune", "kratos", "febio"):
        text = coupling_knowledge(solver)
        low = text.lower()
        # The phrasing that told agents to EXPORT a projection.
        for bad in ("exported flux is an l2 projection",
                    "export the l2 projection",
                    "exported traction is an l2 projection"):
            assert bad not in low, (
                f"solver={solver!r}: served guidance still recommends the "
                f"retired recovery ({bad!r}). Measured, it does not converge "
                f"in the max norm over interior interface nodes.")
