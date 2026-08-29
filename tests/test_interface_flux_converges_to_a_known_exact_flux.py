"""The interface flux recovery, measured against an EXACT flux it was not given.

WHY THIS FILE EXISTS — AND WHY THE FILE NEXT DOOR IS NOT THIS EVIDENCE.

`test_interface_flux_recovery.py` hands the NEUMANN side a flux g and asks for
it back. On that side the participant solves `A u = b_vol + M_Gamma g`, so on
every FREE interface row the residual it recovers from is

    r_i = (A u - b_vol)_i = (M_Gamma g)_i        EXACTLY, by construction,

and the export is `-r_i / w_i` with `w_i = (M_Gamma 1)_i`. That is the
consistent-to-nodal conversion of the P1 boundary mass matrix and nothing else.
Its "error" against -g has the closed form

    Q_i + g(y_i) = -(h^2 / 6) g''(y_i) + O(h^4),

so it is second order for ANY correct assembly, of ANY equation, with ANY
material — the PDE, the conductivity and the solver never enter. Measured: at
n = 8/16/32 FEniCSx, scikit-fem and NGSolve all report 1.9643385e-02,
4.9833201e-03, 1.2496293e-03 and DUNE-fem 1.9643386e-02, 4.9832892e-03,
1.2496477e-03 — and a bare 1-D P1 mass matrix in NumPy, with no FEM, no PDE and
no solver in it, reports 1.9643386e-02, 4.9833201e-03, 1.2494690e-03. Four
codes agreeing to seven significant figures there is four evaluations of one
algebraic expression, not a cross-code check; an independent review further
found the figures unchanged to ten significant figures under a 100x change in
conductivity, under a change of domain, and at nu = 0.49 where P1 triangles
lock and the displacement is substantially wrong. An order read off that
fixture is entailed by the algebra; it is not a convergence measurement. That
test is still worth having — see its own docstring — but for the sign
convention, the weight and the blocked-dof mapping, not for an order.

WHAT THIS FILE MEASURES INSTEAD. The DIRICHLET side, where the interface rows
are constrained and carry a genuine reaction, against an exact interface flux
that is known analytically and that is NOT the number handed in. The
participant is given the interface TEMPERATURE and graded on the exported
FLUX — two different quantities, so no echo can pass.

    T(x, y) = 300 + sin(3x) cos(pi y / Ly)   on [0, X1] x [0, Ly], K constant
    f = -div(K grad T) = K (9 + (pi/Ly)^2) sin(3x) cos(pi y / Ly)

chosen to fit the SHIPPED participant's boundary layout exactly, so the shipped
file is run unmodified apart from its own edit block:
  * dT/dy = 0 at y = 0 and y = Ly, which is the shipped natural top/bottom;
  * T(0, y) = 300 identically, which is the shipped constant `T_OUTER`.
The exact outward flux at the interface x = X1 (outward normal +e_x) is

    q_ex(y) = -K dT/dx|_{x=X1} = -3 K cos(3 X1) cos(pi y / Ly)

MEASURED HERE, driving data/coupling_participants/participant_fenics.py on
8/16/32/64 uniform triangle meshes, max error over the interface nodes,
against |q_ex| <= 1.0224:

    interior nodes   2.889e-01  7.243e-02  1.814e-02  4.556e-03
                     ORDER 1.996  1.998  1.993
    the two ends     1.529e+00  7.142e-01  3.415e-01  1.663e-01
                     ORDER 1.099  1.065  1.038

The recovery is second order in the max norm over interior interface nodes and
FIRST order at the two nodes where the interface meets the outer boundary. The
end nodes are reported and asserted apart throughout, here and in the shipped
participants, because that node's residual row is not this interface's flux
alone. The same measurement on a variant with Dirichlet data on the WHOLE outer
boundary gives interior 2.391e-01 / 6.040e-02 / 1.515e-02 / 3.794e-03 (order
1.985, 1.995, 1.998) and end nodes exactly order 1.000, 1.013, 1.010 — the
interior order is the robust statement, the end-node constant is not.
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

# The manufactured problem. Ly and X1 are chosen so that cos(pi y / Ly) has zero
# derivative on both y-boundaries and sin(3x) vanishes at x = 0.
K, X1, LY = 1.5, 0.6, 0.4
MESHES = (8, 16, 32, 64)


def T_exact(y):
    """The interface temperature the participant is HANDED."""
    return 300.0 + np.sin(3.0 * X1) * np.cos(np.pi * np.asarray(y, float) / LY)


def q_exact(y):
    """The interface flux the participant is GRADED ON. Analytic, and not the
    number it was given: it is handed T and must produce -K dT/dn."""
    return -3.0 * K * np.cos(3.0 * X1) * np.cos(np.pi * np.asarray(y, float) / LY)


EDITS = {
    "SIDE": '"dirichlet"',      # import T, export the recovered flux
    "PARTNER": '"right"',
    "X0, X1": f"0.0, {X1}",
    "Y0, Y1": f"0.0, {LY}",
    "IFACE_X": str(X1),
    "K": str(K),
    "T_OUTER": "300.0",         # == T_exact on the outer edge x = 0
    "T_INIT": "300.0",
    "Q_INIT": "0.0",
}
# The manufactured source, written the way the shipped file's own docstring
# tells an agent to write one.
SRC_LINE = "return np.zeros_like(x)"
SRC_REPL = (f"return {K} * (9.0 + (np.pi / {LY}) ** 2) "
            f"* np.sin(3.0 * x) * np.cos(np.pi * y / {LY})")


def _edit(text: str, edits: dict) -> str:
    for k, v in edits.items():
        pat = re.compile(rf"^({re.escape(k)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        assert pat.search(text), f"edit key {k!r} not found in the shipped script"
        text = pat.sub(lambda m: m.group(1) + v + m.group(3), text, count=1)
    return text


def _fenics_python():
    try:
        from backends.fenics.backend import _find_fenics_python
    except Exception:
        return None
    p = _find_fenics_python()
    return str(p) if p else None


def _fenics_available() -> bool:
    try:
        from core.registry import get_backend, load_all_backends
    except Exception:
        return False
    load_all_backends()
    b = get_backend("fenics")
    return bool(b) and b.check_availability()[0].value == "available"


def _run_once(tmp: Path, n: int):
    """One Dirichlet-side solve at resolution n; returns (y, q_exported)."""
    src = PART_DIR / "participant_fenics.py"
    if not src.is_file():
        pytest.skip("no shipped FEniCSx participant")
    text = _edit(src.read_text(), {**EDITS, "NX, NY": f"{n}, {n}"})
    assert SRC_LINE in text, "F_SRC placeholder not found in the shipped script"
    text = text.replace(SRC_LINE, SRC_REPL)

    wd = tmp / f"n{n}"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "participant.py").write_text(text)

    # The partner's export: the EXACT interface temperature, sampled densely
    # enough that the participant's own np.interp of it is orders below the
    # discretisation error being measured (~3e-8 against ~5e-3 at n = 64).
    ys = np.linspace(0.0, LY, 4001)
    (wd / "imports.json").write_text(json.dumps({"right": {
        "field_name": "temperature",
        "n_points": len(ys),
        "coordinates": [[X1, float(y)] for y in ys],
        "values": [float(v) for v in T_exact(ys)]}}))

    r = subprocess.run([_fenics_python(), "participant.py"], cwd=str(wd),
                       capture_output=True, text=True, timeout=900)
    ep = wd / "exports.json"
    assert ep.is_file(), (f"n={n} wrote no exports.json, rc={r.returncode}\n"
                          f"{r.stdout[-800:]}\n{r.stderr[-800:]}")
    d = json.loads(ep.read_text())
    assert "normal_fluxes" in d, "participant exported no normal_fluxes"
    y = np.array([c[1] for c in d["coordinates"]], float)
    q = np.array(d["normal_fluxes"], float)
    o = np.argsort(y)
    return y[o], q[o]


def test_recovered_flux_converges_to_the_exact_flux_at_second_order(tmp_path):
    """Order >= 1.8 on interior interface nodes, against an ANALYTIC flux.

    This is the primary evidence that the shipped reaction recovery is second
    order. Unlike the echo fixture next door, the graded quantity here is not
    algebraically determined by the data handed in: the participant receives a
    temperature and is scored on a flux, so the PDE, the conductivity and the
    discretisation all sit between the input and the number being measured.
    """
    if not _fenics_available():
        pytest.skip("fenics not available on this install")
    if not _fenics_python():
        pytest.skip("no FEniCSx interpreter resolved")

    interior, ends = [], []
    for n in MESHES:
        y, q = _run_once(tmp_path, n)
        ex = q_exact(y)
        inner = (y > 1e-12) & (y < LY - 1e-12)
        assert inner.sum() >= 3, f"n={n}: interface too small to judge"
        interior.append(float(np.max(np.abs(q[inner] - ex[inner]))))
        ends.append(float(np.max(np.abs(q[~inner] - ex[~inner]))))

    assert all(math.isfinite(e) for e in interior), interior
    assert min(interior) > 0.0, (
        f"interior error is exactly zero {interior} — the recovery cannot be "
        f"exact against this flux, so the comparison is not being made")

    orders = [math.log(interior[i - 1] / interior[i]) / math.log(2.0)
              for i in range(1, len(interior))]
    end_orders = [math.log(ends[i - 1] / ends[i]) / math.log(2.0)
                  for i in range(1, len(ends))]

    assert min(orders) >= 1.8, (
        f"interface flux recovery converges at {orders} against the EXACT "
        f"flux (interior max errors {interior}); measured 1.996/1.998/1.993 "
        f"when this was written. End-node errors {ends}, orders {end_orders} "
        f"(first order by construction, and excluded from the assertion).")


def test_the_echo_fixture_is_an_identity_and_is_labelled_as_one(tmp_path):
    """The sibling test must not sell itself as a convergence measurement.

    Cheap and interpreter-free. It exists because the claim it guards has
    already been made wrongly once, in this repo, in eleven places: an order
    read off a fixture where `r = A u - b_vol` is identically the applied
    interface load is a property of the P1 boundary mass matrix, and the same
    2.00 comes out of NumPy with no PDE in it at all.
    """
    sib = Path(__file__).with_name("test_interface_flux_recovery.py")
    if not sib.is_file():
        pytest.skip("sibling echo test not present")
    doc = sib.read_text()
    head = doc[:doc.find('"""', 3)] if doc.startswith('"""') else doc[:4000]
    low = head.lower()
    assert "identity" in low or "identical" in low, (
        "the echo test's docstring no longer says its residual rows ARE the "
        "applied interface load; without that it reads as a convergence study")
    assert __name__.rsplit(".", 1)[-1] in doc or "known_exact_flux" in doc, (
        "the echo test's docstring must point at this file as the place the "
        "convergence order is actually measured")
