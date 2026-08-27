"""Kratos participant for subdomain B (NEUMANN side) in coupled heat conduction.

Physics: -div(k grad u) = f on [0.625, 1.5] x [0, 1] with k=1000
Interface at x = 0.625 (left edge of this subdomain)
Outer BC: u = 0 on right, top, bottom edges
Role: NEUMANN - imports flux from partner, exports temperature

IMPORTANT: Interface corners (y=0 and y=1) are EXCLUDED from exchange.
They belong to the outer boundary and are constrained by u=0.

FLUX SIGN CONVENTION:
- Partner A exports q_A = -(k_A grad u_A) . n_A where n_A = +e_x
- We apply this value directly as FACE_HEAT_FLUX (no sign change)
- Kratos FluxCondition2D2N enforces: K grad T . n = FACE_HEAT_FLUX
- Our outward normal is n_own = -e_x, so this gives the correct physics
"""
import json
import sys
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── PROBLEM PARAMETERS ───────────────────────────────────────────────────────
PARTNER   = "A"           # Name of the partner participant
X0, X1    = 0.625, 1.5    # Subdomain B: x in [0.625, 1.5]
Y0, Y1    = 0.0, 1.0      # y in [0, 1]
IFACE_X   = 0.625         # Interface at x = 5/8 (left edge of this subdomain)
K         = 1000.0        # Thermal conductivity in subdomain B

# Source term f(x,y) in subdomain B (exactly as given in problem)
def F_SRC(x, y):
    """Source term for subdomain B."""
    return (-9*x**3*y/2500000 + x**3/2500000 - 64901*x**2*y/10000000 
            - 25033*x**2/30000000 - 9*x*y**3/2500000 + 3*x*y**2/2500000 
            - 422790921*x*y/160000000 - 21609971*x/32000000 
            - 64901*y**3/30000000 - 25033*y**2/30000000 
            + 1274009971*y/320000000 + 12989997/12800000)

T_OUTER   = 0.0           # Dirichlet value on outer boundary (u=0 everywhere)
NX, NY    = 8, 8          # Will be set per mesh level via environment
Q_INIT    = 0.0           # Iteration-1 fallback interface flux density
# ─────────────────────────────────────────────────────────────────────────────

# Parse NX, NY from environment
import os
NX = int(os.environ.get('NX', '8'))
NY = int(os.environ.get('NY', '8'))

ON_MAX_X = abs(IFACE_X - X1) < abs(IFACE_X - X0)  # False: interface is left edge (x-min)
OUTER_X = X0 if ON_MAX_X else X1  # OUTER_X = 1.5 (right edge)
S = 1.0 if ON_MAX_X else -1.0  # Outward normal at interface = -e_x (pointing left)
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map partner's samples onto THIS participant's interface points."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


def build_model():
    """Structured triangulation of [X0,X1] x [Y0,Y1]; returns (mp, nid)."""
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
              KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    for j in range(NY + 1):
        for i in range(NX + 1):
            mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / NX,
                             Y0 + (Y1 - Y0) * j / NY, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props)
            eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props)
            eid += 1
    return mp, nid


def main():
    if min(abs(IFACE_X - X0), abs(IFACE_X - X1)) > TOL:
        sys.exit(f"IFACE_X={IFACE_X} is not an x-boundary of this subdomain [{X0},{X1}]")
    if NX < 1 or NY < 1:
        sys.exit(f"NX,NY = {NX},{NY}: need at least one element in each direction")

    mp, nid = build_model()
    
    # Column indices
    i_if = NX if ON_MAX_X else 0  # Interface column (i_if = 0 since interface is left)
    i_out = 0 if ON_MAX_X else NX  # Outer Dirichlet column (i_out = NX, right edge)
    
    # Verify geometry matches node numbering
    for idx, want, what in ((i_if, IFACE_X, "interface"), (i_out, OUTER_X, "outer")):
        got = mp.Nodes[nid[(idx, 0)]].X
        if abs(got - want) > TOL:
            sys.exit(f"internal: {what} column at x={got}, not x={want}")
    
    # Get all interface y-coordinates
    y_if_all = np.array([Y0 + (Y1 - Y0) * j / NY for j in range(NY + 1)])
    
    # EXCLUDE CORNER NODES from interface exchange (y=0 and y=1)
    corner_mask = (np.abs(y_if_all - Y0) < TOL) | (np.abs(y_if_all - Y1) < TOL)
    interior_mask = ~corner_mask
    y_if = y_if_all[interior_mask]
    interior_indices = np.where(interior_mask)[0]
    
    # Import flux from partner (only interior points)
    q_in = sample(read_imports(), "normal_fluxes", Q_INIT, y_if)

    # Set material and source
    for n in mp.Nodes:
        n.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        n.SetSolutionStepValue(KM.HEAT_FLUX, float(F_SRC(n.X, n.Y)))
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

    # Outer Dirichlet boundary (right edge at x=1.5)
    for j in range(NY + 1):
        n = mp.Nodes[nid[(i_out, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
        n.Fix(KM.TEMPERATURE)

    # Top and bottom edges are also outer Dirichlet (u=0)
    for i in range(NX + 1):
        for j_edge, y_edge in [(0, Y0), (NY, Y1)]:
            n = mp.Nodes[nid[(i, j_edge)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
            n.Fix(KM.TEMPERATURE)

    # Interface: apply partner's flux as Neumann condition (only interior edges!)
    # The partner exports q = -(k grad u) . n_partner where n_partner = +e_x
    # We apply this value directly as FACE_HEAT_FLUX (no sign change)
    # Create FluxCondition only for interior edges (exclude corners)
    props = mp.GetProperties()[1]
    cond_id = 1
    for j in interior_indices[:-1]:  # Edges between interior nodes
        mp.Nodes[nid[(i_if, j)]].SetSolutionStepValue(
            KM.FACE_HEAT_FLUX, float(q_in[j - sum(interior_mask[:j])]))
        mp.CreateNewCondition("FluxCondition2D2N", cond_id,
                              [nid[(i_if, j)], nid[(i_if, j + 1)]], props)
        cond_id += 1

    # Add DOFs with reaction variable
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()

    # Extract interface temperatures (only interior points)
    T = np.array([mp.Nodes[nid[(i_if, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                  for j in interior_indices])

    # Compute our own outward normal flux (for export, though Dirichlet partner doesn't use it)
    # q = -(K grad T) . n_own where n_own = -e_x, so q = +K * dT/dx
    num = np.zeros(len(mp.Nodes) + 1)
    den = np.zeros(len(mp.Nodes) + 1)
    for el in mp.Elements:
        nds = el.GetNodes()
        x = [n.X for n in nds]
        y = [n.Y for n in nds]
        t = [n.GetSolutionStepValue(KM.TEMPERATURE) for n in nds]
        det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
        if abs(det) < 1e-30:
            continue
        dTdx = ((y[1]-y[2])*t[0] + (y[2]-y[0])*t[1] + (y[0]-y[1])*t[2]) / det
        area = 0.5 * abs(det)
        qe = -K * S * dTdx  # S = -1, so qe = +K * dTdx
        for n in nds:
            num[n.Id] += area * qe
            den[n.Id] += area
    ids_if = [nid[(i_if, j)] for j in interior_indices]
    Q = np.array([num[i] / den[i] if den[i] > 0 else 0.0 for i in ids_if])

    # Conservation self-check
    hy = (Y1 - Y0) / NY
    load_iface = float(hy * (0.5 * q_in[0] + q_in[1:-1].sum() + 0.5 * q_in[-1]))
    load_vol = 0.0
    for el in mp.Elements:
        nds = el.GetNodes()
        x = [n.X for n in nds]
        y = [n.Y for n in nds]
        det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
        load_vol += (0.5 * abs(det)) * sum(
            n.GetSolutionStepValue(KM.HEAT_FLUX) for n in nds) / 3.0
    react = sum(mp.Nodes[nid[(i_out, j)]].GetSolutionStepValue(KM.REACTION_FLUX)
                for j in range(NY + 1))
    imb_abs = abs(react + load_vol + load_iface)
    scale = max(abs(react), abs(load_vol), abs(load_iface))
    if scale <= 1e-10 * K * max(1.0, abs(T_OUTER)) * (Y1 - Y0):
        bal = f"balance trivial (|imbalance|={imb_abs:.3e})"
    else:
        bal = f"balance |imb|={imb_abs:.3e} rel={imb_abs/scale:.3e}"

    print(f"[kratos B neumann] interface n={len(T)} q_applied=[{q_in.min():.6g},{q_in.max():.6g}] "
          f"T=[{T.min():.6g},{T.max():.6g}] {bal}")

    # Write exports.json LAST - only interior interface points
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(T)),
        "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))

    # Write run log with NDOF
    ndof = len(mp.Nodes)
    with open("run.log", "w") as logfile:
        logfile.write(f"NDOF = {ndof}\n")
        logfile.write(f"Elements = {len(mp.Elements)}\n")
        logfile.write(f"Interface points (interior only) = {len(T)}\n")

    return mp, nid


if __name__ == "__main__":
    main()
