"""Kratos participant for subdomain B (NEUMANN side) - coupled heat conduction.

Subdomain B: (0.625, 1.5) x (0, 1), k=200
Interface at x = 0.625 (left boundary of B)
Outer BCs: u=0 on x=1.5, y=0, y=1
Coupling: imports flux from partner, exports temperature
"""
import json
import sys
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem parameters for subdomain B
PARTNER   = "A"           # partner name
X0, X1    = 0.625, 1.5    # subdomain B x-extent
Y0, Y1    = 0.0, 1.0      # subdomain B y-extent
IFACE_X   = 0.625         # interface at left boundary
K         = 200.0         # thermal conductivity in B


def F_SRC(x, y):
    """Source term in subdomain B.
    
    f(x,y) = -3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
             - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
             - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/640000
    """
    result = (-3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
              - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
              - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/640000)
    return result


T_OUTER   = 0.0           # Dirichlet value on outer boundary (x=1.5, y=0, y=1)
NX, NY    = 20, 16        # will be updated per level
Q_INIT    = 0.0           # iteration-1 fallback flux


def source(x, y):
    """Volumetric source at nodes."""
    return F_SRC(x, y)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


def build_model():
    """Build structured mesh of subdomain B."""
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
    global NX, NY
    
    # Read mesh level
    level = 1
    if Path("level.json").is_file():
        level = json.loads(Path("level.json").read_text()).get("level", 1)
    
    # Set NX, NY based on mesh level
    divisions = [8, 16, 32][level - 1]
    # For subdomain B: width = 0.875, so NX = 0.875 * divisions
    NX = int(round(0.875 * divisions))
    NY = int(divisions)  # height is 1.0

    ON_MAX_X = False  # interface is at x=X0 (left), not x=X1
    OUTER_X = X1      # outer Dirichlet at right boundary
    S = -1.0          # outward normal at interface = -e_x (pointing left)
    TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

    mp, nid = build_model()
    i_if = 0          # interface column index (left)
    i_out = NX        # outer Dirichlet column index (right)
    y_if = np.array([Y0 + (Y1 - Y0) * j / NY for j in range(NY + 1)])

    q_in = sample(read_imports(), "normal_fluxes", Q_INIT, y_if)

    # Set material and source
    for n in mp.Nodes:
        n.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(n.X, n.Y)))
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

    # Outer Dirichlet boundary (right edge, bottom, top)
    for j in range(NY + 1):
        n = mp.Nodes[nid[(i_out, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
        n.Fix(KM.TEMPERATURE)
    # Bottom edge (exclude corners)
    for i in range(1, NX + 1):
        n = mp.Nodes[nid[(i, 0)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
        n.Fix(KM.TEMPERATURE)
    # Top edge (exclude corners)
    for i in range(1, NX + 1):
        n = mp.Nodes[nid[(i, NY)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
        n.Fix(KM.TEMPERATURE)

    # Interface: apply imported flux as Neumann condition
    for j in range(NY + 1):
        mp.Nodes[nid[(i_if, j)]].SetSolutionStepValue(
            KM.FACE_HEAT_FLUX, float(q_in[j]))
    props = mp.GetProperties()[1]
    for j in range(NY):
        mp.CreateNewCondition("FluxCondition2D2N", j + 1,
                              [nid[(i_if, j)], nid[(i_if, j + 1)]], props)

    # Add DOFs with reaction variable
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()

    # Extract interface temperature
    T = np.array([mp.Nodes[nid[(i_if, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                  for j in range(NY + 1)])

    # Compute outward normal flux (using L2 projection of gradient)
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
        qe = -K * S * dTdx  # Outward normal flux
        for n in nds:
            num[n.Id] += area * qe
            den[n.Id] += area
    ids_if = [nid[(i_if, j)] for j in range(NY + 1)]
    Q = np.array([num[i] / den[i] if den[i] > 0 else 0.0 for i in ids_if])

    print(f"[kratos B] interface n={len(T)} q_applied=[{q_in.min():.6g},{q_in.max():.6g}] "
          f"T=[{T.min():.6g},{T.max():.6g}]")

    # Write exports.json
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(T)),
        "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))

    # Write run log with NDOF
    ndof = len(list(mp.Nodes))
    Path("run.log").write_text(f"NDOF = {ndof}\n")

    return mp, nid


if __name__ == "__main__":
    main()
