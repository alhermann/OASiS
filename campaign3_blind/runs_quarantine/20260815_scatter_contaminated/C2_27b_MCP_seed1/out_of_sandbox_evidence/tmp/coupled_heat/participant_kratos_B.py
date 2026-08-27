"""Kratos Multiphysics participant for OASiS couple driver - Subdomain B (NEUMANN side).

Subdomain B: (0.625, 1.5) x (0, 1), k=200
Interface at x=0.625 (left boundary of B)
This is the NEUMANN side: imports flux from partner, exports temperature.

Flux convention: Each participant exports its OUTWARD normal flux.
- For subdomain A, outward at interface is +e_x, so qn_A = -K_A * du/dx
- For subdomain B, outward at interface is -e_x, so qn_B = K_B * du/dx
At convergence: qn_A should equal qn_B (continuity of flux)
"""
import json
from pathlib import Path
import sys
import os

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem parameters for Subdomain B
PARTNER = "fourc_A"
X0, X1 = 0.625, 1.5  # Subdomain B x-extent
Y0, Y1 = 0.0, 1.0    # y-extent
IFACE_X = 0.625      # Interface at left boundary
K = 200.0            # Thermal conductivity in B
T_OUTER = 0.0        # Dirichlet value on outer boundary (x=1.5)
Q_INIT = 0.0         # Iteration-1 fallback interface flux

def get_mesh_params(level):
    """Get mesh divisions for each level."""
    base_divisions = [8, 16, 32][level - 1]
    nx = int(round(0.875 * base_divisions))
    ny = int(round(1.0 * base_divisions))
    return nx, ny

if len(sys.argv) > 1:
    level = int(sys.argv[1])
else:
    level = int(os.environ.get("LEVEL", 1))

nx, ny = get_mesh_params(level)

def imported_flux(y_coords):
    """Import flux values from partner."""
    p = Path("imports.json")
    if not p.is_file():
        print(f"[Kratos B] No imports.json found, using Q_INIT={Q_INIT}", file=sys.stderr)
        return np.full_like(y_coords, float(Q_INIT))
    
    try:
        imp = json.loads(p.read_text())
    except Exception as e:
        print(f"[Kratos B] Error reading imports.json: {e}", file=sys.stderr)
        return np.full_like(y_coords, float(Q_INIT))
    
    if PARTNER not in imp:
        print(f"[Kratos B] Partner {PARTNER} not in imports", file=sys.stderr)
        return np.full_like(y_coords, float(Q_INIT))
    
    d = imp[PARTNER]
    if "normal_fluxes" not in d:
        print(f"[Kratos B] No normal_fluxes in imports", file=sys.stderr)
        return np.full_like(y_coords, float(Q_INIT))
    
    src_y = np.asarray(d["coordinates"], float)[:, 1]
    src_v = np.asarray(d["normal_fluxes"], float).ravel()
    o = np.argsort(src_y)
    result = np.interp(y_coords, src_y[o], src_v[o])
    print(f"[Kratos B] Imported flux: min={result.min():.6f}, max={result.max():.6f}", file=sys.stderr)
    return result

def source_term(x, y):
    """Source term in subdomain B."""
    return (-3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
            - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
            - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/64000)

def solve(q_if_in):
    """Solve the heat equation with Neumann BC at interface."""
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
    
    # Create nodes
    node_list = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = X0 + (X1 - X0) * i / nx
            y = Y0 + (Y1 - Y0) * j / ny
            node = mp.CreateNewNode(cnt, x, y, 0.0)
            nid[(i, j)] = cnt
            node_list.append((cnt, i, j, x, y))
            cnt += 1
    
    n_nodes = cnt - 1
    
    # Set material properties and source term
    for (node_id, i, j, x, y) in node_list:
        node = mp.Nodes[node_id]
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        node.SetSolutionStepValue(KM.HEAT_FLUX, source_term(x, y))
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
    
    # Create triangular elements
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1
    
    # Outer Dirichlet boundary at x=X1 (right side) - u=0
    for j in range(ny + 1):
        n = mp.Nodes[nid[(nx, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    
    # INTERFACE Neumann condition at x=X0 (left side)
    # q_if_in is the flux exported by 4C, which is outward from A (+x direction)
    # For B, this flux ENTERS from the left
    # In Kratos, FACE_HEAT_FLUX is the prescribed OUTWARD flux
    # So if q_if_in enters B, we set FACE_HEAT_FLUX = -q_if_in
    for j in range(ny + 1):
        n = mp.Nodes[nid[(0, j)]]
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, -q_if_in[j])
        # Do NOT fix temperature - it's a natural BC
    
    # Add DOFs
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    
    return mp, nid, n_nodes

def main():
    global nx, ny
    y_if = np.array([Y0 + (Y1 - Y0) * j / ny for j in range(ny + 1)])
    q_in = imported_flux(y_if)
    
    mp, nid, n_nodes = solve(q_in)
    
    # Extract interface temperature
    T_if = np.array([mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                     for j in range(ny + 1)])
    
    # Compute outward normal flux from B's perspective
    # Outward normal at interface for B is -e_x
    # qn_B = -(K grad u) . n_out = -(K grad u) . (-e_x) = K * du/dx
    hx = (X1 - X0) / nx
    
    q_out = np.zeros(ny + 1)
    for j in range(ny + 1):
        T0 = mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
        T1 = mp.Nodes[nid[(1, j)]].GetSolutionStepValue(KM.TEMPERATURE)
        du_dx = (T1 - T0) / hx
        q_out[j] = K * du_dx  # Outward flux = K * du/dx
    
    print(f"[Kratos B] Computed outward flux: min={q_out.min():.6f}, max={q_out.max():.6f}", file=sys.stderr)
    
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
        "values": [float(v) for v in T_if],
        "normal_fluxes": [float(v) for v in q_out],
    }, indent=2))
    
    print(f"[Kratos B NEUMANN] n={len(T_if)} T=[{T_if.min():.6g},{T_if.max():.6g}] q_out=[{q_out.min():.6g},{q_out.max():.6g}]")
    
    with open("run.log", "w") as f:
        f.write(f"NDOF = {n_nodes}\n")
        f.write(f"Level = {level}\n")
        f.write(f"Elements = {2 * nx * ny}\n")

if __name__ == "__main__":
    main()
