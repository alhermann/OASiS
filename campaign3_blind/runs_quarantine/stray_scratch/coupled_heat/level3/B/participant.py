"""Kratos Multiphysics participant for subdomain B (NEUMANN side) - coupled heat conduction.

Subdomain B: (0.625, 1.5) x (0, 1), k = 1000
Interface at x = 0.625 (left edge of this subdomain)
Outer BCs: u = 0 on right, top, bottom; interface gets Neumann from partner
"""
import json
from pathlib import Path
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem parameters for subdomain B
PARTNER   = "A"           # Partner name in couple()
X0, X1    = 0.625, 1.5    # Subdomain B x-range
Y0, Y1    = 0.0, 1.0      # y-range
IFACE_X   = 0.625         # Interface location (x = 5/8, left edge of B)
K         = 1000.0        # Thermal conductivity in B
T_OUTER   = 0.0           # Dirichlet value on outer boundaries
T_INIT    = 0.0           # Iteration-1 fallback temperature
Q_INIT    = 0.0           # Iteration-1 fallback flux

# Read resolution from environment
import os
RESOLUTION = int(os.environ.get('RESOLUTION', 8))
NX = max(1, int((X1 - X0) * RESOLUTION))  # Elements in x for subdomain B
NY = max(1, int((Y1 - Y0) * RESOLUTION))  # Elements in y


def source_B(x, y):
    """Source term f(x,y) in subdomain B."""
    return (-9*x**3*y/2500000 + x**3/2500000 - 64901*x**2*y/10000000 - 25033*x**2/30000000 
            - 9*x*y**3/2500000 + 3*x*y**2/2500000 - 422790921*x*y/160000000 - 21609971*x/32000000 
            - 64901*y**3/30000000 - 25033*y**2/30000000 + 1274009971*y/320000000 + 12989997/12800000)


def imported_flux(y_coords):
    """Import flux from partner and interpolate to our interface nodes."""
    p = Path("imports.json")
    imp = json.loads(p.read_text() or "{}") if p.is_file() else {}
    if PARTNER not in imp or "normal_fluxes" not in imp[PARTNER]:
        return np.full_like(y_coords, float(Q_INIT))
    d = imp[PARTNER]
    src_y = np.asarray(d["coordinates"], float)[:, 1]
    src_q = np.asarray(d["normal_fluxes"], float).ravel()
    o = np.argsort(src_y)
    return np.interp(y_coords, src_y[o], src_q[o])


def solve(q_if_in):
    """Solve the heat equation with Neumann BC on interface."""
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    
    # Add required variables before creating nodes
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, 
              KM.FACE_HEAT_FLUX, KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)
    
    props = mp.CreateNewProperties(1)
    
    nid, cnt = {}, 1
    H = Y1 - Y0
    W = X1 - X0
    for j in range(NY + 1):
        for i in range(NX + 1):
            x = X0 + W * i / NX
            y = Y0 + H * j / NY
            mp.CreateNewNode(cnt, x, y, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    
    eid = 1
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1
    
    # Set material properties and source term
    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        x = node.X
        y = node.Y
        node.SetSolutionStepValue(KM.HEAT_FLUX, source_B(x, y))
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
    
    # Outer boundary conditions (u = 0)
    # Right boundary (x = X1 = 1.5)
    for j in range(NY + 1):
        n = mp.Nodes[nid[(NX, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    
    # Top boundary (y = Y1 = 1.0) - exclude corners already fixed
    for i in range(NX):  # Exclude right corner
        n = mp.Nodes[nid[(i, NY)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    
    # Bottom boundary (y = Y0 = 0.0) - exclude corners
    for i in range(NX):  # Exclude right corner
        n = mp.Nodes[nid[(i, 0)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    
    # INTERFACE: Neumann condition from partner
    for j in range(NY + 1):
        n = mp.Nodes[nid[(0, j)]]  # Left edge (interface)
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(q_if_in[j]))
    
    # Add DOFs with reaction variable for flux recovery
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    
    return mp, nid


def main():
    H = Y1 - Y0
    y_if = np.array([Y0 + H * j / NY for j in range(NY + 1)])
    
    # Import flux from partner
    q_in = imported_flux(y_if)
    
    mp, nid = solve(q_in)
    
    # Extract interface temperatures
    T_if = np.array([mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                     for j in range(NY + 1)])
    
    # Compute outward normal flux using gradient projection
    dx = (X1 - X0) / NX
    q_out = np.zeros(NY + 1)
    for j in range(NY + 1):
        if NX >= 1:
            T_left = mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
            T_right = mp.Nodes[nid[(1, j)]].GetSolutionStepValue(KM.TEMPERATURE)
            dTdx = (T_right - T_left) / dx
            q_out[j] = K * dTdx  # Outward flux (normal points -x)
    
    # Write exports
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
        "values": [float(t) for t in T_if],
        "normal_fluxes": [float(q) for q in q_out],
    }, indent=2))
    
    # Write solution data as simple text file for post-processing
    # Format: x y temperature
    with open("solution.dat", 'w') as f:
        for node in mp.Nodes:
            t = node.GetSolutionStepValue(KM.TEMPERATURE)
            f.write(f"{node.X} {node.Y} {t}\n")
    
    # Count degrees of freedom
    ndof = sum(1 for n in mp.Nodes if not n.IsFixed(KM.TEMPERATURE))
    
    # Write run log
    with open("run.log", "w") as logfile:
        logfile.write(f"NDOF = {ndof}\n")
        logfile.write(f"Total nodes: {len(list(mp.Nodes))}\n")
        logfile.write(f"Interface points: {len(T_if)}\n")
    
    print(f"Kratos B (Neumann): ndof={ndof}, T_interface_mean={T_if.mean():.6f}, q_out_mean={q_out.mean():.6f}")


if __name__ == "__main__":
    main()
