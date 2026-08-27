#!/usr/bin/env python3
"""
Kratos participant for subdomain A (DIRICHLET side) - 3D heat conduction
Interface at x = 0.625, subdomain A is [0, 0.625] x [0, 1] x [0, 1]
"""
import json
import sys
from pathlib import Path
import numpy as np

# Import Kratos
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem definition
PARTNER = "B"
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
Z0, Z1 = 0.0, 1.0
K = 1.0
IFACE_X = 0.625
T_INIT = 0.0

# Mesh resolution (will be read from file)
NX = NY = NZ = 8

def source_A(x, y, z):
    """Source term for subdomain A"""
    return (-15*x**3*y**3/2 + 35*x**3*y**2/2 - 45*x**3*y*z**2/2 + 45*x**3*y*z/2 
            - 10*x**3*y + 35*x**3*z**2/2 - 35*x**3*z/2 + 3775*x**2*y**3/192 
            - 305*x**2*y**2/64 + 3775*x**2*y*z**2/64 - 3775*x**2*y*z/64 
            - 715*x**2*y/48 - 305*x**2*z**2/64 + 305*x**2*z/64 
            - 45*x*y**3*z**2/2 + 45*x*y**3*z/2 - 45*x*y**3/4 
            + 105*x*y**2*z**2/2 - 105*x*y**2*z/2 - 135*x*y**2/16 
            - 255*x*y*z**2/4 + 255*x*y*z/4 + 315*x*y/16 
            - 135*x*z**2/16 + 135*x*z/16 + 3775*y**3*z**2/192 
            - 3775*y**3*z/192 - 305*y**2*z**2/64 + 305*y**2*z/64 
            - 715*y*z**2/48 + 715*y*z/48)

def read_imports():
    """Read imported temperature from partner B"""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER)
    except:
        return None

def interpolate_2d(imp, y_coords, z_coords):
    """Interpolate imported values to (y, z) grid"""
    if not imp or "coordinates" not in imp:
        return np.full(len(y_coords) * len(z_coords), T_INIT)
    
    coords = np.array(imp["coordinates"])
    values = np.array(imp["values"])
    
    result = []
    for y, z in zip(y_coords, z_coords):
        # Simple nearest-neighbor interpolation
        dists = np.sum((coords[:, 1:] - [y, z])**2, axis=1)
        result.append(values[np.argmin(dists)])
    return np.array(result)

def main():
    global NX, NY, NZ
    
    # Read mesh resolution
    try:
        with open("mesh_resolution.txt") as f:
            NX, NY, NZ = map(int, f.read().strip().split())
    except:
        pass
    
    # Create model
    model = KM.Model()
    mp = model.CreateModelPart("domain")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    
    # Add variables
    for v in [KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX, KM.REACTION_FLUX]:
        mp.AddNodalSolutionStepVariable(v)
    
    mp.SetBufferSize(1)
    props = mp.CreateNewProperties(1)
    
    # Create nodes
    nid = {}
    cnt = 1
    for k in range(NZ + 1):
        for j in range(NY + 1):
            for i in range(NX + 1):
                x = X0 + (X1 - X0) * i / NX
                y = Y0 + (Y1 - Y0) * j / NY
                z = Z0 + (Z1 - Z0) * k / NZ
                mp.CreateNewNode(cnt, x, y, z)
                nid[(i, j, k)] = cnt
                cnt += 1
    
    # Set material and source
    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        node.SetSolutionStepValue(KM.HEAT_FLUX, source_A(node.X, node.Y, node.Z))
    
    # Create hexahedral elements
    eid = 1
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                nodes = [nid[(i,j,k)], nid[(i+1,j,k)], nid[(i+1,j+1,k)], nid[(i,j+1,k)],
                        nid[(i,j,k+1)], nid[(i+1,j,k+1)], nid[(i+1,j+1,k+1)], nid[(i,j+1,k+1)]]
                mp.CreateNewElement("LaplacianElement3D8N", eid, nodes, props)
                eid += 1
    
    # Read imported interface temperature
    imp = read_imports()
    
    # Interface grid (interior points only, not on outer boundary)
    # Match the probe point definition: interior of interface
    ny_int, nz_int = NY - 1, NZ - 1
    y_int = [Y0 + (j + 0.5) * (Y1 - Y0) / ny_int for j in range(ny_int)]
    z_int = [Z0 + (k + 0.5) * (Z1 - Z0) / nz_int for k in range(nz_int)]
    
    y_flat, z_flat = [], []
    for y in y_int:
        for z in z_int:
            y_flat.append(y)
            z_flat.append(z)
    
    T_imported = interpolate_2d(imp, y_flat, z_flat)
    
    # Apply outer boundary conditions: u = 0
    # All faces except interface
    for j in range(NY + 1):
        for k in range(NZ + 1):
            # Left face (x = 0)
            mp.Nodes[nid[(0, j, k)]].SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            mp.Nodes[nid[(0, j, k)]].Fix(KM.TEMPERATURE)
    
    for i in range(NX + 1):
        for k in range(NZ + 1):
            # Top (y = 1) and bottom (y = 0)
            mp.Nodes[nid[(i, NY, k)]].SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            mp.Nodes[nid[(i, NY, k)]].Fix(KM.TEMPERATURE)
            mp.Nodes[nid[(i, 0, k)]].SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            mp.Nodes[nid[(i, 0, k)]].Fix(KM.TEMPERATURE)
        for j in range(NY + 1):
            # Front (z = 1) and back (z = 0)
            mp.Nodes[nid[(i, j, NZ)]].SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            mp.Nodes[nid[(i, j, NZ)]].Fix(KM.TEMPERATURE)
            mp.Nodes[nid[(i, j, 0)]].SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            mp.Nodes[nid[(i, j, 0)]].Fix(KM.TEMPERATURE)
    
    # Interface: Dirichlet from imported data
    # Map imported values to interface nodes (interior only)
    t_idx = 0
    for j in range(1, NY):  # exclude y=0 and y=1
        for k in range(1, NZ):  # exclude z=0 and z=1
            node = mp.Nodes[nid[(NX, j, k)]]
            node.SetSolutionStepValue(KM.TEMPERATURE, float(T_imported[t_idx]))
            node.Fix(KM.TEMPERATURE)
            t_idx += 1
    
    # Add DOFs
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    
    # Solve
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    
    # Extract interface data for export
    iface_coords, iface_values, iface_fluxes = [], [], []
    
    for j in range(1, NY):
        for k in range(1, NZ):
            node = mp.Nodes[nid[(NX, j, k)]]
            y = Y0 + j * (Y1 - Y0) / NY
            z = Z0 + k * (Z1 - Z0) / NZ
            
            iface_coords.append([IFACE_X, y, z])
            iface_values.append(node.GetSolutionStepValue(KM.TEMPERATURE))
            
            # Outward normal flux: q = -k * dT/dn
            # For subdomain A, outward normal at interface is +x
            reaction = node.GetSolutionStepValue(KM.REACTION_FLUX)
            dy = (Y1 - Y0) / NY
            dz = (Z1 - Z0) / NZ
            weight = dy * dz
            flux = -reaction / weight
            iface_fluxes.append(flux)
    
    # Write exports
    exports = {
        "field_name": "temperature",
        "n_points": len(iface_coords),
        "coordinates": iface_coords,
        "values": iface_values,
        "normal_fluxes": iface_fluxes
    }
    Path("exports.json").write_text(json.dumps(exports))
    
    # Write log
    with open("run.log", "w") as f:
        f.write(f"NDOF = {len(list(mp.Nodes))}\n")
    
    print(f"Kratos A: {len(iface_coords)} iface pts, NDOF={len(list(mp.Nodes))}")

if __name__ == "__main__":
    main()
