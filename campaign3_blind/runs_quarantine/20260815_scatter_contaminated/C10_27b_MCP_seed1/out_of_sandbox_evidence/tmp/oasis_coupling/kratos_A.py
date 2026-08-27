"""Kratos Multiphysics participant for subdomain A (DIRICHLET side) - 3D heat conduction.

CONTRACT: runs in work_dir with no arguments, reads imports.json, writes exports.json LAST.
Dirichlet side: imports temperature values, exports temperature + outward normal flux.
"""
import json
from pathlib import Path
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem parameters for subdomain A
PARTNER = "B"
X0, X1 = 0.0, 0.625  # Subdomain A: x in [0, 0.625]
Y0, Y1 = 0.0, 1.0
Z0, Z1 = 0.0, 1.0
K = 1.0  # thermal conductivity
IFACE_X = 0.625  # interface location
T_INIT = 0.0  # iteration-1 fallback

# Mesh resolution - will be set based on level
# h = 1/8 -> nx = 0.625/0.125 = 5, ny = 1/0.125 = 8, nz = 8
# h = 1/16 -> nx = 10, ny = 16, nz = 16
# h = 1/32 -> nx = 20, ny = 32, nz = 32
NX, NY, NZ = 5, 8, 8  # default, will be overridden

# Source term for subdomain A
def source_A(x, y, z):
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
    """Read imported data from partner."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER)


def interpolate_imported(imp, y_coords, z_coords):
    """Interpolate imported temperature to interface grid."""
    if not imp or "coordinates" not in imp:
        return np.full(len(y_coords) * len(z_coords), T_INIT)
    
    coords = np.array(imp["coordinates"])
    values = np.array(imp["values"])
    
    # Interpolate in 2D (y, z) on the interface
    result = []
    for y in y_coords:
        for z in z_coords:
            # Find nearest neighbor or interpolate
            dists = np.sqrt((coords[:, 1] - y)**2 + **(coords[:, 2] - z)2)
            if len(dists) > 0:
                idx = np.argmin(dists)
                result.append(values[idx])
            else:
                result.append(T_INIT)
    return np.array(result)


def solve(T_interface):
    """Solve the heat equation in subdomain A."""
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 3
    
    # Add required variables
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX, KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    
    mp.SetBufferSize(1)
    
    # Create properties
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
    
    # Set material properties and source
    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        x, y, z = node.X, node.Y, node.Z
        node.SetSolutionStepValue(KM.HEAT_FLUX, source_A(x, y, z))
    
    # Create hexahedral elements (8-node)
    eid = 1
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                nodes = [
                    nid[(i, j, k)], nid[(i+1, j, k)], nid[(i+1, j+1, k)], nid[(i, j+1, k)],
                    nid[(i, j, k+1)], nid[(i+1, j, k+1)], nid[(i+1, j+1, k+1)], nid[(i, j+1, k+1)]
                ]
                mp.CreateNewElement("LaplacianElement3D8N", eid, nodes, props)
                eid += 1
    
    # Outer boundary conditions: u = 0 on all outer faces
    # Left face (x = X0)
    for j in range(NY + 1):
        for k in range(NZ + 1):
            n = mp.Nodes[nid[(0, j, k)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
    
    # Top face (y = Y1)
    for i in range(NX + 1):
        for k in range(NZ + 1):
            n = mp.Nodes[nid[(i, NY, k)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
    
    # Bottom face (y = Y0)
    for i in range(NX + 1):
        for k in range(NZ + 1):
            n = mp.Nodes[nid[(i, 0, k)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
    
    # Front face (z = Z1)
    for i in range(NX + 1):
        for j in range(NY + 1):
            n = mp.Nodes[nid[(i, j, NZ)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
    
    # Back face (z = Z0)
    for i in range(NX + 1):
        for j in range(NY + 1):
            n = mp.Nodes[nid[(i, j, 0)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
    
    # Interface (x = X1): Dirichlet from imported data
    # Interface nodes EXCLUDING corners that are on outer boundary
    T_iface_idx = 0
    for j in range(NY + 1):
        for k in range(NZ + 1):
            # Skip corners that are on outer boundary
            if (j == 0 or j == NY or k == 0 or k == NZ):
                continue
            n = mp.Nodes[nid[(NX, j, k)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, float(T_interface[T_iface_idx]))
            n.Fix(KM.TEMPERATURE)
            T_iface_idx += 1
    
    # Add DOFs with reaction variable
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    
    # Solver setup
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    
    return mp, nid


def main():
    global NX, NY, NZ
    
    # Read mesh resolution from environment or file
    try:
        with open("mesh_resolution.txt", "r") as f:
            NX, NY, NZ = map(int, f.read().strip().split())
    except:
        pass  # Use defaults
    
    # Read imported data
    imp = read_imports()
    
    # Interface grid for importing
    # Interior interface points only (not on outer boundary)
    ny_int = NY - 1  # exclude boundary
    nz_int = NZ - 1
    y_int = [Y0 + (j + 0.5) * (Y1 - Y0) / ny_int for j in range(ny_int)]
    z_int = [Z0 + (k + 0.5) * (Z1 - Z0) / nz_int for k in range(nz_int)]
    
    # Flatten for interpolation
    y_flat = []
    z_flat = []
    for y in y_int:
        for z in z_int:
            y_flat.append(y)
            z_flat.append(z)
    
    T_interface = interpolate_imported(imp, y_flat, z_flat)
    
    # Solve
    mp, nid = solve(T_interface)
    
    # Extract interface data for export
    # Interface nodes (interior only, matching import grid)
    iface_coords = []
    iface_values = []
    iface_fluxes = []
    
    T_iface_idx = 0
    for j in range(1, NY):  # exclude boundaries
        for k in range(1, NZ):  # exclude boundaries
            node = mp.Nodes[nid[(NX, j, k)]]
            y = Y0 + j * (Y1 - Y0) / NY
            z = Z0 + k * (Z1 - Z0) / NZ
            
            iface_coords.append([IFACE_X, y, z])
            iface_values.append(node.GetSolutionStepValue(KM.TEMPERATURE))
            
            # Outward normal flux: q = -k * dT/dn
            # For subdomain A, outward normal at interface is +x direction
            reaction = node.GetSolutionStepValue(KM.REACTION_FLUX)
            # Reaction is the force needed to maintain the constraint
            # For heat: reaction = -integral(k * grad(T) . n * phi) = integral(q . n * phi)
            # So reaction / weight gives the flux density
            # Weight for interior node on interface: dy * dz
            dy = (Y1 - Y0) / NY
            dz = (Z1 - Z0) / NZ
            weight = dy * dz
            flux = -reaction / weight  # negative because reaction is opposite to flux
            iface_fluxes.append(flux)
            
            T_iface_idx += 1
    
    # Write exports
    exports = {
        "field_name": "temperature",
        "n_points": len(iface_coords),
        "coordinates": iface_coords,
        "values": iface_values,
        "normal_fluxes": iface_fluxes
    }
    
    Path("exports.json").write_text(json.dumps(exports, indent=2))
    
    # Write NDOF to log file
    ndof = sum(1 for n in mp.Nodes if not n.IsFixed(KM.TEMPERATURE))
    with open("run.log", "w") as f:
        f.write(f"NDOF = {len(list(mp.Nodes))}\n")
        f.write(f"Interface points = {len(iface_coords)}\n")
    
    print(f"Kratos A (Dirichlet): {len(iface_coords)} interface points, NDOF = {len(list(mp.Nodes))}")


if __name__ == "__main__":
    main()
