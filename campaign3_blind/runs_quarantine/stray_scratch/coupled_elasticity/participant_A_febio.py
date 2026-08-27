"""FEBio 4 participant for coupled elasticity - Subdomain A (Dirichlet side).

Subdomain A: (0, 0.625) x (0, 1), lambda=450, mu=225 (E=600, nu=1/3)
Interface at x = 0.625 (right edge of A)
Outer BC: u=0 on left, top, bottom edges (including interface corners)
Role: Dirichlet side - imports displacement, exports traction

Using hex8 elements with unit thickness in z for plane strain approximation.
"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

# Problem parameters for Subdomain A
X0, X1 = 0.0, 0.625  # Subdomain A extent
Y0, Y1 = 0.0, 1.0
Z0, Z1 = 0.0, 0.1    # Small thickness for quasi-3D
IFACE_X = 0.625      # Interface is at right edge of A
LAMBDA_A = 450.0
MU_A = 225.0
E_A = 600.0          # Young's modulus
NU_A = 1.0/3.0       # Poisson's ratio

PARTNER = "B"  # Name of partner in couple()
FEBIO_BIN = "/home/alexander/FEBio/bin/febio4"

# Fallback values for iteration 1
U_INIT = [0.0, 0.0]


def read_imports():
    """Read imports.json and return partner's interface data."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER)


def sample_vector(imp, y_coords):
    """Interpolate partner's vector field onto our y-coordinates."""
    if imp is None or "coordinates" not in imp:
        n = len(y_coords)
        return np.array([[U_INIT[0], U_INIT[1]]] * n)
    
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vals = np.asarray(imp.get("values", []), float)
    
    if vals.ndim == 1:
        vals = vals.reshape(-1, 2)
    
    if vals.shape[0] != len(ys):
        n = len(y_coords)
        return np.array([[U_INIT[0], U_INIT[1]]] * n)
    
    order = np.argsort(ys)
    ux_interp = np.interp(y_coords, ys[order], vals[order, 0])
    uy_interp = np.interp(y_coords, ys[order], vals[order, 1])
    
    return np.column_stack([ux_interp, uy_interp])


def build_mesh_and_nodes(nx, ny):
    """Build structured mesh for subdomain A using hex8 elements."""
    xs = np.linspace(X0, X1, nx + 1)
    ys = np.linspace(Y0, Y1, ny + 1)
    zs = np.linspace(Z0, Z1, 2)  # One element thick in z
    
    nodes = []
    node_id = 1
    node_map = {}
    
    for k in range(2):
        for j in range(ny + 1):
            for i in range(nx + 1):
                node_map[(i, j, k)] = node_id
                nodes.append((node_id, xs[i], ys[j], zs[k]))
                node_id += 1
    
    # Elements (hex8)
    elems = []
    elem_id = 1
    for j in range(ny):
        for i in range(nx):
            conn = [
                node_map[(i, j, 0)],       # bottom-front-left
                node_map[(i+1, j, 0)],     # bottom-front-right
                node_map[(i+1, j+1, 0)],   # bottom-back-right
                node_map[(i, j+1, 0)],     # bottom-back-left
                node_map[(i, j, 1)],       # top-front-left
                node_map[(i+1, j, 1)],     # top-front-right
                node_map[(i+1, j+1, 1)],   # top-back-right
                node_map[(i, j+1, 1)],     # top-back-left
            ]
            elems.append((elem_id, conn))
            elem_id += 1
    
    # Interface nodes at x = IFACE_X (right edge), EXCLUDING corners
    # Take only the front face (k=0) for export
    iface_i = nx
    iface_nodes = [node_map[(iface_i, j, 0)] for j in range(1, ny)]  # Exclude j=0 and j=ny
    iface_y = [float(ys[j]) for j in range(1, ny)]
    
    # All interface nodes (both z layers) for BC application
    iface_all = [node_map[(iface_i, j, k)] for k in range(2) for j in range(1, ny)]
    
    # Outer boundary nodes: left edge, top edge, bottom edge, AND interface corners
    outer_nodes = []
    # Left edge (x = X0), all nodes including corners, both z layers
    for k in range(2):
        for j in range(ny + 1):
            outer_nodes.append(node_map[(0, j, k)])
    # Top edge (y = Y1), excluding left corner already counted
    for k in range(2):
        for i in range(1, nx + 1):  # Include interface corner at (nx, ny)
            outer_nodes.append(node_map[(i, ny, k)])
    # Bottom edge (y = Y0), excluding left corner, include interface corner
    for k in range(2):
        for i in range(1, nx + 1):  # Include interface corner at (nx, 0)
            outer_nodes.append(node_map[(i, 0, k)])
    
    return nodes, elems, iface_nodes, iface_all, iface_y, outer_nodes


def write_febio_deck(nodes, elems, iface_nodes, iface_all, outer_nodes, u_iface):
    """Write FEBio 4.0 deck file with hex8 elements."""
    lines = []
    lines.append('<?xml version="1.0" encoding="ISO-8859-1"?>')
    lines.append('<febio_spec version="4.0">')
    lines.append('  <Module type="solid"/>')
    lines.append('  <Control>')
    lines.append('    <analysis>STATIC</analysis>')
    lines.append('    <time_steps>1</time_steps>')
    lines.append('    <step_size>1.0</step_size>')
    lines.append('    <solver type="solid">')
    lines.append('      <symmetric_stiffness>symmetric</symmetric_stiffness>')
    lines.append('    </solver>')
    lines.append('  </Control>')
    lines.append('  <Globals>')
    lines.append('    <Constants>')
    lines.append('      <T>0</T><R>0</R><Fc>0</Fc>')
    lines.append('    </Constants>')
    lines.append('  </Globals>')
    lines.append('  <Material>')
    lines.append(f'    <material id="1" name="MatA" type="isotropic elastic">')
    lines.append('      <density>1.0</density>')
    lines.append(f'      <E>{E_A:.17g}</E>')
    lines.append(f'      <v>{NU_A:.17g}</v>')
    lines.append('    </material>')
    lines.append('  </Material>')
    lines.append('  <Mesh>')
    lines.append('    <Nodes name="Object1">')
    for nid, x, y, z in nodes:
        lines.append(f'      <node id="{nid}">{x:.17g},{y:.17g},{z:.17g}</node>')
    lines.append('    </Nodes>')
    lines.append('    <Elements type="hex8" mat="1" name="PartA">')
    for eid, conn in elems:
        lines.append(f'      <elem id="{eid}">{",".join(str(c) for c in conn)}</elem>')
    lines.append('    </Elements>')
    
    # Node sets
    lines.append('    <NodeSet name="outer">' + ",".join(str(n) for n in outer_nodes) + '</NodeSet>')
    lines.append('    <NodeSet name="iface">' + ",".join(str(n) for n in iface_all) + '</NodeSet>')
    lines.append('    <NodeSet name="iface_export">' + ",".join(str(n) for n in iface_nodes) + '</NodeSet>')
    lines.append('  </Mesh>')
    lines.append('  <MeshDomains>')
    lines.append('    <SolidDomain name="PartA" mat="MatA"/>')
    lines.append('  </MeshDomains>')
    
    # MeshData for prescribed displacement at interface
    lines.append('  <MeshData>')
    lines.append('    <NodeData name="ux_map" node_set="iface" data_type="scalar">')
    for lid, n in enumerate(iface_all, 1):
        # Map to the corresponding export node index
        idx = (lid - 1) % len(iface_nodes)
        lines.append(f'      <node lid="{lid}">{u_iface[idx, 0]:.17g}</node>')
    lines.append('    </NodeData>')
    lines.append('    <NodeData name="uy_map" node_set="iface" data_type="scalar">')
    for lid, n in enumerate(iface_all, 1):
        idx = (lid - 1) % len(iface_nodes)
        lines.append(f'      <node lid="{lid}">{u_iface[idx, 1]:.17g}</node>')
    lines.append('    </NodeData>')
    lines.append('  </MeshData>')
    
    # Boundary conditions
    lines.append('  <Boundary>')
    # Outer boundary: u = 0
    lines.append('    <bc name="outer_fix" type="zero displacement" node_set="outer">')
    lines.append('      <x_dof>1</x_dof>')
    lines.append('      <y_dof>1</y_dof>')
    lines.append('      <z_dof>1</z_dof>')
    lines.append('    </bc>')
    # Interface: prescribed displacement from partner
    lines.append('    <bc name="iface_ux" type="prescribed displacement" node_set="iface">')
    lines.append('      <dof>x</dof>')
    lines.append('      <value lc="1" type="map">ux_map</value>')
    lines.append('      <relative>0</relative>')
    lines.append('    </bc>')
    lines.append('    <bc name="iface_uy" type="prescribed displacement" node_set="iface">')
    lines.append('      <dof>y</dof>')
    lines.append('      <value lc="1" type="map">uy_map</value>')
    lines.append('      <relative>0</relative>')
    lines.append('    </bc>')
    lines.append('  </Boundary>')
    
    # Load controller
    lines.append('  <LoadData>')
    lines.append('    <load_controller id="1" type="loadcurve">')
    lines.append('      <interpolate>LINEAR</interpolate>')
    lines.append('      <extend>CONSTANT</extend>')
    lines.append('      <points>')
    lines.append('        <pt>0,0</pt>')
    lines.append('        <pt>1,1</pt>')
    lines.append('      </points>')
    lines.append('    </load_controller>')
    lines.append('  </LoadData>')
    
    # Output
    lines.append('  <Output>')
    lines.append('    <logfile>')
    lines.append('      <node_data data="ux" delim="," file="node_ux.csv" node_set="iface_export"/>')
    lines.append('      <node_data data="uy" delim="," file="node_uy.csv" node_set="iface_export"/>')
    lines.append('      <element_data data="sx" delim="," file="elem_sx.csv"/>')
    lines.append('      <element_data data="sy" delim="," file="elem_sy.csv"/>')
    lines.append('      <element_data data="sxy" delim="," file="elem_sxy.csv"/>')
    lines.append('    </logfile>')
    lines.append('    <plotfile type="febio">')
    lines.append('      <var type="displacement"/>')
    lines.append('      <var type="stress"/>')
    lines.append('    </plotfile>')
    lines.append('  </Output>')
    lines.append('</febio_spec>')
    
    Path("subdomain_A.feb").write_text("\n".join(lines) + "\n")


def parse_csv_log(filename):
    """Parse FEBio CSV log file."""
    result = {}
    try:
        content = Path(filename).read_text()
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            if not line or line.startswith("*"):
                continue
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    key = int(float(parts[0]))
                    val = float(parts[1])
                    result[key] = val
                except ValueError:
                    continue
    except FileNotFoundError:
        pass
    return result


def compute_traction(iface_nodes, elems):
    """Compute traction at interface nodes from element stresses.
    
    For plane strain: sigma = [[sx, sxy], [sxy, sy]]
    Outward normal at right edge of A: n = [+1, 0]
    Traction t = sigma . n = [sx, sxy]
    Export q_out = -t (outward flux convention per OASiS spec)
    """
    sx = parse_csv_log("elem_sx.csv")
    sxy = parse_csv_log("elem_sxy.csv")
    
    # Build element-to-node mapping
    elem_to_nodes = {}
    for eid, conn in elems:
        elem_to_nodes[eid] = conn
    
    # For each interface node, find adjacent elements and average stress
    traction = np.zeros((len(iface_nodes), 2))
    
    for ni, node_id in enumerate(iface_nodes):
        sx_vals = []
        sxy_vals = []
        
        for eid, conn in elem_to_nodes.items():
            if node_id in conn:
                if eid in sx:
                    sx_vals.append(sx[eid])
                if eid in sxy:
                    sxy_vals.append(sxy[eid])
        
        if sx_vals and sxy_vals:
            sx_avg = np.mean(sx_vals)
            sxy_avg = np.mean(sxy_vals)
            # Outward normal is [+1, 0] (pointing out of subdomain A to the right)
            # Traction = sigma . n = [sx, sxy]
            # Export q_out = -traction (flux convention)
            traction[ni, 0] = -sx_avg
            traction[ni, 1] = -sxy_avg
        else:
            traction[ni, :] = [0.0, 0.0]
    
    return traction


def main():
    global NX, NY
    
    # Get mesh resolution from command line
    if len(sys.argv) > 1:
        NX = int(sys.argv[1])
        NY = int(sys.argv[2])
    else:
        NX = 8
        NY = 8
    
    # Read imports
    imp = read_imports()
    
    # Build mesh
    nodes, elems, iface_nodes, iface_all, iface_y, outer_nodes = build_mesh_and_nodes(NX, NY)
    
    # Sample interface displacement from partner
    u_iface = sample_vector(imp, iface_y)
    
    # Write and run FEBio deck
    write_febio_deck(nodes, elems, iface_nodes, iface_all, outer_nodes, u_iface)
    
    # Run FEBio
    result = subprocess.run(
        [FEBIO_BIN, "-i", "subdomain_A.feb"],
        capture_output=True, text=True, timeout=1800
    )
    
    if "N O R M A L   T E R M I N A T I O N" not in result.stdout:
        print(f"FEBio failed (rc={result.returncode})", file=sys.stderr)
        print(result.stdout[-2000:], file=sys.stderr)
        sys.exit(1)
    
    # Compute traction at interface
    traction = compute_traction(iface_nodes, elems)
    
    # Write exports.json
    exports = {
        "field_name": "traction",
        "n_points": len(iface_nodes),
        "coordinates": [[float(IFACE_X), float(y)] for y in iface_y],
        "values": [[float(t[0]), float(t[1])] for t in traction],
        "normal_fluxes": [[float(t[0]), float(t[1])] for t in traction]
    }
    
    Path("exports.json").write_text(json.dumps(exports, indent=2))
    
    # Write run log
    ndof = 3 * len(nodes)  # 3 DOFs per node (ux, uy, uz)
    with open("run.log", "w") as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"NX = {NX}, NY = {NY}\n")
        f.write(f"Interface nodes = {len(iface_nodes)}\n")
    
    print(f"Subdomain A (FEBio): NDof={ndof}, iface_u_mean={u_iface.mean():.6e}")


if __name__ == "__main__":
    main()
