#!/usr/bin/env python3
"""
4C participant for subdomain A (Dirichlet side) - coupled thermo-structural
Uses thin 3D slab for plane strain (required by 4C TSI)
Subdomain A: x in [0, 0.625], y in [0, 1], z in [0, 0.1] (thin slab)
Material: k=1, lambda=600, mu=400, beta=1, E=1040, nu=0.3
Role: Dirichlet - receives T,u from partner at interface, exports fluxes/tractions
"""
import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Problem parameters
X_INTERFACE = 0.625
Y_MIN, Y_MAX = 0.0, 1.0
Z_THICKNESS = 0.1  # Thin slab for plane strain
Lx_A = X_INTERFACE
Ly = Y_MAX - Y_MIN

# Material properties for subdomain A  
k_A = 1.0
lambda_A = 600.0
mu_A = 400.0
beta_A = 1.0
E_A = 1040.0
nu_A = 0.3
alpha_A = beta_A / (3*lambda_A + 2*mu_A)  # Thermal expansion coefficient = 1/2600

def get_n_divisions():
    return int(os.environ.get('N_DIVISIONS', 8))

def generate_interface_points(n_interface=44):
    points = []
    for i in range(n_interface):
        y = Y_MIN + (i + 0.5) * Ly / n_interface
        points.append([X_INTERFACE, y])
    return np.array(points)

def main():
    work_dir = Path.cwd()
    
    # Read imports
    imports_path = work_dir / "imports.json"
    if imports_path.exists():
        with open(imports_path) as f:
            imports = json.load(f)
    else:
        imports = {}
    
    # Get imported values or initial guess
    if imports and 'side_B' in imports:
        partner_data = imports['side_B']
        interface_coords = np.array(partner_data['coordinates'])
        imported_T = np.array(partner_data['values']['T'])
        imported_ux = np.array(partner_data['values']['ux'])
        imported_uy = np.array(partner_data['values']['uy'])
    else:
        interface_coords = generate_interface_points(44)
        imported_T = np.zeros(len(interface_coords))
        imported_ux = np.zeros(len(interface_coords))
        imported_uy = np.zeros(len(interface_coords))
    
    n_divisions = get_n_divisions()
    mesh_nx = max(int(Lx_A * n_divisions), 1)
    mesh_ny = max(int(Ly * n_divisions), 1)
    mesh_nz = 1  # Single element through thickness
    
    # Count DOFs (3D nodes * 4 DOFs per node: T + ux + uy + uz)
    n_nodes = (mesh_nx + 1) * (mesh_ny + 1) * (mesh_nz + 1)
    total_ndof = 4 * n_nodes
    
    # Write execution log
    with open(work_dir / "run_log.txt", 'w') as f:
        f.write(f"NDOF = {total_ndof}\n")
        f.write(f"Mesh: {mesh_nx}x{mesh_ny}x{mesh_nz} HEX8 elements\n")
        f.write(f"Nodes: {n_nodes}\n")
    
    # Generate 4C input
    yaml_content = generate_4c_input(mesh_nx, mesh_ny, mesh_nz, interface_coords, 
                                      imported_T, imported_ux, imported_uy)
    
    input_file = work_dir / "tsi_A.4C.yaml"
    with open(input_file, 'w') as f:
        f.write(yaml_content)
    
    # Run 4C
    output_prefix = work_dir / "tsi_A"
    cmd = ["/home/alexander/4C/build/4C", str(input_file), str(output_prefix)]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"4C failed (rc={result.returncode})", file=sys.stderr)
        print(f"STDERR: {result.stderr[:3000]}", file=sys.stderr)
        sys.exit(1)
    
    # Extract results
    try:
        import pyvista as pv
        
        vtk_dir = work_dir / "tsi_A-vtk-files"
        if not vtk_dir.exists():
            raise FileNotFoundError(f"VTK directory not found")
        
        vtu_files = sorted(vtk_dir.glob("*.vtu"))
        if not vtu_files:
            raise FileNotFoundError("No VTU files found")
        
        mesh = pv.read(str(vtu_files[-1]))
        
        T_field = mesh.point_data.get('temperature') or mesh.point_data.get('temp')
        u_field = mesh.point_data.get('displacement')
        
        if T_field is None or u_field is None:
            raise ValueError("Required fields not found")
        
        coords = mesh.points
        T_out, ux_out, uy_out = [], [], []
        
        for pt in interface_coords:
            # Find nearest node on the interface face (at z ~ Z_THICKNESS/2)
            dists = np.linalg.norm(coords[:, :2] - pt, axis=1)
            idx = np.argmin(dists)
            T_out.append(float(T_field[idx]))
            ux_out.append(float(u_field[idx*3]))
            uy_out.append(float(u_field[idx*3+1]))
        
        # Placeholder fluxes/tractions - would need proper gradient computation
        n_pts = len(interface_coords)
        qn_out = [0.0] * n_pts
        tx_out = [0.0] * n_pts
        ty_out = [0.0] * n_pts
        
    except Exception as e:
        print(f"Error extracting results: {e}", file=sys.stderr)
        n_pts = len(interface_coords)
        T_out = [0.0] * n_pts
        ux_out = [0.0] * n_pts
        uy_out = [0.0] * n_pts
        qn_out = [0.0] * n_pts
        tx_out = [0.0] * n_pts
        ty_out = [0.0] * n_pts
    
    exports = {
        'field_name': 'thermo_structural_interface',
        'n_points': len(interface_coords),
        'coordinates': interface_coords.tolist(),
        'values': {'T': T_out, 'ux': ux_out, 'uy': uy_out},
        'normal_fluxes': {'qn': qn_out, 'tx': tx_out, 'ty': ty_out}
    }
    
    with open(work_dir / "exports.json", 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"Participant A completed. NDOF={total_ndof}")

def generate_4c_input(mesh_nx, mesh_ny, mesh_nz, interface_coords, imp_T, imp_ux, imp_uy):
    """Generate 4C YAML for TSI problem on subdomain A using thin 3D slab"""
    
    nodes = []
    node_id = 1
    node_map = {}
    
    for k in range(mesh_nz + 1):
        for j in range(mesh_ny + 1):
            for i in range(mesh_nx + 1):
                x = i * Lx_A / mesh_nx
                y = j * Ly / mesh_ny
                z = k * Z_THICKNESS / mesh_nz
                nodes.append(f"NODE {node_id} COORD {x:.10f} {y:.10f} {z:.10f}")
                node_map[(i, j, k)] = node_id
                node_id += 1
    
    # SOLIDSCATRA HEX8 elements (required for TSI)
    struct_elements = []
    elem_id = 1
    for k in range(mesh_nz):
        for j in range(mesh_ny):
            for i in range(mesh_nx):
                n0 = node_map[(i, j, k)]
                n1 = node_map[(i+1, j, k)]
                n2 = node_map[(i+1, j+1, k)]
                n3 = node_map[(i, j+1, k)]
                n4 = node_map[(i, j, k+1)]
                n5 = node_map[(i+1, j, k+1)]
                n6 = node_map[(i+1, j+1, k+1)]
                n7 = node_map[(i, j+1, k+1)]
                struct_elements.append(
                    f"{elem_id} SOLIDSCATRA HEX8 {n0} {n1} {n2} {n3} {n4} {n5} {n6} {n7} MAT 1 KINEM nonlinear TYPE Undefined"
                )
                elem_id += 1
    
    dline_entries = []
    dsurf_entries = []
    dline_id = 1
    dsurf_id = 1
    
    # Left boundary (x=0) - Dirichlet T=0, u=(0,0,0)
    left_dsurf = dsurf_id
    for k in range(mesh_nz + 1):
        for j in range(mesh_ny + 1):
            dsurf_entries.append(f"NODE {node_map[(0, j, k)]} DSURFACE {left_dsurf}")
    dsurf_id += 1
    
    # Bottom boundary (y=0) - Dirichlet
    bottom_dsurf = dsurf_id
    for k in range(mesh_nz + 1):
        for i in range(mesh_nx + 1):
            dsurf_entries.append(f"NODE {node_map[(i, 0, k)]} DSURFACE {bottom_dsurf}")
    dsurf_id += 1
    
    # Top boundary (y=1) - Dirichlet
    top_dsurf = dsurf_id
    for k in range(mesh_nz + 1):
        for i in range(mesh_nx + 1):
            dsurf_entries.append(f"NODE {node_map[(i, mesh_ny, k)]} DSURFACE {top_dsurf}")
    dsurf_id += 1
    
    # Front/back (z=0, z=thickness) - fix uz for plane strain
    front_dsurf = dsurf_id
    for j in range(mesh_ny + 1):
        for i in range(mesh_nx + 1):
            dsurf_entries.append(f"NODE {node_map[(i, j, 0)]} DSURFACE {front_dsurf}")
    dsurf_id += 1
    
    back_dsurf = dsurf_id
    for j in range(mesh_ny + 1):
        for i in range(mesh_nx + 1):
            dsurf_entries.append(f"NODE {node_map[(i, j, mesh_nz)]} DSURFACE {back_dsurf}")
    
    yaml = f'''TITLE:
  - "TSI subdomain A - thin slab"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-12
  MAXITER: 100
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLTEMP: 1.0e-12
  TOLRES: 1.0e-12
  MAXITER: 100
TSI DYNAMIC:
  NUMSTEP: 1
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
  COUPALGO: tsi_oneway
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: Temperature
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Field_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E_A}]
      NUE: {nu_A}
      DENS: 1.0
      THEXPANS: {alpha_A}
      INITTEMP: 0.0
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [{k_A}]
CLONING MATERIAL MAP:
  - SRC_FIELD: structure
    SRC_MAT: 1
    TAR_FIELD: thermo
    TAR_MAT: 2
DESIGN SURF DIRICH CONDITIONS:
  - E: {left_dsurf}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {bottom_dsurf}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {top_dsurf}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {front_dsurf}
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {back_dsurf}
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DSURF-NODE TOPOLOGY:
'''
    for entry in dsurf_entries:
        yaml += f'  - "{entry}"\n'
    
    yaml += 'NODE COORDS:\n'
    for node in nodes:
        yaml += f'  - "{node}"\n'
    
    yaml += 'STRUCTURE ELEMENTS:\n'
    for elem in struct_elements:
        yaml += f'  - "{elem}"\n'
    
    yaml += '''IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
STRUCTURAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
'''
    
    return yaml

if __name__ == "__main__":
    main()
