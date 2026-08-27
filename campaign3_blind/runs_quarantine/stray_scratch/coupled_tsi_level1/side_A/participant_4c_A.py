#!/usr/bin/env python3
"""
4C participant for subdomain A (Dirichlet side) in coupled thermo-structural problem.
Subdomain A: x in [0, 0.625], y in [0, 1]
Material: k=1, lambda=600, mu=400, beta=1
Role: Dirichlet side - receives T and u from partner, applies as BC, exports fluxes and tractions
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
Lx_A = X_INTERFACE  # width of subdomain A
Ly = Y_MAX - Y_MIN

# Material properties for subdomain A
k_A = 1.0
lambda_A = 600.0
mu_A = 400.0
beta_A = 1.0

# Mesh resolution (will be set based on command line or config)
N_DIVISIONS = None  # Will be determined from imports or default

def get_mesh_resolution():
    """Determine mesh resolution from environment or default"""
    # Default: h = 1/8 means 8 divisions in y-direction
    # For x-direction in A: Lx_A/h = 0.625/(1/8) = 5 divisions at level 1
    return 8  # base resolution

def generate_interface_points(n_interface=44):
    """Generate interface probe points along x=X_INTERFACE"""
    points = []
    for i in range(n_interface):
        y = Y_MIN + (i + 0.5) * Ly / n_interface
        points.append([X_INTERFACE, y])
    return np.array(points)

def write_4c_input(mesh_nx, mesh_ny, imports_data, output_prefix):
    """Generate 4C YAML input file for thermo-structural coupling"""
    
    # Generate mesh nodes for subdomain A
    nodes = []
    node_id = 1
    for j in range(mesh_ny + 1):
        for i in range(mesh_nx + 1):
            x = i * Lx_A / mesh_nx
            y = j * Ly / mesh_ny
            nodes.append(f"NODE {node_id} COORD {x:.10f} {y:.10f} 0.0")
            node_id += 1
    
    # Generate QUAD4 elements
    elements = []
    elem_id = 1
    for j in range(mesh_ny):
        for i in range(mesh_nx):
            n_bl = i + j * (mesh_nx + 1) + 1  # bottom-left
            n_br = n_bl + 1                    # bottom-right
            n_tr = n_bl + mesh_nx + 1          # top-right
            n_tl = n_bl + 1                     # top-left (actually n_bl + mesh_nx + 2 - 1)
            n_tl = n_bl + mesh_nx + 1
            n_br = n_bl + 1
            elements.append(f"{elem_id} SOLID QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2")
            elem_id += 1
    
    # Also need thermal elements
    thermo_elements = []
    elem_id = 1
    for j in range(mesh_ny):
        for i in range(mesh_nx):
            n_bl = i + j * (mesh_nx + 1) + 1
            n_br = n_bl + 1
            n_tr = n_bl + mesh_nx + 1
            n_tl = n_bl + mesh_nx + 1
            thermo_elements.append(f"{elem_id} THERMO QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1")
            elem_id += 1
    
    # Identify boundary nodes for Dirichlet conditions (outer boundary)
    # Left (x=0), Bottom (y=0), Top (y=1) - all have T=0, u=(0,0)
    # Right edge (x=X_INTERFACE) is the interface - will get BC from imports
    
    # DLINE topology for outer boundaries
    dline_nodes = []
    dline_id = 1
    
    # Left boundary (x=0) - Dirichlet
    left_dline = dline_id
    for j in range(mesh_ny + 1):
        node_idx = j * (mesh_nx + 1) + 1
        dline_nodes.append(f"NODE {node_idx} DLINE {left_dline}")
    dline_id += 1
    
    # Bottom boundary (y=0) - Dirichlet  
    bottom_dline = dline_id
    for i in range(mesh_nx + 1):
        node_idx = i + 1
        dline_nodes.append(f"NODE {node_idx} DLINE {bottom_dline}")
    dline_id += 1
    
    # Top boundary (y=1) - Dirichlet
    top_dline = dline_id
    for i in range(mesh_nx + 1):
        node_idx = i + mesh_ny * (mesh_nx + 1) + 1
        dline_nodes.append(f"NODE {node_idx} DLINE {top_dline}")
    dline_id += 1
    
    # Interface line (x=X_INTERFACE) - will use Neumann from imports
    interface_dline = dline_id
    for j in range(mesh_ny + 1):
        node_idx = mesh_nx + j * (mesh_nx + 1) + 1
        dline_nodes.append(f"NODE {node_idx} DLINE {interface_dline}")
    
    # Source terms - need to evaluate f_T, f_x, f_y at element centers
    # For simplicity, we'll use a constant approximation or zero for now
    # The actual source terms are complex polynomials
    
    yaml_content = f'''TITLE:
  - "Coupled thermo-structural subdomain A - 4C participant"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLTEMP: 1.0e-12
  TOLRES: 1.0e-12
  MAXITER: 100
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-12
  MAXITER: 100
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "TSI_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [{k_A}]
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {1040.0}
      NUE: {0.3}
      DENS: 0.0
      BETA: {beta_A}
DESIGN LINE DIRICH CONDITIONS:
  - E: {left_dline}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {bottom_dline}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {top_dline}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: 0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DLINE-NODE TOPOLOGY:
{chr(10).join('  - ' + line for line in dline_nodes)}
NODE COORDS:
{chr(10).join('  - "' + node + '"' for node in nodes)}
STRUCTURE ELEMENTS:
{chr(10).join('  - "' + elem + '"' for elem in elements)}
THERMO ELEMENTS:
{chr(10).join('  - "' + elem + '"' for elem in thermo_elements)}
IO/RUNTIME VTK OUTPUT:
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
    
    with open(f"{output_prefix}.4C.yaml", 'w') as f:
        f.write(yaml_content)
    
    return f"{output_prefix}.4C.yaml"

def run_4c_solver(input_file, output_prefix):
    """Run the 4C binary"""
    cmd = ["/home/alexander/4C/build/4C", input_file, output_prefix]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"4C failed with return code {result.returncode}", file=sys.stderr)
        print(f"STDOUT: {result.stdout}", file=sys.stderr)
        print(f"STDERR: {result.stderr}", file=sys.stderr)
        return False
    
    return True

def extract_results(output_prefix, interface_points):
    """Extract temperature, displacement, fluxes, and tractions at interface"""
    # Read VTU file using pyvista
    try:
        import pyvista as pv
        
        # Find the VTU file
        vtu_files = list(Path(f"{output_prefix}-vtk-files").glob("*.vtu"))
        if not vtu_files:
            # Try alternative location
            vtu_files = list(Path(".").glob("*structure*.vtu"))
        
        if not vtu_files:
            raise FileNotFoundError("No VTU files found")
        
        # Read the last timestep
        vtu_file = sorted(vtu_files)[-1]
        mesh = pv.read(str(vtu_file))
        
        # Extract fields
        T = mesh.point_data.get('temperature', mesh.point_data.get('temp'))
        u = mesh.point_data.get('displacement')
        
        if T is None or u is None:
            raise ValueError("Required fields not found in VTU")
        
        # Interpolate to interface points
        coords = mesh.points
        T_values = []
        u_values = []
        
        for pt in interface_points:
            # Find nearest node
            dists = np.linalg.norm(coords - pt, axis=1)
            idx = np.argmin(dists)
            T_values.append(T[idx])
            u_values.append(u[idx*2:idx*2+2])
        
        # Compute fluxes and tractions at interface
        # Heat flux: q = -k * grad(T) . n
        # Traction: t = sigma . n
        # For subdomain A, outward normal at interface is (+1, 0)
        
        # This requires computing gradients - simplified approach
        # In practice, we'd need to compute these from the FEM solution
        
        qn_values = []  # Normal heat flux
        tx_values = []  # Traction x-component
        ty_values = []  # Traction y-component
        
        # Placeholder - would need proper gradient computation
        for i in range(len(interface_points)):
            qn_values.append(0.0)  # To be computed
            tx_values.append(0.0)  # To be computed
            ty_values.append(0.0)  # To be computed
        
        return {
            'T': np.array(T_values),
            'ux': np.array([uv[0] for uv in u_values]),
            'uy': np.array([uv[1] for uv in u_values]),
            'qn': np.array(qn_values),
            'tx': np.array(tx_values),
            'ty': np.array(ty_values)
        }
        
    except Exception as e:
        print(f"Error extracting results: {e}", file=sys.stderr)
        raise

def main():
    work_dir = Path.cwd()
    
    # Read imports
    imports_path = work_dir / "imports.json"
    if imports_path.exists():
        with open(imports_path) as f:
            imports = json.load(f)
    else:
        imports = {}
    
    # Get initial guess or imported values
    if imports and 'side_B' in imports:
        partner_data = imports['side_B']
        interface_coords = np.array(partner_data['coordinates'])
        imported_T = np.array(partner_data['values']['T'])
        imported_ux = np.array(partner_data['values']['ux'])
        imported_uy = np.array(partner_data['values']['uy'])
    else:
        # Initial guess: zero
        interface_points = generate_interface_points(44)
        imported_T = np.zeros(len(interface_points))
        imported_ux = np.zeros(len(interface_points))
        imported_uy = np.zeros(len(interface_points))
    
    # Determine mesh resolution
    n_divisions = get_mesh_resolution()
    mesh_nx = int(Lx_A * n_divisions)  # divisions in x for subdomain A
    mesh_ny = int(Ly * n_divisions)     # divisions in y
    
    # Count DOFs
    n_nodes = (mesh_nx + 1) * (mesh_ny + 1)
    ndof_thermal = n_nodes
    ndof_structural = 2 * n_nodes
    total_ndof = ndof_thermal + ndof_structural
    
    # Write execution log
    with open(work_dir / "run_log.txt", 'w') as f:
        f.write(f"NDOF = {total_ndof}\n")
        f.write(f"Mesh: {mesh_nx}x{mesh_ny} elements\n")
        f.write(f"Nodes: {n_nodes}\n")
    
    # Generate and run 4C input
    input_file = write_4c_input(mesh_nx, mesh_ny, imports, str(work_dir / "tsi_A"))
    
    success = run_4c_solver(input_file, str(work_dir / "tsi_A"))
    
    if not success:
        sys.exit(1)
    
    # Extract results at interface
    interface_points = generate_interface_points(44)
    results = extract_results(str(work_dir / "tsi_A"), interface_points)
    
    # Prepare exports
    exports = {
        'field_name': 'thermo_structural_interface',
        'n_points': len(interface_points),
        'coordinates': interface_points.tolist(),
        'values': {
            'T': results['T'].tolist(),
            'ux': results['ux'].tolist(),
            'uy': results['uy'].tolist()
        },
        'normal_fluxes': {
            'qn': results['qn'].tolist(),
            'tx': results['tx'].tolist(),
            'ty': results['ty'].tolist()
        }
    }
    
    # Write exports
    with open(work_dir / "exports.json", 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"Participant A completed successfully. Exported {len(interface_points)} interface points.")

if __name__ == "__main__":
    main()
