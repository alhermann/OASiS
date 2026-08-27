#!/usr/bin/env python3
"""
Generator script for transient heat conduction problem with 4C.
Solves: dT/dt - div(k grad T) = f on unit square (0,1)x(0,1)
with k=1, volumetric heat capacity=1, T=0 on boundary, T=0 initially.
Uses THERMO QUAD4 elements with Crank-Nicolson (theta=0.5).
"""

import os
import subprocess
import sys
import numpy as np
from pathlib import Path

# Source term f(x,y,t) - exact expression from problem statement
# Converted from Python ** to C-style pow() function for 4C parser
SOURCE_TERM = """-t*pow(x,3)*pow(y,2)*exp(t/2)/2 + t*pow(x,3)*y*exp(t/2)/2 + 2*t*pow(x,3)*exp(t/2) - 3*t*pow(x,2)*pow(y,3)*exp(t/2)/4 + 31*t*pow(x,2)*pow(y,2)*exp(t/2)/20 + 41*t*pow(x,2)*y*exp(t/2)/5 - 31*t*pow(x,2)*exp(t/2)/5 + 3*t*x*pow(y,3)*exp(t/2)/4 + 99*t*x*pow(y,2)*exp(t/2)/20 - 147*t*x*y*exp(t/2)/10 + 21*t*x*exp(t/2)/5 + 3*t*pow(y,3)*exp(t/2) - 31*t*pow(y,2)*exp(t/2)/5 + 16*t*y*exp(t/2)/5 - pow(x,3)*pow(y,2)*exp(t/2) + pow(x,3)*y*exp(t/2) - 3*pow(x,2)*pow(y,3)*exp(t/2)/2 + 31*pow(x,2)*pow(y,2)*exp(t/2)/10 - 8*pow(x,2)*y*exp(t/2)/5 + 3*x*pow(y,3)*exp(t/2)/2 - 21*x*pow(y,2)*exp(t/2)/10 + 3*x*y*exp(t/2)/5"""

def generate_mesh(N):
    """Generate structured NxN quad mesh on unit square."""
    nodes = []
    node_id = 1
    h = 1.0 / N
    
    # Generate nodes in row-major order (y varies slowest)
    node_coords = {}
    for j in range(N + 1):  # y direction
        for i in range(N + 1):  # x direction
            x = i * h
            y = j * h
            node_coords[(i, j)] = node_id
            nodes.append(f'NODE {node_id} COORD {x:.10f} {y:.10f} 0.0')
            node_id += 1
    
    return nodes, node_coords

def generate_elements(N, node_coords):
    """Generate NxN QUAD4 elements with CCW ordering."""
    elements = []
    elem_id = 1
    
    for j in range(N):  # element rows
        for i in range(N):  # element columns
            # CCW ordering: bottom-left, bottom-right, top-right, top-left
            n1 = node_coords[(i, j)]       # bottom-left
            n2 = node_coords[(i+1, j)]     # bottom-right
            n3 = node_coords[(i+1, j+1)]   # top-right
            n4 = node_coords[(i, j+1)]     # top-left
            
            elements.append(f'{elem_id} THERMO QUAD4 {n1} {n2} {n3} {n4} MAT 1')
            elem_id += 1
    
    return elements

def generate_boundary_topology(N, node_coords):
    """Generate DLINE topology for all four edges."""
    dline_nodes = []
    
    # Edge 1: bottom (y=0), left to right
    for i in range(N + 1):
        nid = node_coords[(i, 0)]
        dline_nodes.append(f'NODE {nid} DLINE 1')
    
    # Edge 2: right (x=1), bottom to top
    for j in range(N + 1):
        nid = node_coords[(N, j)]
        dline_nodes.append(f'NODE {nid} DLINE 2')
    
    # Edge 3: top (y=1), right to left
    for i in range(N, -1, -1):
        nid = node_coords[(i, N)]
        dline_nodes.append(f'NODE {nid} DLINE 3')
    
    # Edge 4: left (x=0), top to bottom
    for j in range(N, -1, -1):
        nid = node_coords[(0, j)]
        dline_nodes.append(f'NODE {nid} DLINE 4')
    
    return dline_nodes

def generate_surface_topology(N, node_coords):
    """Generate DSURF topology for all elements (for 2D volumetric source via surface Neumann)."""
    dsurf_nodes = []
    
    # In 2D, each element is a "surface" - assign all nodes to DSURFACE 1
    for j in range(N + 1):
        for i in range(N + 1):
            nid = node_coords[(i, j)]
            dsurf_nodes.append(f'NODE {nid} DSURFACE 1')
    
    return dsurf_nodes

def create_4c_input(N, level, output_prefix):
    """Create .4C.yaml input file for given mesh size."""
    dt = 1.0 / (4 * N)
    num_steps = int(round(0.25 / dt))  # Should be exactly N steps to reach t=0.25
    
    nodes, node_coords = generate_mesh(N)
    elements = generate_elements(N, node_coords)
    dline_topology = generate_boundary_topology(N, node_coords)
    dsurf_topology = generate_surface_topology(N, node_coords)
    
    total_nodes = (N + 1) ** 2
    
    yaml_content = f'''TITLE:
  - "Transient heat conduction - level {level}, N={N}"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo"
THERMAL DYNAMIC:
  DYNAMICTYPE: "OneStepTheta"
  TIMESTEP: {dt:.15e}
  NUMSTEP: {num_steps}
  MAXTIME: 0.25
  INITIALFIELD: "zero_field"
  TOLTEMP: 1.0e-12
  TOLRES: 1.0e-10
  MAXITER: 50
  LINEAR_SOLVER: 1
  RESULTSEVERY: 1
THERMAL DYNAMIC/ONESTEPTHETA:
  THETA: 0.5
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermo_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [1.0]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SOURCE_TERM}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DLINE-NODE TOPOLOGY:
{chr(10).join('  - ' + line for line in dline_topology)}
DSURF-NODE TOPOLOGY:
{chr(10).join('  - ' + line for line in dsurf_topology)}
NODE COORDS:
{chr(10).join('  - "' + line + '"' for line in nodes)}
THERMO ELEMENTS:
{chr(10).join('  - "' + line + '"' for line in elements)}
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
'''
    
    filename = f"{output_prefix}.4C.yaml"
    with open(filename, 'w') as f:
        f.write(yaml_content)
    
    return filename, total_nodes

def run_4c(input_file, output_prefix):
    """Run 4C solver."""
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
    
    cmd = ['/home/alexander/4C/build/4C', input_file, output_prefix]
    
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )
    
    print("STDOUT:")
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
    
    return result.returncode

def find_last_vtu(output_prefix):
    """Find the last VTU file written by 4C."""
    vtu_dir = f"{output_prefix}-vtk-files"
    if not os.path.exists(vtu_dir):
        raise FileNotFoundError(f"VTU directory {vtu_dir} not found")
    
    vtu_files = sorted([f for f in os.listdir(vtu_dir) if f.endswith('.vtu')])
    if not vtu_files:
        raise FileNotFoundError(f"No VTU files found in {vtu_dir}")
    
    return os.path.join(vtu_dir, vtu_files[-1])

def read_vtu(vtu_file):
    """Read VTU file and extract node coordinates and temperature values using pyvista."""
    try:
        import pyvista as pv
    except ImportError:
        # Try alternative approach
        import xml.etree.ElementTree as ET
        import base64
        import zlib
        
        tree = ET.parse(vtu_file)
        root = tree.getroot()
        piece = root.find('.//Piece')
        
        # Get points
        points_elem = piece.find('.//Points/DataArray')
        coords_data = points_elem.text.strip()
        coords_format = points_elem.get('format', 'ascii')
        
        if coords_format == 'binary':
            # Decode base64 and decompress
            coords_bytes = base64.b64decode(coords_data)
            coords_decompressed = zlib.decompress(coords_bytes)
            coords = np.frombuffer(coords_decompressed, dtype=np.float64).reshape(-1, 3)
        else:
            coords_str = coords_data.split()
            coords = np.array([float(c) for c in coords_str]).reshape(-1, 3)
        
        # Get temperature field
        point_data = piece.find('.//PointData')
        temp_elem = None
        for da in point_data.findall('DataArray'):
            name = da.get('Name', '')
            if name == 'temperature':
                temp_elem = da
                break
        
        if temp_elem is None:
            available = [da.get('Name', '') for da in point_data.findall('DataArray')]
            raise ValueError(f"Temperature field not found. Available: {available}")
        
        temp_data = temp_elem.text.strip()
        temp_format = temp_elem.get('format', 'ascii')
        
        if temp_format == 'binary':
            temp_bytes = base64.b64decode(temp_data)
            temp_decompressed = zlib.decompress(temp_bytes)
            temperatures = np.frombuffer(temp_decompressed, dtype=np.float64)
        else:
            temp_str = temp_data.split()
            temperatures = np.array([float(t) for t in temp_str])
        
        return coords, temperatures
    
    # Use pyvista
    mesh = pv.read(vtu_file)
    coords = mesh.points
    temperatures = mesh.point_data['temperature']
    return coords, temperatures

def interpolate_to_probe_points(coords, temperatures, probe_points):
    """Interpolate temperature field to arbitrary probe points using bilinear shape functions."""
    N = len(probe_points)
    results = np.zeros(N)
    
    # Build a simple spatial lookup - for each probe point, find containing element
    unique_x = np.sort(np.unique(np.round(coords[:, 0], 10)))
    unique_y = np.sort(np.unique(np.round(coords[:, 1], 10)))
    
    nx = len(unique_x) - 1  # number of elements in x
    ny = len(unique_y) - 1  # number of elements in y
    
    hx = unique_x[1] - unique_x[0]
    hy = unique_y[1] - unique_y[0]
    
    # Create mapping from (i,j) node index to global node index
    node_index = {}
    idx = 0
    for j in range(len(unique_y)):
        for i in range(len(unique_x)):
            node_index[(i, j)] = idx
            idx += 1
    
    for kp, (px, py) in enumerate(probe_points):
        # Find which element contains this point
        ix = int(px / hx)
        iy = int(py / hy)
        
        # Clamp to valid range
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)
        
        # Local coordinates in reference element [-1,1]x[-1,1]
        xi = 2 * (px - unique_x[ix]) / hx - 1
        eta = 2 * (py - unique_y[iy]) / hy - 1
        
        # Bilinear shape functions
        N1 = 0.25 * (1 - xi) * (1 - eta)  # bottom-left
        N2 = 0.25 * (1 + xi) * (1 - eta)  # bottom-right
        N3 = 0.25 * (1 + xi) * (1 + eta)  # top-right
        N4 = 0.25 * (1 - xi) * (1 + eta)  # top-left
        
        # Get node indices for this element
        n1 = node_index[(ix, iy)]
        n2 = node_index[(ix + 1, iy)]
        n3 = node_index[(ix + 1, iy + 1)]
        n4 = node_index[(ix, iy + 1)]
        
        # Interpolate
        results[kp] = N1 * temperatures[n1] + N2 * temperatures[n2] + \
                      N3 * temperatures[n3] + N4 * temperatures[n4]
    
    return results

def generate_probe_points():
    """Generate 1936 probe points as specified."""
    probe_points = []
    for iy in range(44):
        for ix in range(44):
            x = (ix + 0.5) / 44
            y = (iy + 0.5) / 44
            probe_points.append((x, y))
    return probe_points

def write_csv(filename, probe_points, temperatures):
    """Write solution to CSV with high precision."""
    with open(filename, 'w') as f:
        f.write("x, y, u\n")
        for (x, y), u in zip(probe_points, temperatures):
            f.write(f"{x:.15e}, {y:.15e}, {u:.15e}\n")

def write_log(filename, ndof):
    """Write run log with NDOF count."""
    with open(filename, 'w') as f:
        f.write(f"NDOF = {ndof}\n")

def main():
    # Mesh levels
    levels = [8, 16, 32]
    probe_points = generate_probe_points()
    
    all_solutions = {}
    
    for level_idx, N in enumerate(levels, 1):
        print(f"\n{'='*60}")
        print(f"LEVEL {level_idx}: N = {N}")
        print(f"{'='*60}")
        
        output_prefix = f"level{level_idx}_N{N}"
        
        # Create input file
        input_file, ndof = create_4c_input(N, level_idx, output_prefix)
        print(f"Created input file: {input_file}")
        print(f"Total nodes (NDOF): {ndof}")
        
        # Run 4C
        returncode = run_4c(input_file, output_prefix)
        
        if returncode != 0:
            print(f"ERROR: 4C exited with code {returncode}")
            continue
        
        # Find and read VTU file
        vtu_file = find_last_vtu(output_prefix)
        print(f"Reading VTU file: {vtu_file}")
        
        coords, temperatures = read_vtu(vtu_file)
        print(f"Read {len(temperatures)} nodal temperatures")
        
        # Interpolate to probe points
        probe_temps = interpolate_to_probe_points(coords, temperatures, probe_points)
        
        # Store results
        all_solutions[level_idx] = probe_temps
        
        # Write output files
        csv_file = f"solution_level{level_idx}.csv"
        write_csv(csv_file, probe_points, probe_temps)
        print(f"Wrote {csv_file}")
        
        log_file = f"run_level{level_idx}.log"
        write_log(log_file, ndof)
        print(f"Wrote {log_file}")
    
    # Compute mesh independence
    print(f"\n{'='*60}")
    print("MESH INDEPENDENCE ANALYSIS")
    print(f"{'='*60}")
    
    if len(all_solutions) >= 2:
        # Compare finest two levels
        sol_fine = all_solutions[3]  # N=32
        sol_medium = all_solutions[2]  # N=16
        
        # Relative change
        rel_changes = np.abs(sol_fine - sol_medium) / (np.abs(sol_medium) + 1e-30)
        max_rel_change = np.max(rel_changes)
        
        print(f"Max relative change between N=16 and N=32: {max_rel_change:.6e}")
        
        # Check convergence (threshold 1%)
        converged = max_rel_change < 0.01
        mesh_independence = "CONVERGED" if converged else "NOT_CONVERGED"
    else:
        max_rel_change = float('inf')
        mesh_independence = "NOT_CONVERGED"
    
    # Write RESULT.txt
    csv_files = ", ".join([f"solution_level{k}.csv" for k in range(1, len(levels)+1)])
    with open("RESULT.txt", 'w') as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {csv_files}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print(f"\nWrote RESULT.txt")
    print(f"LEVELS = {len(levels)}")
    print(f"FILES = {csv_files}")
    print(f"MESH_INDEPENDENCE = {mesh_independence}")
    print(f"MAX_REL_CHANGE = {max_rel_change:.6e}")

if __name__ == "__main__":
    main()
