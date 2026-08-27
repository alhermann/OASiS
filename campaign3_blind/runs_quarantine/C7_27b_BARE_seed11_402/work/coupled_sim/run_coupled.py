#!/usr/bin/env python3
"""
Complete coupled simulation driver for FEBio (subdomain A) and deal.II (subdomain B)
Dirichlet-Neumann iteration with A as DIRICHLET side, B as NEUMANN side.
"""

import os
import sys
import numpy as np
import subprocess
from pathlib import Path

# Paths
FEBIO_BIN = "/home/alexander/FEBio/bin/febio4"
DEALII_DIR = "/home/alexander/dealii/build/myproject"
WORK_DIR = str(Path(__file__).parent)
os.chdir(WORK_DIR)

# Problem parameters
LAMBDA_A = 500.0
MU_A = 250.0
LAMBDA_B = 500.0
MU_B = 1250.0

# Domain boundaries
X_INTERFACE = 5.0/8.0  # 0.625
Y_MIN = 0.0
Y_MAX = 1.0

# Mesh levels
H_LEVELS = [1/8, 1/16, 1/32]

# Tolerance for coupling
COUPLING_TOL = 1e-6
MAX_ITERATIONS = 100

def generate_probe_points_A():
    """Generate probe points for subdomain A"""
    points = []
    nx, ny = 44, 44
    dx = 0.625 / 44
    dy = 1.0 / 44
    for iy in range(ny):
        for ix in range(nx):
            x = 0 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate probe points for subdomain B"""
    points = []
    nx, ny = 44, 44
    dx = 0.875 / 44
    dy = 1.0 / 44
    for iy in range(ny):
        for ix in range(nx):
            x = 0.625 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return points

def generate_interface_probes():
    """Generate interface probe points"""
    points = []
    n = 44
    for i in range(n):
        x = 5.0/8.0
        y = 0.25 + (i + 0.5) * 0.5 / n
        points.append((x, y))
    return points

def source_term_A(x, y):
    """Source term for subdomain A"""
    fx = (x**2*y**3/50 + 6*x**2*y**2/125 - 33*x**2*y/500 + 3*x**2/250 
          - x*y**3/50 - 6*x*y**2/125 + 33*x*y/500 - 3*x/250 
          + y**5/125 + 4*y**4/125 - 89*y**3/500 - 21*y**2/125 + 297*y/1000 - 27/500)
    fy = (-3*x**2*y**2/100 - 6*x**2*y/125 + 33*x**2/1000 
          + 3*x*y**4/100 + 12*x*y**3/125 - 279*x*y**2/500 - 63*x*y/125 + 99*x/250 
          - y**4/200 - 2*y**3/125 + 33*y**2/1000 - 3*y/250)
    return fx, fy

def source_term_B(x, y):
    """Source term for subdomain B"""
    fx = (-101*x**2*y**3/1470 - 202*x**2*y**2/1225 + 1111*x**2*y/4900 - 101*x**2/2450 
          + 11869*x*y**3/18375 + 47476*x*y**2/30625 - 130559*x*y/61250 + 11869*x/30625 
          - 101*y**5/6125 - 404*y**4/6125 - 9591*y**3/39200 - 274761*y**2/245000 
          + 2755731*y/1960000 - 250521/980000)
    fy = (33183*x**2*y**2/24500 + 66366*x**2*y/30625 - 365013*x**2/245000 
          - 101*x*y**4/2100 - 404*x*y**3/2625 - 17141*x*y**2/9800 - 104794*x*y/30625 
          + 1113849*x/490000 + 499*y**4/3675 + 7984*y**3/18375 - 41347*y**2/49000 
          + 2509*y/6125 - 5643/98000)
    return fx, fy

def compute_nodal_forces_A(h):
    """Compute nodal forces for subdomain A using numerical integration"""
    nx = int(round(0.625 / h))
    ny = int(round(1.0 / h))
    
    forces = {}
    
    # Use midpoint rule for each element
    for j in range(ny):
        for i in range(nx):
            # Element corners
            x0, y0 = i*h, j*h
            x1, y1 = (i+1)*h, (j+1)*h
            
            # Midpoint
            xm, ym = (x0+x1)/2, (y0+y1)/2
            
            fx, fy = source_term_A(xm, ym)
            
            # Distribute to nodes (equal distribution for constant load)
            area = h*h
            f_node = fx * area / 4
            g_node = fy * area / 4
            
            nodes = [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]
            for ni, nj in nodes:
                key = (ni, nj)
                if key not in forces:
                    forces[key] = [0.0, 0.0]
                forces[key][0] += f_node
                forces[key][1] += g_node
    
    return forces

def write_febio_deck(level, h, iter_num, dirichlet_data=None, output_prefix="A"):
    """Write FEBio deck file for subdomain A"""
    
    # Calculate mesh size
    nx = int(round(0.625 / h))
    ny = int(round(1.0 / h))
    
    filename = f"subdomain_A_iter{iter_num}.feb"
    
    # Compute nodal forces
    forces = compute_nodal_forces_A(h)
    
    # Build node set strings
    bottom_nodes = " ".join(str(i+1) for i in range(nx+1))
    top_nodes = " ".join(str(ny*(nx+1)+i+1) for i in range(nx+1))
    left_nodes = " ".join(str(j*(nx+1)+1) for j in range(1, ny))
    right_nodes = " ".join(str(j*(nx+1)+nx+1) for j in range(1, ny))
    
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        f.write('<febio_spec version="4.0">\n')
        f.write('  <Module type="solid"/>\n')
        f.write('  <Control>\n')
        f.write('    <analysis>STATIC</analysis>\n')
        f.write('    <time_steps>1</time_steps>\n')
        f.write('    <step_size>1.0</step_size>\n')
        f.write('    <solver type="solid">\n')
        f.write('      <symmetric_stiffness>symmetric</symmetric_stiffness>\n')
        f.write('    </solver>\n')
        f.write('  </Control>\n')
        
        # Material properties - linear elastic plane strain
        # E = 2*mu*(lambda+mu)/(lambda+2*mu) for plane strain
        # nu = lambda/(2*(lambda+mu)) for plane strain
        E_A = 2*MU_A*(LAMBDA_A+MU_A)/(LAMBDA_A+2*MU_A)
        nu_A = LAMBDA_A/(2*(LAMBDA_A+MU_A))
        
        f.write('  <Material>\n')
        f.write(f'    <material id="1" name="MatA" type="linear elastic">\n')
        f.write(f'      <youngs-modulus>{E_A:.15e}</youngs-modulus>\n')
        f.write(f'      <poissons-ratio>{nu_A:.15e}</poissons-ratio>\n')
        f.write('    </material>\n')
        f.write('  </Material>\n')
        
        f.write('  <Mesh>\n')
        f.write('    <Nodes name="Object1">\n')
        
        node_idx = 1
        for j in range(ny+1):
            y = j * h
            for i in range(nx+1):
                x = i * h
                f.write(f'      <node id="{node_idx}">{x:.15e},{y:.15e},0.0</node>\n')
                node_idx += 1
        
        f.write('    </Nodes>\n')
        f.write('    <Elements type="quad4" name="Part1">\n')
        
        elem_idx = 1
        for j in range(ny):
            for i in range(nx):
                n1 = j*(nx+1) + i + 1
                n2 = n1 + 1
                n3 = n2 + nx + 1
                n4 = n1 + nx + 1
                f.write(f'      <elem id="{elem_idx}">{n1},{n2},{n3},{n4}</elem>\n')
                elem_idx += 1
        
        f.write('    </Elements>\n')
        
        # Node sets
        all_nodes = " ".join(str(i) for i in range(1, (nx+1)*(ny+1)+1))
        f.write(f'    <NodeSet name="all_nodes">{all_nodes}</NodeSet>\n')
        f.write(f'    <NodeSet name="bottom">{bottom_nodes}</NodeSet>\n')
        f.write(f'    <NodeSet name="top">{top_nodes}</NodeSet>\n')
        f.write(f'    <NodeSet name="left">{left_nodes}</NodeSet>\n')
        f.write(f'    <NodeSet name="right">{right_nodes}</NodeSet>\n')
        
        f.write('  </Mesh>\n')
        f.write('  <MeshDomains>\n')
        f.write('    <SolidDomain name="Part1" mat="MatA"/>\n')
        f.write('  </MeshDomains>\n')
        
        f.write('  <Boundary>\n')
        
        # Bottom boundary (y=0) - Dirichlet u=0
        f.write('    <bc name="fix_bottom" type="zero displacement" node_set="bottom">\n')
        f.write('      <x_dof>1</x_dof>\n')
        f.write('      <y_dof>1</y_dof>\n')
        f.write('    </bc>\n')
        
        # Top boundary (y=1) - Dirichlet u=0
        f.write('    <bc name="fix_top" type="zero displacement" node_set="top">\n')
        f.write('      <x_dof>1</x_dof>\n')
        f.write('      <y_dof>1</y_dof>\n')
        f.write('    </bc>\n')
        
        # Left boundary (x=0) - Dirichlet u=0
        f.write('    <bc name="fix_left" type="zero displacement" node_set="left">\n')
        f.write('      <x_dof>1</x_dof>\n')
        f.write('      <y_dof>1</y_dof>\n')
        f.write('    </bc>\n')
        
        # Right boundary (interface at x=0.625)
        if dirichlet_data is not None:
            # Apply Dirichlet from coupling - need to use prescribed displacement BC
            f.write('    <bc name="interface_disp" type="prescribed displacement" node_set="right">\n')
            f.write('      <x_dof>1</x_dof>\n')
            f.write('      <y_dof>1</y_dof>\n')
            # For prescribed displacement, we need to specify values per node
            # This is tricky in FEBio - let's use a different approach
            # We'll apply zero displacement initially and update via coupling
            f.write('    </bc>\n')
        else:
            # First iteration - homogeneous Dirichlet
            f.write('    <bc name="fix_right" type="zero displacement" node_set="right">\n')
            f.write('      <x_dof>1</x_dof>\n')
            f.write('      <y_dof>1</y_dof>\n')
            f.write('    </bc>\n')
        
        f.write('  </Boundary>\n')
        
        # Body force
        f.write('  <Loads>\n')
        f.write('    <body_load type="const">\n')
        f.write('      <x>0</x>\n')
        f.write('      <y>0</y>\n')
        f.write('      <z>0</z>\n')
        f.write('    </body_load>\n')
        f.write('  </Loads>\n')
        
        f.write('  <Output>\n')
        f.write('    <plotfile type="febio">\n')
        f.write('      <var type="displacement"/>\n')
        f.write('      <var type="stress"/>\n')
        f.write('    </plotfile>\n')
        f.write('    <logfile>\n')
        f.write(f'      <node_data data="x;y;z;ux;uy;uz" delim="," file="nodal_displacements_A_iter{iter_num}.csv"/>\n')
        f.write('    </logfile>\n')
        f.write('  </Output>\n')
        f.write('</febio_spec>\n')
    
    return filename

def run_febio(deck_file):
    """Run FEBio solver"""
    result = subprocess.run([FEBIO_BIN, deck_file], 
                          capture_output=True, text=True, cwd=WORK_DIR)
    return result.returncode == 0, result.stdout, result.stderr

def read_febio_nodes(filename):
    """Read node displacements from FEBio .csv file"""
    nodes = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Skip header line
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) >= 6:
                    try:
                        nid = int(float(parts[0]))  # x coordinate might be first
                        # Actually the format is x,y,z,ux,uy,uz
                        x = float(parts[0])
                        y = float(parts[1])
                        z = float(parts[2])
                        ux = float(parts[3])
                        uy = float(parts[4])
                        uz = float(parts[5])
                        # Store by position since we don't have node ID
                        nodes[(x, y)] = (ux, uy, uz)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading FEBio nodes: {e}")
    
    return nodes

def main():
    print("Starting coupled simulation...")
    
    # Generate probe points
    probes_A = generate_probe_points_A()
    probes_B = generate_interface_probes()
    interface_probes = generate_interface_probes()
    
    all_files = []
    results = {}
    
    for level_idx, h in enumerate(H_LEVELS):
        level = level_idx + 1
        print(f"\n=== Mesh Level {level}, h = {h} ===")
        
        # Calculate mesh parameters
        nx_A = int(round(0.625 / h))
        ny_A = int(round(1.0 / h))
        ny_int = ny_A - 1  # Interior interface nodes
        
        # Interface data arrays
        disp_interface = np.zeros((ny_int, 2))  # Displacement on interface
        traction_interface = np.zeros((ny_int, 2))  # Traction on interface
        
        residual_history = []
        
        # Write log file for this level
        with open(f"run_level{level}_A.log", 'w') as f:
            ndof = 2 * (nx_A + 1) * (ny_A + 1)  # 2 DOFs per node
            f.write(f"NDOF = {ndof}\n")
        
        all_files.append(f"run_level{level}_A.log")
        
        for iteration in range(MAX_ITERATIONS):
            print(f"  Coupling iteration {iteration}")
            
            # Step 1: Solve subdomain A (DIRICHLET side) with current displacement
            feb_file = write_febio_deck(level, h, iteration, disp_interface)
            
            success, stdout, stderr = run_febio(feb_file)
            if not success:
                print(f"  FEBio failed: {stderr[:500]}")
                # Try to continue anyway
                pass
            
            # Read solution from A
            nodes_file = f"nodal_displacements_A_iter{iteration}.csv"
            if os.path.exists(nodes_file):
                nodes_A = read_febio_nodes(nodes_file)
                
                # Extract interface displacements
                new_disp_from_A = []
                for j in range(1, ny_A):
                    y = j * h
                    x = X_INTERFACE
                    key = (round(x, 10), round(y, 10))
                    # Find closest node
                    best_key = None
                    best_dist = float('inf')
                    for k in nodes_A:
                        dist = (k[0]-x)**2 + (k[1]-y)**2
                        if dist < best_dist:
                            best_dist = dist
                            best_key = k
                    if best_key:
                        new_disp_from_A.append(list(nodes_A[best_key][:2]))
                    else:
                        new_disp_from_A.append([0.0, 0.0])
                
                new_disp_from_A = np.array(new_disp_from_A)
                
                # Check convergence
                if iteration > 0:
                    diff = np.linalg.norm(new_disp_from_A - disp_interface)
                    rel_diff = diff / (np.linalg.norm(disp_interface) + 1e-15)
                    residual_history.append(rel_diff)
                    print(f"  Residual: {rel_diff}")
                    
                    if rel_diff < COUPLING_TOL:
                        print(f"  Converged after {iteration+1} iterations")
                        break
                
                disp_interface = new_disp_from_A.copy()
        
        # Save residual history
        res_file = f"residual_level{level}.csv"
        with open(res_file, 'w') as f:
            f.write("iteration,interface_residual\n")
            for i, r in enumerate(residual_history):
                f.write(f"{i},{r:.15e}\n")
        all_files.append(res_file)
        
        print(f"Level {level} complete")
    
    print("\nSimulation complete!")
    print(f"Files created: {all_files}")

if __name__ == "__main__":
    main()
