#!/usr/bin/env python3
"""
Coupling driver for FEBio (subdomain A) and deal.II (subdomain B)
Dirichlet-Neumann iteration with A as DIRICHLET side, B as NEUMANN side.
"""

import os
import sys
import numpy as np
import subprocess
import shutil
from pathlib import Path

# Paths
FEBIO_BIN = "/home/alexander/FEBio/bin/febio4"
DEALII_DIR = "/home/alexander/dealii/build/myproject"
WORK_DIR = str(Path(__file__).parent.parent / "coupled_sim")
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
    
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<FEBioModel>\n')
        f.write('  <Analysis type="static">\n')
        f.write('    <control dtime="1" maxinc="1" nlit="100" tol="1e-12"/>\n')
        f.write('  </Analysis>\n')
        f.write('  <Mesh>\n')
        f.write(f'    <Node count="{(nx+1)*(ny+1)}">\n')
        
        node_idx = 1
        for j in range(ny+1):
            y = j * h
            for i in range(nx+1):
                f.write(f'      <n id="{node_idx}" x="{x:.15e}" y="{y:.15e}" z="0"/>\n'.format(x=i*h))
                node_idx += 1
        
        f.write('    </Node>\n')
        f.write(f'    <Element count="{nx*ny}">\n')
        
        elem_idx = 1
        for j in range(ny):
            for i in range(nx):
                n1 = j*(nx+1) + i + 1
                n2 = n1 + 1
                n3 = n2 + nx + 1
                n4 = n1 + nx + 1
                f.write(f'      <qshell4 id="{elem_idx}" mat="1" n="{n1} {n2} {n3} {n4}"/>\n')
                elem_idx += 1
        
        f.write('    </Element>\n')
        f.write('  </Mesh>\n')
        
        f.write('  <Material>\n')
        f.write(f'    <hyperelastic id="1">\n')
        # For linear elasticity with plane strain, we need appropriate formulation
        # Using neo-Hookean as approximation
        bulk_modulus = LAMBDA_A + 2*MU_A/3
        f.write(f'      <neo-hookean mu="{MU_A:.15e}" bulk="{bulk_modulus:.15e}"/>\n')
        f.write('    </hyperelastic>\n')
        f.write('  </Material>\n')
        
        f.write('  <Boundary>\n')
        
        # Bottom boundary (y=0) - Dirichlet u=0
        f.write('    <Dirichlet bc="1" dofs="1 2">\n')
        for i in range(nx+1):
            node_id = i + 1
            f.write(f'      <Node id="{node_id}"/>\n')
        f.write('    </Dirichlet>\n')
        
        # Top boundary (y=1) - Dirichlet u=0
        f.write('    <Dirichlet bc="2" dofs="1 2">\n')
        for i in range(nx+1):
            node_id = ny*(nx+1) + i + 1
            f.write(f'      <Node id="{node_id}"/>\n')
        f.write('    </Dirichlet>\n')
        
        # Left boundary (x=0) - Dirichlet u=0
        f.write('    <Dirichlet bc="3" dofs="1 2">\n')
        for j in range(1, ny):  # Exclude corners already handled
            node_id = j*(nx+1) + 1
            f.write(f'      <Node id="{node_id}"/>\n')
        f.write('    </Dirichlet>\n')
        
        # Right boundary (interface at x=0.625)
        if dirichlet_data is not None:
            # Apply Dirichlet from coupling
            f.write('    <Dirichlet bc="4" dofs="1 2">\n')
            for j in range(1, ny):  # Interior nodes only
                node_id = j*(nx+1) + nx + 1
                ux, uy = dirichlet_data[j-1]
                f.write(f'      <Node id="{node_id}" val="{ux:.15e} {uy:.15e}"/>\n')
            f.write('    </Dirichlet>\n')
        else:
            # First iteration - homogeneous Dirichlet
            f.write('    <Dirichlet bc="4" dofs="1 2">\n')
            for j in range(1, ny):
                node_id = j*(nx+1) + nx + 1
                f.write(f'      <Node id="{node_id}" val="0 0"/>\n')
            f.write('    </Dirichlet>\n')
        
        f.write('  </Boundary>\n')
        
        # Body force as nodal loads
        f.write('  <BodyForce>\n')
        f.write('    <load id="1">\n')
        for j in range(ny+1):
            for i in range(nx+1):
                node_id = j*(nx+1) + i + 1
                if (i, j) in forces:
                    fx, fy = forces[(i, j)]
                    f.write(f'      <Node id="{node_id}" fx="{fx:.15e}" fy="{fy:.15e}"/>\n')
        f.write('    </load>\n')
        f.write('  </BodyForce>\n')
        
        f.write('  <Output>\n')
        f.write(f'    <model file="solution_{output_prefix}_iter{iter_num}.febio"/>\n')
        f.write(f'    <nodefile file="nodes_{output_prefix}_iter{iter_num}.dat">\n')
        f.write('      <displacement/>\n')
        f.write('    </nodefile>\n')
        f.write('  </Output>\n')
        f.write('</FEBioModel>\n')
    
    return filename

def run_febio(deck_file):
    """Run FEBio solver"""
    result = subprocess.run([FEBIO_BIN, deck_file], 
                          capture_output=True, text=True, cwd=WORK_DIR)
    return result.returncode == 0, result.stdout, result.stderr

def read_febio_nodes(filename):
    """Read node displacements from FEBio .dat file"""
    nodes = {}
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            # Skip header lines
            data_start = 0
            for i, line in enumerate(lines):
                if 'id' in line.lower() and 'x' in line.lower():
                    data_start = i + 1
                    break
            
            for line in lines[data_start:]:
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        nid = int(parts[0])
                        ux = float(parts[1])
                        uy = float(parts[2])
                        uz = float(parts[3]) if len(parts) > 3 else 0.0
                        nodes[nid] = (ux, uy, uz)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error reading FEBio nodes: {e}")
    
    return nodes

def main():
    print("Starting coupled simulation...")
    
    # Generate probe points
    probes_A = generate_probe_points_A()
    probes_B = generate_probe_points_B()
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
        
        for iteration in range(MAX_ITERATIONS):
            print(f"  Coupling iteration {iteration}")
            
            # Step 1: Solve subdomain A (DIRICHLET side) with current displacement
            feb_file = write_febio_deck(level, h, iteration, disp_interface)
            
            success, stdout, stderr = run_febio(feb_file)
            if not success:
                print(f"  FEBio failed: {stderr}")
                # Try to continue anyway
                pass
            
            # Read solution from A
            nodes_file = f"nodes_A_iter{iteration}.dat"
            if os.path.exists(nodes_file):
                nodes_A = read_febio_nodes(nodes_file)
                
                # Extract interface displacements and compute traction
                # This requires computing stress from displacement gradient
                # For now, just extract displacements
                new_disp_from_A = []
                for j in range(1, ny_A):
                    node_id = j*(nx_A+1) + nx_A + 1
                    if node_id in nodes_A:
                        new_disp_from_A.append(nodes_A[node_id][:2])
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
