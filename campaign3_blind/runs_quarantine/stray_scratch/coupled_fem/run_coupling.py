#!/usr/bin/env python3
"""
Dirichlet-Neumann coupling between FEBio (subdomain A) and deal.II (subdomain B).

Subdomain A: (0, 0.625) x (0, 1), lambda=500, mu=250 - DIRICHLET side
Subdomain B: (0.625, 1.5) x (0, 1), lambda=500, mu=1250 - NEUMANN side

Interface at x = 0.625 (5/8)
"""

import numpy as np
import os
import subprocess
from pathlib import Path
import sys

# Problem parameters
INTERFACE_X = 0.625  # 5/8
WIDTH_A = 0.625
WIDTH_B = 0.875  # 1.5 - 0.625
HEIGHT = 1.0
THICKNESS = 0.1  # Small thickness for plane strain approximation

# Material properties
LAMBDA_A, MU_A = 500, 250
LAMBDA_B, MU_B = 500, 1250

# Coupling parameters
TOL = 1e-6
MAX_ITER = 100


class FEBioSolver:
    """FEBio solver wrapper for subdomain A."""
    
    def __init__(self, level, h):
        self.level = level
        self.h = h
        self.work_dir = Path(f"febio_level{level}")
        self.work_dir.mkdir(exist_ok=True)
        
        # Generate mesh - uniform grid
        nx = int(round(WIDTH_A / h))
        ny = int(round(HEIGHT / h))
        
        # Adjust to fit exactly
        hx = WIDTH_A / nx
        hy = HEIGHT / ny
        
        # Create nodes for both z-layers
        # Order: for each z-layer, iterate j (y), then i (x)
        self.nodes = []  # List of (x, y, z)
        self.node_map = {}  # (i, j, k) -> node_id (1-based for FEBio)
        
        idx = 1
        # Bottom layer (z=0)
        for j in range(ny + 1):
            y = j * hy
            if abs(y - HEIGHT) < 1e-10:
                y = HEIGHT
            for i in range(nx + 1):
                x = i * hx
                if abs(x - WIDTH_A) < 1e-10:
                    x = WIDTH_A
                self.nodes.append((x, y, 0.0))
                self.node_map[(i, j, 0)] = idx
                idx += 1
        
        # Top layer (z=THICKNESS)
        for j in range(ny + 1):
            y = j * hy
            if abs(y - HEIGHT) < 1e-10:
                y = HEIGHT
            for i in range(nx + 1):
                x = i * hx
                if abs(x - WIDTH_A) < 1e-10:
                    x = WIDTH_A
                self.nodes.append((x, y, THICKNESS))
                self.node_map[(i, j, 1)] = idx
                idx += 1
        
        self.nx = nx
        self.ny = ny
        self.n_nodes = len(self.nodes)
        self.ndof = 3 * self.n_nodes  # ux, uy, uz per node
        
        # Interface nodes (at x = WIDTH_A, interior only - not corners)
        self.interface_node_ids_bottom = [self.node_map[(nx, j, 0)] for j in range(1, ny)]
        self.interface_node_ids_top = [self.node_map[(nx, j, 1)] for j in range(1, ny)]
        self.interface_node_ids = self.interface_node_ids_bottom + self.interface_node_ids_top
        
        # Outer boundary nodes (excluding interface interior)
        self.outer_boundary_ids = set()
        for k in range(2):  # Both z-layers
            for j in range(ny + 1):
                for i in range(nx + 1):
                    x, y, z = self.nodes[self.node_map[(i, j, k)] - 1]
                    # Check if on outer boundary (not interface interior)
                    if abs(x) < 1e-10 or abs(y) < 1e-10 or abs(y - HEIGHT) < 1e-10:
                        self.outer_boundary_ids.add(self.node_map[(i, j, k)])
                    elif abs(x - WIDTH_A) < 1e-10 and (abs(y) < 1e-10 or abs(y - HEIGHT) < 1e-10):
                        # Interface corners are part of outer boundary
                        self.outer_boundary_ids.add(self.node_map[(i, j, k)])
        
        # All nodes for z-constraint
        self.all_node_ids = list(range(1, self.n_nodes + 1))
        
        # Elements (hex8)
        # For hex8: bottom face nodes CCW, then top face nodes CCW
        self.elements = []
        for j in range(ny):
            for i in range(nx):
                # Bottom face nodes (counter-clockwise starting from lower-left)
                n0 = self.node_map[(i, j, 0)]         # lower-left
                n1 = self.node_map[(i+1, j, 0)]       # lower-right
                n2 = self.node_map[(i+1, j+1, 0)]     # upper-right
                n3 = self.node_map[(i, j+1, 0)]       # upper-left
                # Top face nodes (same ordering)
                n4 = self.node_map[(i, j, 1)]         # lower-left
                n5 = self.node_map[(i+1, j, 1)]       # lower-right
                n6 = self.node_map[(i+1, j+1, 1)]     # upper-right
                n7 = self.node_map[(i, j+1, 1)]       # upper-left
                self.elements.append([n0, n1, n2, n3, n4, n5, n6, n7])
    
    def write_feb_file(self, filename, interface_displacement=None):
        """Write FEBio input file."""
        # Convert lambda, mu to E, nu for isotropic elastic material
        # For plane strain: E = mu*(3*lambda + 2*mu)/(lambda + mu), nu = lambda/(2*(lambda + mu))
        E = MU_A * (3*LAMBDA_A + 2*MU_A) / (LAMBDA_A + MU_A)
        nu = LAMBDA_A / (2 * (LAMBDA_A + MU_A))
        
        with open(filename, 'w') as f:
            f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
            f.write('<febio_spec version="4.0">\n')
            f.write('  <Module type="solid"/>\n')
            f.write('  <Control>\n')
            f.write('    <analysis>STATIC</analysis>\n')
            f.write('    <time_steps>1</time_steps>\n')
            f.write('    <step_size>1</step_size>\n')
            f.write('    <solver type="solid">\n')
            f.write('      <symmetric_stiffness>symmetric</symmetric_stiffness>\n')
            f.write('    </solver>\n')
            f.write('  </Control>\n')
            f.write('  <Material>\n')
            f.write('    <material id="1" name="MaterialA" type="isotropic elastic">\n')
            f.write('      <density>1.0</density>\n')
            f.write(f'      <E>{E:.15e}</E>\n')
            f.write(f'      <v>{nu:.15e}</v>\n')
            f.write('    </material>\n')
            f.write('  </Material>\n')
            f.write('  <Mesh>\n')
            f.write('    <Nodes name="Object1">\n')
            for idx, (x, y, z) in enumerate(self.nodes, 1):
                f.write(f'      <node id="{idx}">{x:.15e},{y:.15e},{z:.15e}</node>\n')
            f.write('    </Nodes>\n')
            f.write('    <Elements type="hex8" name="Part1">\n')
            for e_idx, e in enumerate(self.elements, 1):
                f.write(f'      <elem id="{e_idx}">{" ".join(map(str, e))}</elem>\n')
            f.write('    </Elements>\n')
            
            # Node sets
            f.write(f'    <NodeSet name="outer_boundary">{" ".join(map(str, sorted(self.outer_boundary_ids)))}</NodeSet>\n')
            f.write(f'    <NodeSet name="all_nodes">{" ".join(map(str, self.all_node_ids))}</NodeSet>\n')
            
            if interface_displacement is not None:
                f.write(f'    <NodeSet name="interface">{" ".join(map(str, self.interface_node_ids))}</NodeSet>\n')
            
            f.write('  </Mesh>\n')
            f.write('  <MeshDomains>\n')
            f.write('    <SolidDomain name="Part1" mat="MaterialA"/>\n')
            f.write('  </MeshDomains>\n')
            f.write('  <Boundary>\n')
            # Zero displacement on outer boundary
            f.write('    <bc name="fix_outer" type="zero displacement" node_set="outer_boundary">\n')
            f.write('      <x_dof>1</x_dof>\n')
            f.write('      <y_dof>1</y_dof>\n')
            f.write('      <z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
            
            # Fix z-displacement for all nodes (plane strain)
            f.write('    <bc name="fix_z" type="zero displacement" node_set="all_nodes">\n')
            f.write('      <z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
            
            f.write('  </Boundary>\n')
            f.write('  <Loads>\n')
            # Body load - constant approximation (will be improved later)
            f.write('    <body_load type="const">\n')
            f.write('      <x>0.0</x>\n')
            f.write('      <y>0.0</y>\n')
            f.write('      <z>0</z>\n')
            f.write('    </body_load>\n')
            f.write('  </Loads>\n')
            f.write('  <Output>\n')
            f.write('    <plotfile type="febio">\n')
            f.write('      <var type="displacement"/>\n')
            f.write('      <var type="stress"/>\n')
            f.write('    </plotfile>\n')
            f.write('    <logfile>\n')
            f.write('      <node_data data="x;y;z;ux;uy;uz" delim="," file="nodal_output.csv"/>\n')
            f.write('    </logfile>\n')
            f.write('  </Output>\n')
            f.write('</febio_spec>\n')
    
    def solve(self, interface_displacement=None):
        """Run FEBio solver."""
        feb_file = self.work_dir / "model.feb"
        self.write_feb_file(feb_file, interface_displacement)
        
        # Run FEBio from the work directory
        result = subprocess.run(
            ['/home/alexander/FEBio/bin/febio4', 'model.feb'],
            cwd=str(self.work_dir),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Write log file
        with open(f"run_level{self.level}_A.log", 'w') as f:
            f.write(f"NDOF = {self.ndof}\n")
            f.write(f"FEBio output:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"FEBio errors:\n{result.stderr}\n")
        
        return result.returncode == 0
    
    def read_solution(self):
        """Read solution from FEBio output."""
        csv_file = self.work_dir / "nodal_output.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Solution file not found: {csv_file}")
        
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        # Columns: x, y, z, ux, uy, uz
        displacements = data[:, 3:6]  # ux, uy, uz
        return displacements


def main():
    print("Starting Dirichlet-Neumann coupling simulation...")
    
    # Test FEBio first
    print("\nTesting FEBio solver...")
    test_solver = FEBioSolver(1, 1/8)
    print(f"Mesh: {test_solver.nx}x{test_solver.ny} elements, {test_solver.n_nodes} nodes")
    success = test_solver.solve()
    print(f"FEBio test: {'SUCCESS' if success else 'FAILED'}")
    
    if success:
        disp = test_solver.read_solution()
        print(f"Read {len(disp)} displacement values")
        print(f"Sample displacements: {disp[:3]}")
    
    print("\nDone with initial test.")


if __name__ == "__main__":
    main()
