#!/usr/bin/env python3
"""
Generate FEBio deck file for subdomain A
"""

import os
import sys
import numpy as np

# Problem parameters
LAMBDA_A = 500.0
MU_A = 250.0

# For plane strain linear elasticity:
# E = 2*mu*(lambda+mu)/(lambda+2*mu)
# nu = lambda/(2*(lambda+mu))
E_A = 2*MU_A*(LAMBDA_A+MU_A)/(LAMBDA_A+2*MU_A)
nu_A = LAMBDA_A/(2*(LAMBDA_A+MU_A))

print(f"Subdomain A: E={E_A}, nu={nu_A}")

def source_term_A(x, y):
    """Source term for subdomain A"""
    fx = (x**2*y**3/50 + 6*x**2*y**2/125 - 33*x**2*y/500 + 3*x**2/250 
          - x*y**3/50 - 6*x*y**2/125 + 33*x*y/500 - 3*x/250 
          + y**5/125 + 4*y**4/125 - 89*y**3/500 - 21*y**2/125 + 297*y/1000 - 27/500)
    fy = (-3*x**2*y**2/100 - 6*x**2*y/125 + 33*x**2/1000 
          + 3*x*y**4/100 + 12*x*y**3/125 - 279*x*y**2/500 - 63*x*y/125 + 99*x/250 
          - y**4/200 - 2*y**3/125 + 33*y**2/1000 - 3*y/250)
    return fx, fy

def write_febio_deck(h, iter_num, dirichlet_disp=None, output_prefix="A"):
    """Write FEBio deck file for subdomain A"""
    
    # Calculate mesh size
    nx = int(round(0.625 / h))
    ny = int(round(1.0 / h))
    nz = 1  # Single layer in z for plane strain
    
    filename = f"subdomain_A_iter{iter_num}.feb"
    
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
        
        f.write('  <Material>\n')
        f.write(f'    <material id="1" name="MatA" type="isotropic elastic">\n')
        f.write(f'      <density>1.0</density>\n')
        f.write(f'      <E>{E_A:.15e}</E>\n')
        f.write(f'      <v>{nu_A:.15e}</v>\n')
        f.write('    </material>\n')
        f.write('  </Material>\n')
        
        f.write('  <Mesh>\n')
        f.write('    <Nodes name="Object1">\n')
        
        node_idx = 1
        nodes = []
        # Generate nodes: first layer at z=0, second layer at z=0.1
        for jz in range(nz+1):
            z = jz * 0.1
            for j in range(ny+1):
                y = j * h
                for i in range(nx+1):
                    x = i * h
                    nodes.append((node_idx, x, y, z))
                    node_idx += 1
        
        for nid, x, y, z in nodes:
            f.write(f'      <node id="{nid}">{x:.15e},{y:.15e},{z:.15e}</node>\n')
        
        f.write('    </Nodes>\n')
        f.write('    <Elements type="hex8" name="Part1">\n')
        
        elem_idx = 1
        # Generate hex8 elements
        for jz in range(nz):
            z_offset = jz * (nx+1) * (ny+1)
            for j in range(ny):
                for i in range(nx):
                    # Bottom face (z=0 for this element)
                    n1 = z_offset + j*(nx+1) + i + 1
                    n2 = n1 + 1
                    n3 = n2 + nx + 1
                    n4 = n1 + nx + 1
                    # Top face (z=0.1 for this element)
                    n5 = n1 + (nx+1)*(ny+1)
                    n6 = n2 + (nx+1)*(ny+1)
                    n7 = n3 + (nx+1)*(ny+1)
                    n8 = n4 + (nx+1)*(ny+1)
                    f.write(f'      <elem id="{elem_idx}">{n1},{n2},{n3},{n4},{n5},{n6},{n7},{n8}</elem>\n')
                    elem_idx += 1
        
        f.write('    </Elements>\n')
        
        # Node sets - only use bottom layer (z=0) for boundary conditions
        bottom_nodes = " ".join(str(j*(nx+1)+i+1) for j in [0] for i in range(nx+1))
        top_nodes = " ".join(str(j*(nx+1)+i+1) for j in [ny] for i in range(nx+1))
        left_nodes = " ".join(str(j*(nx+1)+1) for j in range(ny+1))
        right_nodes = " ".join(str(j*(nx+1)+nx+1) for j in range(ny+1))
        
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
        f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
        f.write('    </bc>\n')
        
        # Top boundary (y=1) - Dirichlet u=0
        f.write('    <bc name="fix_top" type="zero displacement" node_set="top">\n')
        f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
        f.write('    </bc>\n')
        
        # Left boundary (x=0) - Dirichlet u=0
        f.write('    <bc name="fix_left" type="zero displacement" node_set="left">\n')
        f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
        f.write('    </bc>\n')
        
        # Right boundary (interface at x=0.625)
        if dirichlet_disp is not None:
            # Apply prescribed displacement from coupling
            # For now, use zero displacement
            f.write('    <bc name="fix_right" type="zero displacement" node_set="right">\n')
            f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
        else:
            f.write('    <bc name="fix_right" type="zero displacement" node_set="right">\n')
            f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
        
        f.write('  </Boundary>\n')
        
        # Body force using non-const type with expressions
        f.write('  <Loads>\n')
        f.write('    <body_load type="non-const">\n')
        f.write('      <x>x^2*y^3/50 + 6*x^2*y^2/125 - 33*x^2*y/500 + 3*x^2/250 - x*y^3/50 - 6*x*y^2/125 + 33*x*y/500 - 3*x/250 + y^5/125 + 4*y^4/125 - 89*y^3/500 - 21*y^2/125 + 297*y/1000 - 27/500</x>\n')
        f.write('      <y>-3*x^2*y^2/100 - 6*x^2*y/125 + 33*x^2/1000 + 3*x*y^4/100 + 12*x*y^3/125 - 279*x*y^2/500 - 63*x*y/125 + 99*x/250 - y^4/200 - 2*y^3/125 + 33*y^2/1000 - 3*y/250</y>\n')
        f.write('      <z>0</z>\n')
        f.write('    </body_load>\n')
        f.write('  </Loads>\n')
        
        f.write('  <Output>\n')
        f.write('    <logfile>\n')
        f.write(f'      <node_data data="x;y;z;ux;uy;uz" file="nodal_displacements_A_iter{iter_num}.txt"/>\n')
        f.write('    </logfile>\n')
        f.write('  </Output>\n')
        f.write('</febio_spec>\n')
    
    return filename, nx, ny

if __name__ == "__main__":
    h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.125
    iter_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    filename, nx, ny = write_febio_deck(h, iter_num)
    print(f"Generated {filename} with nx={nx}, ny={ny}")
