#!/usr/bin/env python3
"""Generate meshes for subdomains A and B."""

import numpy as np

def generate_subdomain_A_mesh(h):
    """Generate mesh for subdomain A: (0, 0.625) x (0, 1)"""
    nx = int(round(0.625 / h))
    ny = int(round(1.0 / h))
    
    # Nodes
    nodes = []
    node_idx = {}
    idx = 0
    for j in range(ny + 1):
        y = j * h
        for i in range(nx + 1):
            x = i * h
            if abs(x - 0.625) < 1e-10:
                x = 0.625
            if abs(y - 1.0) < 1e-10:
                y = 1.0
            nodes.append((x, y))
            node_idx[(i, j)] = idx
            idx += 1
    
    # Elements (quadrilaterals split into triangles for P1)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = node_idx[(i, j)]
            n1 = node_idx[(i+1, j)]
            n2 = node_idx[(i+1, j+1)]
            n3 = node_idx[(i, j+1)]
            # Split into two triangles
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    
    return nodes, elements, nx, ny

def generate_subdomain_B_mesh(h):
    """Generate mesh for subdomain B: (0.625, 1.5) x (0, 1)"""
    width = 1.5 - 0.625  # 0.875
    nx = int(round(width / h))
    ny = int(round(1.0 / h))
    
    # Adjust h to fit exactly
    hx = width / nx
    hy = 1.0 / ny
    
    # Nodes
    nodes = []
    node_idx = {}
    idx = 0
    for j in range(ny + 1):
        y = j * hy
        if abs(y - 1.0) < 1e-10:
            y = 1.0
        for i in range(nx + 1):
            x = 0.625 + i * hx
            if abs(x - 0.625) < 1e-10:
                x = 0.625
            if abs(x - 1.5) < 1e-10:
                x = 1.5
            nodes.append((x, y))
            node_idx[(i, j)] = idx
            idx += 1
    
    # Elements
    elements = []
    for j in range(ny):
        for i in range(nx):
            n0 = node_idx[(i, j)]
            n1 = node_idx[(i+1, j)]
            n2 = node_idx[(i+1, j+1)]
            n3 = node_idx[(i, j+1)]
            elements.append([n0, n1, n2])
            elements.append([n0, n2, n3])
    
    return nodes, elements, nx, ny

def write_febio_mesh(filename, nodes, elements, lambda_val, mu_val):
    """Write FEBio .feb file for linear elasticity."""
    with open(filename, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<FEBioModel>\n')
        f.write('  <Analysis type="static">\n')
        f.write('    <Control>\n')
        f.write('      <Step size="1" num="1"/>\n')
        f.write('    </Control>\n')
        f.write('  </Analysis>\n')
        f.write('  <Mesh>\n')
        f.write('    <Node>\n')
        for n in nodes:
            f.write(f'      <n id="{nodes.index(n)}" x="{n[0]:.15e}" y="{n[1]:.15e}" z="0"/>\n')
        f.write('    </Node>\n')
        f.write('    <Element>\n')
        for e in elements:
            f.write(f'      <e type="tri6" mat="1" n="{e[0]} {e[1]} {e[2]}"/>\n')
        f.write('    </Element>\n')
        f.write('  </Mesh>\n')
        f.write('  <Material>\n')
        f.write('    <Solid>\n')
        f.write('      <LinearElastic shear_modulus="{:.15e}".lambda="{:.15e}"/>'.format(mu_val, lambda_val))
        f.write('    </Solid>\n')
        f.write('  </Material>\n')
        f.write('</FEBioModel>\n')

def write_dealii_mesh(filename, nodes, elements):
    """Write simple mesh format for deal.II."""
    with open(filename, 'w') as f:
        f.write(f"{len(nodes)}\n")
        for n in nodes:
            f.write(f"{n[0]:.15e} {n[1]:.15e}\n")
        f.write(f"{len(elements)}\n")
        for e in elements:
            f.write(f"{e[0]} {e[1]} {e[2]}\n")

if __name__ == "__main__":
    import sys
    level = int(sys.argv[1])
    h = 1.0 / (8 * level)
    
    print(f"Level {level}, h = {h}")
    
    # Generate meshes
    nodes_A, elems_A, nx_A, ny_A = generate_subdomain_A_mesh(h)
    nodes_B, elems_B, nx_B, ny_B = generate_subdomain_B_mesh(h)
    
    print(f"Subdomain A: {len(nodes_A)} nodes, {len(elems_A)} elements")
    print(f"Subdomain B: {len(nodes_B)} nodes, {len(elems_B)} elements")
    
    # Write meshes
    write_febio_mesh(f"mesh_A_level{level}.feb", nodes_A, elems_A, 500, 250)
    write_dealii_mesh(f"mesh_B_level{level}.txt", nodes_B, elems_B)
    
    # Save node coordinates for later use
    np.save(f"nodes_A_level{level}.npy", np.array(nodes_A))
    np.save(f"nodes_B_level{level}.npy", np.array(nodes_B))
    np.save(f"elems_A_level{level}.npy", np.array(elems_A))
    np.save(f"elems_B_level{level}.npy", np.array(elems_B))
