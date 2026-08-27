#!/usr/bin/env python3
"""Test mesh generation for subdomain A."""
import numpy as np
from skfem import MeshQuad, ElementQuad1, Basis

nx = 8

# Create full unit square mesh points
p_x = np.linspace(0, 1, nx + 1)
p_y = np.linspace(0, 1, nx + 1)

# Build element connectivity for quads
elements = []
node_map = {}
node_idx = 0

for i in range(nx):
    for j in range(nx):
        # Element center
        xc = (p_x[i] + p_x[i+1]) / 2
        yc = (p_y[j] + p_y[j+1]) / 2
        
        # Exclude if in subdomain B: x > 0.5 and y < 0.5
        in_B = (xc > 0.5 and yc < 0.5)
        # Exclude if in notch: x > 0.75 and y > 0.75
        in_notch = (xc > 0.75 and yc > 0.75)
        
        if not in_B and not in_notch:
            # Add nodes if not already present
            for ii, jj in [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]:
                key = (ii, jj)
                if key not in node_map:
                    node_map[key] = node_idx
                    node_idx += 1
            
            elem_nodes = [node_map[(i, j)], node_map[(i+1, j)], 
                         node_map[(i+1, j+1)], node_map[(i, j+1)]]
            elements.append(elem_nodes)

# Create coordinate array - shape should be (ndim, nvertices)
coords = np.zeros((2, node_idx))
for (ii, jj), idx in node_map.items():
    coords[0, idx] = p_x[ii]
    coords[1, idx] = p_y[jj]

print(f"Nodes: {node_idx}, Elements: {len(elements)}")

# Create skfem mesh
t = np.array(elements).T  # Shape: (4, n_elements)
m = MeshQuad(coords, t)
print(f"Mesh created: {m.nelements} elements, {m.p.shape[1]} nodes")

# Test boundaries - use & with proper parentheses
def boundary_left(x):
    return np.abs(x[0] - 0.0) < 1e-10

def boundary_right(x):
    return (np.abs(x[0] - 1.0) < 1e-10) & (x[1] >= 0.5 - 1e-10)

def boundary_bottom(x):
    return (np.abs(x[1] - 0.0) < 1e-10) & (x[0] <= 0.5 + 1e-10)

def boundary_top(x):
    return (np.abs(x[1] - 1.0) < 1e-10) & (x[0] <= 0.5 + 1e-10)

def boundary_interface_v(x):
    return (np.abs(x[0] - 0.5) < 1e-10) & (x[1] > 1e-10) & (x[1] < 0.5 - 1e-10)

def boundary_interface_h(x):
    return (np.abs(x[1] - 0.5) < 1e-10) & (x[0] > 0.5 + 1e-10) & (x[0] < 1.0 - 1e-10)

def boundary_notch_v(x):
    return (np.abs(x[0] - 0.75) < 1e-10) & (x[1] > 0.75 - 1e-10)

def boundary_notch_h(x):
    return (np.abs(x[1] - 0.75) < 1e-10) & (x[0] > 0.75 - 1e-10)

m = m.with_boundaries({
    'outer': lambda x: (boundary_left(x) | boundary_right(x) | 
                        boundary_bottom(x) | boundary_top(x) |
                        boundary_notch_v(x) | boundary_notch_h(x)),
    'interface_v': boundary_interface_v,
    'interface_h': boundary_interface_h,
})

e = ElementQuad1()
basis = Basis(m, e)

D_outer = basis.get_dofs('outer').flatten()
D_interface_v = basis.get_dofs('interface_v').flatten()
D_interface_h = basis.get_dofs('interface_h').flatten()

print(f"Outer boundary DOFs: {len(D_outer)}")
print(f"Interface vertical DOFs: {len(D_interface_v)}")
print(f"Interface horizontal DOFs: {len(D_interface_h)}")
print(f"Total interface DOFs: {len(np.unique(np.concatenate([D_interface_v, D_interface_h])))}")
