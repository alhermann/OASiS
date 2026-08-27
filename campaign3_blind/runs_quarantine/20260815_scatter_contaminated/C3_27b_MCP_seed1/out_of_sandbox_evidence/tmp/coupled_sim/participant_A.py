#!/usr/bin/env python3
"""
Participant A (scikit-fem) - DIRICHLET side of coupling
Subdomain A: unit square minus (1/2,1)x(0,1/2) minus (3/4,1)x(3/4,1)
This is an L-shaped non-convex domain.

Receives: u values from subdomain B on interface
Exports: normal flux qn = -k * grad(u) . n_out on interface
"""
import json
import os
import sys
import numpy as np
from skfem import *
from skfem.models.poisson import laplace
from skfem.helpers import dot, grad

# Get work directory from environment or use current
work_dir = os.environ.get('WORK_DIR', os.getcwd())

# Mesh resolution parameter (will be set based on level)
# h = 1/nx where nx = 8, 16, 32 for levels 1, 2, 3
nx = int(os.environ.get('NX', 8))
h = 1.0 / nx

# Material coefficients k in different regions
# Region definitions:
#   R1: (0, 1/2) x (0, 1/2) -> k = 1
#   R2: (1/2, 1) x (0, 1/2) -> this is subdomain B, not part of A
#   R3: (1/2, 1) x (1/2, 1) -> k = 5
#   R4: (0, 1/2) x (1/2, 1) -> k = 2
# Notch: (3/4, 1) x (3/4, 1) -> removed from domain

def get_k(x, y):
    """Return material coefficient k at point (x, y)."""
    if x < 0.5 and y < 0.5:
        return 1.0
    elif x >= 0.5 and y >= 0.5:
        return 5.0
    elif x < 0.5 and y >= 0.5:
        return 2.0
    else:
        # Should not happen for subdomain A
        return 1.0

def get_f(x, y):
    """Return source term f at point (x, y)."""
    if x < 0.5 and y < 0.5:
        # Region 1: (0,1/2)x(0,1/2), k=1
        return (3*x**4*y - 11*x**4/8 + 12*x**3*y**2 - 123*x**3*y/20 - x**3/40 
                + 6*x**2*y**3 - 477*x**2*y**2/20 + 2799*x**2*y/400 + 1423*x**2/800 
                + 6*x*y**4 - 123*x*y**3/20 + 993*x*y**2/200 + 63*x*y/1600 - 609*x/800 
                - 13*y**4/5 + 279*y**3/200 + 1423*y**2/800 - 327*y/320)
    elif x < 0.5 and y >= 0.5:
        # Region 4: (0,1/2)x(1/2,1), k=2
        return (3*x**4*y/4 - 5*x**4/16 + 3*x**3*y**2/2 - 3*x**3*y/80 - 13*x**3/32 
                + 3*x**2*y**3/2 - 153*x**2*y**2/40 - 873*x**2*y/800 + 119*x**2/80 
                + 3*x*y**4/4 - 3*x*y**3/80 - 471*x*y**2/800 + 9*x*y/16 - 3*x/800 
                - 13*y**4/40 - 241*y**3/800 + 37*y**2/40 - 107*y/3200 - 849/3200)
    elif x >= 0.5 and y >= 0.5:
        # Region 3: (1/2,1)x(1/2,1), k=5
        return (6*x**4*y/125 - x**4/50 + 6*x**3*y**2/25 + 69*x**3*y/500 - x**3/8 
                + 12*x**2*y**3/125 - 9*x**2*y**2/25 - 9*x**2*y/40 + 769*x**2/4000 
                + 3*x*y**4/25 + 69*x*y**3/500 - 51*x*y**2/100 - 549*x*y/8000 
                + 2853*x/16000 - y**4/25 - 71*y**3/1000 + 233*y**2/800 
                + 419*y/4000 - 2031/16000)
    else:
        return 0.0

# Generate mesh for subdomain A
# Subdomain A is the unit square with two rectangular regions removed:
#   - (1/2, 1) x (0, 1/2) [subdomain B]
#   - (3/4, 1) x (3/4, 1) [notch]
# We'll create a structured quad mesh and mark elements to exclude

print(f"Creating mesh for subdomain A with nx={nx}, h={h}", file=sys.stderr)

# Create full unit square mesh
p_x = np.linspace(0, 1, nx + 1)
p_y = np.linspace(0, 1, nx + 1)

# Build element connectivity for quads
# Each quad has 4 nodes: bottom-left, bottom-right, top-right, top-left
elements = []
node_map = {}
node_idx = 0

for i in range(nx):
    for j in range(nx):
        # Check if this element is in subdomain A
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

# Create coordinate array
coords = np.zeros((2, node_idx))
for (ii, jj), idx in node_map.items():
    coords[0, idx] = p_x[ii]
    coords[1, idx] = p_y[jj]

# Create skfem mesh
from skfem import MeshQuad
m = MeshQuad(coords.T, np.array(elements).T)

print(f"Subdomain A mesh: {m.nelements} elements, {m.nox} nodes", file=sys.stderr)

# Define boundaries
# Outer boundary: x=0, x=1 (where y>=0.5 or y<=0.5 but x<=0.5), y=0 (where x<=0.5), y=1 (where x<=0.5)
# Also the notch faces: x=0.75 for y>0.75, y=0.75 for x>0.75
# Interface: x=0.5 for y<0.5, y=0.5 for x>0.5

def boundary_left(x):
    return np.abs(x[0] - 0.0) < 1e-10

def boundary_right(x):
    # Right edge x=1, but only where y >= 0.5 (the part that's in A)
    return np.abs(x[0] - 1.0) < 1e-10 & (x[1] >= 0.5 - 1e-10)

def boundary_bottom(x):
    # Bottom edge y=0, but only where x <= 0.5 (the part that's in A)
    return np.abs(x[1] - 0.0) < 1e-10 & (x[0] <= 0.5 + 1e-10)

def boundary_top(x):
    # Top edge y=1, but only where x <= 0.5 (the part that's in A)
    return np.abs(x[1] - 1.0) < 1e-10 & (x[0] <= 0.5 + 1e-10)

def boundary_interface_vertical(x):
    # Interface leg 1: x=0.5, 0 < y < 0.5
    return np.abs(x[0] - 0.5) < 1e-10 & (x[1] > 1e-10) & (x[1] < 0.5 - 1e-10)

def boundary_interface_horizontal(x):
    # Interface leg 2: y=0.5, 0.5 < x < 1
    return np.abs(x[1] - 0.5) < 1e-10 & (x[0] > 0.5 + 1e-10) & (x[0] < 1.0 - 1e-10)

def boundary_notch_vertical(x):
    # Notch face: x=0.75, y > 0.75
    return np.abs(x[0] - 0.75) < 1e-10 & (x[1] > 0.75 - 1e-10)

def boundary_notch_horizontal(x):
    # Notch face: y=0.75, x > 0.75
    return np.abs(x[1] - 0.75) < 1e-10 & (x[0] > 0.75 - 1e-10)

m = m.with_boundaries({
    'outer': lambda x: (boundary_left(x) | boundary_right(x) | 
                        boundary_bottom(x) | boundary_top(x) |
                        boundary_notch_vertical(x) | boundary_notch_horizontal(x)),
    'interface_v': boundary_interface_vertical,
    'interface_h': boundary_interface_horizontal,
})

# Create basis
e = ElementQuad1()
basis = Basis(m, e)

# Assembly with piecewise constant k
@BilinearForm
def stiffness(u, v, w):
    k_val = np.zeros(w['x'].shape[1])
    for i in range(w['x'].shape[1]):
        k_val[i] = get_k(w['x'][0, i], w['y'][0, i])
    return k_val * dot(grad(u), grad(v))

@LinearForm
def load(v, w):
    f_val = np.zeros(w['x'].shape[1])
    for i in range(w['x'].shape[1]):
        f_val[i] = get_f(w['x'][0, i], w['y'][0, i])
    return f_val * v

K = stiffness.assemble(basis)
f_vec = load.assemble(basis)

# Read imports from partner (subdomain B)
imports_path = os.path.join(work_dir, 'imports.json')
if os.path.exists(imports_path):
    with open(imports_path, 'r') as fp:
        imports = json.load(fp)
    has_imports = True
else:
    imports = {}
    has_imports = False

# Get interface u values from imports (or use initial guess of 0)
if has_imports and 'B' in imports and 'values' in imports['B']:
    imported_u = np.array(imports['B']['values'])
    imported_coords = np.array(imports['B']['coordinates'])
    print(f"Received {len(imported_u)} interface values from B", file=sys.stderr)
else:
    # Initial guess: u = 0 on interface
    imported_u = None
    imported_coords = None
    print("No imports received, using u=0 on interface", file=sys.stderr)

# Identify interface DOFs
D_outer = basis.get_dofs('outer').flatten()
D_interface_v = basis.get_dofs('interface_v').flatten()
D_interface_h = basis.get_dofs('interface_h').flatten()
D_interface = np.concatenate([D_interface_v, D_interface_h])
D_interface = np.unique(D_interface)

# All Dirichlet DOFs
if imported_u is not None:
    # Need to interpolate imported values to our interface nodes
    # First, get coordinates of our interface nodes
    interface_node_coords = m.p[:, D_interface].T
    
    # Interpolate imported values to our nodes
    # Using nearest neighbor interpolation
    u_interface = np.zeros(len(D_interface))
    for i, coord in enumerate(interface_node_coords):
        # Find nearest imported point
        dists = np.sqrt(np.sum((imported_coords - coord)**2, axis=1))
        nearest = np.argmin(dists)
        u_interface[i] = imported_u[nearest]
    
    # Create full solution vector with BC values
    g = np.zeros(K.shape[0])
    g[D_interface] = u_interface
    D_all = np.concatenate([D_outer, D_interface])
else:
    g = np.zeros(K.shape[0])
    D_all = np.concatenate([D_outer, D_interface])

# Solve with Dirichlet BCs
u = solve(*condense(K, f_vec, D=D_all, x=g))

print(f"Solved subdomain A: max(u)={u.max():.6e}, min(u)={u.min():.6e}", file=sys.stderr)

# Compute normal flux on interface: qn = -k * grad(u) . n_out
# For subdomain A:
#   On vertical leg (x=0.5): n_out = (+1, 0), so qn = -k * du/dx
#   On horizontal leg (y=0.5): n_out = (0, +1), so qn = -k * du/dy

# Evaluate gradient at interface points
# We need to compute flux at the same points we'll export

# Export interface data
# Interface probe points: 44 points on each leg
# Leg 1: (x, y) = (1/2, 1/8 + (i+0.5)*(1/4)/44) for i = 0..43
# Leg 2: (x, y) = (5/8 + (i+0.5)*(1/4)/44, 1/2) for i = 0..43

export_coords = []
export_values = []
export_fluxes = []

# Leg 1: vertical segment at x=0.5
for i in range(44):
    x = 0.5
    y = 1/8 + (i + 0.5) * (1/4) / 44
    export_coords.append([x, y])
    
    # Interpolate u at this point
    u_val = basis.interpolate(u)(np.array([[x], [y]])).flatten()[0]
    export_values.append(u_val)
    
    # Compute flux: need gradient at this point
    # For P1 elements, gradient is constant per element
    # Find which element contains this point and evaluate gradient there
    grad_u = basis.projection(grad(u))(np.array([[x], [y]])).reshape(-1, 2)[0]
    k_val = get_k(x, y)
    
    # Normal is (+1, 0) for vertical leg
    qn = -k_val * grad_u[0]  # -k * du/dx
    export_fluxes.append(qn)

# Leg 2: horizontal segment at y=0.5
for i in range(44):
    x = 5/8 + (i + 0.5) * (1/4) / 44
    y = 0.5
    export_coords.append([x, y])
    
    u_val = basis.interpolate(u)(np.array([[x], [y]])).flatten()[0]
    export_values.append(u_val)
    
    grad_u = basis.projection(grad(u))(np.array([[x], [y]])).reshape(-1, 2)[0]
    k_val = get_k(x, y)
    
    # Normal is (0, +1) for horizontal leg
    qn = -k_val * grad_u[1]  # -k * du/dy
    export_fluxes.append(qn)

export_coords = np.array(export_coords)
export_values = np.array(export_values)
export_fluxes = np.array(export_fluxes)

# Write exports
exports = {
    'field_name': 'u',
    'n_points': len(export_coords),
    'coordinates': export_coords.tolist(),
    'values': export_values.tolist(),
    'normal_fluxes': export_fluxes.tolist()
}

exports_path = os.path.join(work_dir, 'exports.json')
with open(exports_path, 'w') as fp:
    json.dump(exports, fp, indent=2)

print(f"Exported {len(export_coords)} interface points", file=sys.stderr)
print(f"Flux stats: min={export_fluxes.min():.6e}, max={export_fluxes.max():.6e}, mean={export_fluxes.mean():.6e}", file=sys.stderr)

# Write run log
log_path = os.path.join(work_dir, 'run.log')
with open(log_path, 'w') as fp:
    fp.write(f"NDOF = {K.shape[0]}\n")
    fp.write(f"NELEMENTS = {m.nelements}\n")
    fp.write(f"MESH_SIZE = {nx}\n")
    fp.write(f"MAX_U = {u.max():.10e}\n")
    fp.write(f"MIN_U = {u.min():.10e}\n")

print("Participant A completed successfully", file=sys.stderr)
sys.exit(0)
