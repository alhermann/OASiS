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
nx = int(os.environ.get('NX', 8))
h = 1.0 / nx

print(f"=== Participant A (skfem) ===", file=sys.stderr)
print(f"Work dir: {work_dir}", file=sys.stderr)
print(f"Mesh resolution: nx={nx}, h={h}", file=sys.stderr)

# Material coefficients k in different regions - vectorized version
def get_k_vec(x, y):
    """Return material coefficient k at points (x, y) - x and y are arrays."""
    k_vals = np.ones_like(x)
    mask1 = (x < 0.5) & (y < 0.5)
    k_vals[mask1] = 1.0
    mask3 = (x >= 0.5) & (y >= 0.5)
    k_vals[mask3] = 5.0
    mask4 = (x < 0.5) & (y >= 0.5)
    k_vals[mask4] = 2.0
    return k_vals

def get_f_vec(x, y):
    """Return source term f at points (x, y) - x and y are arrays."""
    f_vals = np.zeros_like(x)
    
    mask1 = (x < 0.5) & (y < 0.5)
    f_vals[mask1] = (3*x[mask1]**4*y[mask1] - 11*x[mask1]**4/8 + 12*x[mask1]**3*y[mask1]**2 
                     - 123*x[mask1]**3*y[mask1]/20 - x[mask1]**3/40 
                     + 6*x[mask1]**2*y[mask1]**3 - 477*x[mask1]**2*y[mask1]**2/20 
                     + 2799*x[mask1]**2*y[mask1]/400 + 1423*x[mask1]**2/800 
                     + 6*x[mask1]*y[mask1]**4 - 123*x[mask1]*y[mask1]**3/20 
                     + 993*x[mask1]*y[mask1]**2/200 + 63*x[mask1]*y[mask1]/1600 
                     - 609*x[mask1]/800 - 13*y[mask1]**4/5 + 279*y[mask1]**3/200 
                     + 1423*y[mask1]**2/800 - 327*y[mask1]/320)
    
    mask4 = (x < 0.5) & (y >= 0.5)
    f_vals[mask4] = (3*x[mask4]**4*y[mask4]/4 - 5*x[mask4]**4/16 + 3*x[mask4]**3*y[mask4]**2/2 
                     - 3*x[mask4]**3*y[mask4]/80 - 13*x[mask4]**3/32 
                     + 3*x[mask4]**2*y[mask4]**3/2 - 153*x[mask4]**2*y[mask4]**2/40 
                     - 873*x[mask4]**2*y[mask4]/800 + 119*x[mask4]**2/80 
                     + 3*x[mask4]*y[mask4]**4/4 - 3*x[mask4]*y[mask4]**3/80 
                     - 471*x[mask4]*y[mask4]**2/800 + 9*x[mask4]*y[mask4]/16 
                     - 3*x[mask4]/800 - 13*y[mask4]**4/40 - 241*y[mask4]**3/800 
                     + 37*y[mask4]**2/40 - 107*y[mask4]/3200 - 849/3200)
    
    mask3 = (x >= 0.5) & (y >= 0.5)
    f_vals[mask3] = (6*x[mask3]**4*y[mask3]/125 - x[mask3]**4/50 + 6*x[mask3]**3*y[mask3]**2/25 
                     + 69*x[mask3]**3*y[mask3]/500 - x[mask3]**3/8 
                     + 12*x[mask3]**2*y[mask3]**3/125 - 9*x[mask3]**2*y[mask3]**2/25 
                     - 9*x[mask3]**2*y[mask3]/40 + 769*x[mask3]**2/4000 
                     + 3*x[mask3]*y[mask3]**4/25 + 69*x[mask3]*y[mask3]**3/500 
                     - 51*x[mask3]*y[mask3]**2/100 - 549*x[mask3]*y[mask3]/8000 
                     + 2853*x[mask3]/16000 - y[mask3]**4/25 - 71*y[mask3]**3/1000 
                     + 233*y[mask3]**2/800 + 419*y[mask3]/4000 - 2031/16000)
    
    return f_vals

def get_k(x, y):
    return get_k_vec(np.array([x]), np.array([y]))[0]

def get_f(x, y):
    return get_f_vec(np.array([x]), np.array([y]))[0]

# Generate mesh for subdomain A
p_x = np.linspace(0, 1, nx + 1)
p_y = np.linspace(0, 1, nx + 1)

elements = []
node_map = {}
node_idx = 0

for i in range(nx):
    for j in range(nx):
        xc = (p_x[i] + p_x[i+1]) / 2
        yc = (p_y[j] + p_y[j+1]) / 2
        
        in_B = (xc > 0.5 and yc < 0.5)
        in_notch = (xc > 0.75 and yc > 0.75)
        
        if not in_B and not in_notch:
            for ii, jj in [(i, j), (i+1, j), (i+1, j+1), (i, j+1)]:
                key = (ii, jj)
                if key not in node_map:
                    node_map[key] = node_idx
                    node_idx += 1
            
            elem_nodes = [node_map[(i, j)], node_map[(i+1, j)], 
                         node_map[(i+1, j+1)], node_map[(i, j+1)]]
            elements.append(elem_nodes)

coords = np.zeros((2, node_idx))
for (ii, jj), idx in node_map.items():
    coords[0, idx] = p_x[ii]
    coords[1, idx] = p_y[jj]

t = np.array(elements).T
m = MeshQuad(coords, t)

print(f"Subdomain A mesh: {m.nelements} elements, {m.p.shape[1]} nodes", file=sys.stderr)

# Define boundaries
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

# Assembly with piecewise constant k
@BilinearForm
def stiffness(u, v, w):
    x_coords = w['x'][0, :]
    y_coords = w['x'][1, :]
    k_vals = get_k_vec(x_coords, y_coords)
    return k_vals * dot(grad(u), grad(v))

@LinearForm
def load(v, w):
    x_coords = w['x'][0, :]
    y_coords = w['x'][1, :]
    f_vals = get_f_vec(x_coords, y_coords)
    return f_vals * v

K = stiffness.assemble(basis)
f_vec = load.assemble(basis)

# Read imports from partner
imports_path = os.path.join(work_dir, 'imports.json')
if os.path.exists(imports_path):
    with open(imports_path, 'r') as fp:
        imports = json.load(fp)
    has_imports = True
else:
    imports = {}
    has_imports = False

# Get interface u values from imports
if has_imports and 'B' in imports and 'values' in imports['B']:
    imported_u = np.array(imports['B']['values'])
    imported_coords = np.array(imports['B']['coordinates'])
    print(f"Received {len(imported_u)} interface values from B", file=sys.stderr)
else:
    imported_u = None
    imported_coords = None
    print("No imports received, using u=0 on interface", file=sys.stderr)

# Identify interface DOFs
D_outer = basis.get_dofs('outer').flatten()
D_interface_v = basis.get_dofs('interface_v').flatten()
D_interface_h = basis.get_dofs('interface_h').flatten()
D_interface = np.unique(np.concatenate([D_interface_v, D_interface_h]))

# Create BC vector
g = np.zeros(K.shape[0])
if imported_u is not None:
    interface_node_coords = m.p[:, D_interface].T
    
    for i, coord in enumerate(interface_node_coords):
        dists = np.sqrt(np.sum((imported_coords - coord)**2, axis=1))
        nearest = np.argmin(dists)
        g[D_interface[i]] = imported_u[nearest]

D_all = np.unique(np.concatenate([D_outer, D_interface]))

# Solve
u = solve(*condense(K, f_vec, D=D_all, x=g))

print(f"Solved subdomain A: max(u)={u.max():.6e}, min(u)={u.min():.6e}", file=sys.stderr)

# Export interface data at probe points
export_coords = []
export_values = []
export_fluxes = []

# For P1 elements, we can compute gradient analytically per element
# Create a gradient basis for evaluating grad(u) at arbitrary points
grad_basis = InteriorBasis(m, ElementQuad1(), derivative=1)

# Compute gradient field as a function
# For each element, the gradient is constant
# We'll evaluate it by finding which element contains each point

def eval_grad_at_point(x, y):
    """Evaluate gradient of u at point (x, y) using finite difference."""
    h_fd = 1e-6
    u_center = basis.interpolate(u)(np.array([[x], [y]])).flatten()[0]
    u_right = basis.interpolate(u)(np.array([[x + h_fd], [y]])).flatten()[0]
    u_up = basis.interpolate(u)(np.array([[x], [y + h_fd]])).flatten()[0]
    du_dx = (u_right - u_center) / h_fd
    du_dy = (u_up - u_center) / h_fd
    return np.array([du_dx, du_dy])

# Leg 1: vertical segment at x=0.5
for i in range(44):
    x = 0.5
    y = 1/8 + (i + 0.5) * (1/4) / 44
    export_coords.append([x, y])
    
    u_val = basis.interpolate(u)(np.array([[x], [y]])).flatten()[0]
    export_values.append(u_val)
    
    grad_u = eval_grad_at_point(x, y)
    k_val = get_k(x, y)
    
    # Normal is (+1, 0) for vertical leg (outward from A)
    qn = -k_val * grad_u[0]
    export_fluxes.append(qn)

# Leg 2: horizontal segment at y=0.5
for i in range(44):
    x = 5/8 + (i + 0.5) * (1/4) / 44
    y = 0.5
    export_coords.append([x, y])
    
    u_val = basis.interpolate(u)(np.array([[x], [y]])).flatten()[0]
    export_values.append(u_val)
    
    grad_u = eval_grad_at_point(x, y)
    k_val = get_k(x, y)
    
    # Normal is (0, +1) for horizontal leg (outward from A)
    qn = -k_val * grad_u[1]
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
print(f"Flux stats: min={export_fluxes.min():.6e}, max={export_fluxes.max():.6e}", file=sys.stderr)

# Write run log
log_path = os.path.join(work_dir, 'run.log')
with open(log_path, 'w') as fp:
    fp.write(f"NDOF = {K.shape[0]}\n")
    fp.write(f"NELEMENTS = {m.nelements}\n")
    fp.write(f"MESH_SIZE = {nx}\n")

print("Participant A completed successfully", file=sys.stderr)
sys.exit(0)
