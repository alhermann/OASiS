#!/usr/bin/env python3
"""
scikit-fem participant for subdomain A - DIRICHLET side
Subdomain A: unit square minus (1/2,1)x(0,1/2) minus (3/4,1)x(3/4,1)
L-shaped non-convex domain with bent interface
"""
import json
from pathlib import Path
import numpy as np
from skfem import (Basis, BilinearForm, ElementQuad1, FacetBasis, LinearForm,
                   MeshQuad, condense, solve)
from skfem.helpers import dot, grad

# Problem parameters
NX = int(__import__('os').environ.get('NX', 8))
PARTNER = "B"
SIDE = "dirichlet"

# Interface probe points (same for both participants)
def get_interface_points():
    """Generate 88 interface probe points: 44 on each leg."""
    points = []
    # Leg 1: vertical at x=0.5, y from 1/8 to 3/8
    for i in range(44):
        x = 0.5
        y = 1/8 + (i + 0.5) * (1/4) / 44
        points.append((x, y))
    # Leg 2: horizontal at y=0.5, x from 5/8 to 7/8
    for i in range(44):
        x = 5/8 + (i + 0.5) * (1/4) / 44
        y = 0.5
        points.append((x, y))
    return points

INTERFACE_POINTS = get_interface_points()

# Material coefficient function
def get_k(x, y):
    if x < 0.5 and y < 0.5:
        return 1.0
    elif x >= 0.5 and y >= 0.5:
        return 5.0
    elif x < 0.5 and y >= 0.5:
        return 2.0
    return 1.0

def get_f(x, y):
    if x < 0.5 and y < 0.5:
        return (3*x**4*y - 11*x**4/8 + 12*x**3*y**2 - 123*x**3*y/20 - x**3/40 
                + 6*x**2*y**3 - 477*x**2*y**2/20 + 2799*x**2*y/400 + 1423*x**2/800 
                + 6*x*y**4 - 123*x*y**3/20 + 993*x*y**2/200 + 63*x*y/1600 - 609*x/800 
                - 13*y**4/5 + 279*y**3/200 + 1423*y**2/800 - 327*y/320)
    elif x < 0.5 and y >= 0.5:
        return (3*x**4*y/4 - 5*x**4/16 + 3*x**3*y**2/2 - 3*x**3*y/80 - 13*x**3/32 
                + 3*x**2*y**3/2 - 153*x**2*y**2/40 - 873*x**2*y/800 + 119*x**2/80 
                + 3*x*y**4/4 - 3*x*y**3/80 - 471*x*y**2/800 + 9*x*y/16 - 3*x/800 
                - 13*y**4/40 - 241*y**3/800 + 37*y**2/40 - 107*y/3200 - 849/3200)
    elif x >= 0.5 and y >= 0.5:
        return (6*x**4*y/125 - x**4/50 + 6*x**3*y**2/25 + 69*x**3*y/500 - x**3/8 
                + 12*x**2*y**3/125 - 9*x**2*y**2/25 - 9*x**2*y/40 + 769*x**2/4000 
                + 3*x*y**4/25 + 69*x*y**3/500 - 51*x*y**2/100 - 549*x*y/8000 
                + 2853*x/16000 - y**4/25 - 71*y**3/1000 + 233*y**2/800 
                + 419*y/4000 - 2031/16000)
    return 0.0

# Generate mesh for subdomain A
p_x = np.linspace(0, 1, NX + 1)
p_y = np.linspace(0, 1, NX + 1)

elements = []
node_map = {}
node_idx = 0

for i in range(NX):
    for j in range(NX):
        xc = (p_x[i] + p_x[i+1]) / 2
        yc = (p_y[j] + p_y[j+1]) / 2
        
        # Exclude subdomain B and notch
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
mesh = MeshQuad(coords, t)

print(f"Subdomain A: {mesh.nelements} elements, {mesh.p.shape[1]} nodes", flush=True)

# Define boundaries with tolerance
TOL = 1e-9

def boundary_outer(x):
    left = np.abs(x[0] - 0.0) < TOL
    right = (np.abs(x[0] - 1.0) < TOL) & (x[1] >= 0.5 - TOL)
    bottom = (np.abs(x[1] - 0.0) < TOL) & (x[0] <= 0.5 + TOL)
    top = (np.abs(x[1] - 1.0) < TOL) & (x[0] <= 0.5 + TOL)
    notch_v = (np.abs(x[0] - 0.75) < TOL) & (x[1] > 0.75 - TOL)
    notch_h = (np.abs(x[1] - 0.75) < TOL) & (x[0] > 0.75 - TOL)
    return left | right | bottom | top | notch_v | notch_h

def boundary_interface(x):
    # Vertical leg: x=0.5, 0<y<0.5
    v_leg = (np.abs(x[0] - 0.5) < TOL) & (x[1] > TOL) & (x[1] < 0.5 - TOL)
    # Horizontal leg: y=0.5, 0.5<x<1
    h_leg = (np.abs(x[1] - 0.5) < TOL) & (x[0] > 0.5 + TOL) & (x[0] < 1.0 - TOL)
    return v_leg | h_leg

mesh = mesh.with_boundaries({
    'outer': boundary_outer,
    'interface': boundary_interface,
})

elem = ElementQuad1()
basis = Basis(mesh, elem)

# Get DOFs
D_outer = basis.get_dofs('outer').flatten()
D_interface = basis.get_dofs('interface').flatten()

# Assembly
@BilinearForm
def stiffness(u, v, w):
    k_vals = np.array([get_k(w['x'][0,i], w['x'][1,i]) for i in range(w['x'].shape[1])])
    return k_vals * dot(grad(u), grad(v))

@LinearForm
def source(v, w):
    f_vals = np.array([get_f(w['x'][0,i], w['x'][1,i]) for i in range(w['x'].shape[1])])
    return f_vals * v

A = stiffness.assemble(basis)
b = source.assemble(basis)

# Read imports
imp_path = Path("imports.json")
if imp_path.is_file():
    try:
        imports = json.loads(imp_path.read_text())
        imp_data = imports.get(PARTNER)
    except:
        imp_data = None
else:
    imp_data = None

# Initial guess
U_INIT = 0.0

# Apply Dirichlet BC on interface from imports
sol = basis.zeros()
if imp_data and 'values' in imp_data:
    imported_u = np.array(imp_data['values'])
    imported_coords = np.array(imp_data['coordinates'])
    
    # Interpolate to our interface nodes
    iface_node_coords = mesh.p[:, D_interface].T
    for i, coord in enumerate(iface_node_coords):
        dists = np.sqrt(np.sum((imported_coords - coord)**2, axis=1))
        nearest = np.argmin(dists)
        sol[D_interface[i]] = imported_u[nearest]
else:
    print("No imports, using u=0 on interface", flush=True)

# All Dirichlet DOFs
D_all = np.unique(np.concatenate([D_outer, D_interface]))

# Solve
u = solve(*condense(A, b, D=D_all, x=sol))

print(f"Solved A: max={u.max():.6e}, min={u.min():.6e}", flush=True)

# Export at interface probe points
export_coords = []
export_values = []
export_fluxes = []

# Finite difference gradient evaluation
h_fd = 1e-7
def eval_grad(x, y):
    u_c = basis.interpolate(u)(np.array([[x], [y]])).flatten()[0]
    u_r = basis.interpolate(u)(np.array([[x+h_fd], [y]])).flatten()[0]
    u_u = basis.interpolate(u)(np.array([[x], [y+h_fd]])).flatten()[0]
    return np.array([(u_r-u_c)/h_fd, (u_u-u_c)/h_fd])

for x, y in INTERFACE_POINTS:
    export_coords.append([x, y])
    u_val = basis.interpolate(u)(np.array([[x], [y]])).flatten()[0]
    export_values.append(float(u_val))
    
    grad_u = eval_grad(x, y)
    k_val = get_k(x, y)
    
    # Determine normal direction
    if abs(x - 0.5) < TOL:
        # Vertical leg: n = (+1, 0) outward from A
        qn = -k_val * grad_u[0]
    else:
        # Horizontal leg: n = (0, +1) outward from A
        qn = -k_val * grad_u[1]
    export_fluxes.append(float(qn))

# Write exports
exports = {
    "field_name": "u",
    "n_points": len(export_coords),
    "coordinates": export_coords,
    "values": export_values,
    "normal_fluxes": export_fluxes
}

Path("exports.json").write_text(json.dumps(exports, indent=2))

print(f"Exported {len(export_coords)} points", flush=True)

# Write run log
with open("run.log", "w") as fp:
    fp.write(f"NDOF = {A.shape[0]}\n")
    fp.write(f"NELEMENTS = {mesh.nelements}\n")
    fp.write(f"MESH_SIZE = {NX}\n")

print("Participant A completed", flush=True)
