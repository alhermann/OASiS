#!/usr/bin/env python3
"""
Participant B (FEniCSx/dolfinx) - NEUMANN side of coupling
Subdomain B: rectangle (1/2, 1) x (0, 1/2)

Receives: normal flux qn from subdomain A on interface
Exports: u values on interface
"""
import json
import os
import sys
import numpy as np

# Get work directory
work_dir = os.environ.get('WORK_DIR', os.getcwd())

# Mesh resolution parameter
nx = int(os.environ.get('NX', 8))

print(f"=== Participant B (fenics) ===", file=sys.stderr)
print(f"Work dir: {work_dir}", file=sys.stderr)
print(f"Mesh resolution: nx={nx}", file=sys.stderr)

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import meshtags
import ufl

# Create mesh for subdomain B: (0.5, 1) x (0, 0.5)
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [0.5, 0.0],
    [1.0, 0.5],
    [nx, nx],
    mesh.CellType.triangle
)

V = fem.functionspace(domain, ("Lagrange", 1))

print(f"Subdomain B mesh: {domain.topology.index_map(domain.topology.dim).size_global} elements, {V.dofmap.index_map.size_global} DOFs", file=sys.stderr)

# Material coefficient k = 5/2 in subdomain B
k = fem.Constant(domain, default_scalar_type(2.5))

# Source term f in subdomain B
x_coords = ufl.SpatialCoordinate(domain)
f_ufl = (24*x_coords[0]**4*x_coords[1]/125 - 11*x_coords[0]**4/125 + 48*x_coords[0]**3*x_coords[1]**2/25 
         - 51*x_coords[0]**3*x_coords[1]/125 - 67*x_coords[0]**3/250 
         + 48*x_coords[0]**2*x_coords[1]**3/125 - 306*x_coords[0]**2*x_coords[1]**2/125 
         + 36*x_coords[0]**2*x_coords[1]/125 + 811*x_coords[0]**2/2000 
         + 24*x_coords[0]*x_coords[1]**4/25 - 51*x_coords[0]*x_coords[1]**3/125 
         - 471*x_coords[0]*x_coords[1]**2/250 + 657*x_coords[0]*x_coords[1]/1000 
         + 603*x_coords[0]/4000 - 8*x_coords[1]**4/25 + 9*x_coords[1]**3/250 
         + 2971*x_coords[1]**2/2000 - 2487*x_coords[1]/8000 - 801/4000)

# Define boundaries
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

def boundary_left(x):
    return np.abs(x[0] - 0.5) < 1e-10

def boundary_right(x):
    return np.abs(x[0] - 1.0) < 1e-10

def boundary_bottom(x):
    return np.abs(x[1] - 0.0) < 1e-10

def boundary_top(x):
    return np.abs(x[1] - 0.5) < 1e-10

left_facets = mesh.locate_entities_boundary(domain, fdim, boundary_left)
right_facets = mesh.locate_entities_boundary(domain, fdim, boundary_right)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, boundary_bottom)
top_facets = mesh.locate_entities_boundary(domain, fdim, boundary_top)

# Outer boundary: right and bottom edges
outer_facets = np.unique(np.concatenate([right_facets, bottom_facets]))

# Interface: left and top edges  
interface_facets = np.unique(np.concatenate([left_facets, top_facets]))

# Dirichlet BC on outer boundary: u = 0
dofs_outer = fem.locate_dofs_topological(V, fdim, outer_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs_outer, V)

# Read imports from partner
imports_path = os.path.join(work_dir, 'imports.json')
if os.path.exists(imports_path):
    with open(imports_path, 'r') as fp:
        imports = json.load(fp)
    has_imports = True
else:
    imports = {}
    has_imports = False

# Get interface flux from imports
if has_imports and 'A' in imports and 'normal_fluxes' in imports['A']:
    imported_flux = np.array(imports['A']['normal_fluxes'])
    imported_coords = np.array(imports['A']['coordinates'])
    print(f"Received {len(imported_flux)} interface flux values from A", file=sys.stderr)
else:
    imported_flux = None
    imported_coords = None
    print("No imports received, using zero flux on interface", file=sys.stderr)

# Weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = k * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f_ufl * v * ufl.dx

# Add Neumann boundary term if we have imports
if imported_flux is not None:
    # Identify interface DOFs
    dofs_interface = fem.locate_dofs_topological(V, fdim, interface_facets)
    
    # Get coordinates of interface DOFs
    interface_dof_coords = domain.geometry.x[:, dofs_interface]
    
    # Interpolate imported flux to our interface DOFs
    flux_at_dofs = np.zeros(len(dofs_interface))
    for i, coord in enumerate(interface_dof_coords):
        dists = np.sqrt(np.sum((imported_coords - coord)**2, axis=1))
        nearest = np.argmin(dists)
        flux_at_dofs[i] = imported_flux[nearest]
    
    # Create a Function with these values
    g = fem.Function(V)
    g.x.array[:] = 0.0
    g.x.array[dofs_interface] = flux_at_dofs
    
    # Mark interface facets
    facet_tags = meshtags(domain, fdim, interface_facets, 1)
    ds_interface = ufl.Measure("ds", domain, facet_tags)
    
    # Add Neumann term - the flux from A is applied directly
    L += g * v * ds_interface(1)

# Solve
problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solve", 
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
uh.name = "u"

print(f"Solved subdomain B: max(u)={uh.x.array.max():.6e}, min(u)={uh.x.array.min():.6e}", file=sys.stderr)

# Export interface data at probe points
export_coords = []
export_values = []
export_fluxes = []

# Leg 1: vertical segment at x=0.5
for i in range(44):
    x = 0.5
    y = 1/8 + (i + 0.5) * (1/4) / 44
    export_coords.append([x, y])

# Leg 2: horizontal segment at y=0.5
for i in range(44):
    x = 5/8 + (i + 0.5) * (1/4) / 44
    y = 0.5
    export_coords.append([x, y])

export_coords = np.array(export_coords)

# Interpolate solution at export points
export_values = []
for coord in export_coords:
    u_val = uh.eval(coord.reshape(-1, 1))[0]
    export_values.append(float(u_val))

export_values = np.array(export_values)

# Compute normal flux for export (with respect to B's outward normal)
# For subdomain B:
#   On vertical leg (x=0.5): n_out = (-1, 0), so qn = -k * grad(u) . (-1, 0) = k * du/dx
#   On horizontal leg (y=0.5): n_out = (0, -1), so qn = -k * grad(u) . (0, -1) = k * du/dy

# Create a function to evaluate gradient
from dolfinx.fem import Expression
grad_u_expr = ufl.expr(ufl.grad(uh))

export_fluxes = []
for i, coord in enumerate(export_coords):
    x, y = coord
    # Evaluate gradient at this point
    # Find cell containing the point
    cell_finder = domain.geometry.create_entity_collapser()
    
    # Simpler approach: use the fact that for P1 elements, gradient is constant per cell
    # We'll approximate by evaluating at nearby nodes
    # For now, compute from finite difference approximation
    
    # Actually, let's compute the flux properly
    # The gradient of a P1 function can be computed element-wise
    # For simplicity, we'll use a central difference approximation
    
    h_eval = 1e-6
    u_center = uh.eval(coord.reshape(-1, 1))[0]
    
    if i < 44:
        # Vertical leg - need du/dx
        u_right = uh.eval(np.array([[x + h_eval], [y]]))[0]
        du_dx = (u_right - u_center) / h_eval
        # n_out = (-1, 0), so qn = k * du/dx
        qn = 2.5 * du_dx
    else:
        # Horizontal leg - need du/dy
        u_above = uh.eval(np.array([[x], [y + h_eval]]))[0]
        du_dy = (u_above - u_center) / h_eval
        # n_out = (0, -1), so qn = k * du/dy
        qn = 2.5 * du_dy
    
    export_fluxes.append(float(qn))

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
    fp.write(f"NDOF = {V.dofmap.index_map.size_global}\n")
    fp.write(f"NELEMENTS = {domain.topology.index_map(domain.topology.dim).size_global}\n")
    fp.write(f"MESH_SIZE = {nx}\n")

print("Participant B completed successfully", file=sys.stderr)
sys.exit(0)
