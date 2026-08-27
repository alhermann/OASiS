"""FEniCSx (dolfinx) participant for subdomain B (Neumann side) - OASiS couple driver.

Subdomain B: Rectangle (0.5, 1) x (0, 0.5) with uniform k = 5/2 = 2.5

Interface (two legs from B's perspective):
- Leg 1: x = 0.5, 0 < y < 0.5 (vertical) - normal from B is (-1, 0) [points left]
- Leg 2: y = 0.5, 0.5 < x < 1 (horizontal) - normal from B is (0, -1) [points down]

Outer BC: u=0 on all outer boundaries (x=1, y=0)
Note: The interface corners (0.5, 0), (0.5, 0.5), (1, 0.5) are treated as:
- (0.5, 0): meets outer boundary y=0 -> u=0 applies
- (1, 0.5): meets outer boundary x=1 -> u=0 applies  
- (0.5, 0.5): interior corner where two interface legs meet -> part of interface
"""
import json
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# ── CONFIGURATION ───────────────────────────────────────────────────────────
SIDE      = "neumann"     # B is Neumann side
PARTNER   = "side_A"      # partner name in couple() call

# Read mesh level from file if present
config_path = Path("level_config.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    NX = config.get("nx", 8)
else:
    NX = 8  # default

# Subdomain B geometry: (0.5, 1) x (0, 0.5)
X0, X1    = 0.5, 1.0
Y0, Y1    = 0.0, 0.5

# Conductivity in subdomain B (uniform)
K         = 2.5  # 5/2


def F_SRC(x, y):
    """Source term in subdomain B: (0.5, 1) x (0, 0.5).
    
    From problem statement, this region has k = 5/2 and source:
    24*x**4*y/125 - 11*x**4/125 + 48*x**3*y**2/25 - 51*x**3*y/125 - 67*x**3/250 
    + 48*x**2*y**3/125 - 306*x**2*y**2/125 + 36*x**2*y/125 + 811*x**2/2000 
    + 24*x*y**4/25 - 51*x*y**3/125 - 471*x*y**2/250 + 657*x*y/1000 
    + 603*x/4000 - 8*y**4/25 + 9*y**3/250 + 2971*y**2/2000 - 2487*y/8000 - 801/4000
    """
    return (24*x**4*y/125 - 11*x**4/125 + 48*x**3*y**2/25 - 51*x**3*y/125 - 67*x**3/250 
            + 48*x**2*y**3/125 - 306*x**2*y**2/125 + 36*x**2*y/125 + 811*x**2/2000 
            + 24*x*y**4/25 - 51*x*y**3/125 - 471*x*y**2/250 + 657*x*y/1000 
            + 603*x/4000 - 8*y**4/25 + 9*y**3/250 + 2971*y**2/2000 - 2487*y/8000 - 801/4000)


T_OUTER   = 0.0           # Dirichlet value on outer boundary
NX_B, NY_B = NX // 2, NX // 2  # Mesh resolution for subdomain B (half the width)
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux

# Tolerance
TOL = 1e-9

# ── END CONFIGURATION ───────────────────────────────────────────────────────


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, coords):
    """Map the partner's samples onto THIS participant's interface points."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(coords[0]), float(fallback))
    
    ys_partner = np.array([c[1] for c in imp["coordinates"]], float)
    xs_partner = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    
    if vs.size != len(ys_partner):
        return np.full(len(coords[0]), float(fallback))
    
    # Sort by coordinate for interpolation
    o = np.argsort(xs_partner)
    xs_sorted = xs_partner[o]
    ys_sorted = ys_partner[o]
    vs_sorted = vs[o]
    
    result = np.zeros(len(coords[0]))
    for i, (xx, yy) in enumerate(zip(coords[0], coords[1])):
        # Determine which leg this point is on
        if abs(xx - 0.5) < 1e-6:  # Vertical leg (x=0.5)
            result[i] = np.interp(yy, ys_sorted, vs_sorted)
        else:  # Horizontal leg (y=0.5)
            result[i] = np.interp(xx, xs_sorted, vs_sorted)
    
    return result


# Build mesh for subdomain B: rectangle (0.5, 1) x (0, 0.5)
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX_B, NY_B], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

xy = V.tabulate_dof_coordinates()

# Identify interface nodes (two legs from B's perspective)
# Leg 1: x = 0.5, 0 < y < 0.5 (vertical) - EXCLUDE corners (0.5, 0) and (0.5, 0.5)
# Leg 2: y = 0.5, 0.5 < x < 1 (horizontal) - EXCLUDE corners (0.5, 0.5) and (1, 0.5)

iface_leg1 = np.where((np.abs(xy[:, 0] - 0.5) < TOL) & 
                      (xy[:, 1] > TOL) & (xy[:, 1] < 0.5 - TOL))[0]
iface_leg2 = np.where((np.abs(xy[:, 1] - 0.5) < TOL) & 
                      (xy[:, 0] > 0.5 + TOL) & (xy[:, 0] < 1.0 - TOL))[0]

# Sort each leg
iface_leg1 = iface_leg1[np.argsort(xy[iface_leg1, 1])]
iface_leg2 = iface_leg2[np.argsort(xy[iface_leg2, 0])]

# Combine: leg 1 first, then leg 2
iface_dofs = np.concatenate([iface_leg1, iface_leg2])
iface_x = xy[iface_dofs, 0]
iface_y = xy[iface_dofs, 1]

print(f"[fenics B] NDOF = {V.dofmap.index_map.size_global}, interface dofs = {len(iface_dofs)}")

# Write run log
with open("run.log", "w") as f:
    f.write(f"NDOF = {V.dofmap.index_map.size_global}\n")
    f.write(f"interface_dofs = {len(iface_dofs)}\n")

# Identify outer boundary DOFs (u=0)
# Outer boundary: x=1, y=0
outer_facets_x1 = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], 1.0))
outer_facets_y0 = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[1], 0.0))
outer_facets = np.concatenate([outer_facets_x1, outer_facets_y0])
outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)

# Interface facets for Neumann BC application
facets_if_leg1 = dmesh.locate_entities_boundary(domain, fdim, 
                                                 lambda x: np.isclose(x[0], 0.5))
facets_if_leg2 = dmesh.locate_entities_boundary(domain, fdim, 
                                                 lambda x: np.isclose(x[1], 0.5))
facets_if = np.concatenate([facets_if_leg1, facets_if_leg2])

# Tag interface facets
tags_if = dmesh.meshtags(domain, fdim, np.sort(facets_if),
                         np.full(len(facets_if), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags_if)(7)

# Forms
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
k_const = fem.Constant(domain, default_scalar_type(K))
a = k_const * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Source term
f_src = fem.Function(V)
f_src.interpolate(lambda X: F_SRC(X[0], X[1]))
L = f_src * v * ufl.dx

# Outer Dirichlet BC
bcs = [fem.dirichletbc(default_scalar_type(T_OUTER), outer_dofs, V)]

# Read imports
imp = read_imports()

if SIDE == "dirichlet":
    g = fem.Function(V)
    g.x.array[iface_dofs] = sample(imp, "values", T_INIT, (iface_x, iface_y))
    bcs.append(fem.dirichletbc(g, iface_dofs))
else:
    # Neumann side: import flux from partner and apply it
    q_if = sample(imp, "normal_fluxes", Q_INIT, (iface_x, iface_y))
    g = fem.Function(V)
    g.x.array[iface_dofs] = q_if
    # Apply flux as Neumann BC: add integral(g * v) ds_interface
    # Note: The partner exports flux with respect to THEIR outward normal.
    # For conservation, we apply the SAME number (sign convention handles itself).
    L += g * v * ds_if

# Solve
uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="cpl",
                   petsc_options={"ksp_type": "preonly", "pc_type": "lu"}).solve()

# Export interface data
# Values: temperature at interface
T_export = uh.x.array[iface_dofs]

# Flux: For Neumann side, we need to compute the flux
# q_out = -(k grad u) . n_out where n_out is outward normal from subdomain B
# On leg 1 (x=0.5): n_out = (-1, 0) [points left, out of B]
# On leg 2 (y=0.5): n_out = (0, -1) [points down, out of B]

# Use L2 projection to recover flux
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)

# For leg 1 (x=0.5): normal is (-1, 0), so flux = -k * (-du/dx) = k * du/dx
# For leg 2 (y=0.5): normal is (0, -1), so flux = -k * (-du/dy) = k * du/dy

# We'll compute the gradient and project
grad_u = ufl.grad(uh)

# Project flux components
qx_proj = LinearProblem(p_ * w_ * ufl.dx, 
                        k_const * grad_u[0] * w_ * ufl.dx,
                        petsc_options_prefix="flx_x",
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}).solve()
qy_proj = LinearProblem(p_ * w_ * ufl.dx, 
                        k_const * grad_u[1] * w_ * ufl.dx,
                        petsc_options_prefix="flx_y",
                        petsc_options={"ksp_type": "preonly", "pc_type": "lu"}).solve()

# Extract flux at interface points
# On leg 1: q_out = k * du/dx (since n = (-1, 0))
# On leg 2: q_out = k * du/dy (since n = (0, -1))
Q = np.zeros(len(iface_dofs))
for i, (leg_idx, dof_idx) in enumerate(zip(range(len(iface_dofs)), iface_dofs)):
    if i < len(iface_leg1):  # Leg 1 (vertical, x=0.5)
        # Normal is (-1, 0), flux = -(-k*du/dx) = k*du/dx
        Q[i] = qx_proj.x.array[dof_idx]
    else:  # Leg 2 (horizontal, y=0.5)
        # Normal is (0, -1), flux = -(-k*du/dy) = k*du/dy
        Q[i] = qy_proj.x.array[dof_idx]

print(f"[fenics B] Exported {len(iface_dofs)} interface points")
print(f"[fenics B] T_range = [{T_export.min():.6g}, {T_export.max():.6g}]")
print(f"[fenics B] q_range = [{Q.min():.6g}, {Q.max():.6g}]")

# Write exports.json LAST
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(iface_x[i]), float(iface_y[i])] for i in range(len(iface_dofs))],
    "values": [float(t) for t in T_export],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
