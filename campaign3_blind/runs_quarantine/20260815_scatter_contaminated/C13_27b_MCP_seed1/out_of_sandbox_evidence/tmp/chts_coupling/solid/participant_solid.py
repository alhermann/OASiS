"""FEniCSx (dolfinx) participant for conjugate heat transfer coupling.

This is the SOLID side (Neumann-type participant):
- IMPORTS: heat flux from gas at interface (x = 0.001 m)
- EXPORTS: interface temperature to gas

Physics: steady heat conduction -div(ks grad T) = 0
Domain: rectangle (0, 0.001) x (0, 0.0006) meters
BCs:
  - x = 0: T = 400 K (Dirichlet)
  - y = 0, y = 0.0006: insulated (natural, no BC needed)
  - x = 0.001: Neumann BC from imported flux (heat INFLOW from gas)

Sign convention: imported flux is positive when energy flows INTO the solid
through the interface (from gas to solid). This matches the outward normal
at x=0.001 which points in +x direction.
"""
import json
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# Problem parameters
PARTNER   = "gas"       # partner participant name
X0, X1    = 0.0, 0.001  # solid domain x-extent [m]
Y0, Y1    = 0.0, 0.0006 # solid domain y-extent [m]
IFACE_X   = 0.001       # interface location (right boundary of solid)
KS        = 0.02        # thermal conductivity W/(m K)
T_LEFT    = 400.0       # Dirichlet temperature at x=0 [K]
NX, NY    = 40, 24      # mesh divisions (meets minimum requirements)
Q_INIT    = 0.0         # iteration-1 fallback flux [W/m^2]

# Derived
OUTER_X = X0  # the non-interface x-boundary (left side)


def read_imports():
    """Read imports.json; return None if not present or empty."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text() or "{}")
        return data.get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample_flux(imp, fallback, y_coords):
    """Interpolate partner's flux values onto our interface nodes."""
    n = len(y_coords)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    qs = np.asarray(imp.get("normal_fluxes", []), float).ravel()
    
    if qs.size != ys.size or len(ys) < 2:
        return np.full(n, float(fallback))
    
    # Sort by y and interpolate
    o = np.argsort(ys)
    return np.interp(y_coords, ys[o], qs[o])


# Create mesh
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.quadrilateral)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

# Get DOF coordinates and identify interface DOFs
xy = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-12)[0]
# Sort by y-coordinate for consistent ordering
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]
y_if = xy[iface_dofs, 1]

if len(iface_dofs) == 0:
    sys.exit(f"ERROR: No interface DOFs found at x={IFACE_X}")

# Write NDOF to log file
ndof = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
with open("run_level1_A.log", "w") as f:
    f.write(f"NDOF = {ndof}\n")
    f.write(f"Interface DOFs: {len(iface_dofs)}\n")
    f.write(f"Mesh: {NX}x{NY} quadrilaterals\n")

# Weak form: -div(ks grad T) = 0  =>  a(u,v) = ks*grad(u)*grad(v)*dx
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a_form = fem.Constant(domain, default_scalar_type(KS)) * \
         ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L_form = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx  # no source term

# Left boundary Dirichlet BC (x = 0)
left_facets = dmesh.locate_entities_boundary(domain, fdim,
                                              lambda x: np.isclose(x[0], OUTER_X))
left_dofs = fem.locate_dofs_topological(V, fdim, left_facets)
bc_left = fem.dirichletbc(default_scalar_type(T_LEFT), left_dofs, V)

# Interface measure for Neumann BC
iface_facets = dmesh.locate_entities_boundary(domain, fdim,
                                               lambda x: np.isclose(x[0], IFACE_X))
tags_iface = dmesh.meshtags(domain, fdim, np.sort(iface_facets),
                            np.full(len(iface_facets), 1, dtype=np.int32))
ds_iface = ufl.Measure("ds", domain=domain, subdomain_data=tags_iface)(1)

# Read imports and apply Neumann BC
imp = read_imports()
q_imported = sample_flux(imp, Q_INIT, y_if)

# Create function for Neumann BC values
q_func = fem.Function(V)
q_func.x.array[iface_dofs] = q_imported

# Add Neumann term to L: integral of q*v on interface
# Sign: positive q means heat flowing INTO domain through interface
# Outward normal at x=IFACE_X is +x, so q*n = q (positive = inflow)
L_form += q_func * v * ds_iface

# Solve
problem = LinearProblem(a_form, L_form, bcs=[bc_left],
                        petsc_options_prefix="solid_cpl",
                        petsc_options={"ksp_type": "preonly",
                                       "pc_type": "lu"})
T_h = problem.solve()

# Extract interface temperature
T_iface = T_h.x.array[iface_dofs]

print(f"[solid] NDOF={ndof}, interface_n={len(T_iface)}, "
      f"T=[{T_iface.min():.4f},{T_iface.max():.4f}]K, mean={T_iface.mean():.4f}K")

# Compute heat flux through solid at x=0 for verification
# Use reaction forces: the Dirichlet reaction gives us the flux
from dolfinx.fem import petsc as _fp

# Assemble residual without BCs to get reactions
Amat = _fp.assemble_matrix(fem.form(a_form))
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_form))
bvec.ghostUpdate()

r = Amat.createVecLeft()
Amat.mult(T_h.x.petsc_vec, r)
r.axpy(-1.0, bvec)

# Reaction at left boundary DOFs gives flux
left_reaction = r.array[left_dofs]

# Weight functions for left boundary
tags_left = dmesh.meshtags(domain, fdim, np.sort(left_facets),
                           np.full(len(left_facets), 2, dtype=np.int32))
ds_left = ufl.Measure("ds", domain=domain, subdomain_data=tags_left)(2)
wvec = _fp.assemble_vector(fem.form(v * ds_left))
wvec.ghostUpdate()
w_left = wvec.array[left_dofs]

flux_left = np.sum(-left_reaction[w_left > 1e-15] / w_left[w_left > 1e-15])
# Total heat entering at x=0 divided by height
flux_density_left = flux_left / (Y1 - Y0)

print(f"[solid] Heat flux density at x=0: {flux_density_left:.6f} W/m^2")

# Export interface data
# Coordinates at interface
coords = [[float(IFACE_X), float(y)] for y in y_if]

# For flux export: compute -ks*grad(T).n at interface
# Outward normal at interface is +x, so we want -ks*dT/dx
# Use L2 projection for accuracy
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)
grad_x = fem.Constant(domain, default_scalar_type(-KS)) * T_h.dx(0)
flux_proj = LinearProblem(p_ * w_ * ufl.dx, grad_x * w_ * ufl.dx,
                          petsc_options_prefix="flux_proj",
                          petsc_options={"ksp_type": "preonly",
                                         "pc_type": "lu"}).solve()
q_export = flux_proj.x.array[iface_dofs]

# Write exports.json LAST
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": coords,
    "values": [float(t) for t in T_iface],
    "normal_fluxes": [float(q) for q in q_export],
}, indent=2))

print(f"[solid] Exported T_interface, flux range [{q_export.min():.4f},{q_export.max():.4f}] W/m^2")
