#!/home/alexander/miniconda3/envs/fenics/bin/python
"""FEniCSx participant for subdomain A (Dirichlet side) - transient heat with coupling"""
import json
import sys
from pathlib import Path
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, default_scalar_type
from dolfinx.fem import petsc as fp
from petsc4py import PETSc

# Parameters from command line or defaults
NX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 1/32

# Problem setup
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K = 1.0
RHO_CP = 1.0
T_END = 0.25
PARTNER = "B"

NY = int(round(NX * (Y1-Y0)/(X1-X0)))

def source_A(x, t):
    """Source term for subdomain A"""
    exp_t2 = np.exp(t/2.0)
    r = np.zeros_like(x[0])
    r += (-384*t*x[0]**3*x[1]**3 + 1184*t*x[0]**3*x[1]**2 + 3808*t*x[0]**3*x[1] - 4736*t*x[0]**3
          - 316*t*x[0]**2*x[1]**3 - 199*t*x[0]**2*x[1]**2 + 4307*t*x[0]**2*x[1] + 796*t*x[0]**2
          + 5148*t*x[0]*x[1]**3 - 14883*t*x[0]*x[1]**2 + 3255*t*x[0]*x[1] + 2700*t*x[0]
          + 1264*t*x[1]**3 + 796*t*x[1]**2 - 2060*t*x[1]
          - 768*x[0]**3*x[1]**3 + 2368*x[0]**3*x[1]**2 - 1600*x[0]**3*x[1]
          - 632*x[0]**2*x[1]**3 - 398*x[0]**2*x[1]**2 + 1030*x[0]**2*x[1]
          + 1080*x[0]*x[1]**3 - 1350*x[0]*x[1]**2 + 270*x[0]*x[1])
    return r * exp_t2 / 1280.0

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER)
    except:
        return None

def interp(imp, key, fallback, y):
    if imp is None or "coordinates" not in imp:
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if len(vs) != len(ys):
        return np.full(len(y), float(fallback))
    idx = np.argsort(ys)
    return np.interp(y, ys[idx], vs[idx])

# Create mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]], [NX, NY], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

# Find interface and outer boundary DOFs
xy = V.tabulate_dof_coordinates()
iface_mask = np.abs(xy[:, 0] - IFACE_X) < 1e-10
outer_mask = (~iface_mask) & (np.isclose(xy[:, 0], X0) | np.isclose(xy[:, 1], Y0) | np.isclose(xy[:, 1], Y1))

iface_dofs = np.where(iface_mask)[0].astype(np.int32)
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]
y_if = xy[iface_dofs, 1]

outer_dofs = np.where(outer_mask)[0].astype(np.int32)

if len(iface_dofs) == 0:
    print(f"ERROR: No interface DOFs at x={IFACE_X}", flush=True)
    sys.exit(1)

# Tag boundaries for measures
def is_outer(x):
    return (np.isclose(x[0], X0) | np.isclose(x[1], Y0) | np.isclose(x[1], Y1)) & ~np.isclose(x[0], IFACE_X)

def is_iface(x):
    return np.isclose(x[0], IFACE_X)

outer_facets = mesh.locate_entities_boundary(domain, fdim, is_outer)
iface_facets = mesh.locate_entities_boundary(domain, fdim, is_iface)

tags = mesh.meshtags(domain, fdim, 
                     np.concatenate([outer_facets, iface_facets]),
                     np.concatenate([np.ones(len(outer_facets), dtype=np.int32),
                                     np.full(len(iface_facets), 2, dtype=np.int32)]))
ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)

# Weak form setup
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
k_c = fem.Constant(domain, default_scalar_type(K))
rho_c = fem.Constant(domain, default_scalar_type(RHO_CP))
dt_c = fem.Constant(domain, default_scalar_type(DT))
theta = 0.5

spatial = lambda w: ufl.inner(k_c * ufl.grad(w), ufl.grad(v)) * ufl.dx

T_n = fem.Function(V)
T_h = fem.Function(V)
T_n.x.array[:] = 0.0
T_h.x.array[:] = 0.0

# System matrix (time-independent)
a_form = (rho_c/dt_c) * ufl.inner(u, v) * ufl.dx + theta * spatial(u)
a_fem = fem.form(a_form)
A = fp.assemble_matrix(a_fem)
A.assemble()

b = fp.create_vector(V)
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# Time stepping
n_steps = int(np.ceil(T_END / DT))
t = 0.0

# Read imports once (coupling iteration happens outside time loop in this simplified version)
imp = read_imports()
iface_temps = interp(imp, "values", 0.0, y_if)

for step in range(n_steps):
    t = (step + 1) * DT
    
    # Source term at current time
    f_func = fem.Function(V)
    f_func.interpolate(lambda x: source_A(x, t))
    
    # RHS form
    L_form = ((rho_c/dt_c) * ufl.inner(T_n, v) * ufl.dx 
              + (1-theta) * spatial(T_n) 
              + f_func * v * ufl.dx)
    L_fem = fem.form(L_form)
    
    # Boundary conditions - use fem.Constant for scalar values
    zero_const = fem.Constant(domain, default_scalar_type(0.0))
    bc_outer = fem.dirichletbc(zero_const, outer_dofs, V)
    
    # Interface Dirichlet BC (from partner)
    g_iface = fem.Function(V)
    g_iface.x.array[iface_dofs] = iface_temps
    bc_iface = fem.dirichletbc(g_iface, iface_dofs)
    
    bcs = [bc_outer, bc_iface]
    
    # Assemble and solve
    with b.localForm() as loc:
        loc.set(0.0)
    fp.assemble_vector(b, L_fem)
    fp.apply_lifting(b, [a_fem], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fp.set_bc(b, bcs)
    
    ksp.solve(b, T_h.x.petsc_vec)
    T_h.x.scatter_forward()
    
    # Update T_n
    T_n.x.array[:] = T_h.x.array

# Compute outward normal flux at interface using reaction forces
Amat = fp.assemble_matrix(a_fem)
Amat.assemble()
bvec = fp.assemble_vector(L_fem)
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(T_h.x.petsc_vec, r)
r.axpy(-1.0, bvec)

wvec = fp.assemble_vector(fem.form(v * ds(2)))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# Handle corners
suspect = ~ok
good = np.where(ok)[0]
if len(good) > 0 and np.any(suspect):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

T = T_h.x.array[iface_dofs]

print(f"[A] t={t:.4f} T=[{T.min():.4g},{T.max():.4g}] q=[{Q.min():.4g},{Q.max():.4g}]", flush=True)

# Write exports
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(iface_dofs),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [float(v) for v in T],
    "normal_fluxes": [float(q) for q in Q],
}))

# Write log
ndof = V.dofmap.index_map.size_global
Path("run.log").write_text(f"NDOF = {ndof}\n")

print(f"[A] NDOF={ndof}", flush=True)
