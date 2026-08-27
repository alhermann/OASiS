"""FEniCSx (dolfinx) participant for coupled transient heat conduction.

Subdomain A: (0, 0.625) x (0, 1), k=1, Dirichlet side (imports T from B, exports flux)
Equation: du/dt - div(k grad u) = f with Crank-Nicolson time stepping
Outer BC: u=0 on all outer boundaries
Interface at x=0.625: receives temperature from partner, imposes as Dirichlet BC
"""
import json
import sys
from pathlib import Path
import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import fem, mesh, la, default_scalar_type
from dolfinx.fem import petsc as fp
from petsc4py import PETSc

# Problem parameters
SIDE = "dirichlet"  # A is Dirichlet side
PARTNER = "B"
X0, X1 = 0.0, 0.625  # Subdomain A extent
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K = 1.0  # conductivity in A
RHO_CP = 1.0  # heat capacity
NX, NY = 8, 8  # Will be set based on resolution
DT = 1/32  # Will be set based on resolution
T_END = 0.25
T_INIT = 0.0  # Initial condition
Q_INIT = 0.0  # Fallback interface flux

def source_term_A(x, t):
    """Source term for subdomain A"""
    exp_t2 = np.exp(t/2.0)
    result = np.zeros_like(x[0])
    
    # Terms with t
    result += (-384*t*x[0]**3*x[1]**3 + 1184*t*x[0]**3*x[1]**2 + 3808*t*x[0]**3*x[1] - 4736*t*x[0]**3
               - 316*t*x[0]**2*x[1]**3 - 199*t*x[0]**2*x[1]**2 + 4307*t*x[0]**2*x[1] + 796*t*x[0]**2
               + 5148*t*x[0]*x[1]**3 - 14883*t*x[0]*x[1]**2 + 3255*t*x[0]*x[1] + 2700*t*x[0]
               + 1264*t*x[1]**3 + 796*t*x[1]**2 - 2060*t*x[1])
    
    # Terms without t
    result += (-768*x[0]**3*x[1]**3 + 2368*x[0]**3*x[1]**2 - 1600*x[0]**3*x[1]
               - 632*x[0]**2*x[1]**3 - 398*x[0]**2*x[1]**2 + 1030*x[0]**2*x[1]
               + 1080*x[0]*x[1]**3 - 1350*x[0]*x[1]**2 + 270*x[0]*x[1])
    
    return result * exp_t2 / 1280.0

def read_imports():
    """Read imports.json; returns None if not available (iteration 1)"""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get(PARTNER)
    except (json.JSONDecodeError, KeyError):
        return None

def sample_interface(imp, key, fallback, y_coords):
    """Interpolate partner's data onto our interface points"""
    if imp is None or "coordinates" not in imp:
        return np.full(len(y_coords), float(fallback))
    
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    
    if len(vs) != len(ys) or len(ys) == 0:
        return np.full(len(y_coords), float(fallback))
    
    # Sort by y for interpolation
    idx = np.argsort(ys)
    return np.interp(y_coords, ys[idx], vs[idx])

# Read command line args for resolution and dt
if len(sys.argv) >= 3:
    NX = int(sys.argv[1])
    DT = float(sys.argv[2])
    NY = int(NX * (Y1-Y0) / (X1-X0))  # Keep aspect ratio

print(f"[FEniCSx A] NX={NX}, NY={NY}, DT={DT}", flush=True)

# Create mesh
domain = mesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

# Find interface DOFs
xy = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]
y_if = xy[iface_dofs, 1]

if len(iface_dofs) == 0:
    sys.exit(f"No interface DOFs found at x={IFACE_X}")

# Tag boundaries
def outer_boundary(x):
    """All outer boundaries (not interface)"""
    return (np.isclose(x[0], X0) | np.isclose(x[1], Y0) | 
            np.isclose(x[1], Y1)) & ~np.isclose(x[0], IFACE_X)

def interface_boundary(x):
    return np.isclose(x[0], IFACE_X)

outer_facets = mesh.locate_entities_boundary(domain, fdim, outer_boundary)
iface_facets = mesh.locate_entities_boundary(domain, fdim, interface_boundary)

tags = mesh.meshtags(domain, fdim, 
                     np.concatenate([outer_facets, iface_facets]),
                     np.concatenate([np.full(len(outer_facets), 1, dtype=np.int32),
                                     np.full(len(iface_facets), 2, dtype=np.int32)]))
ds = ufl.Measure("ds", domain=domain, subdomain_data=tags)

# Outer boundary DOFs (u=0)
outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)

# Weak form
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
k_const = fem.Constant(domain, default_scalar_type(K))
rho_cp_const = fem.Constant(domain, default_scalar_type(RHO_CP))
dt_const = fem.Constant(domain, default_scalar_type(DT))
theta = 0.5  # Crank-Nicolson

# Spatial operator: int(k*grad(u)*grad(v))
def spatial_form(w):
    return ufl.inner(k_const * ufl.grad(w), ufl.grad(v)) * ufl.dx

# Time-dependent source
class SourceWrapper:
    def __init__(self, func):
        self.func = func
        self.t = 0.0
    
    def interpolate(self, V):
        self.V = V
        return fem.Function(V)
    
    def update(self, t):
        self.t = t
        x = ufl.SpatialCoordinate(domain)
        self.fem_func.interpolate(lambda x: self.func(x, self.t))

src_func = SourceWrapper(lambda x, t: source_term_A(x, t))
f_src = src_func.interpolate(V)

# Crank-Nicolson weak form
# (rho_cp/dt)*(u - u_n)*v + theta*spatial(u) + (1-theta)*spatial(u_n) = f*v
T_n = fem.Function(V, name="T_n")
T_h = fem.Function(V, name="T_h")
T_n.x.array[:] = T_INIT
T_h.x.array[:] = T_INIT

# Assemble system matrix once (time-independent)
a_form = (rho_cp_const/dt_const) * ufl.inner(u, v) * ufl.dx + theta * spatial_form(u)
L_form = ((rho_cp_const/dt_const) * ufl.inner(T_n, v) * ufl.dx 
          + (1-theta) * spatial_form(T_n) + f_src * v * ufl.dx)

a_fem = fem.form(a_form)
A = fp.assemble_matrix(a_fem, bcs=[])
A.assemble()

b = fp.create_vector(V)
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# Time stepping
n_steps = int(np.ceil(T_END / DT))
t = 0.0

# Read initial interface data
imp = read_imports()

for step in range(n_steps):
    t = (step + 1) * DT
    
    # Update source term
    src_func.update(t)
    f_src.interpolate(lambda x: source_term_A(x, t))
    
    # Build boundary conditions
    bc_outer = fem.dirichletbc(default_scalar_type(0.0), outer_dofs, V)
    
    # Interface BC (Dirichlet side receives temperature from partner)
    g_iface = fem.Function(V)
    iface_temps = sample_interface(imp, "values", T_INIT, y_if)
    g_iface.x.array[iface_dofs] = iface_temps
    bc_iface = fem.dirichletbc(g_iface, iface_dofs, V)
    
    bcs = [bc_outer, bc_iface]
    
    # Assemble RHS
    with b.localForm() as loc:
        loc.set(0.0)
    fp.assemble_vector(b, L_form)
    fp.apply_lifting(b, [a_fem], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fp.set_bc(b, bcs)
    
    # Solve
    ksp.solve(b, T_h.x.petsc_vec)
    T_h.x.scatter_forward()
    
    # Update T_n
    T_n.x.array[:] = T_h.x.array

# Extract interface values and compute outward normal flux
# Outward normal at interface (x=IFACE_X) for subdomain A is +e_x (pointing right)
# Flux q = -k * du/dx (outward)

# Use reaction flux method for accurate flux recovery
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)

# Assemble unconstrained residual to get reaction forces
Amat = fp.assemble_matrix(fem.form((rho_cp_const/dt_const) * ufl.inner(p_, w_) * ufl.dx + 
                                     theta * spatial_form(p_)), bcs=[])
Amat.assemble()
bvec = fp.assemble_vector(fem.form(((rho_cp_const/dt_const) * ufl.inner(T_h, w_) * ufl.dx + 
                                     (1-theta) * spatial_form(T_h) + f_src * w_ * ufl.dx)))
bvec.ghostUpdate()

r = Amat.createVecLeft()
Amat.mult(T_h.x.petsc_vec, r)
r.axpy(-1.0, bvec)

# Weight vector for interface
wvec = fp.assemble_vector(fem.form(w_ * ds(2)))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

# Compute flux: q_i = -r_i / w_i (reaction force divided by weight)
Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# Handle corner nodes (where interface meets outer boundary)
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good) > 0:
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Get interface temperatures
T = T_h.x.array[iface_dofs]

print(f"[FEniCSx A] Final t={t:.6f}, T_range=[{T.min():.6g},{T.max():.6g}], q_range=[{Q.min():.6g},{Q.max():.6g}]", flush=True)

# Write exports.json LAST
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [float(val) for val in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

# Write run log with NDOF
ndof = V.dofmap.index_map.size_global
Path("run.log").write_text(f"NDOF = {ndof}\n")

print(f"[FEniCSx A] NDOF={ndof}, exported {len(T)} interface points", flush=True)
