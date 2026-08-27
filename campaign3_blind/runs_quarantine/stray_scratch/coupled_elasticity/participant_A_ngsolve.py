"""NGSolve participant for coupled elasticity - Subdomain A (NEUMANN side).

Subdomain A: (0,1) x (0,0.625), lambda=480, mu=1200
Role: NEUMANN - receives traction from partner, exports displacement
Interface: y = 5/8 (top edge of this subdomain)
Outer BC: u = 0 on all outer boundaries (left, right, bottom)
"""
import json
from pathlib import Path
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

# ── PROBLEM PARAMETERS ────────────────────────────────────────────────────────
SIDE      = "neumann"     # A is the NEUMANN side
PARTNER   = "B"           # name of partner in couple(...)
X0, X1    = 0.0, 1.0      # subdomain A x-range
Y0, Y1    = 0.0, 0.625    # subdomain A y-range
IFACE_Y   = 0.625         # interface at y = 5/8
LAMBDA    = 480.0         # Lamé parameter λ
MU        = 1200.0        # shear modulus μ
NX, NY    = 8, 5          # mesh divisions (will be updated per level)
UX_INIT   = 0.0           # iteration-1 fallback displacement
TX_INIT   = 0.0           # iteration-1 fallback traction
# ─────────────────────────────────────────────────────────────────────────────

TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    """Read imports.json and return partner's data."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample_vector(imp, key, fallback, x_coords):
    """Interpolate partner's vector samples onto this participant's x-coordinates.
    
    For vector fields, we must interpolate each component separately!
    """
    if not imp or not imp.get("coordinates"):
        return np.full((len(x_coords), 2), float(fallback))
    
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float)
    
    if vs.ndim == 1:
        return np.column_stack([np.interp(x_coords, xs, vs), 
                                np.zeros_like(x_coords)])
    
    result = np.zeros((len(x_coords), 2))
    for comp in range(2):
        result[:, comp] = np.interp(x_coords, xs, vs[:, comp])
    return result


# Read imported data
imp = read_imports()

# ── MESH GENERATION ───────────────────────────────────────────────────────────
geo = SplineGeometry()
# AddRectangle edge order: bottom, right, top, left
# Interface is at top (y=Y1=0.625), outer BC on bottom, left, right
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=("outer", "outer", "interface", "outer"))
mesh = Mesh(geo.GenerateMesh(maxh=min((X1-X0)/NX, (Y1-Y0)/NY)))

# ── FE SPACE (vector P1) ─────────────────────────────────────────────────────
fes = VectorH1(mesh, order=1, dirichlet="outer")
u, v = fes.TnT()

# Strain and stress operators for plane strain
def Strain(u):
    """Symmetric gradient (small strain tensor)."""
    return 0.5 * (grad(u) + grad(u).trans)

def Stress(u):
    """Constitutive law: sigma = 2*mu*eps + lambda*tr(eps)*I"""
    eps = Strain(u)
    return 2 * MU * eps + LAMBDA * Trace(eps) * Id(2)

# ── ASSEMBLY ─────────────────────────────────────────────────────────────────
a = BilinearForm(fes)
a += InnerProduct(Stress(u), Strain(v)) * dx

# Body force source term for subdomain A
f = LinearForm(fes)
f += CoefficientFunction((-63*x**4*y/3125 + 99*x**4/5000 + 168*x**3*y/15625 - 33*x**3/3125 + 
                          216*x**2*y**2/625 - 1557*x**2*y/3125 - 99*x**2/5000 - 288*x*y**2/3125 + 
                          1992*x*y/15625 + 33*x/3125 - 36*y**2/625 + 54*y/625)) * v[0] * dx
f += CoefficientFunction(-108*x**5/15625 + 72*x**4/15625 - 18*x**3*y**2/625 + 153*x**3*y/1250 - 
                         279*x**3/3125 + 36*x**2*y**2/3125 - 153*x**2*y/3125 + 486*x**2/15625 + 
                         9*x*y**2/625 - 153*x*y/2500 + 63*x/1250 - 12*y**2/3125 + 51*y/3125 - 
                         42/3125) * v[1] * dx

# ── INTERFACE DOF IDENTIFICATION ─────────────────────────────────────────────
# Get vertex-to-dof mapping (order=1: one dof per vertex per component)
vdof = np.array([[fes.GetDofNrs(NodeId(VERTEX, i))[comp] 
                  for comp in range(2)] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

# Interface vertices (at y = IFACE_Y), sorted by x
iface_v = np.where(np.abs(vxy[:, 1] - IFACE_Y) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 0])]
x_if = vxy[iface_v, 0]
y_if = vxy[iface_v, 1]

# Interface dofs (both components)
iface_dofs_x = vdof[iface_v, 0]  # x-component dofs
iface_dofs_y = vdof[iface_v, 1]  # y-component dofs

# Outer boundary dofs (for Dirichlet BC)
outer_mask = (np.abs(vxy[:, 0] - X0) < TOL) | (np.abs(vxy[:, 0] - X1) < TOL) | \
             (np.abs(vxy[:, 1] - Y0) < TOL)
outer_v = np.where(outer_mask)[0]

# ── APPLY IMPORTED TRACTION (NEUMANN BC) ─────────────────────────────────────
if SIDE == "neumann":
    # Import traction from partner
    tx_if = sample_vector(imp, "normal_fluxes", TX_INIT, x_if)
    
    # Create trace functions for traction using the vector space
    gfun = GridFunction(fes)
    gfun.vec[:] = 0.0
    
    # Set traction values at interface dofs
    for d, t in zip(iface_dofs_x, tx_if[:, 0]):
        gfun.vec[int(d)] = float(t)
    for d, t in zip(iface_dofs_y, tx_if[:, 1]):
        gfun.vec[int(d)] = float(t)
    
    # Apply traction as Neumann BC: + integral(g . v) ds_interface
    # The natural boundary term in elasticity weak form is +(sigma.n).v
    f += gfun[0] * v[0] * ds("interface")
    f += gfun[1] * v[1] * ds("interface")

# ── SOLVE ────────────────────────────────────────────────────────────────────
gfu = GridFunction(fes)
gfu.vec[:] = 0.0  # Dirichlet BC is u=0 on outer boundary

with TaskManager():
    a.Assemble()
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * res

# ── EXPORT DISPLACEMENT AND TRACTION ─────────────────────────────────────────
# Extract displacement at interface
ux_if = np.array([gfu.vec[int(d)] for d in iface_dofs_x], float)
uy_if = np.array([gfu.vec[int(d)] for d in iface_dofs_y], float)
disp_if = np.column_stack([ux_if, uy_if])

# Compute traction t = sigma . n_outward at interface
# For NEUMANN side, outward normal at top interface: n = (0, +1)
# t = sigma . n = [sigma_xy, sigma_yy]^T at top edge

# Use scalar H1 spaces for projection
fesq_x = H1(mesh, order=1)
fesq_y = H1(mesh, order=1)
px, wx = fesq_x.TnT()
py, wy = fesq_y.TnT()

mx = BilinearForm(fesq_x)
mx += px * wx * dx
my = BilinearForm(fesq_y)
my += py * wy * dx

# t_x = sigma_xy = mu*(du_x/dy + du_y/dx)
ftx = LinearForm(fesq_x)
ftx += MU * (grad(gfu)[0, 1] + grad(gfu)[1, 0]) * wx * dx

# t_y = sigma_yy = lambda*(du_x/dx + du_y/dy) + 2*mu*du_y/dy
fty = LinearForm(fesq_y)
fty += (LAMBDA * (grad(gfu)[0, 0] + grad(gfu)[1, 1]) + 2*MU*grad(gfu)[1, 1]) * wy * dx

with TaskManager():
    mx.Assemble()
    my.Assemble()
    ftx.Assemble()
    fty.Assemble()

qh_tx = GridFunction(fesq_x)
qh_ty = GridFunction(fesq_y)

qh_tx.vec.data = mx.mat.Inverse(fesq_x.FreeDofs(), inverse="sparsecholesky") * ftx.vec
qh_ty.vec.data = my.mat.Inverse(fesq_y.FreeDofs(), inverse="sparsecholesky") * fty.vec

# Sample traction at interface nodes
tx_export = np.zeros(len(iface_v))
ty_export = np.zeros(len(iface_v))
for i, vi in enumerate(iface_v):
    pt = mesh.vertices[vi].point
    tx_export[i] = qh_tx(mesh(*pt))
    ty_export[i] = qh_ty(mesh(*pt))

traction_if = np.column_stack([tx_export, ty_export])

# ── WRITE EXPORTS ────────────────────────────────────────────────────────────
Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(xx), float(IFACE_Y)] for xx in x_if],
    "values": disp_if.tolist(),
    "normal_fluxes": traction_if.tolist(),
}, indent=2))

# ── WRITE LOG FILE ───────────────────────────────────────────────────────────
log_content = f"NDOF = {fes.ndof}\n"
log_content += f"Elements = {mesh.ne}\n"
log_content += f"Interface points = {len(iface_v)}\n"
Path("run.log").write_text(log_content)

print(f"Subdomain A (NGSolve) complete: {fes.ndof} DOFs, {len(iface_v)} interface points")
