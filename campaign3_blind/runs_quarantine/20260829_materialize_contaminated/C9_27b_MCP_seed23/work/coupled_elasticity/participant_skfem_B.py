"""scikit-fem VECTOR participant for subdomain B (Dirichlet side) - horizontal interface.

Plane-strain linear elasticity -div(sigma(u)) = f on subdomain B = (0,1) x (0.625, 1.5).
Interface is HORIZONTAL at y = IFACE_Y.

CONTRACT: runs in work_dir with no arguments, reads imports.json, writes exports.json LAST.

SIGN CONVENTION:
normal_fluxes = q_out = -(sigma . n_own) where n_own = S * e_y (outward normal at interface)
"""
import json
from pathlib import Path
import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, ElementVector,
                   FacetBasis, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

# ── PROBLEM SPECIFIC PARAMETERS ─
SIDE      = "dirichlet"   # B is Dirichlet side (imports displacement, exports traction)
PARTNER   = "sideA"       # name of partner in couple(...)
X0, X1    = 0.0, 1.0      # subdomain B x-range
Y0, Y1    = 0.625, 1.5    # subdomain B y-range
IFACE_Y   = 0.625         # shared interface (must be Y0 or Y1)
LAM       = 480.0         # Lamé parameter lambda
MU        = 240.0         # Lamé parameter mu (shear modulus)
# Outer boundary: u = 0 everywhere
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)

def B_SRC(x, y):
    """Body force for subdomain B."""
    fx = (1737*x**4*y/30625 - 9357*x**4/245000 - 4632*x**3*y/153125 + 3119*x**3/153125 
          + 396*x**2*y**2/875 - 57969*x**2*y/61250 + 86847*x**2/245000 
          - 528*x*y**2/4375 + 40962*x*y/153125 - 16034*x/153125 
          - 66*y**2/875 + 519*y/3500 - 369/7000)
    fy = (2316*x**5/153125 - 1544*x**4/153125 + 1158*x**3*y**2/30625 + 9201*x**3*y/61250 
          - 53561*x**3/245000 - 2316*x**2*y**2/153125 - 9201*x**2*y/153125 + 59737*x**2/612500 
          - 579*x*y**2/30625 - 9201*x*y/122500 + 9477*x/98000 
          + 772*y**2/153125 + 3067*y/153125 - 3159/122500)
    return fx, fy

# Mesh resolution - will be set via file
NX, NY    = 8, 7         # base resolution
UI_X, UI_Y = 0.0, 0.0    # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0    # iteration-1 fallback interface traction

# Check for mesh resolution file
res_file = Path("mesh_resolution.json")
if res_file.is_file():
    try:
        with open(res_file) as f:
            res = json.load(f)
            NX = res.get("nx", NX)
            NY = res.get("ny", NY)
    except:
        pass

TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

ON_BOTTOM = abs(IFACE_Y - Y0) < abs(IFACE_Y - Y1)   # interface is this side's y-min?
OUTER_Y = Y1 if ON_BOTTOM else Y0
S = -1.0 if ON_BOTTOM else 1.0            # outward normal at interface = S * e_y


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, x):
    """Map the partner's VECTOR samples onto this participant's x-coordinates."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(x), 1))
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != xs.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(x), 1))
    o = np.argsort(xs)
    return np.column_stack([np.interp(x, xs[o], vs[o, c])
                            for c in range(vs.shape[1])])


def u_dirichlet(x, y):
    """The prescribed displacement on the non-interface boundary."""
    return (UDX[0] + UDX[1] * x + UDX[2] * y + UDX[3] * y * y,
            UDY[0] + UDY[1] * x + UDY[2] * y + UDY[3] * y * y)


imp = read_imports()

# ── MESH ──
mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))
elem = ElementVector(ElementTriP1())
basis = Basis(mesh, elem)
nd = basis.nodal_dofs                      # (2, nnodes): node -> (x, y) dof

px, py = mesh.p[0], mesh.p[1]

# Interface nodes: y ≈ IFACE_Y
iface_n = np.where(np.abs(py - IFACE_Y) < TOL)[0]
iface_n = iface_n[np.argsort(px[iface_n])]             # sorted by x
x_if = px[iface_n]

# Outer boundary nodes: x=0, x=1, or y=OUTER_Y
outer_n = np.where((np.abs(px - X0) < TOL) |
                   (np.abs(px - X1) < TOL) | (np.abs(py - OUTER_Y) < TOL))[0]

# Interface corners: where interface meets x=0 or x=1
corner_mask = (np.abs(px[iface_n] - X0) < TOL) | (np.abs(px[iface_n] - X1) < TOL)

# Interface BC nodes (exclude corners)
iface_bc_n = iface_n[~corner_mask]
iface_bc_dofs = np.concatenate([nd[0, iface_bc_n], nd[1, iface_bc_n]])
outer_dofs = np.concatenate([nd[0, outer_n], nd[1, outer_n]])


@BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2.0 * MU * ddot(eu, ev) + LAM * trace(eu) * trace(ev)


@LinearForm
def body_force(v, w):
    bx, by = B_SRC(w.x[0], w.x[1])
    return bx * v[0] + by * v[1]


@LinearForm
def traction(v, w):
    return w["t"][0] * v[0] + w["t"][1] * v[1]


@LinearForm
def unit_load(v, w):
    return 1.0 * v[0] + 1.0 * v[1]


A = stiffness.assemble(basis)          # UNCONSTRAINED
b = body_force.assemble(basis)         # volume load
b_vol = b                              # keep for traction recovery

fbi = FacetBasis(mesh, elem,
                 facets=mesh.facets_satisfying(
                     lambda p: np.abs(p[1] - IFACE_Y) < TOL))

sol = basis.zeros()
ux_d, uy_d = u_dirichlet(px[outer_n], py[outer_n])
sol[nd[0, outer_n]] = ux_d
sol[nd[1, outer_n]] = uy_d
D = outer_dofs

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), x_if)
    keep = ~corner_mask
    sol[nd[0, iface_bc_n]] = u_if[keep, 0]
    sol[nd[1, iface_bc_n]] = u_if[keep, 1]
    D = np.unique(np.concatenate([outer_dofs, iface_bc_dofs]))
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), x_if)
    gnod = basis.zeros()
    gnod[nd[0, iface_n]] = t_if[:, 0]
    gnod[nd[1, iface_n]] = t_if[:, 1]
    # Apply partner's traction UNCHANGED
    b = b + asm(traction, fbi, t=fbi.interpolate(gnod))

sol = solve(*condense(A, b, x=sol, D=D))

# Interface traction export q_out = -(sigma . n_own)
r = A @ sol - b_vol                    # r = A u_h - b_vol
wgt = unit_load.assemble(fbi)          # w_i = int_Gamma phi_i ds

idx = np.column_stack([nd[0, iface_n], nd[1, iface_n]])    # (nnode, 2) dofs
wi = wgt[idx]                          # the SAME w_i in both columns
Q = np.zeros_like(wi)
ok = np.abs(wi) > 1e-14
Q[ok] = -r[idx][ok] / wi[ok]

# Handle corners: use nearest interior node
suspect = np.isin(iface_n, outer_n) | ~ok.all(axis=1)
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Write exports
Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_n)),
    "coordinates": [[float(xx), float(IFACE_Y)] for xx in x_if],
    "values": [[float(sol[nd[0, i]]), float(sol[nd[1, i]])] for i in iface_n],
    "normal_fluxes": [[float(q0), float(q1)] for q0, q1 in Q],
}, indent=2))

# Write run log with NDOF
ndof = A.shape[0]
n_elements = mesh.t.shape[1]  # number of triangles
with open("run.log", "w") as f:
    f.write(f"NDOF = {ndof}\n")
    f.write(f"NX = {NX}, NY = {NY}\n")
    f.write(f"n_elements = {n_elements}\n")
    f.write(f"n_nodes = {mesh.p.shape[1]}\n")

# Write VTU for post-processing
import meshio
cells = [("triangle", mesh.t.T)]
points = np.column_stack([px, py, np.zeros_like(px)])
nodal_dofs = basis.nodal_dofs
ux = sol[nodal_dofs[0]]
uy = sol[nodal_dofs[1]]
displacement = np.column_stack([ux, uy, np.zeros_like(ux)])
meshio.Mesh(points, cells, point_data={"displacement": displacement}).write("result.vtu")
