"""scikit-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm,
                   MeshTri, condense, solve)
from skfem.helpers import dot, grad

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6      # this subdomain
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.6           # shared interface (must be X0 or X1)
K         = 0.8           # conductivity


def F_SRC(x, y):
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)
    """
    return np.zeros_like(x)
T_OUTER   = 320.0         # Dirichlet value on the NON-interface x-boundary
NX, NY    = 24, 16        # this subdomain's own mesh
T_INIT    = 310.0          # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, y):
    """Interpolate the partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


imp = read_imports()

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin
mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))
elem = ElementTriP1()
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs[0]                  # node index -> dof (P1: identity)

px, py = mesh.p[0], mesh.p[1]
iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(py[iface_n])]             # sorted by y
y_if = py[iface_n]
iface_dofs = n2d[iface_n]
outer_dofs = n2d[np.where(np.abs(px - OUTER_X) < TOL)[0]]


@BilinearForm
def stiffness(u, v, w):
    return K * dot(grad(u), grad(v))


@LinearForm
def source(v, w):
    """The loading functional, int_Omega f v dx.

    `w.x` is the (2, nelems, nqp) array of GLOBAL coordinates of the quadrature
    points, so F_SRC is evaluated exactly where the integration rule needs it
    and a polynomial source is integrated to quadrature accuracy — no detour
    through a P1 interpolant of the source, and no constant standing in for a
    field that varies over the element."""
    return F_SRC(w.x[0], w.x[1]) * v
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end


@LinearForm
def flux_load(v, w):
    return w["g"] * v


@LinearForm
def unit_load(v, w):
    return 1.0 * v          # w_i = int_Gamma phi_i ds


# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin
A = stiffness.assemble(basis)          # UNCONSTRAINED: condense() below does
b = source.assemble(basis)             # not modify A or b in place
# THE VOLUME LOAD ALONE, kept for the flux recovery at the bottom. The Neumann
# branch adds the partner's interface term into `b`; subtracting that combined
# vector is what made the reaction look like zero on that side.
b_vol = b
fbasis = FacetBasis(mesh, elem,
                    facets=mesh.facets_satisfying(
                        lambda p: np.abs(p[0] - IFACE_X) < TOL))

sol = basis.zeros()
sol[outer_dofs] = T_OUTER
D = outer_dofs
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    sol[iface_dofs] = T_if
    D = np.concatenate([outer_dofs, iface_dofs])
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gnod = basis.zeros()                   # P1 trace of the partner's samples
    gnod[iface_dofs] = q_if
    b = b + flux_load.assemble(fbasis, g=fbasis.interpolate(gnod))
    # APPLY the partner's number unchanged (+ integral(g*v) ds_interface)
    # `b_vol` above still holds the VOLUME load alone — the flux recovery
    # below subtracts that, not this, and the distinction is the whole point.

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin
sol = solve(*condense(A, b, x=sol, D=D))
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end

# Outward normal flux density q = -(k grad T).n on the interface.
#
# WHY NOT AN L2 PROJECTION OF THE GRADIENT. That is what this file used to do:
# project -k dT/dx over the whole subdomain and sample it at the interface. The
# gradient of a P1 solution is only O(h) accurate ON the boundary — the
# superconvergence points are interior — and the boundary trace is exactly what
# the coupling reads. Measured against a manufactured solution with a known
# exact interface flux, the projection converges at order ~1 while the
# consistent flux below converges at ~2, so the recovery, not the physics and
# not the partner, was setting the answer.
#
# THE CONSISTENT (REACTION) FLUX. From
#     a(u,v) - (f,v) = int_dOmega (k grad u . n) v ds = -int_Gamma qn v ds
# it follows that for every basis function phi_i on the interface
#     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
# with r the UNCONSTRAINED residual: A and b above are assembled with NO
# boundary condition applied and skfem's condense() returns copies, so the
# constrained rows of A still carry the reaction. Dividing by
# w_i = int_Gamma phi_i ds turns the functional into a density the partner can
# interpolate pointwise.
# ONE FORMULA, BOTH SIDES. An earlier version used the reaction on the
# Dirichlet side and an L2-projected gradient on the Neumann side, reasoning
# that the Neumann interface dofs are free so r comes out ~0 there. That holds
# only when the residual is taken against a load that ALREADY CONTAINS the
# interface term. Subtract the VOLUME load alone and those same rows carry
# exactly the interface functional the partner applied:
#     (A u - b_vol)_i = int_Gamma g phi_i ds
# On the Dirichlet side there is no interface term, so b == b_vol and the two
# cases are one expression.
#
# MEASURED (FEniCSx, same formulation) against a known imposed flux
# q = 2 + 3 sin(4y) on 8/16/32/64/128 uniform triangle meshes, interior
# interface nodes: the projected gradient stalls at max 2.6 and does NOT
# converge (order 0.00; 0.93 away from the ends, 0.50 in rms), while the
# reaction against b_vol converges at order 2.00 in all three norms.
r = A @ sol - b_vol                    # r = A u_h - b_vol, no bc applied
wgt = unit_load.assemble(fbasis)       # w_i = int_Gamma phi_i ds

Q = np.zeros(len(iface_dofs))
ok = np.abs(wgt[iface_dofs]) > 1e-14
Q[ok] = -r[iface_dofs][ok] / wgt[iface_dofs][ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux.
# Take the nearest interior interface node rather than exporting a corner
# value that is physically a different quantity. This holds on both sides.
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(t) for t in sol[iface_dofs]],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
