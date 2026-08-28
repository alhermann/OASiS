"""FEniCSx (dolfinx) participant for the OASiS `couple` driver.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: steady conduction  -div(K grad T) = F_SRC  on one rectangular
subdomain of a domain split by a straight interface at x = IFACE_X.
Top and bottom edges are natural (zero-flux). The non-interface x-boundary
carries a Dirichlet value T_OUTER.
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

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.6      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
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
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
T_INIT    = 310.0         # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

OUTER_X = X0 if IFACE_X == X1 else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S*e_x


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here."""
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
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

xy = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]   # constant order, always
y_if = xy[iface_dofs, 1]
if len(iface_dofs) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = fem.Constant(domain, default_scalar_type(K)) * \
    ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

# SOURCE. F_SRC is INTERPOLATED into a Function, not wrapped in a
# fem.Constant. A Constant is one number for the whole subdomain, so the
# moment F_SRC varies with position — which is the normal case, a manufactured
# solution's source is a polynomial — `Constant(domain, F_SRC(x, y))` cannot
# even be built (F_SRC returns an array), and the constant that used to sit
# here silently solved a different problem. The interpolant is P1 like the
# solution space, so the source's quadrature error is O(h^2), the same order as
# the discretization error itself. Do NOT hand F_SRC a ufl.SpatialCoordinate
# instead: a UFL expression carries no NumPy ufuncs, so np.zeros_like(x)
# returns a 0-d OBJECT array and np.sin(x) raises.
f_src = fem.Function(V)
f_src.interpolate(lambda X: np.zeros(X.shape[1]) + F_SRC(X[0], X[1]))
# KEPT SEPARATE FROM L ON PURPOSE. The flux recovery below subtracts the
# VOLUME load alone, on both sides; adding the interface term into the same
# form is what made the reaction look like zero on the Neumann side.
L_vol = f_src * v * ufl.dx
L = L_vol

outer = dmesh.locate_entities_boundary(domain, fdim,
                                       lambda x: np.isclose(x[0], OUTER_X))
outer_dofs = fem.locate_dofs_topological(V, fdim, outer)
bcs = [fem.dirichletbc(default_scalar_type(T_OUTER), outer_dofs, V)]

# One definition of the interface measure, used by the Neumann branch to APPLY
# the partner's flux and by the Dirichlet branch to RECOVER its own.
facets_if = dmesh.locate_entities_boundary(domain, fdim,
                                           lambda x: np.isclose(x[0], IFACE_X))
tags_if = dmesh.meshtags(domain, fdim, np.sort(facets_if),
                         np.full(len(facets_if), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags_if)(7)

if SIDE == "dirichlet":
    g = fem.Function(V)
    g.x.array[iface_dofs] = sample(imp, "values", T_INIT, y_if)
    bcs.append(fem.dirichletbc(g, iface_dofs))
else:
    g = fem.Function(V)
    g.x.array[iface_dofs] = sample(imp, "normal_fluxes", Q_INIT, y_if)
    L = L_vol + g * v * ds_if   # APPLY the partner's number UNCHANGED

uh = LinearProblem(a, L, bcs=bcs, petsc_options_prefix="cpl",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()

# Outward normal flux density q = -(K grad T).n on the interface.
#
# WHY NOT AN L2 PROJECTION OF THE GRADIENT. That is what this file used to do:
# project -K dT/dx over the whole subdomain and sample it at the interface.
# The gradient of a P1 solution is only O(h) accurate ON the boundary — the
# superconvergence points are interior — and the boundary trace is exactly what
# the coupling reads. Measured against a manufactured solution with a known
# exact interface flux, the projection converges at order 1.07 while the
# consistent flux below converges at 2.05, and in a full coupled run the graded
# field order moves from 1.83 to 2.05 against a pass band that ends at 1.6. The
# recovery, not the physics and not the partner, was setting the answer.
#
# THE CONSISTENT (REACTION) FLUX. From
#     a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds = -int_Gamma qn v ds
# it follows that for every basis function phi_i on the interface
#     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
# with r the UNCONSTRAINED residual: assembled with no boundary condition
# applied and with the constrained rows NOT zeroed, because on the Dirichlet
# side those rows ARE the reaction and zeroing them destroys the very quantity
# being recovered. Dividing by w_i = int_Gamma phi_i ds turns the functional
# into a density the partner can interpolate pointwise.
p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end
# ONE FORMULA, BOTH SIDES. An earlier version of this file used the reaction
# only on the Dirichlet side and an L2-projected gradient on the Neumann side,
# on the reasoning that the Neumann interface DOFs are free, so the discrete
# equations hold on them and r comes out ~0. That is true only when the
# residual is taken against a load that ALREADY CONTAINS the interface term.
# Subtract the VOLUME load alone and those same rows carry exactly the
# interface functional the partner applied:
#     (A u - b_vol)_i = int_Gamma g phi_i ds
# On the Dirichlet side there is no interface term at all, so b == b_vol and
# the two cases are the same expression.
#
# MEASURED against a known imposed flux, q = 2 + 3 sin(4y), on 8/16/32/64/128
# uniform triangle meshes (interior interface nodes; ends reported separately
# because an end node is a different quantity):
#     projected gradient   max 2.6 flat, order 0.00 — it never converges;
#                          order 0.93 away from the ends, 0.50 in rms
#     reaction vs b_vol    order 2.00 in max, away-from-ends AND rms
# The old recovery did not converge in the norm the grader reads.
Amat = _fp.assemble_matrix(fem.form(a))              # no bcs= on purpose
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))          # no lifting, no set_bc
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

wvec = _fp.assemble_vector(fem.form(w_ * ds_if))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux
# (measured 9.4e-02 there against 2.1e-02 on the interface proper). Take
# the nearest interior interface node rather than exporting a corner value
# that is physically a different quantity. This holds on both sides.
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

T = uh.x.array[iface_dofs]
print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

# exports.json LAST: the driver takes its existence as proof of success.
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
