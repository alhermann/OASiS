"""DUNE-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from dune.grid import structuredGrid
from dune.fem import assemble
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.fem.operator import galerkin as operator_galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 conditional, dot, ds, dx, grad, lt)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
#    this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.6
K         = 0.8


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
T_OUTER   = 320.0
NX, NY    = 24, 16
T_INIT    = 310.0
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

OUTER_X = X0 if IFACE_X == X1 else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S * e_x
EPS = 1e-8                                 # boundary-indicator tolerance


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

gridView = structuredGrid([X0, Y0], [X1, Y1], [NX, NY])
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)

# dof -> coordinate map: nodal interpolation of the coordinate functions
xd = np.array(space.interpolate(x[0], name="xcoord").as_numpy)
yd = np.array(space.interpolate(x[1], name="ycoord").as_numpy)
iface_dofs = np.where(np.abs(xd - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(yd[iface_dofs])]
y_if = yd[iface_dofs]
outer_dofs = np.where(np.abs(xd - OUTER_X) < 1e-10)[0]

u, v = TrialFunction(space), TestFunction(space)
a = K * dot(grad(u), grad(v)) * dx

# SOURCE. F_SRC is sampled at the nodes and carried by a DISCRETE FUNCTION,
# the same device this file uses for the interface data, for two DUNE-specific
# reasons. A UFL expression built straight out of F_SRC folds to a bare 0 when
# F_SRC returns zero, and `0*v*dx` is a domainless UFL Zero that assemble()
# cannot integrate; a discrete function always carries its grid, so the shipped
# zero assembles like any other value. And its dofs are run-time data, so
# editing F_SRC never re-triggers the C++ JIT. `Constant(F_SRC, ...)`, which
# used to stand here, kept the form safe but held ONE NUMBER for the whole
# subdomain: a source that varies with position could not be written at all.
# order=1, so this is the P1 interpolant of the source; its quadrature error is
# O(h^2), the same order as the discretization error itself. Do NOT hand F_SRC
# the symbolic SpatialCoordinate `x` instead: a UFL expression carries no NumPy
# ufuncs, so np.sin(x) raises and np.zeros_like(x) returns a 0-d OBJECT array —
# the source collapses to a constant and this subdomain solves the wrong
# problem with no error raised.
ffun = space.interpolate(0, name="f_src")
fdofs = ffun.as_numpy
fdofs[:] = np.broadcast_to(np.asarray(F_SRC(xd, yd), float), xd.shape)
# THE VOLUME LOAD ALONE, kept under its own name. The Neumann branch adds the
# partner's interface term into `b`; the flux recovery at the bottom subtracts
# THIS, on both sides — subtracting the combined form is what made the reaction
# look like zero on the Neumann side.
b_vol = ffun * v * dx
b = b_vol

bcs = [DirichletBC(space, T_OUTER, conditional(lt(abs(x[0] - OUTER_X), EPS), 1, 0))]

# coupling data carrier: a discrete function whose interface dofs hold the
# imported samples (the FORM never changes between iterations -> no re-JIT)
gfun = space.interpolate(0, name="iface_data")
gdofs = gfun.as_numpy
gdofs[:] = 0.0

if SIDE == "dirichlet":
    gdofs[iface_dofs] = sample(imp, "values", T_INIT, y_if)
    bcs.append(DirichletBC(space, gfun,
                           conditional(lt(abs(x[0] - IFACE_X), EPS), 1, 0)))
else:
    gdofs[iface_dofs] = sample(imp, "normal_fluxes", Q_INIT, y_if)
    # APPLY the partner's number unchanged: + int(g*v) ds on the interface
    b = b + conditional(lt(abs(x[0] - IFACE_X), EPS), gfun * v, 0.0) * ds

scheme = galerkin([a == b] + bcs, solver="cg")
uh = space.interpolate(0, name="temperature")
scheme.solve(target=uh)

# Outward normal flux density q = -(k grad T).n on the interface.
#
# WHY NOT AN L2 PROJECTION OF THE GRADIENT. That is what this file used to do:
# project -k dT/dx over the whole subdomain and sample it at the interface. The
# gradient of a P1/Q1 solution is only O(h) accurate ON the boundary — the
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
# with r the UNCONSTRAINED residual. `scheme` cannot supply it: it carries the
# DirichletBCs and so overwrites exactly the constrained rows that ARE the
# reaction. A second operator built from the SAME form MINUS the bcs gives the
# unconstrained residual in one application. Dividing by
# w_i = int_Gamma phi_i ds turns the functional into a density the partner can
# interpolate pointwise.
# ONE FORMULA, BOTH SIDES. An earlier version used the reaction on the
# Dirichlet side and an L2-projected gradient on the Neumann side, reasoning
# that the Neumann interface dofs are free so r comes out ~0 there. That holds
# only when the residual is taken against a load that ALREADY CONTAINS the
# interface term. Against the VOLUME load alone those same rows carry exactly
# the interface functional the partner applied. On the Dirichlet side there is
# no interface term, so b == b_vol and the two cases are one expression.
#
# MEASURED (FEniCSx, same formulation) against a known imposed flux
# q = 2 + 3 sin(4y) on 8/16/32/64/128 uniform triangle meshes, interior
# interface nodes: the projected gradient stalls at max 2.6 and does NOT
# converge (order 0.00; 0.93 away from the ends, 0.50 in rms), while the
# reaction against b_vol converges at order 2.00 in all three norms.
op_free = operator_galerkin([a == b_vol])   # volume load, no DirichletBC
rfun = space.interpolate(0, name="residual")
op_free(uh, rfun)                           # r = A u_h - b_vol
r = np.array(rfun.as_numpy)

wfun = assemble(conditional(lt(abs(x[0] - IFACE_X), EPS), v, 0.0) * ds)
wt = np.array(wfun.as_numpy)                # w_i = int_Gamma phi_i ds

Q = np.zeros(len(iface_dofs))
ok = np.abs(wt[iface_dofs]) > 1e-14
Q[ok] = -r[iface_dofs][ok] / wt[iface_dofs][ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux.
# Take the nearest interior interface node rather than exporting a corner
# value that is physically a different quantity.
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]
T_dofs = np.array(uh.as_numpy)

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(t) for t in T_dofs[iface_dofs]],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
