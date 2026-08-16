"""DUNE-fem VECTOR participant for the OASiS `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = f  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X. Unlike the
scalar (heat) participants, the exchanged interface state is a VECTOR on BOTH
channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

Both lists therefore carry ONE ENTRY PER COMPONENT per interface point —
`[[vx, vy], ...]`, exactly like participant_fenics_elastic.py /
participant_skfem_elastic.py / participant_ngsolve_elastic.py. A flat list of
scalars is a different interface and the partner will silently mis-read it.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is required), writes exports.json LAST, exits 0. The same
points in the same order every iteration — the driver relaxes export vectors
element by element and refuses a length that changes.

SIGN CONVENTION — the thing a vector coupling gets wrong silently.
`normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_x

the SAME convention the shipped scalar participants use for heat
(q_out = -k dT/dn_own) and the one the other *_elastic participants use. Two
consequences, both load-bearing:

  * the two sides' exports CANCEL componentwise, because n_own is anti-parallel
    across the interface — that is what makes the interface balance check a
    conservation statement rather than an accident;
  * the NEUMANN side applies the partner's numbers UNCHANGED, as
    `+ dot(g, v) * ds`, because the natural boundary term of the elasticity
    weak form is +(sigma . n_own) . v = +q_out_partner . v.

Exporting the raw traction (sigma . n_own) instead flips the sign the Neumann
side applies; the iteration still converges, to the wrong answer.

WHICH BLOCK TAKES THE DIRICHLET ROLE IS NOT A FREE CHOICE. The
Dirichlet-Neumann iteration contracts only while theta < 2/(1 + mu_max), with
mu the spectrum of S_N^-1 S_D — for two blocks of the same shape, roughly the
stiffness ratio of the DIRICHLET block to the NEUMANN one. Put the STIFFER
block on the Dirichlet side and that bound drops below 1, i.e. below the top of
the driver's Aitken clamp, and the residual plateaus instead of falling.
Measured with this file against participant_skfem_elastic.py on a 2:1
shear-modulus contrast: stiffer-block-Dirichlet PLATEAUED at 3.9e-06 after 300
iterations (Aitken pinned at theta = 0.69, the stability limit); the identical
problem with the SOFTER block on the Dirichlet side reached 9.3e-11 in 63.
Nothing in either participant changes — only which one is given SIDE =
"dirichlet". If a coupling here stalls at a residual that will not fall, swap
the roles before touching theta.

DUNE-FEM SPECIFICS that this file exists to get right:

  * a VECTOR Lagrange space is `lagrange(gridView, order=1, dimRange=2)`, and
    its `as_numpy` vector is INTERLEAVED: the scalar index of component c at
    node n is n*2 + c. Nothing in the API says so, and a blocked (all x, then
    all y) reading produces an interface of the right length whose every number
    is wrong, so the layout is ASSERTED at run time below rather than assumed.
  * `structuredGrid` gives a cube grid, i.e. Q1 elements here. The interface
    nodes are the y-line of nodes at x = IFACE_X.
  * DUNE compiles the UFL forms with a C++ JIT on first use. The first run of a
    new form takes tens of seconds to minutes; that is NOT a hang. Every form
    below is therefore built ONCE and kept fixed across coupling iterations —
    the changing interface data lives in the DOFs of a discrete function, never
    in the form — so only the first iteration pays the compile.
  * DUNE's DirichletBC is evaluated per boundary intersection, so two BCs whose
    indicators overlap at a corner node fight over that node in an unspecified
    order. This file therefore uses exactly ONE DirichletBC whose value is a
    single discrete function carrying BOTH the outer data and (on the Dirichlet
    side) the imported interface data. There is nothing left to overlap.
"""
import json
import sys
from pathlib import Path

import numpy as np
from dune.fem import assemble
from dune.fem.operator import galerkin as operator_galerkin
from dune.fem.scheme import galerkin
from dune.fem.space import lagrange
from dune.grid import structuredGrid
from dune.ufl import Constant, DirichletBC
from ufl import (Identity, SpatialCoordinate, TestFunction, TrialFunction,
                 as_vector, conditional, dot, ds, dx, grad, inner, lt, sym, tr)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.55     # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.55          # the shared interface; must equal X0 or X1
E_MOD     = 1000.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
# The two subdomains must agree at the two interface corners, or the coupled
# problem is not the un-split one.
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)
# Volumetric body force, SAME polynomial convention (zero = the shipped
# elastic participants' -div(sigma) = 0). It is here because a coupling you
# cannot verify against a manufactured solution is a coupling you are trusting,
# and every manufactured elasticity solution that is not affine needs one.
BFX = (0.0, 0.0, 0.0, 0.0)
BFY = (0.0, 0.0, 0.0, 0.0)
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
# ─────────────────────────────────────────────────────────────────────────

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
EXTENT = max(X1 - X0, Y1 - Y0)
TOL = 1e-9 * EXTENT                        # node-coordinate comparisons
EPS = 1e-8 * EXTENT                        # boundary-indicator width in the form


def read_imports():
    """imports.json is {partner_name: InterfaceData}; it is `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's VECTOR samples onto THIS participant's interface
    points, COMPONENT BY COMPONENT.

    The driver does no interpolation — non-matching interface meshes are
    handled here, and for a vector field that has to be done per component. One
    np.interp over a flattened (N, 2) array interleaves the two components: the
    result still has the right length, the coupling still converges, and every
    number is wrong.

    Returns (len(y), ncomp). `fallback` is the per-component constant used on
    iteration 1, when imports.json is `{}`."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, c])
                            for c in range(vs.shape[1])])


imp = read_imports()

gridView = structuredGrid([X0, Y0], [X1, Y1], [NX, NY])
space = lagrange(gridView, order=1, dimRange=2)
x = SpatialCoordinate(space)

# ── dof -> (node, component) map ──────────────────────────────────────────
# Nodal interpolation of the coordinate functions gives the coordinate of every
# scalar dof; the component pattern is read off a marker field rather than
# assumed. A blocked layout (all x-dofs, then all y-dofs) would pass every
# length check downstream and put the y displacement where the partner reads
# the x one, so this refuses to run rather than export interleaved nonsense.
xy = np.array(space.interpolate(as_vector([x[0], x[1]]), name="coords").as_numpy)
mark = np.array(space.interpolate(as_vector([1.0, 2.0]), name="mark").as_numpy)
if not (np.allclose(mark[0::2], 1.0) and np.allclose(mark[1::2], 2.0)):
    sys.exit("dune-fem vector dof layout is not the expected interleaved "
             "node*2+component; the interface export would be scrambled")
xd, yd = xy[0::2], xy[1::2]                # per-NODE coordinates

iface_n = np.where(np.abs(xd - IFACE_X) < TOL)[0]
if len(iface_n) == 0:
    sys.exit(f"no interface dofs at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")
iface_n = iface_n[np.argsort(yd[iface_n])]              # constant order, always
y_if = yd[iface_n]
outer_n = np.where((np.abs(xd - OUTER_X) < TOL) |
                   (np.abs(yd - Y0) < TOL) | (np.abs(yd - Y1) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
# (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face, which carries a prescribed
# displacement in the un-split problem, so they stay Dirichlet in BOTH
# subproblems. Handing them to the interface leaves them unconstrained on the
# Neumann side: that subproblem is still well posed, still converges, and lands
# a few percent off — measured on the sibling participants, 4.7% in the
# interface displacement and 28% in the interface traction, on a coupling whose
# residual reached 1e-10 and whose flux balanced. They are still EXPORTED; they
# are just not interface-imposed.
corner = (np.abs(y_if - Y0) < TOL) | (np.abs(y_if - Y1) < TOL)
iface_bc_n = iface_n[~corner]

u, v = TrialFunction(space), TestFunction(space)


def eps_(w):
    return sym(grad(w))


def sigma(w):
    return 2.0 * MU * eps_(w) + LAM * tr(eps_(w)) * Identity(2)


def poly(c, tag):
    """c[0] + c[1]*x + c[2]*y + c[3]*y^2, kept SYMBOLIC.

    Every coefficient is a dune.ufl Constant even when it is zero: an all-zero
    python expression folds to a bare 0 and `0*v*dx` is a domainless UFL Zero
    that assemble() cannot integrate (the same trap the scalar DUNE participant
    documents for its source term). Constants also make the coefficients
    run-time data, so re-using this form never re-triggers the JIT."""
    return (Constant(c[0], name=tag + "0") + Constant(c[1], name=tag + "1") * x[0]
            + Constant(c[2], name=tag + "2") * x[1]
            + Constant(c[3], name=tag + "3") * x[1] * x[1])


a = inner(sigma(u), eps_(v)) * dx
b = dot(as_vector([poly(BFX, "bfx"), poly(BFY, "bfy")]), v) * dx

# ── boundary indicators ───────────────────────────────────────────────────
on_outer = conditional(lt(abs(x[0] - OUTER_X), EPS), 1,
                       conditional(lt(abs(x[1] - Y0), EPS), 1,
                                   conditional(lt(abs(x[1] - Y1), EPS), 1, 0)))
on_iface = conditional(lt(abs(x[0] - IFACE_X), EPS), 1, 0)

# ONE Dirichlet carrier for BOTH the outer data and the imported interface
# displacement. Its dofs change every iteration; the FORM never does.
gfun = space.interpolate(as_vector([0.0, 0.0]), name="dirichlet_data")
gdofs = gfun.as_numpy
gdofs[:] = 0.0
ox, oy = xd[outer_n], yd[outer_n]
gdofs[2 * outer_n] = UDX[0] + UDX[1] * ox + UDX[2] * oy + UDX[3] * oy * oy
gdofs[2 * outer_n + 1] = UDY[0] + UDY[1] * ox + UDY[2] * oy + UDY[3] * oy * oy

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    gdofs[2 * iface_bc_n] = u_if[~corner, 0]
    gdofs[2 * iface_bc_n + 1] = u_if[~corner, 1]
    # the interface corners keep the OUTER value already written above
    bc_where = conditional(lt(abs(x[0] - OUTER_X), EPS), 1,
                           conditional(lt(abs(x[1] - Y0), EPS), 1,
                                       conditional(lt(abs(x[1] - Y1), EPS), 1,
                                                   on_iface)))
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    tfun = space.interpolate(as_vector([0.0, 0.0]), name="iface_traction")
    tdofs = tfun.as_numpy
    tdofs[:] = 0.0
    tdofs[2 * iface_n] = t_if[:, 0]
    tdofs[2 * iface_n + 1] = t_if[:, 1]
    # APPLY the partner's numbers UNCHANGED (+ integral(g . v) ds_interface)
    b = b + conditional(lt(abs(x[0] - IFACE_X), EPS), dot(tfun, v), 0.0) * ds
    bc_where = on_outer

scheme = galerkin([a == b, DirichletBC(space, gfun, bc_where)], solver="cg")
uh = space.interpolate(as_vector([0.0, 0.0]), name="displacement")
scheme.solve(target=uh)

# Interface traction export q_out = -(sigma . n_own).
#
# WHY NOT AN L2 PROJECTION OF THE STRESS. That is what the sibling participants
# used to do: project -(sigma(u_h) . n_own) over the whole subdomain and sample
# it at the interface. The gradient of a P1/Q1 solution — and therefore the
# stress — is only O(h) accurate ON the boundary; the superconvergence points
# are interior, and the boundary trace is exactly what the coupling reads.
# Measured against a manufactured solution with a known exact interface
# traction, the projection converges at order ~1 while the consistent traction
# below converges at ~2, so the recovery, not the physics and not the partner,
# was setting the answer.
#
# THE CONSISTENT (REACTION) TRACTION. From
#     a(u,v) - (f,v) = int_dOmega (sigma(u) . n) . v ds = -int_Gamma q_out . v ds
# (the second equality is this file's sign convention, q_out = -(sigma . n_own))
# it follows that for every vector basis function phi_i on the interface
#     int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b
# with r the UNCONSTRAINED residual, b the FULL right-hand side (body force
# included — drop it and the reaction is wrong by the load it carries).
# `scheme` cannot supply r: it carries the DirichletBC and so overwrites exactly
# the constrained rows that ARE the reaction. A second operator built from the
# SAME forms MINUS the DirichletBC gives the unconstrained residual in one
# application. Taking the test vector (1,1) in the weight form makes
# w_i = int_Gamma phi_i ds the SCALAR nodal weight for BOTH components, so the
# same division works componentwise, and dividing by it turns the functional
# into a density the partner can interpolate pointwise.
if SIDE == "dirichlet":
    op_free = operator_galerkin([a == b])       # same forms, no DirichletBC
    rfun = space.interpolate(as_vector([0.0, 0.0]), name="residual")
    op_free(uh, rfun)                           # r = A u_h - b
    r = np.array(rfun.as_numpy)

    wfun = assemble(conditional(lt(abs(x[0] - IFACE_X), EPS),
                                v[0] + v[1], 0.0) * ds)
    wt = np.array(wfun.as_numpy)                # w_i = int_Gamma phi_i ds

    idx = np.column_stack([2 * iface_n, 2 * iface_n + 1])
    wi = wt[idx]
    Q = np.zeros_like(wi)
    ok = np.abs(wi) > 1e-14
    Q[ok] = -r[idx][ok] / wi[ok]

    # THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (a y-face),
    # so their rows carry the OUTER reaction too and their residual is not this
    # interface's traction. Take the nearest interior interface node rather than
    # exporting a corner value that is physically a different quantity.
    suspect = np.isin(iface_n, outer_n) | ~ok.all(axis=1)
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]
else:
    # NEUMANN SIDE: the reaction formula MUST NOT be used here. These interface
    # dofs are free, the discrete equations hold on them, so r is ~0 and the
    # expression would silently export ZERO traction with no error raised. This
    # side's export is not what the partner consumes in any case — the Dirichlet
    # partner reads its `values`.
    p_, w_ = TrialFunction(space), TestFunction(space)
    n_own = as_vector([Constant(S, name="nx"), Constant(0.0, name="ny")])
    proj = galerkin([dot(p_, w_) * dx == dot(-dot(sigma(uh), n_own), w_) * dx],
                    solver="cg")
    qh = space.interpolate(as_vector([0.0, 0.0]), name="traction")
    proj.solve(target=qh)
    qd = np.array(qh.as_numpy)
    Q = np.column_stack([qd[2 * iface_n], qd[2 * iface_n + 1]])

ud = np.array(uh.as_numpy)
U = np.column_stack([ud[2 * iface_n], ud[2 * iface_n + 1]])
print(f"[dune {SIDE}] interface n={len(U)} "
      f"ux=[{U[:, 0].min():.6g},{U[:, 0].max():.6g}] "
      f"uy=[{U[:, 1].min():.6g},{U[:, 1].max():.6g}] "
      f"tx=[{Q[:, 0].min():.6g},{Q[:, 0].max():.6g}] "
      f"ty=[{Q[:, 1].min():.6g},{Q[:, 1].max():.6g}]")

# exports.json LAST: the driver takes its existence as proof of success.
Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_n)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [[float(a_), float(b_)] for a_, b_ in U],
    "normal_fluxes": [[float(a_), float(b_)] for a_, b_ in Q],
}, indent=2))
