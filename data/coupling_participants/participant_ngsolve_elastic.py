"""NGSolve VECTOR participant for the OASiS `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = 0  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X. Unlike the
scalar (heat) participants, the exchanged interface state is a VECTOR on BOTH
channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

SIGN CONVENTION — the thing a vector coupling gets wrong silently.
`normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_x

the SAME convention the shipped scalar participants use for heat
(q_out = -k dT/dn_own). The two sides' exports therefore CANCEL componentwise,
and the NEUMANN side applies the partner's numbers UNCHANGED
(`f += InnerProduct(g, v) * ds("interface")`), because the natural boundary
term of the elasticity weak form is +(sigma . n_own) . v = +q_out_partner . v.

NGSolve's VectorH1 is BLOCKED BY COMPONENT: GetDofNrs(NodeId(VERTEX, i))
returns (dof of u_x, dof of u_y) and those two indices are nv apart, not
adjacent. Writing nodal values with an adjacency assumption silently scatters
u_y into the u_x block.
"""
import json
from pathlib import Path

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, CF, GridFunction, InnerProduct,
                     LinearForm, Mesh, NodeId, TaskManager, VectorH1, ds, dx,
                     grad)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.55     # this subdomain
Y0, Y1    = 0.0, 0.4
IFACE_X   = 0.55          # shared interface (must be X0 or X1)
E_MOD     = 1000.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*y*y
UDX = (0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0)


def B_SRC(x, y):
    """Body force per unit volume, (b_x, b_y), as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants: with displacement prescribed
    on the whole outer boundary and no body force, the only solution is
    u = 0 everywhere, and the coupling will converge beautifully to it.

    If your problem states a body force, or gives you a manufactured solution
    whose source term you derived, put it here. `x` and `y` are NumPy arrays,
    so build the answer with NumPy and return two arrays of the same shape:

        return (2.0 * MU * np.pi**2 * np.sin(np.pi * x) * np.cos(np.pi * y),
                np.zeros_like(x))
    """
    return np.zeros_like(x), np.zeros_like(y)
NX, NY    = 24, 16        # this subdomain's own mesh (netgen maxh derived below)
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
# ─────────────────────────────────────────────────────────────────────────

MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)    # netgen's scalar mesh size
ORDER = 1                                     # nodal == vertex dofs

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

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
    """Map the partner's VECTOR samples onto this participant's y-coordinates,
    COMPONENT BY COMPONENT. One np.interp over a flattened (N, 2) array
    interleaves the components: right length, converges, every number wrong.

    Returns (len(y), ncomp)."""
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

# ── mesh: SplineGeometry.AddRectangle edge order is bottom, right, top, left ──
# Everything that is not the interface carries the prescribed displacement, so
# it all gets the same boundary name.
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("outer", "interface", "outer", "outer") if ON_RIGHT else
                      ("outer", "outer", "outer", "interface")))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = VectorH1(mesh, order=ORDER,
               dirichlet=("outer|interface" if SIDE == "dirichlet" else "outer"))
u, v = fes.TnT()

# vertex -> (x-dof, y-dof). VectorH1 blocks by COMPONENT, so these are nv apart.
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[:2] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]           # sorted by y
y_if = vxy[iface_v, 1]
outer_v = np.where((np.abs(vxy[:, 0] - OUTER_X) < TOL) |
                   (np.abs(vxy[:, 1] - Y0) < TOL) |
                   (np.abs(vxy[:, 1] - Y1) < TOL))[0]
# THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES: they
# sit on a y-face, which carries a prescribed displacement in the un-split
# problem. Leaving them to the interface leaves them unconstrained on the
# Neumann side — the subproblem is still well posed, still converges, and lands
# a few percent off. They are still EXPORTED, just not interface-imposed.
corner = (np.abs(y_if - Y0) < TOL) | (np.abs(y_if - Y1) < TOL)


def eps_of(g):
    exx, eyy = g[0, 0], g[1, 1]
    return exx, eyy, 0.5 * (g[0, 1] + g[1, 0])


a = BilinearForm(fes)
gu, gv = grad(u), grad(v)
eu = eps_of(gu)
ev = eps_of(gv)
a += (2.0 * MU * (eu[0] * ev[0] + eu[1] * ev[1] + 2.0 * eu[2] * ev[2])
      + LAM * (eu[0] + eu[1]) * (ev[0] + ev[1])) * dx
f = LinearForm(fes)
# BODY FORCE. B_SRC is sampled at the vertices and carried by a GridFunction,
# which IS a CoefficientFunction, so the linear form's source varies in space
# instead of being the constant it used to be. ORDER = 1, so this is the P1
# interpolant of the source; its quadrature error is O(h^2), the same order as
# the P1 discretization error itself. Do NOT hand B_SRC ngsolve's symbolic x, y
# instead: a CoefficientFunction carries no NumPy ufuncs, so np.sin(x) raises
# and np.zeros_like(x) returns a 0-d OBJECT array — the source collapses to a
# constant and the subdomain solves the wrong problem with no error raised.
# Nodal writes go through vdof, because VectorH1 blocks BY COMPONENT.
gfb = GridFunction(fes)
gfb.vec[:] = 0.0
bx, by = B_SRC(vxy[:, 0], vxy[:, 1])
bvals = gfb.vec.FV().NumPy()
bvals[vdof[:, 0]] = np.broadcast_to(np.asarray(bx, float), (mesh.nv,))
bvals[vdof[:, 1]] = np.broadcast_to(np.asarray(by, float), (mesh.nv,))
f += InnerProduct(gfb, v) * dx

gfu = GridFunction(fes)                    # also carries the Dirichlet data
gfu.vec[:] = 0.0

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
    for k, vtx in enumerate(iface_v):
        if corner[k]:
            continue
        gfu.vec[int(vdof[vtx, 0])] = float(u_if[k, 0])
        gfu.vec[int(vdof[vtx, 1])] = float(u_if[k, 1])
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for k, vtx in enumerate(iface_v):
        gfun.vec[int(vdof[vtx, 0])] = float(t_if[k, 0])
        gfun.vec[int(vdof[vtx, 1])] = float(t_if[k, 1])
    # APPLY the partner's numbers UNCHANGED
    f += InnerProduct(gfun, v) * ds("interface")

# The outer boundary is written LAST so it wins at the two interface corners.
ox, oy = vxy[outer_v, 0], vxy[outer_v, 1]
oux = UDX[0] + UDX[1] * ox + UDX[2] * oy + UDX[3] * oy * oy
ouy = UDY[0] + UDY[1] * ox + UDY[2] * oy + UDY[3] * oy * oy
for k, vtx in enumerate(outer_v):
    gfu.vec[int(vdof[vtx, 0])] = float(oux[k])
    gfu.vec[int(vdof[vtx, 1])] = float(ouy[k])

with TaskManager():
    a.Assemble()
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(),
                                  inverse="sparsecholesky") * res

    # Interface traction export q_out = -(sigma . n_own).
    #
    # WHY NOT AN L2 PROJECTION OF THE STRESS. That is what this file used to do:
    # project -(sigma(u_h) . n_own) over the whole subdomain and sample it at
    # the interface. The gradient of a P1 solution — and therefore the stress —
    # is only O(h) accurate ON the boundary; the superconvergence points are
    # interior, and the boundary trace is exactly what the coupling reads.
    # Measured against a manufactured solution with a known exact interface
    # traction, the projection converges at order ~1 while the consistent
    # traction below converges at ~2, so the recovery, not the physics and not
    # the partner, was setting the answer.
    #
    # THE CONSISTENT (REACTION) TRACTION. From
    #     a(u,v) - (f,v) = int_dOmega (sigma(u).n).v ds = -int_Gamma q_out.v ds
    # (the second equality is this file's sign convention) it follows that for
    # every vector basis function phi_i on the interface
    #     int_Gamma q_out . phi_i ds = -r_i,   r = A u_h - b
    # with r the UNCONSTRAINED residual: NGSolve's a.mat and f.vec are exactly
    # that — the Dirichlet condition lives in fes.FreeDofs() at solve time and
    # never touches the assembled operator, so the constrained rows still carry
    # the reaction. The test vector (1,1) in the weight form makes
    # w_i = int_Gamma phi_i ds the SCALAR nodal weight for BOTH components.
    if SIDE == "dirichlet":
        rvec = f.vec.CreateVector()
        rvec.data = a.mat * gfu.vec - f.vec       # r = A u_h - b, no bc applied
        fw = LinearForm(fes)
        fw += InnerProduct(CF((1.0, 1.0)), v) * ds("interface")
        fw.Assemble()

        Q = np.zeros((len(iface_v), 2))
        ok = np.ones((len(iface_v), 2), bool)
        for k, vtx in enumerate(iface_v):
            for c in (0, 1):
                d = int(vdof[vtx, c])
                wi = float(fw.vec[d])
                if abs(wi) > 1e-14:
                    Q[k, c] = -float(rvec[d]) / wi
                else:
                    ok[k, c] = False

        # THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (a
        # y-face), so their rows carry the OUTER reaction too and their residual
        # is not this interface's traction. Take the nearest interior interface
        # node rather than exporting a corner value that is physically a
        # different quantity.
        suspect = np.isin(iface_v, outer_v) | ~ok.all(axis=1)
        good = np.where(~suspect)[0]
        if len(good):
            for i in np.where(suspect)[0]:
                Q[i] = Q[good[np.argmin(np.abs(good - i))]]
    else:
        # NEUMANN SIDE: the reaction formula MUST NOT be used here. These
        # interface dofs are free, the discrete equations hold on them, so r is
        # ~0 and the expression would silently export ZERO traction with no
        # error raised. This side's export is not what the partner consumes in
        # any case — the Dirichlet partner reads its `values`.
        fesq = VectorH1(mesh, order=ORDER)
        p, w = fesq.TnT()
        m = BilinearForm(fesq)
        m += InnerProduct(p, w) * dx
        m.Assemble()
        exx, eyy, exy = eps_of(grad(gfu))
        sxx = 2.0 * MU * exx + LAM * (exx + eyy)
        sxy = 2.0 * MU * exy
        fq = LinearForm(fesq)
        fq += (-S) * (sxx * w[0] + sxy * w[1]) * dx
        fq.Assemble()
        qh = GridFunction(fesq)
        qh.vec.data = m.mat.Inverse(fesq.FreeDofs(),
                                    inverse="sparsecholesky") * fq.vec
        qdof = np.array([fesq.GetDofNrs(NodeId(VERTEX, int(i)))[:2]
                         for i in iface_v], int)
        Q = np.array([[float(qh.vec[int(d0)]), float(qh.vec[int(d1)])]
                      for d0, d1 in qdof])

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [[float(gfu.vec[int(vdof[i, 0])]), float(gfu.vec[int(vdof[i, 1])])]
               for i in iface_v],
    "normal_fluxes": [[float(q0), float(q1)] for q0, q1 in Q],
}, indent=2))
