"""NGSolve path-walk participant: scalar or vector, either role, either axis.

Everything problem-specific comes from cfg.json. See wcommon.py for what a walk
is and what it is not.

FLUX / TRACTION RECOVERY. The Dirichlet side exports the CONSISTENT (reaction)
flux, not an L2 projection of the gradient. Measured on D1 with the same meshes,
driver and tolerance: the projection gives an interface-trace order of 1.33 and a
field order of 1.83, the consistent recovery gives 2.73 and 2.05, and it costs
three extra coupling iterations out of thirty. The recovery, not the physics and
not the partner, sets the graded order. NGSolve's assembled ``a.mat`` and
``f.vec`` are exactly the unconstrained system -- the essential condition lives in
``fes.FreeDofs()`` at solve time and never touches the operator -- so the
constrained rows still carry the reaction.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

from ngsolve import (H1, VectorH1, BilinearForm, LinearForm, GridFunction,   # noqa: E402
                     CoefficientFunction, TaskManager, NodeId, VERTEX,
                     grad, dx, ds, x, y, z, Id, Sym, Trace, InnerProduct)
from ngsolve.meshes import MakeStructured2DMesh                  # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
vec = cfg["physics"] == "vector"
ncomp = dim if vec else 1
S = W.outward_sign(ext, axis, xi)
free_axes = [i for i in range(dim) if i != axis]
COORD = (x, y, z)[:dim]

# MakeStructured2DMesh names its four edges bottom / right / top / left. The
# interface is whichever one carries x_axis = xi; everything else is outer.
EDGE = {(0, 1.0): "right", (0, -1.0): "left",
        (1, 1.0): "top", (1, -1.0): "bottom"}
iface_bc = EDGE[(axis, S)]
outer_bc = "|".join(e for e in ("bottom", "right", "top", "left")
                    if e != iface_bc)

mesh = MakeStructured2DMesh(
    quads=False, nx=n[0], ny=n[1],
    mapping=lambda a, b: (ext[0][0] + (ext[0][1] - ext[0][0]) * a,
                          ext[1][0] + (ext[1][1] - ext[1][0]) * b))

dirich = outer_bc + ("|" + iface_bc if cfg["side"] == "dirichlet" else "")
fes = (VectorH1 if vec else H1)(mesh, order=1, dirichlet=dirich)
u, v = fes.TnT()

nv = mesh.nv
vxy = np.array([mesh.vertices[i].point for i in range(nv)], float)
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i)) for i in range(nv)], int)

iv = np.where(np.abs(vxy[:, axis] - xi) < 1e-9)[0]
iv = iv[np.lexsort(tuple(vxy[iv, a] for a in reversed(free_axes)))]
ipts = vxy[iv]
idofs = vdof[iv]
outer_v = np.zeros(nv, bool)
for a in range(dim):
    for val in ext[a]:
        if a == axis and abs(val - xi) < 1e-9:
            continue
        outer_v |= np.abs(vxy[:, a] - val) < 1e-9
outer_dofs = set(vdof[outer_v].ravel().tolist())

# ── operator ──────────────────────────────────────────────────────────
src = cfg["source"]
if vec:
    lam, mu = float(cfg["lam"]), float(cfg["mu"])

    def eps(w):
        return Sym(grad(w))

    a = BilinearForm(fes)
    a += (2 * mu * InnerProduct(eps(u), eps(v))
          + lam * Trace(eps(u)) * Trace(eps(v))) * dx
    fvec = [eval(e, {"x": x, "y": y, "z": z, "exp": None}) for e in src]
    fexpr = CoefficientFunction(tuple(fvec))
    f = LinearForm(fes)
    f += InnerProduct(fexpr, v) * dx
else:
    K = np.asarray(cfg["K"], float)
    Kc = CoefficientFunction(tuple(K.ravel()), dims=(dim, dim))
    a = BilinearForm(fes)
    a += InnerProduct(Kc * grad(u), grad(v)) * dx
    if cfg.get("reaction"):
        a += float(cfg["reaction"]) * u * v * dx
    fexpr = eval(src, {"x": x, "y": y, "z": z})
    f = LinearForm(fes)
    f += fexpr * v * dx

gfu = GridFunction(fes)
gfu.vec[:] = 0.0

imp = W.read_imports(cfg["partner"])
if cfg["side"] == "dirichlet":
    g = W.sample(imp, "values", ipts, 0.0, ncomp, free_axes)
    # THE INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY ON BOTH SIDES. They lie
    # on an outer face as well as on the interface, and in the un-split problem
    # they carry the prescribed datum. Overwriting them with the partner's value
    # makes the two subproblems disagree there by O(1) forever: the driver
    # residual then oscillates near 1 and never falls, with both fields already
    # close to correct. They are still exported, just not interface-imposed.
    for v_i, row, vals in zip(iv, idofs, g):
        if outer_v[v_i]:
            continue
        for d, val in zip(row, np.atleast_1d(vals)):
            gfu.vec[int(d)] = float(val)
else:
    q = W.sample(imp, "normal_fluxes", ipts, 0.0, ncomp, free_axes)
    gq = GridFunction(fes)
    gq.vec[:] = 0.0
    for row, vals in zip(idofs, q):
        for d, val in zip(row, np.atleast_1d(vals)):
            gq.vec[int(d)] = float(val)
    # APPLY THE PARTNER'S NUMBER UNCHANGED. Both sides export with respect to
    # their OWN outward normal, so the two exports carry opposite signs and the
    # natural term for this side is exactly the partner's number.
    f += (InnerProduct(gq, v) if vec else gq * v) * ds(iface_bc)

with TaskManager():
    a.Assemble()
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="umfpack") * res

    if cfg["side"] == "dirichlet":
        r = f.vec.CreateVector()
        r.data = a.mat * gfu.vec - f.vec        # unconstrained residual
        fw = LinearForm(fes)
        one = CoefficientFunction(tuple([1.0] * dim)) if vec else 1.0
        fw += (InnerProduct(one, v) if vec else v) * ds(iface_bc)
        fw.Assemble()
        Q = np.zeros((len(iv), ncomp))
        for i, row in enumerate(idofs):
            for c, d in enumerate(row):
                w_i = fw.vec[int(d)]
                Q[i, c] = -r[int(d)] / w_i if abs(w_i) > 1e-14 else np.nan
        # An interface node that ALSO lies on the outer Dirichlet boundary
        # carries the OUTER reaction as well, so its residual is not this
        # interface's flux. Take the nearest interior interface node.
        bad = np.array([any(int(d) in outer_dofs for d in row) for row in idofs])
        bad |= ~np.isfinite(Q).all(axis=1)
        good = np.where(~bad)[0]
        for i in np.where(bad)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]
    else:
        # The reaction formula MUST NOT be used here: these dofs are free, the
        # discrete equations hold on them, r is ~0, and the expression would
        # silently export ZERO with no error raised. The Dirichlet partner reads
        # this side's `values`, not its flux, so an O(h) projection is harmless.
        fq = (VectorH1 if vec else H1)(mesh, order=1)
        p, w = fq.TnT()
        m = BilinearForm(fq)
        m += (InnerProduct(p, w) if vec else p * w) * dx
        m.Assemble()
        if vec:
            sig = 2 * mu * Sym(grad(gfu)) + lam * Trace(Sym(grad(gfu))) * Id(dim)
            nvecc = CoefficientFunction(tuple(S if i == axis else 0.0
                                              for i in range(dim)))
            rhs = InnerProduct(-(sig * nvecc), w)
        else:
            gr = grad(gfu)
            kn = sum(float(K[axis, j]) * gr[j] for j in range(dim))
            rhs = (-S * kn) * w
        lf = LinearForm(fq)
        lf += rhs * dx
        lf.Assemble()
        qh = GridFunction(fq)
        qh.vec.data = m.mat.Inverse(fq.FreeDofs(), inverse="umfpack") * lf.vec
        qd = np.array([fq.GetDofNrs(NodeId(VERTEX, int(i))) for i in iv], int)
        Q = np.array([[qh.vec[int(d)] for d in row] for row in qd], float)

U = np.array([[gfu.vec[int(d)] for d in row] for row in idofs], float)
allU = np.array([[gfu.vec[int(d)] for d in row] for row in vdof], float)

W.write_nodes("nodes.csv", vxy, allU)
W.write_log(cfg, fes.ndof, f"ne = {mesh.ne}  nv = {mesh.nv}")
print(f"[ngsolve {cfg['sidename']} {cfg['side']}] ndof={fes.ndof} "
      f"iface_n={len(iv)} u=[{U.min():.6g},{U.max():.6g}] "
      f"q=[{Q.min():.6g},{Q.max():.6g}]")
W.write_exports(ipts, U, Q, "displacement" if vec else "temperature")
