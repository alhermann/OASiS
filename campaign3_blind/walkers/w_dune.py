"""DUNE-fem path-walk participant: scalar conduction OR elasticity, either role.

THE VECTOR BRANCH IS THE ONE THAT DID NOT EXIST. ``participant_dune.py`` is
scalar-only, and C12 pairs DUNE with FEBio -- which has no heat module at all, so
that cell cannot be scalar. Neither shipped participant can serve it. The
elasticity branch below is the DUNE vector participant written for it: a
``dimRange = dim`` Lagrange space, plane-strain stress, displacement continuity
and traction equilibrium across the interface.

The scalar branch additionally carries a reaction term, which the different-
operators cell needs on one side only, and a full conductivity tensor.

DOF ADDRESSING. dune-fem does not hand out a vertex-to-dof map, so it is built
by interpolating the coordinate functions themselves into the same space: the
value of ``interpolate(x_k)`` at a dof IS that dof's k-th coordinate. That is
exact for Lagrange spaces of any order and needs no assumption about ordering.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

import ufl                                                       # noqa: E402
from dune.grid import structuredGrid                             # noqa: E402
from dune.fem.space import lagrange                              # noqa: E402
from dune.fem.scheme import galerkin                             # noqa: E402
from dune.ufl import DirichletBC                                 # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
vec = cfg["physics"] == "vector"
ncomp = dim if vec else 1
free_axes = [i for i in range(dim) if i != axis]
S = W.outward_sign(ext, axis, xi)

gv = structuredGrid([e[0] for e in ext], [e[1] for e in ext], list(n))
space = lagrange(gv, order=1, dimRange=ncomp) if vec else lagrange(gv, order=1)
u = ufl.TrialFunction(space)
v = ufl.TestFunction(space)
xc = ufl.SpatialCoordinate(space)

# dof -> coordinate, by interpolating the coordinate functions themselves.
if vec:
    dofx = np.array([space.interpolate(
        ufl.as_vector([xc[k]] * ncomp), name=f"c{k}").as_numpy[:]
        for k in range(dim)]).T
else:
    dofx = np.array([space.interpolate(xc[k], name=f"c{k}").as_numpy[:]
                     for k in range(dim)]).T
NDOF = dofx.shape[0]
if vec:
    node_of = {}
    for j in range(NDOF):
        node_of.setdefault(tuple(np.round(dofx[j], 12)), []).append(j)
    pts = np.array([list(k) for k in node_of])
    dofs = np.array([node_of[tuple(np.round(p, 12))] for p in pts], int)
else:
    pts = dofx
    dofs = np.arange(NDOF).reshape(-1, 1)

ifm = np.abs(pts[:, axis] - xi) < 1e-9
inode = np.where(ifm)[0]
inode = inode[np.lexsort(tuple(pts[inode, a] for a in reversed(free_axes)))]
ipts = pts[inode]
outer = np.zeros(len(pts), bool)
for a in range(dim):
    for val in ext[a]:
        if a == axis and abs(val - xi) < 1e-9:
            continue
        outer |= np.abs(pts[:, a] - val) < 1e-9

# ── operator ──────────────────────────────────────────────────────────
src = cfg["source"]
NS = {"x": xc[0], "y": xc[1], "z": (xc[2] if dim > 2 else None),
      "exp": ufl.exp, "sin": ufl.sin, "cos": ufl.cos, "sqrt": ufl.sqrt}
if vec:
    lam, mu = float(cfg["lam"]), float(cfg["mu"])

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sig(w):
        return 2 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(dim)

    a_form = ufl.inner(sig(u), eps(v)) * ufl.dx
    fexpr = ufl.as_vector([eval(e, dict(NS)) for e in src])
    L_form = ufl.inner(fexpr, v) * ufl.dx
else:
    K = ufl.as_matrix([[float(k) for k in row] for row in cfg["K"]])
    a_form = ufl.inner(K * ufl.grad(u), ufl.grad(v)) * ufl.dx
    if cfg.get("reaction"):
        a_form = a_form + float(cfg["reaction"]) * u * v * ufl.dx
    L_form = eval(src, dict(NS)) * v * ufl.dx

imp = W.read_imports(cfg["partner"])
zero = ufl.as_vector([0.0] * ncomp) if vec else 0.0
on_outer = None
for a in range(dim):
    for val in ext[a]:
        if a == axis and abs(val - xi) < 1e-9:
            continue
        c = abs(xc[a] - val) < 1e-8
        on_outer = c if on_outer is None else ufl.Or(on_outer, c)
bcs = [DirichletBC(space, zero, on_outer)]

# The imported interface datum enters as a discrete function whose dofs are set
# directly, which is exact and needs no evaluation of a Python callable inside
# the assembly loop.
gh = space.interpolate(zero, name="gh")
gh_np = gh.as_numpy
if cfg["side"] == "dirichlet":
    g = W.sample(imp, "values", ipts, 0.0, ncomp, free_axes)
    for i, row in zip(inode, g):
        if outer[i]:
            continue        # the interface ends keep the OUTER datum, both sides
        for c, d in enumerate(dofs[i]):
            gh_np[int(d)] = float(np.atleast_1d(row)[c])
    on_iface = abs(xc[axis] - xi) < 1e-8
    bcs.append(DirichletBC(space, gh, on_iface))
else:
    q = W.sample(imp, "normal_fluxes", ipts, 0.0, ncomp, free_axes)
    for i, row in zip(inode, q):
        for c, d in enumerate(dofs[i]):
            gh_np[int(d)] = float(np.atleast_1d(row)[c])
    # APPLY THE PARTNER'S NUMBER UNCHANGED, on the interface facets only. There
    # is no named facet subset here, so the interface is selected inside the
    # integrand: the two participants export with respect to their own outward
    # normals, so the natural term for this side is exactly the partner's number.
    ind = ufl.conditional(abs(xc[axis] - xi) < 1e-8, 1.0, 0.0)
    L_form = L_form + ind * (ufl.inner(gh, v) if vec else gh * v) * ufl.ds

# CG with SSOR: this build's iterative backend offers none / sor / ssor /
# gauss-seidel / jacobi as preconditioners and rejects anything else at
# construction, so the choice is not free.
scheme = galerkin([a_form == L_form] + bcs, space,
                  solver="cg", parameters={
                      "linear.tolerance": 1e-13,
                      "linear.preconditioning.method": "ssor",
                      "linear.maxiterations": 100000,
                      "nonlinear.verbose": False,
                      "linear.verbose": False})
uh = space.interpolate(zero, name="uh")
scheme.solve(target=uh)
U = uh.as_numpy[:]

# ── export ────────────────────────────────────────────────────────────
if cfg["side"] == "dirichlet":
    # CONSISTENT (reaction) flux: int_Gamma qn phi_i ds = -(A u - b)_i with the
    # system assembled WITHOUT the boundary condition, so the constrained rows
    # still carry the reaction. A second scheme with no DirichletBC gives
    # exactly that residual. An L2 projection of the gradient is only O(h) on the
    # boundary and the boundary trace is what the coupling reads.
    free = galerkin(a_form == L_form, space)
    res = space.interpolate(zero, name="res")
    free(uh, res)
    r = res.as_numpy[:]
    h = [(ext[a][1] - ext[a][0]) / n[a] for a in range(dim)]
    w = np.ones(len(inode))
    for a in free_axes:
        e_ = (np.abs(ipts[:, a] - ext[a][0]) < 1e-9) | \
             (np.abs(ipts[:, a] - ext[a][1]) < 1e-9)
        w *= np.where(e_, 0.5 * h[a], h[a])
    Q = np.zeros((len(inode), ncomp))
    for k, i in enumerate(inode):
        for c, d in enumerate(dofs[i]):
            Q[k, c] = -r[int(d)] / w[k]
    bad = outer[inode]
    good = np.where(~bad)[0]
    for k in np.where(bad)[0]:
        Q[k] = Q[good[np.argmin(np.abs(good - k))]]
else:
    # The reaction formula must NOT be used on the Neumann side: those dofs are
    # free, so the residual is ~0 there and it would silently export zero. The
    # Dirichlet partner reads this side's values, not its flux.
    off = np.array(ipts)
    off[:, axis] -= S * (ext[axis][1] - ext[axis][0]) / n[axis]
    key = {tuple(np.round(p, 10)): i for i, p in enumerate(pts)}
    near = np.array([key[tuple(np.round(p, 10))] for p in off], int)
    dxh = (ext[axis][1] - ext[axis][0]) / n[axis]
    if vec:
        Q = np.zeros((len(inode), ncomp))
        for k, (i, j) in enumerate(zip(inode, near)):
            du = np.array([U[dofs[i][c]] - U[dofs[j][c]]
                           for c in range(ncomp)]) / dxh
            Q[k] = -(2 * mu * du + 0.0) * S       # normal-derivative part only
    else:
        kk = float(cfg["K"][axis][axis])
        Q = (-kk * S * (U[dofs[inode, 0]] - U[dofs[near, 0]])
             / dxh).reshape(-1, 1)

vals = np.array([[U[d] for d in dofs[i]] for i in range(len(pts))], float)
W.write_nodes("nodes.csv", pts, vals)
W.write_log(cfg, NDOF, f"gridView elements = {gv.size(0)}")
print(f"[dune {cfg['sidename']} {cfg['side']}] NDOF={NDOF} "
      f"iface_n={len(inode)} u=[{vals.min():.6g},{vals.max():.6g}] "
      f"q=[{Q.min():.6g},{Q.max():.6g}]")
W.write_exports(ipts, vals[inode], Q, "displacement" if vec else "temperature")
