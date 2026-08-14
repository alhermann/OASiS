"""scikit-fem path-walk participant: scalar or vector, either role, either axis.

Also serves the NON-RECTANGULAR subdomain of the notched instance. No shipped
participant can: every one of them meshes a rectangle from an extent, and the
notched cell's subdomain A is the unit square minus the other subdomain minus a
corner square. Here it is a tensor mesh with the elements in those two regions
removed, which keeps every material line and the notch on mesh lines at every
level, so nothing is paid in geometric error.

The interface of that subdomain is a BENT polyline with two outward normals, so
it cannot be identified by "the facets at x = xi". It is identified by the two
legs and the export is ordered along the polyline by arclength, which is also how
the partner reads it -- the corner where the legs meet then needs no special case.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

from skfem import (MeshTri, ElementTriP1, ElementVector, Basis, FacetBasis,   # noqa: E402
                   BilinearForm, LinearForm, asm, condense, solve, enforce)
from skfem.helpers import dot, grad, sym_grad, ddot, trace, eye  # noqa: E402
import scipy.sparse as sps                                       # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
vec = cfg["physics"] == "vector"
ncomp = dim if vec else 1
free_axes = [i for i in range(dim) if i != axis] if axis is not None else [0, 1]
BENT = cfg.get("bent")                 # [[axis, value, (lo, hi) of the free ax]]

# ── mesh ──────────────────────────────────────────────────────────────
axes = [np.linspace(lo, hi, k + 1) for (lo, hi), k in zip(ext, n)]
m = MeshTri.init_tensor(*axes)
for box in cfg.get("remove", []):
    c = m.p[:, m.t].mean(axis=1)
    inside = np.ones(m.t.shape[1], bool)
    for a, (lo, hi) in enumerate(box):
        inside &= (c[a] > lo) & (c[a] < hi)
    m = m.remove_elements(np.nonzero(inside)[0])

e = ElementVector(ElementTriP1()) if vec else ElementTriP1()
basis = Basis(m, e)
ndof = basis.N
P = m.p.T                                            # (nv, dim) node coords


def _node_dofs():
    """scalar-dof index per node per component, for P1 (and vector P1)."""
    dl = basis.doflocs                               # (dim, N)
    key = {tuple(np.round(p, 12)): i for i, p in enumerate(P)}
    out = np.full((len(P), ncomp), -1, int)
    seen = {}
    for j in range(dl.shape[1]):
        k = tuple(np.round(dl[:, j], 12))
        i = key.get(k)
        if i is None:
            continue
        c = seen.get(i, 0)
        out[i, c] = j
        seen[i] = c + 1
    return out


ndofs = _node_dofs()

# ── the interface facets and nodes ────────────────────────────────────
if BENT:
    def on_iface(pts):
        hit = np.zeros(pts.shape[1], bool)
        for a, val, (lo, hi) in BENT:
            hit |= ((np.abs(pts[a] - val) < 1e-9)
                    & (pts[1 - a] > lo - 1e-9) & (pts[1 - a] < hi + 1e-9))
        return hit
else:
    def on_iface(pts):
        return np.abs(pts[axis] - xi) < 1e-9

ifacets = m.facets_satisfying(on_iface, boundaries_only=True)
fb = FacetBasis(m, e, facets=ifacets)

inode = np.where(on_iface(P.T))[0]


def _order(idx):
    """Order interface nodes: by the free coordinate, or leg by leg along a bent
    polyline (leg 1 first, then leg 2, exactly as the task prescribes). The
    corner where the legs meet belongs to both and is emitted once."""
    if not BENT:
        return idx[np.lexsort(tuple(P[idx, a] for a in reversed(free_axes)))]
    seen, flat = set(), []
    for a, val, (lo, hi) in BENT:
        sel = idx[(np.abs(P[idx, a] - val) < 1e-9)
                  & (P[idx, 1 - a] > lo - 1e-9) & (P[idx, 1 - a] < hi + 1e-9)]
        for i in sel[np.argsort(P[sel, 1 - a])]:
            if int(i) not in seen:
                seen.add(int(i))
                flat.append(int(i))
    return np.array(flat, int)


inode = _order(inode)
ipts = P[inode]

# THE OUTER BOUNDARY, AND WHO OWNS THE INTERFACE CORNERS.
#
# A node is OUTER if it lies on a face of this subdomain that is NOT part of the
# interface. The two ends of the interface lie on such a face as well as on the
# interface, and they belong to the OUTER boundary on BOTH sides -- in the
# un-split problem they carry the prescribed datum, so they stay essential in
# both subproblems.
#
# Handing them to the interface instead is not a small error. On the Dirichlet
# side they then take the partner's imported value while the Neumann side holds
# them at the outer datum, and the two subproblems disagree at those nodes by
# O(1) forever: measured here on C9, the driver residual oscillated between 0.39
# and 1.23 for 25 iterations and never fell, while both fields were already
# within 2% of the manufactured solution. They are still EXPORTED; they are just
# not interface-imposed.
def _is_iface_face(a, val):
    if BENT:
        return any(ax == a and abs(v - val) < 1e-9 for ax, v, _ in BENT)
    return a == axis and abs(val - xi) < 1e-9


outer = np.zeros(len(P), bool)
for a in range(dim):
    for val in ext[a]:
        if _is_iface_face(a, val):
            continue
        outer |= np.abs(P[:, a] - val) < 1e-9
# The faces of a REMOVED region are outer boundary -- for the notch, whose two
# faces carry the stated datum -- EXCEPT where the removed region is the other
# SUBDOMAIN, whose faces are the interface itself.
#
# Without that exception the whole interface is marked outer, every interface
# node is constrained to the outer datum instead of to the partner's value, and
# NOTHING SAYS SO: the partitioned iteration still converges (55 iterations to
# 8.5e-07 here), because the two subproblems agree on a consistent wrong
# interface state. Measured: the coupled error then sat at 1.5e-04 and 1.3e-04
# on two levels, an observed order of 0.21 against a solution whose own RMS is
# 2.5e-04. A converged residual is not evidence of a correct boundary set.
for box in cfg.get("remove", []):
    for a, (lo, hi) in enumerate(box):
        for val in (lo, hi):
            if _is_iface_face(a, val):
                continue
            on = np.abs(P[:, a] - val) < 1e-9
            for b, (blo, bhi) in enumerate(box):
                if b != a:
                    on &= (P[:, b] > blo - 1e-9) & (P[:, b] < bhi + 1e-9)
            outer |= on
inode_bc = np.array([i for i in inode if not outer[i]], int)

# ── operator ──────────────────────────────────────────────────────────
src = cfg["source"]
if vec:
    lam, mu = float(cfg["lam"]), float(cfg["mu"])

    @BilinearForm
    def bilin(u, v, w):
        return (2.0 * mu * ddot(sym_grad(u), sym_grad(v))
                + lam * trace(sym_grad(u)) * trace(sym_grad(v)))

    fun = W.make_vec_fun(src, dim)

    @LinearForm
    def lin(v, w):
        f = fun(*[w.x[i] for i in range(dim)])
        return sum(f[i] * v[i] for i in range(dim))
else:
    K = np.asarray(cfg["K"], float)
    react = float(cfg.get("reaction") or 0.0)
    # A SUBDOMAIN MAY CARRY SEVERAL MATERIALS. The notched cell's subdomain A
    # spans three material cells with three conductivities and three different
    # sources; every shipped participant assumes one of each. Both are selected
    # per QUADRATURE POINT from the box the point falls in, so nothing depends on
    # how the mesh happened to be cut.
    CELLS = cfg.get("cells")
    if CELLS:
        _funs = [W.make_fun(c["source"], dim) for c in CELLS]

        def _mask(w, box):
            m = np.ones_like(w.x[0])
            for a, (lo, hi) in enumerate(box):
                m = m * ((w.x[a] > lo) & (w.x[a] < hi))
            return m

        @BilinearForm
        def bilin(u, v, w):
            gu, gv = grad(u), grad(v)
            gg = sum(gu[i] * gv[i] for i in range(dim))
            out = 0.0
            for c in CELLS:
                out = out + c["k"] * _mask(w, c["box"]) * gg
            return out

        @LinearForm
        def lin(v, w):
            out = 0.0
            for c, f in zip(CELLS, _funs):
                out = out + (_mask(w, c["box"])
                             * f(*[w.x[i] for i in range(dim)]) * v)
            return out
    else:
        @BilinearForm
        def bilin(u, v, w):
            gu, gv = grad(u), grad(v)
            return sum(K[i, j] * gu[j] * gv[i]
                       for i in range(dim) for j in range(dim)) + react * u * v

        fun = W.make_fun(src, dim)

        @LinearForm
        def lin(v, w):
            return fun(*[w.x[i] for i in range(dim)]) * v


A = asm(bilin, basis)
b = asm(lin, basis)

imp = W.read_imports(cfg["partner"])
xg = np.zeros(ndof)
D = np.unique(ndofs[outer][ndofs[outer] >= 0])

if cfg["side"] == "dirichlet":
    g = W.sample(imp, "values", ipts, 0.0, ncomp, free_axes)
    for i, row in zip(inode, g):
        if outer[i]:
            continue                       # the corners keep the outer datum
        for c in range(ncomp):
            if ndofs[i, c] >= 0:
                xg[ndofs[i, c]] = float(np.atleast_1d(row)[c])
    D = np.unique(np.concatenate(
        [D, ndofs[inode_bc][ndofs[inode_bc] >= 0]]))
else:
    q = W.sample(imp, "normal_fluxes", ipts, 0.0, ncomp, free_axes)
    qv = np.zeros(ndof)
    for i, row in zip(inode, q):
        for c in range(ncomp):
            if ndofs[i, c] >= 0:
                qv[ndofs[i, c]] = float(np.atleast_1d(row)[c])

    if vec:
        @LinearForm
        def lneu(v, w):
            return sum(w["q"][i] * v[i] for i in range(dim))
    else:
        @LinearForm
        def lneu(v, w):
            return w["q"] * v
    # APPLY THE PARTNER'S NUMBER UNCHANGED: both sides export with respect to
    # their own outward normal, so the natural term is the partner's number.
    b = b + asm(lneu, fb, q=fb.interpolate(qv))

xsol = solve(*condense(A, b, x=xg, D=D))

# ── export ────────────────────────────────────────────────────────────
if cfg["side"] == "dirichlet":
    # CONSISTENT (reaction) flux: int_Gamma qn phi_i ds = -(A u - b)_i on the
    # constrained rows, with A and b assembled WITHOUT the boundary condition.
    # An L2 projection of the gradient is only O(h) on the boundary and the
    # boundary trace is exactly what the coupling reads.
    r = A @ xsol - b
    if vec:
        @LinearForm
        def lw(v, w):
            return sum(v[i] for i in range(dim))
    else:
        @LinearForm
        def lw(v, w):
            return v
    wv = asm(lw, fb)
    Q = np.zeros((len(inode), ncomp))
    for k, i in enumerate(inode):
        for c in range(ncomp):
            j = ndofs[i, c]
            Q[k, c] = -r[j] / wv[j] if abs(wv[j]) > 1e-14 else np.nan
    outer_set = set(np.unique(ndofs[outer][ndofs[outer] >= 0]).tolist())
    bad = np.array([any(int(ndofs[i, c]) in outer_set for c in range(ncomp))
                    for i in inode])
    bad |= ~np.isfinite(Q).all(axis=1)
    good = np.where(~bad)[0]
    if len(good):
        for k in np.where(bad)[0]:
            Q[k] = Q[good[np.argmin(np.abs(good - k))]]
else:
    # The reaction formula must NOT be used on the Neumann side: those dofs are
    # free, so r ~ 0 and it would silently export zero. The Dirichlet partner
    # reads this side's `values`, so an O(h) projection here is harmless.
    S = W.outward_sign(ext, axis, xi) if axis is not None else 1.0
    gb = Basis(m, ElementTriP1())
    if vec:
        Q = np.zeros((len(inode), ncomp))
    else:
        @BilinearForm
        def mass(u, v, w):
            return u * v

        @LinearForm
        def rhs(v, w):
            return -S * sum(K[axis, j] * w["gu"].grad[j] for j in range(dim)) * v
        sb = Basis(m, ElementTriP1())
        scal = np.zeros(sb.N)
        for i in range(len(P)):
            if ndofs[i, 0] >= 0:
                scal[i] = xsol[ndofs[i, 0]]
        qh = solve(asm(mass, sb), asm(rhs, sb, gu=sb.interpolate(scal)))
        Q = qh[inode].reshape(-1, 1)
    if vec:
        sig = None
        # projected traction, componentwise, from the P1 stress
        vb = Basis(m, ElementVector(ElementTriP1()))

        @BilinearForm
        def vmass(u, v, w):
            return dot(u, v)

        @LinearForm
        def vrhs(v, w):
            gu = w["uu"].grad
            eps = 0.5 * (gu + np.swapaxes(gu, 0, 1))
            tr = sum(eps[i, i] for i in range(dim))
            out = 0.0
            nrm = [S if i == axis else 0.0 for i in range(dim)]
            for i in range(dim):
                si = sum((2 * mu * eps[i, j] + (lam * tr if i == j else 0.0))
                         * nrm[j] for j in range(dim))
                out = out + (-si) * v[i]
            return out
        th = solve(asm(vmass, vb), asm(vrhs, vb, uu=vb.interpolate(xsol)))
        Q = np.array([[th[ndofs[i, c]] for c in range(ncomp)] for i in inode])

U = np.array([[xsol[ndofs[i, c]] for c in range(ncomp)] for i in inode], float)
allU = np.array([[xsol[ndofs[i, c]] if ndofs[i, c] >= 0 else 0.0
                  for c in range(ncomp)] for i in range(len(P))], float)
keep = (ndofs[:, 0] >= 0)

W.write_nodes("nodes.csv", P[keep], allU[keep])
W.write_log(cfg, ndof, f"n_elements = {m.t.shape[1]}")
print(f"[skfem {cfg['sidename']} {cfg['side']}] NDOF={ndof} "
      f"iface_n={len(inode)} u=[{U.min():.6g},{U.max():.6g}] "
      f"q=[{np.nanmin(Q):.6g},{np.nanmax(Q):.6g}]")
W.write_exports(ipts, U, Q, "displacement" if vec else "temperature")
