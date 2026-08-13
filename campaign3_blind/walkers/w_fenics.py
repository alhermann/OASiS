"""FEniCSx (dolfinx) path-walk participant.

Scalar conduction (steady or transient), elasticity, or the THERMOELASTIC pair --
temperature and displacement transmitted together across one interface -- in
either role, on either axis. Also serves a subdomain whose interface is BENT: the
notched cell's subdomain B is a rectangle two of whose four edges are interface,
with two different outward normals.

WHAT DOES NOT SHIP FOR ANY OF THIS. ``participant_fenics.py`` is steady scalar
conduction on a rectangle with a constant source and a straight interface at
x = const; ``participant_fenics_elastic.py`` is its vector twin. No transient
participant of any kind ships, none takes a spatially varying source, none has a
bent interface, and none exchanges two physical fields at once.

FLUX / TRACTION RECOVERY. The Dirichlet side exports the CONSISTENT (reaction)
flux from the unconstrained discrete residual, not an L2 projection of the
gradient. The gradient of a P1 solution is only O(h) accurate ON the boundary --
the superconvergence points are interior -- and the boundary trace is exactly
what the coupling reads. Measured on the same meshes and driver: projection gives
an interface-trace order of 1.33 and a field order of 1.83; the consistent
recovery gives 2.73 and 2.05.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

import ufl                                                       # noqa: E402
from dolfinx import default_scalar_type, fem, mesh as dmesh      # noqa: E402
from dolfinx.fem import petsc as _fp                             # noqa: E402
from dolfinx.fem.petsc import LinearProblem                      # noqa: E402
from mpi4py import MPI                                           # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
phys = cfg["physics"]
vec = phys == "vector"
ncomp = {"scalar": 1, "vector": dim}[phys]
free_axes = [i for i in range(dim) if i != axis] if axis is not None else [0, 1]
BENT = cfg.get("bent")
S = W.outward_sign(ext, axis, xi) if axis is not None else 1.0
TR = cfg.get("transient")

if dim == 2:
    domain = dmesh.create_rectangle(
        MPI.COMM_WORLD, [[ext[0][0], ext[1][0]], [ext[0][1], ext[1][1]]],
        [n[0], n[1]], dmesh.CellType.triangle)
else:
    domain = dmesh.create_box(
        MPI.COMM_WORLD, [[e[0] for e in ext], [e[1] for e in ext]],
        list(n), dmesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 1, (ncomp,)) if vec
                      else ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

# tabulate_dof_coordinates() has ONE ROW PER NODE (dof block); the scalar array
# index of component c at node j is j*ncomp + c.
XY = V.tabulate_dof_coordinates()[:, :dim]


def _on_iface(P):
    if BENT:
        hit = np.zeros(len(P), bool)
        for a, val, (lo, hi) in BENT:
            hit |= ((np.abs(P[:, a] - val) < 1e-9)
                    & (P[:, 1 - a] > lo - 1e-9) & (P[:, 1 - a] < hi + 1e-9))
        return hit
    return np.abs(P[:, axis] - xi) < 1e-9


def _iface_facets(px):
    if BENT:
        hit = np.zeros(px.shape[1], bool)
        for a, val, (lo, hi) in BENT:
            hit |= ((np.isclose(px[a], val))
                    & (px[1 - a] > lo - 1e-9) & (px[1 - a] < hi + 1e-9))
        return hit
    return np.isclose(px[axis], xi)


def _is_iface_face(a, val):
    if BENT:
        return any(ax == a and abs(v - val) < 1e-9 for ax, v, _ in BENT)
    return a == axis and abs(val - xi) < 1e-9


inode = np.where(_on_iface(XY))[0]
if BENT:
    seen, flat = set(), []
    for a, val, (lo, hi) in BENT:
        sel = inode[(np.abs(XY[inode, a] - val) < 1e-9)
                    & (XY[inode, 1 - a] > lo - 1e-9)
                    & (XY[inode, 1 - a] < hi + 1e-9)]
        for i in sel[np.argsort(XY[sel, 1 - a])]:
            if int(i) not in seen:
                seen.add(int(i))
                flat.append(int(i))
    inode = np.array(flat, int)
else:
    inode = inode[np.lexsort(tuple(XY[inode, a] for a in reversed(free_axes)))]
ipts = XY[inode]
if len(inode) == 0:
    sys.exit(f"no interface dofs found; this subdomain spans {ext}")

# THE INTERFACE ENDS BELONG TO THE OUTER BOUNDARY ON BOTH SIDES: they lie on an
# outer face too, and in the un-split problem they carry the prescribed datum.
outer_n = np.zeros(len(XY), bool)
for a in range(dim):
    for val in ext[a]:
        if _is_iface_face(a, val):
            continue
        outer_n |= np.abs(XY[:, a] - val) < 1e-9
for box in cfg.get("remove", []):
    for a, (lo, hi) in enumerate(box):
        for val in (lo, hi):
            on = np.abs(XY[:, a] - val) < 1e-9
            for b, (blo, bhi) in enumerate(box):
                if b != a:
                    on &= (XY[:, b] > blo - 1e-9) & (XY[:, b] < bhi + 1e-9)
            outer_n |= on
inode_bc = np.array([i for i in inode if not outer_n[i]], int)

facets = dmesh.locate_entities_boundary(domain, fdim,
                                        lambda px: _iface_facets(px))
tags = dmesh.meshtags(domain, fdim, np.sort(facets),
                      np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)

xc = ufl.SpatialCoordinate(domain)
NS = {"x": xc[0], "y": xc[1], "z": (xc[2] if dim > 2 else None),
      "exp": ufl.exp, "sin": ufl.sin, "cos": ufl.cos, "sqrt": ufl.sqrt}
u_, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)

if vec:
    lam, mu = float(cfg["lam"]), float(cfg["mu"])

    def eps(w):
        return ufl.sym(ufl.grad(w))

    def sig(w):
        return 2 * mu * eps(w) + lam * ufl.tr(eps(w)) * ufl.Identity(dim)

    a_form = ufl.inner(sig(u_), eps(v_)) * ufl.dx
    fexpr = ufl.as_vector([eval(e, dict(NS)) for e in cfg["source"]])
    L_form = ufl.inner(fexpr, v_) * ufl.dx
else:
    K = ufl.as_matrix([[float(k) for k in row] for row in cfg["K"]])
    a_form = ufl.inner(K * ufl.grad(u_), ufl.grad(v_)) * ufl.dx
    if cfg.get("reaction"):
        a_form = a_form + float(cfg["reaction"]) * u_ * v_ * ufl.dx
    tsym = None
    if TR:
        tsym = fem.Constant(domain, default_scalar_type(0.0))
        NS = dict(NS, t=tsym)
    L_form = eval(cfg["source"], dict(NS)) * v_ * ufl.dx

imp = W.read_imports(cfg["partner"])
zero = (np.zeros(ncomp, dtype=default_scalar_type) if vec
        else default_scalar_type(0.0))
g_out = fem.Function(V)
g_out.x.array[:] = 0.0
bcs = [fem.dirichletbc(g_out, np.where(outer_n)[0].astype(np.int32))]

g_if = fem.Function(V)
g_if.x.array[:] = 0.0
if cfg["side"] == "dirichlet":
    g = W.sample(imp, "values", ipts, 0.0, ncomp, free_axes)
    for c in range(ncomp):
        g_if.x.array[ncomp * inode + c] = g[:, c]
    bcs.append(fem.dirichletbc(g_if, inode_bc.astype(np.int32)))
else:
    q = W.sample(imp, "normal_fluxes", ipts, 0.0, ncomp, free_axes)
    for c in range(ncomp):
        g_if.x.array[ncomp * inode + c] = q[:, c]
    # APPLY THE PARTNER'S NUMBER UNCHANGED: the two participants export with
    # respect to their own outward normals, so the natural term for this side is
    # exactly the partner's number.
    L_form = L_form + (ufl.inner(g_if, v_) if vec else g_if * v_) * ds_if

OPTS = {"ksp_type": "preonly", "pc_type": "lu"}

if TR:
    # CRANK-NICOLSON. The partner's interface datum is a whole space-time trace:
    # the driver moves n interface points with nsteps components each and relaxes
    # them unchanged, because it treats exchanged values as opaque numbers on
    # coordinates. Calling the driver once per time step instead would need the
    # participants to carry state across driver invocations, which the contract
    # does not provide.
    nsteps = int(round(TR["t_end"] / TR["dt"]))
    dt = TR["t_end"] / nsteps
    uh = fem.Function(V)
    uh.x.array[:] = 0.0                       # the initial datum is u = 0
    unew = fem.Function(V)
    m_form = u_ * v_ * ufl.dx
    stiff = a_form
    trace_u = np.zeros((len(inode), nsteps))
    trace_q = np.zeros((len(inode), nsteps))
    gs = W.sample(imp, "values" if cfg["side"] == "dirichlet"
                  else "normal_fluxes", ipts, 0.0, nsteps, free_axes)
    for s in range(nsteps):
        t0, t1 = s * dt, (s + 1) * dt
        tsym.value = t0
        L0 = fem.form(eval(cfg["source"], dict(NS, t=tsym)) * v_ * ufl.dx)
        f0 = _fp.assemble_vector(L0)
        f0.ghostUpdate()
        tsym.value = t1
        Lt = eval(cfg["source"], dict(NS, t=tsym)) * v_ * ufl.dx
        A_form = m_form + 0.5 * dt * stiff
        b_form = (uh * v_ * ufl.dx - 0.5 * dt * ufl.replace(
            stiff, {u_: uh}) + 0.5 * dt * Lt)
        b_form = b_form + 0.5 * dt * eval(cfg["source"],
                                          dict(NS, t=fem.Constant(
                                              domain,
                                              default_scalar_type(t0)))) \
            * v_ * ufl.dx
        step_bcs = [fem.dirichletbc(g_out, np.where(outer_n)[0].astype(np.int32))]
        if cfg["side"] == "dirichlet":
            gstep = fem.Function(V)
            gstep.x.array[:] = 0.0
            gstep.x.array[inode] = gs[:, s]
            step_bcs.append(fem.dirichletbc(gstep, inode_bc.astype(np.int32)))
        else:
            gq = fem.Function(V)
            gq.x.array[:] = 0.0
            gq.x.array[inode] = gs[:, s]
            b_form = b_form + dt * gq * v_ * ds_if
        unew = LinearProblem(A_form, b_form, bcs=step_bcs,
                             petsc_options_prefix=f"tr{s}",
                             petsc_options=OPTS).solve()
        uh.x.array[:] = unew.x.array
        trace_u[:, s] = uh.x.array[inode]
        if cfg["side"] == "dirichlet":
            Aq = _fp.assemble_matrix(fem.form(A_form))
            Aq.assemble()
            bq = _fp.assemble_vector(fem.form(b_form))
            bq.ghostUpdate()
            r = Aq.createVecLeft()
            Aq.mult(uh.x.petsc_vec, r)
            r.axpy(-1.0, bq)
            wv = _fp.assemble_vector(fem.form(v_ * ds_if))
            wv.ghostUpdate()
            wi = wv.array[inode]
            ok = np.abs(wi) > 1e-14
            col = np.zeros(len(inode))
            col[ok] = -r.array[inode][ok] / wi[ok] / (0.5 * dt)
            trace_q[:, s] = col
    U = trace_u
    Q = trace_q
    sol = uh
else:
    uh = LinearProblem(a_form, L_form, bcs=bcs, petsc_options_prefix="cpl",
                       petsc_options=OPTS).solve()
    sol = uh
    if cfg["side"] == "dirichlet":
        Amat = _fp.assemble_matrix(fem.form(a_form))     # no bcs= on purpose
        Amat.assemble()
        bvec = _fp.assemble_vector(fem.form(L_form))     # no lifting, no set_bc
        bvec.ghostUpdate()
        r = Amat.createVecLeft()
        Amat.mult(uh.x.petsc_vec, r)
        r.axpy(-1.0, bvec)
        one = (fem.Constant(domain, np.ones(ncomp, dtype=default_scalar_type))
               if vec else None)
        wv = _fp.assemble_vector(fem.form(
            (ufl.inner(one, v_) if vec else v_) * ds_if))
        wv.ghostUpdate()
        idx = np.column_stack([ncomp * inode + c for c in range(ncomp)])
        wi = wv.array[idx]
        Q = np.zeros_like(wi, dtype=float)
        ok = np.abs(wi) > 1e-14
        Q[ok] = -r.array[idx][ok] / wi[ok]
        bad = outer_n[inode] | ~ok.all(axis=1)
        good = np.where(~bad)[0]
        for i in np.where(bad)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]
    else:
        # The reaction formula must NOT be used here: these dofs are free, r is
        # ~0, and it would silently export zero. The Dirichlet partner reads this
        # side's values, so an O(h) projection here is harmless.
        p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)
        nrm = ufl.as_vector([default_scalar_type(S if i == axis else 0.0)
                             for i in range(dim)])
        if vec:
            rhs = ufl.inner(-ufl.dot(sig(uh), nrm), w_) * ufl.dx
            mass = ufl.inner(p_, w_) * ufl.dx
        else:
            rhs = (-ufl.dot(K * ufl.grad(uh), nrm)) * w_ * ufl.dx
            mass = p_ * w_ * ufl.dx
        qh = LinearProblem(mass, rhs, petsc_options_prefix="flx",
                           petsc_options=OPTS).solve()
        Q = np.column_stack([qh.x.array[ncomp * inode + c]
                             for c in range(ncomp)])
    U = np.column_stack([sol.x.array[ncomp * inode + c] for c in range(ncomp)])

allU = np.column_stack([sol.x.array[ncomp * np.arange(len(XY)) + c]
                        for c in range(ncomp)])
W.write_nodes("nodes.csv", XY, allU)
W.write_log(cfg, V.dofmap.index_map.size_global * ncomp,
            f"num_cells = {domain.topology.index_map(domain.topology.dim).size_local}")
print(f"[fenics {cfg['sidename']} {cfg['side']}] "
      f"NDOF={V.dofmap.index_map.size_global * ncomp} iface_n={len(inode)} "
      f"u=[{allU.min():.6g},{allU.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")
W.write_exports(ipts, U, Q, "displacement" if vec else "temperature")
