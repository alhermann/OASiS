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
# the thermoelastic pair exchanges (T, ux, uy): one scalar and one vector
# field through ONE interface state
ncomp = {"scalar": 1, "vector": dim, "thermoelastic": 1 + dim}[phys]
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

if phys == "thermoelastic":
    # ── the THERMOELASTIC pair, NEUMANN side ──────────────────────────
    #
    # Two subproblems, solved in the sequence the coupling has: the heat
    # equation first (it is independent of u), then elasticity with the
    # just-computed DISCRETE temperature entering the stress. The weak form of
    # -div(sigma_el - beta*T*I) = f_u is
    #     int sigma_el(u):eps(v) dx = int f_u.v dx + int beta*T_h*div(v) dx
    #                                 + int_Gamma t.v ds
    # so the thermal term is a VOLUME term in T_h and needs no derivative of
    # the discrete temperature. The imported traction is sigma_tot.n of the
    # partner, thermal part included, applied UNCHANGED -- the two outward
    # normals are opposite, so it is this side's natural datum.
    #
    # DIRICHLET is not served: C1 prescribes 4C as the Dirichlet side and no
    # cell puts FEniCSx on the Dirichlet side of a thermoelastic exchange.
    if cfg["side"] != "neumann":
        sys.exit("the thermoelastic fenics participant serves the NEUMANN "
                 "role only (C1 prescribes 4C as the Dirichlet side)")
    ST = fem.functionspace(domain, ("Lagrange", 1))
    SU = fem.functionspace(domain, ("Lagrange", 1, (dim,)))
    XYs = ST.tabulate_dof_coordinates()[:, :dim]

    def _sets(X):
        ino = np.where(np.abs(X[:, axis] - xi) < 1e-9)[0]
        fa = [i for i in range(dim) if i != axis]
        ino = ino[np.lexsort(tuple(X[ino, a] for a in reversed(fa)))]
        out = np.zeros(len(X), bool)
        for a in range(dim):
            for val in ext[a]:
                if a == axis and abs(val - xi) < 1e-9:
                    continue
                out |= np.abs(X[:, a] - val) < 1e-9
        return ino, out

    inoT, outT = _sets(XYs)
    XYu = SU.tabulate_dof_coordinates()[:, :dim]
    inoU, outU = _sets(XYu)
    iptsT = XYs[inoT]

    fac = dmesh.locate_entities_boundary(
        domain, fdim, lambda px: np.isclose(px[axis], xi))
    tg = dmesh.meshtags(domain, fdim, np.sort(fac),
                        np.full(len(fac), 7, dtype=np.int32))
    dsw = ufl.Measure("ds", domain=domain, subdomain_data=tg)(7)

    xcte = ufl.SpatialCoordinate(domain)
    NSx = {"x": xcte[0], "y": xcte[1], "exp": ufl.exp, "sin": ufl.sin,
           "cos": ufl.cos, "sqrt": ufl.sqrt}
    kk = float(cfg["k"])
    lam_, mu_, beta_ = (float(cfg["lam"]), float(cfg["mu"]),
                        float(cfg["beta"]))

    imp_te = W.read_imports(cfg["partner"])
    fa_te = [i for i in range(dim) if i != axis]
    q = W.sample(imp_te, "normal_fluxes", iptsT, 0.0, 1 + dim, fa_te)

    # heat: -div(k grad T) = f_T, natural datum = the partner's flux
    tT, vT = ufl.TrialFunction(ST), ufl.TestFunction(ST)
    gqs = fem.Function(ST)
    gqs.x.array[:] = 0.0
    gqs.x.array[inoT] = q[:, 0]
    aT = kk * ufl.inner(ufl.grad(tT), ufl.grad(vT)) * ufl.dx
    LT = eval(cfg["source_T"], dict(NSx)) * vT * ufl.dx + gqs * vT * dsw
    bcT = [fem.dirichletbc(default_scalar_type(0.0),
                           np.where(outT)[0].astype(np.int32), ST)]
    Th = LinearProblem(aT, LT, bcs=bcT, petsc_options_prefix="teT",
                       petsc_options={"ksp_type": "preonly",
                                      "pc_type": "lu"}).solve()

    # elasticity with the thermal stress of the DISCRETE temperature
    uu, vu = ufl.TrialFunction(SU), ufl.TestFunction(SU)

    def epsw(w):
        return ufl.sym(ufl.grad(w))

    au = ufl.inner(2 * mu_ * epsw(uu) + lam_ * ufl.tr(epsw(uu))
                   * ufl.Identity(dim), epsw(vu)) * ufl.dx
    fvec = ufl.as_vector([eval(e, dict(NSx)) for e in cfg["source_u"]])
    gts = fem.Function(SU)
    gts.x.array[:] = 0.0
    for c in range(dim):
        gts.x.array[dim * inoU + c] = q[:, 1 + c]
    Lu = (ufl.inner(fvec, vu) * ufl.dx
          + beta_ * Th * ufl.div(vu) * ufl.dx
          + ufl.inner(gts, vu) * dsw)
    gzs = fem.Function(SU)
    gzs.x.array[:] = 0.0
    bcu = [fem.dirichletbc(gzs, np.where(outU)[0].astype(np.int32))]
    Uh = LinearProblem(au, Lu, bcs=bcu, petsc_options_prefix="teU",
                       petsc_options={"ksp_type": "preonly",
                                      "pc_type": "lu"}).solve()

    Vv = np.column_stack([Th.x.array[inoT]]
                         + [Uh.x.array[dim * inoU + c] for c in range(dim)])
    allV = np.column_stack([Th.x.array]
                           + [Uh.x.array[dim * np.arange(len(XYs)) + c]
                              for c in range(dim)])
    ndof_te = ST.dofmap.index_map.size_global \
        + dim * SU.dofmap.index_map.size_global
    W.write_nodes("nodes.csv", XYs, allV)
    W.write_log(cfg, ndof_te,
                f"num_cells = "
                f"{domain.topology.index_map(domain.topology.dim).size_local}")
    print(f"[fenics {cfg['sidename']} neumann thermoelastic] NDOF={ndof_te} "
          f"iface_n={len(inoT)} T=[{Th.x.array.min():.6g},"
          f"{Th.x.array.max():.6g}] u=[{Uh.x.array.min():.6g},"
          f"{Uh.x.array.max():.6g}]")
    # this side's flux export is not consumed (the partner is the Dirichlet
    # side and reads `values`); zeros are honest
    W.write_exports(iptsT, Vv, np.zeros_like(Vv), "thermoelastic")
    sys.exit(0)

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
    # CRANK-NICOLSON with WAVEFORM exchange: each coupling iteration
    # integrates the WHOLE time window, and the exchanged object is the entire
    # space-time interface trace -- n points with nsteps components -- which
    # the driver moves and relaxes unchanged because it treats values as
    # opaque numbers on coordinates. Convention shared with the deal.II side:
    # values slot s is the field at t_{s+1}; flux slot s is the STEP-AVERAGED
    # flux of step s, exported as -r_i/(w_i*dt) from the step residual and
    # applied by the partner as dt*q_s*phi_i. Step-average vs midpoint differs
    # by O(dt^2), so CN accuracy is preserved.
    #
    # TWO DEFECTS OF THE FIRST VERSION OF THIS BRANCH, fixed and worth
    # recording because both produce a converged coupling with a wrong flux:
    # it divided the residual by 0.5*dt -- exporting TWICE the step-averaged
    # flux, which the partner then applies in full -- and it assembled the
    # step residual AFTER uh had been overwritten with u^{n+1}, so the
    # "residual" was of a system whose right side no longer contained u^n.
    nsteps = int(round(TR["t_end"] / TR["dt"]))
    dt = TR["t_end"] / nsteps
    dtc = fem.Constant(domain, default_scalar_type(dt))
    t0c = fem.Constant(domain, default_scalar_type(0.0))
    t1c = fem.Constant(domain, default_scalar_type(0.0))
    uh = fem.Function(V)
    uh.x.array[:] = 0.0                       # the initial datum is u = 0
    gq = fem.Function(V)                      # Neumann: step-s flux density
    gq.x.array[:] = 0.0

    F0 = eval(cfg["source"], dict(NS, t=t0c))
    F1 = eval(cfg["source"], dict(NS, t=t1c))
    stiff_u = ufl.replace(a_form, {u_: uh})
    # LinearProblem takes UFL forms; the residual assembly takes COMPILED
    # ones. Keep both, built from the same expressions, so they cannot drift.
    A_ufl = u_ * v_ * ufl.dx + 0.5 * dtc * a_form
    b_terms = (uh * v_ * ufl.dx - 0.5 * dtc * stiff_u
               + 0.5 * dtc * (F0 + F1) * v_ * ufl.dx)
    b_un_form = fem.form(b_terms)             # residual side: NO flux term
    b_ufl = (b_terms + dtc * gq * v_ * ds_if
             if cfg["side"] == "neumann" else b_terms)
    A_form = fem.form(A_ufl)
    wvec = _fp.assemble_vector(fem.form(v_ * ds_if))
    wvec.ghostUpdate()
    wi = wvec.array[inode]
    okw = np.abs(wi) > 1e-14

    Amat = _fp.assemble_matrix(A_form)        # unconstrained; dtc is fixed,
    Amat.assemble()                           # so assembled once
    gs = W.sample(imp, "values" if cfg["side"] == "dirichlet"
                  else "normal_fluxes", ipts, 0.0, nsteps, free_axes)
    gstep = fem.Function(V)
    trace_u = np.zeros((len(inode), nsteps))
    trace_q = np.zeros((len(inode), nsteps))
    r = Amat.createVecLeft()
    for st in range(nsteps):
        t0c.value = st * dt
        t1c.value = (st + 1) * dt
        step_bcs = [fem.dirichletbc(g_out,
                                    np.where(outer_n)[0].astype(np.int32))]
        if cfg["side"] == "dirichlet":
            gstep.x.array[:] = 0.0
            gstep.x.array[inode] = gs[:, st]
            step_bcs.append(fem.dirichletbc(gstep,
                                            inode_bc.astype(np.int32)))
        else:
            gq.x.array[:] = 0.0
            gq.x.array[inode] = gs[:, st]
        # the UNCONSTRAINED right side, with uh still holding u^n -- this is
        # what the flux residual is taken against
        b_un = _fp.assemble_vector(b_un_form)
        b_un.ghostUpdate()
        unew = LinearProblem(A_ufl, b_ufl, bcs=step_bcs,
                             petsc_options_prefix=f"tr{st}",
                             petsc_options=OPTS).solve()
        if cfg["side"] == "dirichlet":
            Amat.mult(unew.x.petsc_vec, r)
            r.axpy(-1.0, b_un)
            col = np.zeros(len(inode))
            col[okw] = -r.array[inode][okw] / wi[okw] / dt
            bad = outer_n[inode] | ~okw
            good = np.where(~bad)[0]
            for i in np.where(bad)[0]:
                col[i] = col[good[np.argmin(np.abs(good - i))]]
            trace_q[:, st] = col
        uh.x.array[:] = unew.x.array
        trace_u[:, st] = uh.x.array[inode]
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
        if axis is None:
            # A BENT interface has a different outward normal on each leg, so
            # there is no single n to project onto -- and none is needed: the
            # Dirichlet partner reads this side's VALUES, never its flux.
            # Exporting zeros is honest; projecting onto a normal that does not
            # exist produced an integrand with no integration domain at all.
            Q = np.zeros((len(inode), ncomp))
        else:
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
