"""Kratos Multiphysics participant for the OASiS `couple` driver (NEUMANN side).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is mandatory), writes exports.json LAST and exits 0.
Needs KratosMultiphysics + ConvectionDiffusionApplication importable in the
interpreter named in `command`. DO NOT GUESS THAT INTERPRETER: take it from
`discover(query='list')`, which reports the one this install actually imports
Kratos with. A system python3 is a common trap — it usually exists, so nothing
looks wrong, and it raises ModuleNotFoundError because the Kratos build targets
a different Python version than the system one.

THE OTHER HALF OF participant_kratos.py. That file is the DIRICHLET side: it
imports the partner's `values` (interface temperature), fixes them as nodal
TEMPERATURE, and exports the consistent REACTION_FLUX. This file is the NEUMANN
side: it imports the partner's `normal_fluxes` and applies them as the natural
boundary condition, then exports the interface TEMPERATURE its solve produced,
which is what a Dirichlet partner consumes.

Physics: steady conduction  -div(K grad T) = f  on one rectangular subdomain of
a domain split by a straight interface at x = IFACE_X. Top and bottom edges are
natural (zero flux). The non-interface x-boundary carries a Dirichlet value
T_OUTER.

  x = OUTER_X : Dirichlet T = T_OUTER
  x = IFACE_X : INTERFACE, natural BC = the partner's imported flux
  top/bottom  : insulated (natural, nothing to do)

THE SIGN, WHICH IS THE ONLY THING A NEUMANN PARTICIPANT CAN GET SILENTLY WRONG.
Every participant exports its outward normal flux density with respect to ITS
OWN outward normal,
      q = -(K grad T) . n_own,
so on a shared interface the two sides carry OPPOSITE signs. Integrating the
weak form on THIS subdomain,
      int_Omega K grad T . grad v  =  int_Omega f v  +  int_dOmega (K grad T . n) v ds,
and on the interface n = n_own = -n_partner, so
      K grad T . n_own = -q_own = +q_partner.
The partner's number is therefore applied UNCHANGED — no minus sign anywhere.
Kratos's FluxCondition2D2N enforces exactly  K grad T . n = FACE_HEAT_FLUX
(verified by a patch test: T fixed to 0 at x=0, FACE_HEAT_FLUX=1 on x=1, K=1,
no source -> T(1)=+1.000000, i.e. dT/dx=+1 not -1), so

      FACE_HEAT_FLUX  :=  the imported `normal_fluxes`, verbatim.

If you ever flip that sign to "make the temperatures look right", you have
built a coupling that drives heat the wrong way across the interface and still
converges. The self-check printed at the end of this script is the guard: it
evaluates the discrete divergence theorem on this subdomain and must come out
at round-off.
"""
import json
import sys
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER   = "left"        # the partner's `name` in your couple(...) call
X0, X1    = 0.6, 1.0      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 1.6           # conductivity of THIS subdomain


def F_SRC(x, y):
    """Volumetric source f in  -div(K grad T) = f, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above.
    A CONSTANT CANNOT REPRESENT A POLYNOMIAL SOURCE: if your problem states
    one, or you derived it from a manufactured solution, a single number here
    silently solves a different problem. With the whole outer boundary
    prescribed and no source, the answer degenerates to the profile between
    the outer values.

    Unlike its FEniCSx and DUNE siblings, this one is called ONCE PER NODE
    with SCALAR coordinates (see the SetSolutionStepValue loop below), so
    write it with plain math or NumPy scalars — do not assume arrays:

        return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    """
    return 0.0 * x

T_OUTER   = 300.0         # Dirichlet value on the NON-interface x-boundary
NX, NY    = 20, 16        # this subdomain's OWN mesh; need not match the partner
Q_INIT    = 0.0           # iteration-1 fallback interface flux density


def source(x, y):
    """Volumetric source at the nodes — the interpolation point for F_SRC.

    Sampling at the nodes is the P1 interpolant of the source, an O(h^2) load
    error, the same order as the discretization error, so it does not touch
    the second-order rate.
    """
    return F_SRC(x, y)
# ─────────────────────────────────────────────────────────────────────────

ON_MAX_X = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_MAX_X else X1
S = 1.0 if ON_MAX_X else -1.0          # outward normal at the interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    # `or []` and not `.get(key, [])`: a partner that writes the key with an
    # explicit null gets [] here instead of a TypeError out of np.asarray, and
    # falls through to the fallback like any other unusable import.
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


def build_model():
    """Structured triangulation of [X0,X1] x [Y0,Y1]; returns (mp, nid)."""
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    # FluxCondition2D2N reads its load from the SURFACE source variable named
    # here; leave it FACE_HEAT_FLUX or the condition assembles nothing and the
    # interface silently becomes insulated.
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
              KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    for j in range(NY + 1):
        for i in range(NX + 1):
            mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / NX,
                             Y0 + (Y1 - Y0) * j / NY, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(NY):
        for i in range(NX):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props)
            eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props)
            eid += 1
    return mp, nid


def main():
    """Solve once and write exports.json. Returns (model_part, node_index_map)
    so a verification script can `import` this file, call main(), and integrate
    the volume field against a manufactured solution — the coupling itself only
    ever needs the file handshake."""
    if min(abs(IFACE_X - X0), abs(IFACE_X - X1)) > TOL:
        sys.exit(f"IFACE_X={IFACE_X} is not an x-boundary of this subdomain "
                 f"[{X0},{X1}] — nothing is shared with the partner")
    # NX < 1 would put the interface column and the outer Dirichlet column on
    # the SAME nodes; the Dirichlet condition wins, the imported flux is
    # discarded, and the run still exits 0 with a plausible-looking export.
    if NX < 1 or NY < 1:
        sys.exit(f"NX,NY = {NX},{NY}: need at least one element in each "
                 "direction, and NX >= 1 so the interface and the outer "
                 "Dirichlet boundary do not land on the same nodes")

    mp, nid = build_model()
    i_if = NX if ON_MAX_X else 0            # column index of the interface
    i_out = 0 if ON_MAX_X else NX           # column index of x = OUTER_X
    # The two column indices above are the ONLY place the geometry meets the
    # node numbering. If an edit to build_model ever reorders the grid they
    # would silently point at interior columns, the interface flux would be
    # applied inside the domain and the run would still exit 0. Check them
    # against the coordinates they are supposed to name.
    for idx, want, what in ((i_if, IFACE_X, "interface"),
                            (i_out, OUTER_X, "outer Dirichlet")):
        got = mp.Nodes[nid[(idx, 0)]].X
        if abs(got - want) > TOL:
            sys.exit(f"internal: the {what} column sits at x={got}, not "
                     f"x={want} — the mesh and the column indices disagree")
    y_if = np.array([Y0 + (Y1 - Y0) * j / NY for j in range(NY + 1)])

    q_in = sample(read_imports(), "normal_fluxes", Q_INIT, y_if)

    for n in mp.Nodes:
        n.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(n.X, n.Y)))
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

    for j in range(NY + 1):                 # outer Dirichlet boundary
        n = mp.Nodes[nid[(i_out, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_OUTER))
        n.Fix(KM.TEMPERATURE)

    # ── the interface: the partner's flux, applied UNCHANGED (see the header) ──
    for j in range(NY + 1):
        mp.Nodes[nid[(i_if, j)]].SetSolutionStepValue(
            KM.FACE_HEAT_FLUX, float(q_in[j]))
    props = mp.GetProperties()[1]
    for j in range(NY):
        mp.CreateNewCondition("FluxCondition2D2N", j + 1,
                              [nid[(i_if, j)], nid[(i_if, j + 1)]], props)

    # AddDof with a REACTION variable: without the second argument the fixed
    # dofs have nowhere to store their reaction and it is silently discarded.
    # This side does not export the reaction, but the conservation self-check
    # below is built from it.
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    # arg 4 is CalculateReactionsFlag: True, or the self-check has no reactions.
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()

    T = np.array([mp.Nodes[nid[(i_if, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                  for j in range(NY + 1)])

    # ── this side's own outward normal flux, q = -(K grad T).n = -K*S*dT/dx ──
    #
    # THE REACTION FORMULA MUST NOT BE USED HERE, and this is the one place a
    # Neumann participant can copy the Dirichlet one and get a plausible-looking
    # zero. These interface dofs are FREE: the discrete equations hold on them,
    # so r = A u - b is ~0 there (round-off), and -r/w would export a flux of
    # zero with no error raised anywhere. What this side knows exactly is the
    # flux it was HANDED; what it computes is the trace of its own gradient.
    #
    # So the export is an L2 projection of the elementwise P1 gradient onto the
    # nodes, with a lumped mass matrix (an area-weighted average of the
    # surrounding constant element gradients). Like the projections in
    # participant_fenics.py and participant_ngsolve.py it is only O(h) accurate
    # ON the boundary — the superconvergence points of a P1 gradient are
    # interior. That is acceptable *here and only here*, because a Dirichlet
    # partner reads this participant's `values`, never its `normal_fluxes`;
    # the number that has to be second-order is the flux the DIRICHLET side
    # exports, and that one is the consistent REACTION_FLUX recovery.
    num = np.zeros(len(mp.Nodes) + 1)
    den = np.zeros(len(mp.Nodes) + 1)
    for el in mp.Elements:
        nds = el.GetNodes()
        x = [n.X for n in nds]
        y = [n.Y for n in nds]
        t = [n.GetSolutionStepValue(KM.TEMPERATURE) for n in nds]
        det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
        if abs(det) < 1e-30:
            continue
        dTdx = ((y[1]-y[2])*t[0] + (y[2]-y[0])*t[1] + (y[0]-y[1])*t[2]) / det
        area = 0.5 * abs(det)
        qe = -K * S * dTdx
        for n in nds:
            num[n.Id] += area * qe
            den[n.Id] += area
    ids_if = [nid[(i_if, j)] for j in range(NY + 1)]
    Q = np.array([num[i] / den[i] if den[i] > 0 else 0.0 for i in ids_if])

    # ── CONSERVATION SELF-CHECK: the discrete divergence theorem ──────────────
    # Summing the unconstrained residual r = A u - b over ALL nodes gives
    # sum_i r_i = -sum_i b_i, because sum_i A_ij = int K grad(sum_i phi_i).grad
    # phi_j = 0 (the phi_i are a partition of unity). r vanishes on free rows,
    # so over the fixed rows alone
    #       sum_{outer} r_i  +  int_Omega f dOmega  +  int_Gamma q_applied ds  =  0
    # exactly, at round-off, for ANY mesh. It is not a discretisation check: it
    # fails only if the flux was applied with the wrong sign or magnitude, or
    # not applied at all — which is precisely the failure mode this side has.
    hy = (Y1 - Y0) / NY
    load_iface = float(hy * (0.5 * q_in[0] + q_in[1:-1].sum() + 0.5 * q_in[-1]))
    load_vol = 0.0
    for el in mp.Elements:
        nds = el.GetNodes()
        x = [n.X for n in nds]
        y = [n.Y for n in nds]
        det = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
        load_vol += (0.5 * abs(det)) * sum(
            n.GetSolutionStepValue(KM.HEAT_FLUX) for n in nds) / 3.0
    react = sum(mp.Nodes[nid[(i_out, j)]].GetSolutionStepValue(KM.REACTION_FLUX)
                for j in range(NY + 1))
    imb_abs = abs(react + load_vol + load_iface)
    scale = max(abs(react), abs(load_vol), abs(load_iface))
    # On iteration 1 with Q_INIT = 0 and no source there is no heat flow at all,
    # every term is round-off, and a RATIO of round-off to round-off is O(1)
    # while meaning nothing. Say "trivial" instead of printing a 1.0 that reads
    # as a 100% conservation error.
    if scale <= 1e-10 * K * max(1.0, abs(T_OUTER)) * (Y1 - Y0):
        bal = f"balance trivial (no heat flow yet, |imbalance|={imb_abs:.3e})"
    else:
        bal = (f"balance |sum(reactions)+vol+iface| = {imb_abs:.3e} abs / "
               f"{imb_abs / scale:.3e} rel")

    print(f"[kratos neumann] interface n={len(T)} "
          f"q_applied=[{q_in.min():.6g},{q_in.max():.6g}] "
          f"T=[{T.min():.6g},{T.max():.6g}] {bal}")

    # exports.json LAST: the driver takes its existence as proof of success.
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(T)),
        "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))
    return mp, nid


if __name__ == "__main__":
    main()
