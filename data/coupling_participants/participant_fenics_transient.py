"""FEniCSx (dolfinx) TRANSIENT participant for the OASiS `couple` driver.

Transient conduction  rho_c dT/dt - div(K grad T) = F_SRC(x, y, t)  on ONE
rectangular subdomain of a domain split by a straight interface at x = IFACE_X.
Time integration is the theta-scheme; THETA = 0.5 is Crank-Nicolson and is
SECOND ORDER, which is what makes an order-2 space-time result reachable at all.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST, exits 0.

╔══════════════════════════════════════════════════════════════════════════╗
║ THE TIME-COUPLING DECISION — THIS FILE IMPLEMENTS A WAVEFORM EXCHANGE.    ║
╚══════════════════════════════════════════════════════════════════════════╝
ONE run of this script marches the WHOLE coupling window T_START -> T_END and
exchanges the ENTIRE interface trace in one exports.json:

    values[i][n]        = T at interface point i at time level t^(n+1)
    normal_fluxes[i][n] = the THETA-AVERAGED outward normal flux density at
                          interface point i over step n  (see THE FLUX below)

both of shape (n_points, N_STEPS). The driver's fixed-point iteration therefore
converges the whole space-TIME interface trace at once — classical Dirichlet-
Neumann WAVEFORM RELAXATION. `values` and `normal_fluxes` carry the same layout
every iteration, which is what the driver requires.

WHY, AND NOT "COUPLE ONCE PER TIME STEP". Coupling once per time step means:
exchange, sub-iterate to convergence, THEN advance. Inside ONE `couple` call
that is not merely inconvenient, it is unreachable, for three checkable reasons:

  1. THE EXCHANGE HAS NO TIME AXIS. The driver rebuilds every payload through
     `InterfaceData.to_dict()`, which emits exactly five keys — field_name,
     n_points, coordinates, values, normal_fluxes. Any extra key you write
     (`"times"`, `"step"`) is dropped on the round trip and the partner never
     sees it. A per-step protocol needs to say WHICH step a payload belongs to,
     and there is nowhere to say it.
  2. A PARTICIPANT WITH HIDDEN TIME STATE BREAKS THE DRIVER'S OWN PROBES. The
     driver runs each participant SEVERAL TIMES ON IDENTICAL imports — once per
     replicate for `noise_replicates`, and again in
     `probe_interface_sensitivity`. A script that advances an internal step
     counter answers the same question differently every time, so the driver
     measures a residual "noise floor" that is really the time march, and the
     sensitivity probe reads a response that is really the next time step.
     Both mis-readings are silent. A waveform participant has no such state:
     measured on the coupled runs below, the probe's repeat-run noise came out
     EXACTLY 0.0 for both participants, so its interface sensitivity (S = 1.0
     and 3.7-5.0) means what it says.
  3. THE DRIVER'S CONVERGENCE TEST IS "THE EXPORT STOPPED CHANGING". A
     participant that advances time whenever it likes satisfies that only by
     accident: one flat time step makes the residual dip, the driver declares
     convergence at step 3 of 20, and the reported answer is at the wrong TIME
     with nothing anywhere saying so. And because the driver is a JACOBI sweep
     (both participants see the PREVIOUS iteration's exports), two participants
     deciding independently when to advance can desynchronize in time.

WHAT PER-TIME-STEP COUPLING WOULD COST IF YOU WANT IT ANYWAY. It is a legitimate
design and it stays inside this contract, but the time loop moves OUT of the
participant and into a script you write around the driver: call `couple(...)`
N_STEPS times; before call n, write the step's t^n / t^(n+1) into a small
control file this script reads, and let the participant restart from the field
dump it writes (see FIELD DUMP below). Every call is then a plain steady-looking
fixed point over one step and every guard above still works. The price:
  * N_STEPS driver invocations instead of one, each paying its own start-up,
    its own sensitivity probe (one extra solve per participant per call) and
    its own noise-floor replicates if you use them;
  * restart files, i.e. real state on disk between calls, so the participant is
    no longer a pure function of imports.json — the property this file relies on;
  * you own the outer loop, so you own its correctness (a per-step call that
    fails to converge must abort the march, not continue).
  What you buy: exchange memory O(n_points) instead of O(n_points x N_STEPS);
  a residual that measures ONE step, so a hard step cannot hide behind easy
  ones; and per-step iteration counts.

WHAT THE WAVEFORM CHOICE COSTS HERE. exports.json holds 2 x n_points x N_STEPS
numbers — measured on the finest level of this file's own verification, 97
points x 80 steps = 15520 numbers and 415 KB of JSON per participant per
iteration — and every driver iteration re-solves the ENTIRE window, so the
total linear-solve count is n_iter x N_STEPS. Measured on the manufactured
two-material problem below, with h and dt halved together over four levels: the
whole window converged to an export residual of 5.1e-10 to 5.8e-10 in 31 driver
iterations AT EVERY LEVEL (37 s, 39 s, 44 s, 79 s of wall clock, so at these
sizes the cost per iteration is still dominated by interpreter start-up). The
waveform iteration count did not move with h or dt, which is what makes this
cost predictable; it does grow with the WINDOW LENGTH, so for a long horizon
split the run into several windows (each one `couple` call, restarting from the
field dump) rather than one enormous window.

WHICH SIDE TAKES THE DIRICHLET ROLE DECIDES WHETHER THIS CONVERGES AT ALL, and
that is a property of the PAIR, not of this file. For two subdomains the
Dirichlet-Neumann iteration contracts like

    rho ~ (K_dirichlet / K_neumann) * (width_neumann / width_dirichlet)

so THE DIRICHLET ROLE BELONGS ON THE SOFTER SIDE: small K, or a wide
subdomain. Measured on the problem below (K = 1.7 across a width of 0.4 against
K = 0.35 across a width of 0.6, so rho ~ 7.3 the wrong way round), with nothing
changed between the runs but the two SIDE strings:
  * Dirichlet on the K = 1.7 side: the driver's residual GREW, from 0.76 at
    iteration 2 to ~1.2, and stalled there — NOT CONVERGED in 40 iterations, at
    all three refinement levels tried. The participants were reported
    `responsive`, the exports were finite, and every level failed identically;
    the only signal was the residual history.
  * Dirichlet on the K = 0.35 side: converged to 5.8e-10 in 31 iterations at
    every level, order 2.00 in space-time.
If the roles are FIXED because one backend can only take one of them, the
remedy is relaxation, not hope: run the driver with accelerator="constant" and
theta0 ~ 1/(1+rho) — measured on the diverging assignment above, theta0 = 0.12
turned it into a converging run, 9.7e-10 in 282 iterations (327 s) against 31
iterations (37 s) for the same problem with the roles the right way round, and
a final-time field error of 1.04e-03 against 1.16e-03, i.e. the same answer for
ten times the work. Aitken did not rescue it on its own: its theta is clamped
to [0.05, 1], and relaxing a Jacobi sweep whose map has complex eigenvalues is
not the scalar sequence Aitken extrapolates.

THE TWO PARTICIPANTS MUST BE GIVEN THE SAME T_START, T_END, N_STEPS AND THETA.
Point 1 above is also a hole: the payload cannot carry the time grid, so a
partner configured with a different window produces a trace this script cannot
tell from a correct one unless its LENGTH differs. The length IS checked (loud
exit). Equal dt is NOT checkable from the payload and is on you.

THE FLUX IS THETA-AVERAGED, AND THAT IS NOT A DETAIL. The theta-scheme's
discrete residual on a constrained row is

    r_i = (M/dt + THETA*A) u^(n+1) - (M/dt - (1-THETA)*A) u^n
          - THETA*F^(n+1) - (1-THETA)*F^n
        = -[ THETA * int_Gamma q^(n+1) phi_i ds
             + (1-THETA) * int_Gamma q^n phi_i ds ]  + O(dt^2)

so the reaction recovers the THETA-WEIGHTED AVERAGE of the flux over the step,
NOT the flux at t^(n+1). That is exactly the quantity the Neumann side needs:
its own theta-scheme right-hand side asks for THETA*N^(n+1) + (1-THETA)*N^n,
i.e. the same average, applied UNCHANGED as `+ int g v ds`. So the two sides
match term for term with no time interpolation anywhere. Reading r_i as "the
flux at t^(n+1)" instead and handing that over is a first-order error in dt
injected straight into an otherwise second-order coupling — it does not look
like a bug, it looks like a scheme that stalled at order 1.

FIELD DUMP. The contract carries interface data only, so a transient run that
returns nothing but the interface is useless: this script writes
`field_final.npz` (dof coordinates, the final-time nodal field, the nodal
volume weights int phi_i dx, and the time grid) BEFORE exports.json. Anything
written after exports.json is a contract violation — the driver takes that file
appearing as proof the run finished.

  AND THE DUMP LEFT ON DISK AFTER A PROBED RUN IS NOT THE CONVERGED FIELD.
  run_coupling(probe=True) ends by calling probe_interface_sensitivity, which
  re-runs every participant TWICE more — once on the same imports, once with
  every exchanged number nudged by 1e-3 relative — and restores only
  imports.json and exports.json afterwards. Any OTHER file a participant wrote
  is left over from that PERTURBED run. Measured: reading field_final.npz
  straight after a probed coupling put a floor of 2.1e-4 under the field L2
  error, so the measured space-time order read 1.98, then 0.32, then -0.12 as
  the mesh was refined — a converged coupling reporting a stalling solver. The
  exports (and therefore every interface quantity) were correct throughout.
  Either pass probe=False, or re-run each participant ONCE in its work_dir
  after the driver returns: the driver leaves the converged imports.json in
  place, and this participant is a pure function of it.

MEASURED (this file, through OASiS's own `run_coupling`, on a manufactured
two-material problem built for it: a rectangle split by a straight interface,
k = 1.7 / rho_c = 2.1 against k = 0.35 / rho_c = 0.8, an exact solution
quadratic in x and cosine in y times 1 - exp(-2.5 t), zero initial condition,
exact Dirichlet data on the outer x-faces, natural y-faces, THETA = 0.5, tol
1e-9; h AND dt halved together over four levels, errors at the FINAL TIME):

    quantity                              lvl0      lvl3     orders
    L2 field error, both subdomains     1.16e-03  1.82e-05  2.00 2.00 2.00
    L2 interface temperature            1.55e-03  2.45e-05  1.99 2.00 2.00
    max interface temperature           5.63e-03  1.25e-04  1.81 1.83 1.85
    L2 interface flux, INTERIOR nodes   5.30e-03  8.72e-05  1.97 1.98 1.98
    L2 interface flux, WHOLE interface  1.22e-02  5.00e-04  1.55 1.54 1.52
    max interface flux, WHOLE interface 5.21e-02  6.34e-03  1.04 1.01 0.99
    interface flux balance, relative    6.35e-02  8.45e-03  ~1

THE TWO INTERFACE END NODES ARE FIRST ORDER HERE, and they set every whole-
interface norm. Measured with the exact interface data fed in, so the coupling
is not involved: interior nodes 1.97, 1.99, the two end nodes 1.03, 1.02, and
the end-node error is 6.8x the interior one at lvl0 and 25x at lvl2. Two O(h)
entries in an otherwise O(h^2) vector is what puts the whole-interface L2 at
~1.5 — report the interior number and the whole-interface number SEPARATELY, or
the recovery looks like a 1.5-order scheme. Copying a neighbour into them does
not fix it either; that value is O(h) at the end too.
  The mechanism is NOT established, and one measurement says it is not the
  obvious one. The obvious candidate is the half-width support at an end node
  making -r_i/w_i a one-sided average, but the SAME recovery in
  heat_iface_dealii_transient.cc, on the SAME manufactured problem, is 2.00 at
  the end nodes as well as inside — the difference there is Q1 quadrilaterals
  against these P1 triangles. So this is a property of the corner
  discretisation, not of the reaction formula. Treat the end nodes as suspect,
  and measure rather than assume on your own mesh.
  (The shipped corner guard is a DIFFERENT thing: it replaces interface nodes
  that also lie on an OUTER DIRICHLET boundary, whose residual is not this
  interface's flux at all. With OUTER_FACES = "x" it never fires, and the end
  nodes above are ordinary interface nodes that are simply less accurate.)

THE FLUX BALANCE IS ONLY FIRST ORDER, and that is the Neumann side's gradient
recovery, not a leak. The two exports cancel to 6.4e-2 relative at lvl0 and
8.5e-3 at lvl3 — order 1, the order of the projected gradient the Neumann side
exports. OASiS's conservation check reads the trace as one component per time
level and balances each on its own, so a coarse run produces one finding per
time step; they all go away at lvl1 and finer. Conservation of what is APPLIED
is exact by construction: the Neumann side integrates the Dirichlet side's
density unchanged.

CROSS-CODE. The same protocol against the deal.II transient participant
(participant_dealii_transient.py, FEniCSx on the Dirichlet side and deal.II on
the Neumann side) converged in 30-31 iterations and measured 2.00, 2.00 in the
final-time field L2 over three levels, 1.99 on the interior interface flux.

NON-MATCHING INTERFACE MESHES go through `sample_trace` and work: 37 interface
points against 25, converged in 30 iterations, the balance check took its
integral path and reported nothing. One level only, so no order from it — the
errors were 1.2x (field L2) to 7x (interface flux L2) those of the matching run
at the same level, which is the price of interpolating the trace twice per
iteration.
"""
import json
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from mpi4py import MPI
from petsc4py import PETSc

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material, BCs and time
#    window. As shipped this is the LEFT / Dirichlet side; the payload that
#    served this script gives the exact block for the RIGHT / Neumann side.
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.6      # this subdomain's x-extent
Y0, Y1    = 0.0, 0.4      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 0.8           # conductivity
RHO_C     = 1.0           # VOLUMETRIC heat capacity rho*c  (NOT c alone)
NX, NY    = 24, 16        # this subdomain's OWN mesh; need not match the partner
T_START   = 0.0           # coupling window start  ─┐ BOTH participants must be
T_END     = 1.0           # coupling window end     ├─ given the SAME three
N_STEPS   = 20            # steps in the window     ─┘ numbers AND the same THETA
THETA     = 0.5           # 0.5 = Crank-Nicolson (2nd order) | 1.0 = backward
                          # Euler (L-stable, 1st order — caps the space-time
                          # order at 1 when you refine dt with h)
OUTER_FACES = "x"         # "x"  : only the non-interface x-face is Dirichlet,
                          #        the two y-faces are natural (zero flux)
                          # "all": the WHOLE non-interface boundary is Dirichlet


# The three problem functions. Each is called with NUMPY ARRAYS x, y and must
# return an array of the same shape — write `0.0 * x + c` for a constant, not
# `c`, or dolfinx sees a scalar where it needs an array.
def T_INITIAL(x, y):
    """T(x, y, T_START). Must agree with T_OUTER(., T_START) on the Dirichlet
    faces, or the first step carries an initial layer that Crank-Nicolson
    answers with oscillations rather than with second order."""
    return 0.0 * x + 300.0


def T_OUTER(x, y, t):
    """Dirichlet datum on the non-interface boundary selected by OUTER_FACES."""
    return 0.0 * x + 320.0


def F_SRC(x, y, t):
    """Volumetric source (per unit volume, same units as rho_c dT/dt)."""
    return 0.0 * x


Q_GUESS   = 0.0           # iteration-1 fallback interface flux. There is no
                          # T_GUESS: the interface temperature at t = T_START is
                          # KNOWN (it is T_INITIAL), so the Dirichlet side's
                          # iteration-1 trace is that value held constant.
# ─────────────────────────────────────────────────────────────────────────

DT = (T_END - T_START) / N_STEPS
TIMES = T_START + DT * np.arange(N_STEPS + 1)      # t^0 ... t^N
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


def sample_trace(imp, key, fallback, y):
    """Map the partner's interface TRACE onto THIS participant's points.

    Returns (len(y), N_STEPS): one column per time step, interpolated in y
    COLUMN BY COLUMN. The driver does no interpolation — non-matching interface
    meshes are handled here — and for a trace that has to be done per time
    level. One np.interp over the flattened (m, N_STEPS) array interleaves the
    time levels: the result still has the right length, the coupling still
    converges, and every number is wrong.

    `fallback` is a scalar or a (len(y),) array, held constant in time.
    """
    fb = np.asarray(fallback, float).ravel()
    if fb.size != len(y):
        fb = np.full(len(y), float(fb.ravel()[0]))
    if not imp or not imp.get("coordinates"):
        return np.tile(fb[:, None], (1, N_STEPS))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:                       # a partner that exported one column
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size:
        return np.tile(fb[:, None], (1, N_STEPS))
    if vs.shape[1] != N_STEPS:
        # LOUD, because it is the one time-window error the payload can reveal.
        sys.exit(f"partner '{PARTNER}' exported a trace with {vs.shape[1]} time "
                 f"levels, this participant's window has N_STEPS={N_STEPS}. The "
                 f"exchange carries no time axis, so a trace of the WRONG "
                 f"LENGTH is the only symptom a mismatched window shows. Give "
                 f"both participants the same T_START/T_END/N_STEPS/THETA.")
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, n])
                            for n in range(N_STEPS)])


imp = read_imports()

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

# ── outer (non-interface) Dirichlet boundary ──────────────────────────────
if OUTER_FACES == "all":
    def _on_outer(x):
        return (np.isclose(x[0], OUTER_X) | np.isclose(x[1], Y0)
                | np.isclose(x[1], Y1))
else:
    def _on_outer(x):
        return np.isclose(x[0], OUTER_X)

outer_facets = dmesh.locate_entities_boundary(domain, fdim, _on_outer)
outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)

# WITH OUTER_FACES = "all" THE TWO INTERFACE END NODES BELONG TO THE OUTER
# BOUNDARY, ON BOTH SIDES: (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face,
# which carries prescribed data in the un-split problem, so they stay outer-
# Dirichlet in BOTH subproblems and are NOT interface-imposed. They are still
# exported. With OUTER_FACES = "x" those faces are natural and the end nodes
# ARE interface nodes like any other.
end_node = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_dofs = (iface_dofs[~end_node] if OUTER_FACES == "all"
                 else iface_dofs).astype(np.int32)

# ── forms: theta-scheme, ONE matrix for the whole march ───────────────────
u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
kc = fem.Constant(domain, default_scalar_type(K))
rc = fem.Constant(domain, default_scalar_type(RHO_C))
dtc = fem.Constant(domain, default_scalar_type(DT))
th = fem.Constant(domain, default_scalar_type(THETA))
omth = fem.Constant(domain, default_scalar_type(1.0 - THETA))

u_n = fem.Function(V)        # solution at t^n
uh = fem.Function(V)         # solution at t^(n+1)
f_old = fem.Function(V)      # F_SRC(., t^n)
f_new = fem.Function(V)      # F_SRC(., t^(n+1))
g_out = fem.Function(V)      # outer Dirichlet datum at t^(n+1)
g_if = fem.Function(V)       # interface datum for the CURRENT step

a = (rc / dtc) * u * v * ufl.dx + th * kc * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ((rc / dtc) * u_n * v * ufl.dx
     - omth * kc * ufl.dot(ufl.grad(u_n), ufl.grad(v)) * ufl.dx
     + th * f_new * v * ufl.dx + omth * f_old * v * ufl.dx)

bcs = [fem.dirichletbc(g_out, outer_dofs)]

# One definition of the interface measure, used by the Neumann branch to APPLY
# the partner's flux and by the Dirichlet branch to RECOVER its own.
facets_if = dmesh.locate_entities_boundary(domain, fdim,
                                           lambda x: np.isclose(x[0], IFACE_X))
tags_if = dmesh.meshtags(domain, fdim, np.sort(facets_if),
                         np.full(len(facets_if), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags_if)(7)

if SIDE == "dirichlet":
    # The trace this side IMPOSES: partner temperature at t^1 ... t^N.
    imp_trace = sample_trace(imp, "values",
                             T_INITIAL(xy[iface_dofs, 0], xy[iface_dofs, 1]),
                             y_if)
    bcs.append(fem.dirichletbc(g_if, iface_bc_dofs))
else:
    # The trace this side APPLIES: partner's THETA-AVERAGED flux per step,
    # UNCHANGED (see THE FLUX in the module docstring).
    imp_trace = sample_trace(imp, "normal_fluxes", Q_GUESS, y_if)
    L += g_if * v * ds_if

af, Lf = fem.form(a), fem.form(L)

# THE THETA-SCHEME MATRIX DOES NOT DEPEND ON t. Assemble and factorize it ONCE:
# with a fixed dt and a fixed set of constrained dofs, re-assembling per step
# buys nothing and pays N_STEPS LU factorizations for it.
A = _fp.assemble_matrix(af, bcs=bcs)
A.assemble()
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType("preonly")
ksp.getPC().setType("lu")

# Dirichlet side: the SAME form assembled with NO boundary condition, kept for
# the reaction below. Neumann side: an L2 mass matrix for the flux projection.
if SIDE == "dirichlet":
    A_free = _fp.assemble_matrix(af)          # no bcs= on purpose
    A_free.assemble()
    wvec = _fp.assemble_vector(fem.form(v * ds_if))   # w_i = int_Gamma phi_i ds
    wvec.ghostUpdate()
    wi = wvec.array[iface_dofs]
    ok = np.abs(wi) > 1e-14
    # An interface node that ALSO lies on the outer Dirichlet boundary carries
    # the OUTER reaction as well, so its residual is not this interface's flux.
    # Take the nearest interior interface node rather than exporting a corner
    # value that is physically a different quantity.
    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    fixup = [(i, good[np.argmin(np.abs(good - i))])
             for i in np.where(suspect)[0]] if len(good) else []
else:
    p_, w_ = ufl.TrialFunction(V), ufl.TestFunction(V)
    Mf = fem.form(p_ * w_ * ufl.dx)
    qLf = fem.form(-kc * default_scalar_type(S) * uh.dx(0) * w_ * ufl.dx)
    qL0f = fem.form(-kc * default_scalar_type(S) * u_n.dx(0) * w_ * ufl.dx)
    Mmat = _fp.assemble_matrix(Mf)
    Mmat.assemble()
    ksp_m = PETSc.KSP().create(domain.comm)
    ksp_m.setOperators(Mmat)
    ksp_m.setType("preonly")
    ksp_m.getPC().setType("lu")
    qh = fem.Function(V)

    def _project_flux(form_):
        rhs = _fp.assemble_vector(form_)
        rhs.ghostUpdate()
        ksp_m.solve(rhs, qh.x.petsc_vec)
        qh.x.scatter_forward()
        rhs.destroy()
        return qh.x.array[iface_dofs].copy()

# ── initial condition ─────────────────────────────────────────────────────
u_n.interpolate(lambda X: T_INITIAL(X[0], X[1]))
f_old.interpolate(lambda X: F_SRC(X[0], X[1], TIMES[0]))
T_out = np.zeros((len(iface_dofs), N_STEPS))
Q_out = np.zeros((len(iface_dofs), N_STEPS))
if SIDE == "neumann":
    q_prev = _project_flux(qL0f)          # flux at t^0, for the first average

# ── the march. ONE run = the WHOLE window (waveform) ──────────────────────
for n in range(N_STEPS):
    t_new = TIMES[n + 1]
    f_new.interpolate(lambda X: F_SRC(X[0], X[1], t_new))
    g_out.x.array[outer_dofs] = T_OUTER(xy[outer_dofs, 0], xy[outer_dofs, 1],
                                        t_new)
    # The imported trace enters HERE, one column per step: column n is the
    # partner's datum for the step t^n -> t^(n+1).
    g_if.x.array[iface_dofs] = imp_trace[:, n]

    b = _fp.assemble_vector(Lf)
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b_free = b.copy() if SIDE == "dirichlet" else None   # no lifting, no set_bc
    _fp.apply_lifting(b, [af], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    _fp.set_bc(b, bcs)
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()

    # Outward normal flux density q = -(K grad T).n on the interface, THETA-
    # averaged over this step.
    #
    # WHY NOT AN L2 PROJECTION OF THE GRADIENT. The gradient of a P1 solution is
    # only O(h) accurate ON the boundary — the superconvergence points are
    # interior — and the boundary trace is exactly what the coupling reads.
    # Measured on the manufactured transient two-material problem, refining h
    # and dt together, with the SHIPPED file against one copy of it whose
    # Dirichlet side exports the projected gradient instead and nothing else
    # changed:
    #     interface flux, interior nodes, L2   1.98  ->  1.11
    #     interface temperature, L2            2.00  ->  1.24
    #     FINAL-TIME FIELD L2, BOTH SUBDOMAINS 2.00  ->  1.37 (and falling)
    # The coupling converged just as well either way — 29-30 iterations against
    # 31, residual 7e-10 to 9e-10 against 5e-10, every driver check silent — so
    # nothing in the run tells you which of those two answers you got.
    #
    # THE CONSISTENT (REACTION) FLUX. From
    #     a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds = -int_Gamma qn v ds
    # applied to the theta-scheme's own discrete equation, for every basis
    # function phi_i on the interface
    #     int_Gamma [THETA q^(n+1) + (1-THETA) q^n] phi_i ds = -r_i,
    #     r = (M/dt + THETA A) u^(n+1) - (M/dt - (1-THETA) A) u^n
    #         - THETA F^(n+1) - (1-THETA) F^n
    # with r the UNCONSTRAINED residual: assembled with no boundary condition
    # applied and with the constrained rows NOT zeroed, because on the Dirichlet
    # side those rows ARE the reaction and zeroing them destroys the very
    # quantity being recovered. Dividing by w_i = int_Gamma phi_i ds turns the
    # functional into a density the partner can interpolate pointwise.
    if SIDE == "dirichlet":
        r = A_free.createVecLeft()
        A_free.mult(uh.x.petsc_vec, r)
        r.axpy(-1.0, b_free)
        q = np.zeros(len(iface_dofs))
        q[ok] = -r.array[iface_dofs][ok] / wi[ok]
        for i, j in fixup:
            q[i] = q[j]
        r.destroy()
        b_free.destroy()
    else:
        # NEUMANN SIDE: the reaction formula MUST NOT be used here. These
        # interface DOFs are free, the discrete equations hold on them, so r is
        # ~0 and the expression would silently export ZERO flux with no error
        # raised. Measured on this side of the manufactured problem, every step:
        # max|r| on the interface rows is 1.6e-16, which divided by w_i is a
        # flux density of 1e-14 where the true flux is 1.0. This side's export
        # is not what the partner consumes in any case — the Dirichlet partner
        # reads its `values` — but it IS what the conservation check balances,
        # so it must still be the THETA-average, on the same step grid.
        q_new = _project_flux(qLf)
        q = THETA * q_new + (1.0 - THETA) * q_prev
        q_prev = q_new

    T_out[:, n] = uh.x.array[iface_dofs]
    Q_out[:, n] = q

    u_n.x.array[:] = uh.x.array
    f_old.x.array[:] = f_new.x.array
    b.destroy()

# FIELD DUMP, BEFORE exports.json (see the docstring). `weights` are the nodal
# volume weights int phi_i dx, so a mass-lumped L2 norm of any nodal field is
# sqrt(sum(weights * field**2)) with no mesh work outside this script.
wvol = _fp.assemble_vector(fem.form(v * ufl.dx))
wvol.ghostUpdate()
np.savez("field_final.npz", coordinates=xy[:, :2], temperature=uh.x.array.copy(),
         weights=wvol.array.copy(), times=TIMES, iface_dofs=iface_dofs,
         side=SIDE, theta=THETA)

print(f"[fenics-transient {SIDE}] iface n={len(iface_dofs)} steps={N_STEPS} "
      f"dt={DT:.6g} theta={THETA} "
      f"T(t_end)=[{T_out[:, -1].min():.6g},{T_out[:, -1].max():.6g}] "
      f"q(last step)=[{Q_out[:, -1].min():.6g},{Q_out[:, -1].max():.6g}]")

# exports.json LAST: the driver takes its existence as proof of success.
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [[float(t) for t in row] for row in T_out],
    "normal_fluxes": [[float(q) for q in row] for row in Q_out],
}, indent=2))
