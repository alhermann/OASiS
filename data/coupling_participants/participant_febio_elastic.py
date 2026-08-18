"""FEBio 4 VECTOR participant for the OASiS `couple` driver.

Plane-strain linear elasticity  -div(sigma(u)) = b  on ONE rectangular
subdomain of a domain split by a straight interface at x = IFACE_X, with the
body force b given by B_SRC in the edit block. Like the
other *_elastic participants (and unlike the scalar ones), the exchanged
interface state is a VECTOR on BOTH channels:

    values        = displacement       u = (u_x, u_y)   at the interface nodes
    normal_fluxes = interface traction export            (SIGN CONVENTION below)

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

FEBio HAS NO SCRIPTING API: it is XML-in / logfile-out. This module is a
WRAPPER that each coupling iteration (1) reads imports.json, (2) writes a
complete FEBio 4.0 .feb deck with the imported interface data baked in PER
NODE, (3) runs `febio4 -i deck.feb`, (4) parses the ASCII <logfile>, (5) writes
exports.json.

PLANE STRAIN IN A 3-D CODE. FEBio solves 3-D solids only, so the subdomain is
meshed as ONE layer of hex8 elements of thickness ZTHICK with u_z = 0 on every
node. That is plane strain exactly, not approximately: the solution is
z-invariant, so the two z-layers of nodes carry identical (u_x, u_y) and the
slab collapses onto the 2-D interface line the other participants speak. The
exported points are therefore the DISTINCT y values of the interface, with
2-D coordinates [IFACE_X, y] — the same list the *_elastic siblings export.

SIGN CONVENTION — the thing a vector coupling gets wrong silently.
`normal_fluxes` is exported as

    q_out = -(sigma . n_own)                       n_own = S * e_x

the SAME convention the shipped scalar participants use for heat
(q_out = -k dT/dn_own) and the one participant_fenics_elastic.py and
participant_skfem_elastic.py use. Two consequences, both load-bearing:

  * the two sides' exports CANCEL componentwise, because n_own is anti-parallel
    across the interface — that is what makes the interface balance check a
    conservation statement rather than an accident;
  * the NEUMANN side applies the partner's numbers UNCHANGED, because the
    natural boundary term of the elasticity weak form is
    +(sigma . n_own) . v = +q_out_partner . v.

Exporting the raw traction (sigma . n_own) instead flips the sign the Neumann
side applies; the iteration still converges, to the wrong answer.

THE TRACTION EXPORT IS PER NODE, AND IT IS THE REACTION — NOT A DOMAIN AVERAGE
AND NOT A STRESS PROJECTION.

The scalar participant_febio.py exports
    q_out = np.full(len(iface_nodes), -mean(element sx) * S)
i.e. ONE domain-averaged sigma_xx broadcast onto every interface point. For its
own 1-D bar that is the exact answer (sigma_xx really is constant there). For
any problem whose interface traction VARIES ALONG THE INTERFACE — a
shear-modulus jump, a non-uniform load, anything two-dimensional — it is not
even first-order accurate: the exported profile is flat, so its error does not
shrink when the mesh is refined, and the Neumann partner is handed a boundary
condition that is simply wrong at both ends of the interface. MEASURED on the
manufactured two-material problem below, whose interface traction is
t_x = K*y (exactly linear in y): the domain-average export sits at the midpoint
value, its relative L2 error along the interface is 45.4% / 45.0% / 44.8% /
44.8% on the four refinement levels — CONVERGENCE ORDER 0.01, 0.00, 0.00 — and
its worst pointwise relative error is 50.0% on every mesh. Refining does not
help, because the error is not a discretisation error.

What is exported instead is the variationally consistent (reaction) traction,
the same recovery every other participant in this corpus uses. From

    a(u,v) - (f,v) = int_dOmega (sigma(u) . n) . v ds = -int_Gamma q_out . v ds

(the second equality is this file's sign convention, q_out = -(sigma . n_own))
it follows that for every vector basis function phi_i on the interface

    int_Gamma q_out . phi_i ds = -r_i,    r = A u_h - b

with r the UNCONSTRAINED residual — assembled with no boundary condition
applied and with the constrained rows NOT zeroed, because on the Dirichlet side
those rows ARE the reaction. FEBio exposes exactly that vector: the node log
variables "Rx","Ry","Rz" return FESolidSolver2::m_Fr at PRESCRIBED dofs, and
m_Fr is accumulated in FEResidualVector::Assemble as -(sum of the assembled
element vectors) = F_internal - F_external = (A u_h - b)_i. So

    q_i = -R_i / w_i,      w_i = int_Gamma phi_i ds

with w_i the interface nodal weight, computed here in closed form from the
quad4 interface faces (2x2 Gauss on a bilinear face is exact for a planar
quad). The two z-layers of one interface y are summed in BOTH numerator and
denominator, which is the correct collapse of the slab: q = -(R_bot+R_top) /
(w_bot+w_top).

Why not the element stresses. FEBio can log element sx/sxy, and averaging those
onto the interface is the obvious FEBio-shaped alternative. It is a stress
recovered from the GRADIENT of a trilinear solution and evaluated ON a
boundary, which is only O(h) accurate there (the superconvergence points are
interior, and the boundary trace is exactly what the coupling reads). Both
routes were run on the problem below and measured AT MID-INTERFACE, away from
the end effects discussed there, on the four meshes:

    reaction        (this file, Dirichlet side)  1.5e-5 1.3e-5 2.8e-6 4.4e-7
    element stress  (this file, Neumann side)    6.2e-3 3.1e-3 1.6e-3 7.8e-4

The element-stress route is order 1.00 flat (0.98, 1.01, 1.00). The reaction
route drops away much faster (0.23, 2.23, 2.68 — it is running into this
build's own arithmetic floor, see below, so the last figures are a floor and
not an order) and is ~1800x more accurate on the finest mesh. The recovery, not
the physics and not the partner, sets the answer.

MEASURED, ON A MANUFACTURED TWO-MATERIAL PROBLEM. Unit square split at x = 0.5,
plane strain, a 3x SHEAR-MODULUS JUMP across the interface (mu 400 -> 1200 at
equal lam = 600), exact solution u_x = (a + b x) y, u_y = c x^2 + p x + q per
subdomain — zero body force, and an interface traction t_x = K y that VARIES
LINEARLY ALONG THE INTERFACE, t_y = const. Dirichlet-Neumann against
participant_skfem_elastic.py (P1 triangles) through
core.coupling_driver.run_coupling with Aitken relaxation; four uniformly
refined meshes 8x8 / 16x16 / 32x32 / 64x64 per subdomain; relative L2 errors:

  FEBio = Dirichlet side (it EXPORTS the traction)
    u, over the whole coupled domain   6.212e-4 1.536e-4 3.866e-5 9.756e-6
                                       ORDER 2.02  1.99  1.99
    t, along the interface             6.572e-2 2.329e-2 8.240e-3 2.914e-3
                                       ORDER 1.50  1.50  1.50
  FEBio = Neumann side (it APPLIES the traction)
    u, over the whole coupled domain   6.070e-4 1.476e-4 3.663e-5 9.167e-6
                                       ORDER 2.04  2.01  2.00
    t, exported by the skfem partner   ORDER 1.50  1.50  1.50

DISPLACEMENT IS ORDER 2. THE TRACTION IS ORDER 1.5, AND IT IS NOT THE RECOVERY.
The traction error is a TWO-NODE BOUNDARY LAYER at the two points where the
interface meets the outer Dirichlet boundary: an O(h) amplitude over an O(h)
width integrates to O(h^1.5) in L2 whatever happens in between. Away from it
the exported traction is 1.5e-5 / 1.3e-5 / 2.8e-6 / 4.4e-7 relative at
mid-interface — better than order 2, down to this build's floor. Handed the
EXACT interface displacement instead of a partner's (same participant, run
alone), the recovery reproduces the exact traction at EVERY interface node to
3.0e-7 / 3.2e-7 / 3.3e-7 / 3.9e-7 relative: exact to the floor, with no
h-dependence at all. Running FEBio on BOTH sides gives the same 1.50, so the
layer is not the P1 partner either. Of the two layer nodes, the interface
CORNER is O(h) BY CONSTRUCTION — it is the nearest-interior copy the siblings
also use, measured 12.0% / 6.0% / 3.0% / 1.5%, i.e. exactly h.

CONSERVATION, MEASURED THREE WAYS ON THE SAME RUNS.
  * The consistent nodal load this file bakes into the deck reproduces
    int_Gamma q_partner . phi_i ds over the loaded nodes to 3.8e-16 / 1.7e-16 /
    3.3e-16 / 5.3e-16 relative. The exchange itself is exact.
  * The NET interface force carried by the exported traction matches the exact
    analytic net force to 3.0e-4 / 1.1e-4 / 3.4e-5 / 9.1e-6 (order 1.42 ->
    1.89, heading for 2).
  * The two sides' INDEPENDENTLY recovered tractions cancel in the resultant to
    6.7e-2 / 3.3e-2 / 1.6e-2 / 8.2e-3, order 1.0, against a scikit-fem partner
    whose own recovery is the O(h) stress projection; with FEBio on the Neumann
    side (its O(h) element-stress recovery) 1.2e-2 / 5.7e-3 / 2.8e-3 / 1.4e-3.
    That number is NOT a coupling defect — it is the first-order recovery on
    the far side, which is precisely what keeps the check from being vacuous.

THE NEUMANN SIDE APPLIES CONSISTENT NODAL FORCES, NOT A SURFACE-LOAD MAP.
FEBio's <surface_load type="traction"> reads its vec3 from a SurfaceData map,
and that map is PER FACE, not per face node: FEBioMeshDataSection creates the
FESurfaceMap with the default FMT_MULT storage and FESurfaceMap::setValue
writes the one parsed vector into all m_maxFaceNodes slots of the facet. A
profile that varies along the interface is therefore applied piecewise constant
— an O(h) representation of the boundary data. (That is read off the source,
NOT measured here: the surface-load route was never run, so treat the O(h) as
the standard argument rather than as a number from this test.) What this file
does instead is exactly what `L += inner(g, v) * ds` does in the
FEniCSx/scikit-fem siblings, evaluated here in closed form:

    F_i = int_Gamma t_h . phi_i ds = sum_j M_ij t_j ,
    M = the quad4 surface mass matrix of the interface faces,

with t_h the bilinear interpolant of the partner's NODAL traction samples. The
result is applied as <nodal_load type="nodal_force"> through a vec3 NodeData
map. Same variational statement, no interpolation loss, and it is a standard
FEBio load.

THE BODY FORCE GOES IN THE SAME WAY, AND NOT AS <body_load>. B_SRC in the edit
block is a NumPy function of (x, y) — the same contract the FEniCSx and
scikit-fem siblings use — and it is integrated HERE,

    F_i = int_Omega b . phi_i dV,     3x3x3 Gauss on each hex8,

then applied over ALL nodes through the same vec3 NodeData map machinery, as a
SECOND <nodal_load>. On the Neumann side the interface entry is left exactly as
it was: FEBio adds every model load into the same residual, so the partner's
numbers still reach the solver UNCHANGED, which is what the coupling contract
says. Why not the deck's own body load:

  * it cannot take a polynomial. FEBio 4 registers "const" and "non-const" as
    FEBio-2 loads obsolete since 3.0, and the current "body force"
    (FEGenericBodyForce) carries ONE vec3 `force` parameter. Replacing a
    polynomial source by a constant is not an approximation of the problem, it
    is a different problem. The parameter does accept a math STRING, but then
    the source has to be written twice in two languages, and everything a
    NumPy callable can do that a math string cannot — a table, a np.where, an
    interpolant — is lost;
  * and it would not mean what it says. FEElasticSolidDomain::BodyForce sets
    the integrand to -H[a] * density * f * J0 and assembles it the way INTERNAL
    forces are assembled, so the deck number is multiplied by the material
    density AND enters with the opposite sign: <force>f</force> applies a
    physical body force of MINUS f. Measured, not inferred — the two routes
    agree only after the deck value is negated;
  * nothing is given up by integrating here: the rule above is exact through
    degree 5 per direction, and the statement is the same `inner(b, v) * dx`
    the siblings assemble.

b is a force per unit VOLUME, the same number the 2-D siblings take. The slab's
volume integral, its stiffness and its interface load each carry one factor of
ZTHICK, so it cancels and the solution stays thickness-independent.

AND THE DIRICHLET SIDE MUST TAKE IT BACK OUT OF THE REACTION. m_Fr, the vector
behind the Rx/Ry log data, is accumulated ONLY in
FEResidualVector::Assemble(en, elm, fe) — the ELEMENT path — so internal forces
and element-based loads land in it. A nodal load does not go through that path:
FENodalLoad::LoadVector calls the scalar FEResidualVector::Assemble(node, dof,
f), which adds to m_R when the equation number is >= 0 and otherwise DROPS f
(it only hands it to the rigid solver). At a prescribed dof m_Fr is therefore
F_int alone, while the reaction identity needs r = A u_h - b with the FULL b.
The traction export below subtracts the body-force entries by hand — the same
array that was written into the deck, so nothing is approximated. Omitting that
correction leaves an error of b_i / w_i ~ |b| h / 2, i.e. O(h), on a quantity
that is otherwise second order.

That is read off the source, and then MEASURED both ways. The assembly path
first: one constant load applied once as this file's <nodal_load> and once as
FEBio's own <body_load> gives displacements identical to 1e-15 absolute
(1.7e-12 relative) — so the consistent vector below IS the load FEBio would
have integrated itself — and reactions differing by exactly the consistent body
load, to 5e-15 on values of 5.5e-4. Then the export, on the manufactured
solution u_x = K x y (y - 1), u_y = 0, whose trace on all three OUTER faces is
identically zero (UDX = UDY = 0 represents it exactly) and whose source
b = (-2 mu K x, -(mu + lam) K (2y - 1)) is a genuine polynomial. Exported
traction against the closed-form -(sigma . n_own), the exact interface
displacement handed in through imports.json, meshes 8/16/32/64, relative L2
over the interface minus its two corners:

    correction ON    2.51e-2  6.39e-3  1.61e-3  4.06e-4   ORDER 1.97 1.99 1.99
    correction OFF   5.61e-2  3.32e-2  1.80e-2  9.40e-3   ORDER 0.76 0.88 0.94

Second order with it, first order without it, and 23x worse on the finest mesh.
With a CONSTANT source instead (quadratic manufactured u, where the Q1 solution
is nodally exact) the corrected export sits at the finite-strain floor
discussed below — 6.3e-6 relative on all four meshes, no h-dependence — while
the uncorrected one runs 9.75e-2 4.84e-2 2.41e-2 1.20e-2, order 1.01 1.01 1.00.

WHAT IS APPROXIMATED, HONESTLY.

  * FEBio has no small-strain material. `isotropic elastic` is
    St.Venant-Kirchhoff: Cauchy stress (1/J) F (lam tr(E) I + 2 mu E) F^T with E
    the Green-Lagrange strain. It is linear elasticity only as |grad u| -> 0,
    with a relative error O(|grad u|). If your displacements are LARGE this
    participant is solving finite strain, which may be what you want — but then
    it is no longer the same equation the linear siblings solve, and the
    coupling is only consistent if the partner is also finite-strain.
  * THERE IS AN ACCURACY FLOOR IN THE RECOVERED TRACTION, and it is a
    U-shaped function of the displacement amplitude, so it cannot be made
    arbitrarily small. Above it sits the O(|grad u|) nonlinearity just
    described; below it sits roundoff, because the finite-strain kinematics
    forms F from CURRENT nodal positions X + u with X = O(1), so |grad u|
    carries an absolute roundoff ~1e-16 and therefore a relative one
    ~1e-16/|grad u|. Measured at 32x32 with |grad u| = 2.8e-6 / 2.8e-7 /
    2.8e-8 / 2.8e-9: 3.3e-6 / 3.3e-7 / 5.0e-7 / 5.4e-6 relative error in the
    recovered traction — a minimum of ~3e-7 around |grad u| ~ 3e-7 (the exact
    minimum was not bracketed further). Consequences:
    scale your problem so |grad u| lands near that minimum, and do not ask the
    `couple` driver for a residual tolerance below ~1e-7 when FEBio is the
    DIRICHLET side — there the noisy reaction IS the exported quantity the
    driver's residual is measured on. Measured: at tol=1e-8 the driver reported
    NOT CONVERGED on the two finest meshes, residual stalled at 3.2e-8 and
    4.8e-8; at tol=1e-7 the same four runs converge in 37/39/41/43 Aitken
    iterations and every error above is unchanged to four digits. The stall is
    arithmetic, not an unsettled interface — but the driver is right to refuse
    to certify it, so ask for a tolerance the solver can actually reach. With
    FEBio only on the NEUMANN side, tol=1e-8 converged on all four meshes.
  * hex8 is TRILINEAR. In-plane that is Q1, not the P1 of the scikit-fem/FEniCSx
    siblings, so the two sides of a coupling have different (both O(h^2))
    discretisation errors. That is normal for a partitioned coupling and is not
    an inconsistency.
  * This build of FEBio has NO pardiso (NumCore registers it only under
    #ifdef PARDISO, and FENewtonSolver's default type is the string "pardiso"
    regardless). LINSOLVE below is therefore set to "skyline", which this build
    always has. Leaving it unset works only where pardiso was compiled in.
  * The Neumann side's OWN traction export is the O(h) element-stress recovery
    (see above), not the reaction — the reaction is identically zero at free
    dofs. It is not what the Dirichlet partner consumes; it exists so the
    interface balance check is an independent statement rather than an echo.

RELAXATION IS NOT PER COMPONENT — see the long note in
participant_skfem_elastic.py. The driver applies ONE theta to the whole
interface state and the optimal theta is set by the WORST component, so
subdomains of the same length and Poisson ratio (proportional
Steklov-Poincare operators) are the well-behaved case; anything else wants
`accelerator="aitken"`.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side.
SIDE      = "dirichlet"   # "dirichlet" (import u, export traction) | "neumann"
PARTNER   = "right"       # the partner's `name` in your couple(...) call
X0, X1    = 0.0, 0.5      # this subdomain's x-extent
Y0, Y1    = 0.0, 1.0      # this subdomain's y-extent
ZTHICK    = 0.05          # slab thickness; ANY positive value, plane strain
IFACE_X   = 0.5           # the shared interface; must equal X0 or X1
E_MOD     = 1040.0        # Young's modulus
NU        = 0.3           # Poisson ratio (PLANE STRAIN)
# Prescribed displacement on this subdomain's WHOLE non-interface boundary
# (its outer x-face and both y-faces), as a QUADRATIC polynomial in (x, y):
#     u_x = UDX[0] + UDX[1]*x + UDX[2]*y + UDX[3]*x*x + UDX[4]*x*y + UDX[5]*y*y
#     u_y = UDY[0] + UDY[1]*x + UDY[2]*y + UDY[3]*x*x + UDY[4]*x*y + UDY[5]*y*y
# (a strict superset of the 4-term form the FEniCSx/scikit-fem siblings ship:
#  their (c0, c1, c2, c3) is this file's (UD[0], UD[1], UD[2], UD[5]). The x*x
#  and x*y terms are what a two-material solution with a traction that VARIES
#  along the interface needs — with only the 4-term form and no body force the
#  exact solution is affine and the interface traction is constant, which is
#  the one case the broken domain-average export gets right.)
# The two subdomains must agree at the two interface corners, or the coupled
# problem is not the un-split one.
UDX = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
UDY = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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

    Write it exactly as you would for the FEniCSx or scikit-fem sibling, in the
    same units. It does NOT become a <body_load>, which could not carry a
    polynomial anyway: this file integrates B_SRC against the element shape
    functions and applies the consistent nodal force vector (see the header).
    """
    return np.zeros_like(x), np.zeros_like(y)
NX, NY    = 16, 16        # this subdomain's OWN mesh; need not match the partner
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction export
FEBIO     = "febio4"      # the FEBio binary path `discover(query='list')` prints
LINSOLVE  = "skyline"     # NOT "pardiso": many builds ship without it (see above)
# ─────────────────────────────────────────────────────────────────────────

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

DECK = "cpl.feb"
LOG_U = "cpl_u.csv"        # ux, uy at the interface nodes
LOG_R = "cpl_r.csv"        # Rx, Ry at the interface nodes (Dirichlet side only)
LOG_E = "cpl_e.csv"        # sx, sxy per element (Neumann side only)


def _n(v):
    """Full-precision XML number. NEVER use repr()/!r: numpy 2 scalars
    stringify as 'np.float64(0.0)' and FEBio rejects the deck."""
    return format(float(v), ".17g")


# ---------------------------------------------------------------- imports
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
    """Map the partner's VECTOR samples onto THIS participant's interface
    points, COMPONENT BY COMPONENT.

    The driver does no interpolation — non-matching interface meshes are
    handled here, and for a vector field that has to be done per component. One
    np.interp over a flattened (N, 2) array interleaves the two components: the
    result still has the right length, the coupling still converges, and every
    number is wrong.

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


def u_dirichlet(x, y):
    """The prescribed displacement on the non-interface boundary."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (UDX[0] + UDX[1] * x + UDX[2] * y
            + UDX[3] * x * x + UDX[4] * x * y + UDX[5] * y * y,
            UDY[0] + UDY[1] * x + UDY[2] * y
            + UDY[3] * x * x + UDY[4] * x * y + UDY[5] * y * y)


# ------------------------------------------------------------------- mesh
class Mesh:
    """One layer of hex8 over [X0,X1] x [Y0,Y1] x [0,ZTHICK]. 1-based ids."""

    def __init__(self):
        self.xs = np.linspace(X0, X1, NX + 1)
        self.ys = np.linspace(Y0, Y1, NY + 1)
        self.zs = np.array([0.0, ZTHICK])

        # NODE NUMBERING IS A PERFORMANCE DECISION, NOT A COSMETIC ONE.
        # FEBio numbers equations in node order and the fallback direct solver
        # is `skyline`, whose cost is O(n * bandwidth^2). Numbering the slab
        # layer-by-layer (all of z=0, then all of z=ZTHICK) puts the two nodes
        # of every through-thickness edge (NX+1)(NY+1) apart, which on a 64x64
        # mesh is a bandwidth of ~12700 dofs and a solve that does not finish.
        # Running the THICKNESS index fastest, then the shorter in-plane
        # direction, keeps the bandwidth at ~6*min(NX,NY) dofs. Measured on
        # 64x64: minutes -> ~2 s per solve, same answer to the last digit.
        fast_i = NX <= NY
        if fast_i:
            def nid(i, j, k):
                return 1 + k + 2 * i + 2 * (NX + 1) * j
        else:
            def nid(i, j, k):
                return 1 + k + 2 * j + 2 * (NY + 1) * i

        self.nid = nid
        self.nodes = [(nid(i, j, k), self.xs[i], self.ys[j], self.zs[k])
                      for k in range(2) for j in range(NY + 1)
                      for i in range(NX + 1)]
        self.nodes.sort()
        self.xyz = np.zeros((len(self.nodes) + 1, 3))     # 1-based lookup
        for (n, x, y, z) in self.nodes:
            self.xyz[n] = (x, y, z)
        self.elems = []
        e = 1
        for j in range(NY):
            for i in range(NX):
                self.elems.append((e, [nid(i, j, 0), nid(i + 1, j, 0),
                                       nid(i + 1, j + 1, 0), nid(i, j + 1, 0),
                                       nid(i, j, 1), nid(i + 1, j, 1),
                                       nid(i + 1, j + 1, 1), nid(i, j + 1, 1)]))
                e += 1
        # the column of elements that touches the interface, in ascending y
        i_col = NX - 1 if ON_RIGHT else 0
        self.iface_elems = [1 + i_col + NX * j for j in range(NY)]

        i_if = NX if ON_RIGHT else 0
        # THE EXPORTED POINTS: the distinct interface y values, ascending. Same
        # points in the same order every iteration — the driver relaxes export
        # vectors entry by entry, so any reordering is silently wrong.
        self.y_if = self.ys.copy()
        # per exported point, the (bottom, top) node ids of the two z-layers
        self.iface_pair = [(nid(i_if, j, 0), nid(i_if, j, 1))
                           for j in range(NY + 1)]
        self.iface_all = [n for pair in self.iface_pair for n in pair]
        # THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
        # (IFACE_X, Y0) and (IFACE_X, Y1) sit on a y-face, which carries a
        # prescribed displacement in the un-split problem, so they stay
        # Dirichlet in BOTH subproblems. Handing them to the interface instead
        # leaves them unconstrained on the Neumann side: that subproblem is
        # still well posed, still converges, and lands a few percent off —
        # measured in the siblings, 4.7% in the interface displacement and 28%
        # in the interface traction on a coupling whose residual reached 1e-10
        # and whose flux balanced. They are still EXPORTED; they are just not
        # interface-imposed. (They also must not appear in two prescribed-
        # displacement BCs at once, which FEBio would not resolve for you.)
        self.interior_j = [j for j in range(NY + 1)
                           if abs(self.ys[j] - Y0) > TOL
                           and abs(self.ys[j] - Y1) > TOL]
        self.iface_free = [n for j in self.interior_j
                           for n in self.iface_pair[j]]
        # the WHOLE non-interface boundary: outer x-face + both y-faces
        outer = set()
        i_out = 0 if ON_RIGHT else NX
        for k in range(2):
            for j in range(NY + 1):
                outer.add(nid(i_out, j, k))
            for i in range(NX + 1):
                outer.add(nid(i, 0, k))
                outer.add(nid(i, NY, k))
        self.outer = sorted(outer)
        # interface quad4 faces (used for the nodal weights and, on the Neumann
        # side, for the consistent nodal forces)
        self.faces = [[nid(i_if, j, 0), nid(i_if, j + 1, 0),
                       nid(i_if, j + 1, 1), nid(i_if, j, 1)]
                      for j in range(NY)]

    # ---- interface surface integrals -----------------------------------
    def _face_gauss(self, face):
        """2x2 Gauss on the bilinear quad4: (shape values, weight*detJ) per
        point. Exact for the Q1 mass matrix of a planar quad."""
        p = self.xyz[face]                          # (4, 3)
        g = 1.0 / np.sqrt(3.0)
        out = []
        for xi in (-g, g):
            for eta in (-g, g):
                N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                     (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                dNx = 0.25 * np.array([-(1 - eta), (1 - eta),
                                       (1 + eta), -(1 + eta)])
                dNe = 0.25 * np.array([-(1 - xi), -(1 + xi),
                                       (1 + xi), (1 - xi)])
                jac = np.cross(dNx @ p, dNe @ p)
                out.append((N, float(np.linalg.norm(jac))))
        return out

    def iface_weights(self):
        """w[node] = int_Gamma phi_node ds over the interface faces."""
        w = np.zeros(len(self.nodes) + 1)
        for f in self.faces:
            for N, dj in self._face_gauss(f):
                w[f] += N * dj
        return w

    def iface_mass_apply(self, t_node):
        """F_i = int_Gamma t_h . phi_i ds with t_h the BILINEAR interpolant of
        the nodal traction samples: the consistent nodal force vector, i.e.
        exactly `inner(g, v) * ds` of the FEniCSx/scikit-fem siblings.

        t_node maps node id -> (tx, ty). Returns node id -> (Fx, Fy)."""
        F = np.zeros((len(self.nodes) + 1, 2))
        for f in self.faces:
            tv = np.array([t_node[n] for n in f])       # (4, 2)
            for N, dj in self._face_gauss(f):
                F[f] += np.outer(N, (N @ tv)) * dj
        return F

    # ---- volume integral of the body force -----------------------------
    def body_load(self):
        """F_i = int_Omega B_SRC . phi_i dV over the hex8 elements: the
        consistent nodal force vector of the body force, i.e. exactly
        `inner(b, v) * dx` of the FEniCSx/scikit-fem siblings, evaluated here
        because the deck cannot be handed a non-constant source (header).

        3x3x3 Gauss — exact through degree 5 per direction. The integrand is
        b times a shape function, so a source polynomial up to degree 4 per
        direction is integrated EXACTLY and anything else to that order.
        Returns a (nnodes + 1, 2) array indexed by node id, so row 0 is the
        1-based padding row and stays zero.

        B_SRC may return arrays (the documented contract) or two plain floats
        (a constant body force); both broadcast."""
        conn = np.array([c for (_, c) in self.elems], dtype=int)   # (ne, 8)
        P = self.xyz[conn]                                         # (ne, 8, 3)
        # the corner signs of hex8 in THIS file's connectivity order: the z=0
        # face counter-clockwise, then the z=ZTHICK face above it.
        sg = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                       (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], float)
        g, w5, w8 = np.sqrt(0.6), 5.0 / 9.0, 8.0 / 9.0
        gp, gw = (-g, 0.0, g), (w5, w8, w5)
        Fx = np.zeros(len(self.nodes) + 1)
        Fy = np.zeros(len(self.nodes) + 1)
        for xi, wi in zip(gp, gw):
            for et, we in zip(gp, gw):
                for ze, wz in zip(gp, gw):
                    N = 0.125 * ((1 + sg[:, 0] * xi) * (1 + sg[:, 1] * et)
                                 * (1 + sg[:, 2] * ze))             # (8,)
                    dN = 0.125 * np.column_stack([
                        sg[:, 0] * (1 + sg[:, 1] * et) * (1 + sg[:, 2] * ze),
                        sg[:, 1] * (1 + sg[:, 0] * xi) * (1 + sg[:, 2] * ze),
                        sg[:, 2] * (1 + sg[:, 0] * xi) * (1 + sg[:, 1] * et)])
                    jac = np.einsum("eai,aj->eij", P, dN)           # (ne, 3, 3)
                    dv = np.abs(np.linalg.det(jac)) * (wi * we * wz)
                    xg = np.einsum("a,eai->ei", N, P)               # (ne, 3)
                    bx, by = B_SRC(xg[:, 0], xg[:, 1])
                    np.add.at(Fx, conn,
                              N[None, :] * (np.asarray(bx, float) * dv)[:, None])
                    np.add.at(Fy, conn,
                              N[None, :] * (np.asarray(by, float) * dv)[:, None])
        return np.column_stack([Fx, Fy])


# -------------------------------------------------------------- deck text
def write_deck(mesh, u_if, f_if, f_bd):
    """u_if: node id -> (ux, uy) for the interface Dirichlet set (dirichlet
    side). f_if: node id -> (Fx, Fy) consistent nodal forces (neumann side).
    f_bd: the body force's consistent nodal forces as a (nnodes + 1, 2) array
    indexed by node id, or None when B_SRC is identically zero — in which case
    no map and no load are written at all and the deck is the one this file
    produced before B_SRC existed."""
    ux_o, uy_o = u_dirichlet(mesh.xyz[mesh.outer, 0], mesh.xyz[mesh.outer, 1])
    L = ['<?xml version="1.0" encoding="ISO-8859-1"?>',
         '<febio_spec version="4.0">',
         '  <Module type="solid"/>',
         '  <Control><analysis>STATIC</analysis><time_steps>1</time_steps>'
         '<step_size>1</step_size>'
         '<solver type="solid"><symmetric_stiffness>symmetric'
         '</symmetric_stiffness><dtol>1e-12</dtol><etol>1e-12</etol>'
         '<rtol>0</rtol>'
         f'<linear_solver type="{LINSOLVE}"/></solver></Control>',
         '  <Globals><Constants><T>0</T><R>0</R><Fc>0</Fc></Constants></Globals>',
         f'  <Material><material id="1" name="Mat" type="isotropic elastic">'
         f'<density>1</density><E>{_n(E_MOD)}</E><v>{_n(NU)}</v>'
         f'</material></Material>',
         '  <Mesh>',
         '    <Nodes name="Object1">']
    L += [f'      <node id="{n}">{_n(x)},{_n(y)},{_n(z)}</node>'
          for (n, x, y, z) in mesh.nodes]
    L += ['    </Nodes>',
          '    <Elements type="hex8" mat="1" name="Part1">']
    L += [f'      <elem id="{e}">{",".join(str(c) for c in conn)}</elem>'
          for (e, conn) in mesh.elems]
    L += ['    </Elements>',
          '    <NodeSet name="all_nodes">'
          + ",".join(str(n[0]) for n in mesh.nodes) + '</NodeSet>',
          '    <NodeSet name="outer">'
          + ",".join(str(n) for n in mesh.outer) + '</NodeSet>',
          '    <NodeSet name="iface_free">'
          + ",".join(str(n) for n in mesh.iface_free) + '</NodeSet>',
          '    <NodeSet name="iface_all">'
          + ",".join(str(n) for n in mesh.iface_all) + '</NodeSet>',
          '  </Mesh>',
          '  <MeshDomains><SolidDomain name="Part1" mat="Mat"/></MeshDomains>',
          '  <MeshData>']
    # PER-NODE maps. `lid` is the 1-based index INTO THE NODE SET as it was
    # declared above — NOT the node id.
    for name, col in (("ox_map", ux_o), ("oy_map", uy_o)):
        L.append(f'    <NodeData name="{name}" node_set="outer" '
                 f'data_type="scalar">')
        L += [f'      <node lid="{lid}">{_n(v)}</node>'
              for lid, v in enumerate(col, 1)]
        L.append('    </NodeData>')
    if SIDE == "dirichlet":
        for name, comp in (("ix_map", 0), ("iy_map", 1)):
            L.append(f'    <NodeData name="{name}" node_set="iface_free" '
                     f'data_type="scalar">')
            L += [f'      <node lid="{lid}">{_n(u_if[n][comp])}</node>'
                  for lid, n in enumerate(mesh.iface_free, 1)]
            L.append('    </NodeData>')
    else:
        L.append('    <NodeData name="f_map" node_set="iface_free" '
                 'data_type="vec3">')
        L += [f'      <node lid="{lid}">{_n(f_if[n][0])},{_n(f_if[n][1])},0'
              f'</node>' for lid, n in enumerate(mesh.iface_free, 1)]
        L.append('    </NodeData>')
    if f_bd is not None:
        # The body force, over ALL nodes — including the prescribed ones. At a
        # prescribed dof the load changes nothing (the constraint replaces the
        # equation), but leaving those nodes out would be a DIFFERENT load
        # vector, and this array is also what the reaction export subtracts.
        # `mesh.nodes` in this order IS the all_nodes node set declared above,
        # so lid and node id line up.
        L.append('    <NodeData name="b_map" node_set="all_nodes" '
                 'data_type="vec3">')
        L += [f'      <node lid="{lid}">{_n(f_bd[nd[0]][0])},'
              f'{_n(f_bd[nd[0]][1])},0</node>'
              for lid, nd in enumerate(mesh.nodes, 1)]
        L.append('    </NodeData>')
    L.append('  </MeshData>')

    L += ['  <Boundary>',
          # PLANE STRAIN: u_z = 0 on every node of the single element layer.
          '    <bc name="planar" type="zero displacement" node_set="all_nodes">'
          '<x_dof>0</x_dof><y_dof>0</y_dof><z_dof>1</z_dof></bc>',
          '    <bc name="ox" type="prescribed displacement" node_set="outer">'
          '<dof>x</dof><value lc="1" type="map">ox_map</value>'
          '<relative>0</relative></bc>',
          '    <bc name="oy" type="prescribed displacement" node_set="outer">'
          '<dof>y</dof><value lc="1" type="map">oy_map</value>'
          '<relative>0</relative></bc>']
    if SIDE == "dirichlet":
        L += ['    <bc name="ix" type="prescribed displacement" '
              'node_set="iface_free">'
              '<dof>x</dof><value lc="1" type="map">ix_map</value>'
              '<relative>0</relative></bc>',
              '    <bc name="iy" type="prescribed displacement" '
              'node_set="iface_free">'
              '<dof>y</dof><value lc="1" type="map">iy_map</value>'
              '<relative>0</relative></bc>']
    L.append('  </Boundary>')

    loads = []
    if SIDE == "neumann":
        # APPLY the partner's numbers UNCHANGED, as the consistent nodal force
        # vector of +int_Gamma q_out_partner . v ds (see the header).
        loads += ['    <nodal_load name="iface_f" type="nodal_force" '
                  'node_set="iface_free">',
                  '      <value lc="1" type="map">f_map</value>',
                  '    </nodal_load>']
    if f_bd is not None:
        # A SECOND load, not a merged one. FEBio accumulates every model load
        # into the same residual (FEMechModel::ExternalForces walks the list
        # and each FENodalLoad adds its own values), so the two ADD on the
        # interface nodes they share, and the entry above still carries the
        # partner's numbers verbatim — which is what makes the deck auditable
        # against the coupling contract.
        loads += ['    <nodal_load name="body_f" type="nodal_force" '
                  'node_set="all_nodes">',
                  '      <value lc="1" type="map">b_map</value>',
                  '    </nodal_load>']
    if loads:
        L += ['  <Loads>'] + loads + ['  </Loads>']

    L += ['  <LoadData><load_controller id="1" type="loadcurve">'
          '<interpolate>LINEAR</interpolate><extend>CONSTANT</extend>'
          '<points><pt>0,0</pt><pt>1,1</pt></points>'
          '</load_controller></LoadData>',
          '  <Output><logfile>',
          f'    <node_data data="ux;uy" delim="," file="{LOG_U}" '
          f'node_set="iface_all"/>']
    if SIDE == "dirichlet":
        L.append(f'    <node_data data="Rx;Ry" delim="," file="{LOG_R}" '
                 f'node_set="iface_all"/>')
    else:
        L.append(f'    <element_data data="sx;sxy" delim="," file="{LOG_E}"/>')
    L += ['  </logfile></Output>', '</febio_spec>']
    Path(DECK).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------ log parsing
def parse_log(path, ncol):
    """FEBio ASCII logfile: '*Step ...' / '*Data =' blocks, then 'id,v1,v2,...'.
    Returns {node id: (v1, ..., vncol)} for the LAST step in the file."""
    out = {}
    txt = Path(path).read_text()
    blocks = txt.split("*Step")
    body = blocks[-1] if len(blocks) > 1 else txt
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split(",")
        if len(parts) < ncol + 1:
            continue
        try:
            out[int(float(parts[0]))] = tuple(float(p) for p in parts[1:ncol + 1])
        except ValueError:
            continue
    return out


# -------------------------------------------------------------------- run
mesh = Mesh()
imp = read_imports()
y_if = mesh.y_if
if len(y_if) == 0:
    sys.exit(f"no interface nodes at x={IFACE_X}: this subdomain spans "
             f"[{X0},{X1}], so nothing is shared with the partner")

u_if, f_if = {}, {}
if SIDE == "dirichlet":
    u_line = sample(imp, "values", (UI_X, UI_Y), y_if)
    for j, (nb, nt) in enumerate(mesh.iface_pair):
        u_if[nb] = u_if[nt] = (float(u_line[j, 0]), float(u_line[j, 1]))
else:
    t_line = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
    t_node = {}
    for j, (nb, nt) in enumerate(mesh.iface_pair):
        t_node[nb] = t_node[nt] = (float(t_line[j, 0]), float(t_line[j, 1]))
    Fc = mesh.iface_mass_apply(t_node)
    for n in mesh.iface_all:
        f_if[n] = (float(Fc[n, 0]), float(Fc[n, 1]))

# The body force, integrated once for the whole subdomain. None when B_SRC is
# the shipped zero, so an unused feature costs the deck nothing.
f_bd = mesh.body_load()
if not np.any(f_bd):
    f_bd = None

write_deck(mesh, u_if, f_if, f_bd)
for f in (LOG_U, LOG_R, LOG_E):
    Path(f).unlink(missing_ok=True)

r = subprocess.run([FEBIO, "-i", DECK], capture_output=True, text=True,
                   timeout=3600)
if "N O R M A L   T E R M I N A T I O N" not in (r.stdout or ""):
    sys.stderr.write(f"FEBio did not terminate normally (rc={r.returncode})\n"
                     f"{(r.stdout or '')[-1500:]}\n")
    sys.exit(1)

ulog = parse_log(LOG_U, 2)
if not ulog:
    sys.stderr.write(f"empty FEBio node logfile {LOG_U}\n")
    sys.exit(2)
U = np.array([[0.5 * (ulog[nb][c] + ulog[nt][c]) for c in (0, 1)]
              for (nb, nt) in mesh.iface_pair], float)

if SIDE == "dirichlet":
    # THE CONSISTENT (REACTION) TRACTION — see the header. Rx/Ry are m_Fr at
    # PRESCRIBED dofs, which is exactly r = A u_h - b there.
    rlog = parse_log(LOG_R, 2)
    if not rlog:
        sys.stderr.write(f"empty FEBio reaction logfile {LOG_R}: this build "
                         f"did not produce the Rx/Ry node log data\n")
        sys.exit(2)
    w = mesh.iface_weights()
    R = np.array([[rlog[nb][c] + rlog[nt][c] for c in (0, 1)]
                  for (nb, nt) in mesh.iface_pair], float)
    W = np.array([w[nb] + w[nt] for (nb, nt) in mesh.iface_pair], float)
    # r = A u_h - b, and b INCLUDES THE BODY FORCE. FEBio's Rx/Ry give m_Fr,
    # which a <nodal_load> never reaches at a prescribed dof (see the header),
    # so what came out of the log is A u_h alone and the consistent body load
    # has to come off here. It is the same array that went into the deck — an
    # exact bookkeeping correction, not a model of one.
    if f_bd is not None:
        R -= np.array([[f_bd[nb][c] + f_bd[nt][c] for c in (0, 1)]
                       for (nb, nt) in mesh.iface_pair], float)
    Q = np.zeros_like(R)
    ok = np.abs(W) > 1e-14
    Q[ok] = -R[ok] / W[ok, None]

    # THE TWO INTERFACE CORNERS ARE ON THE OUTER DIRICHLET BOUNDARY (a y-face),
    # so their rows carry the OUTER reaction too and their residual is not this
    # interface's traction. Take the nearest interior interface node rather than
    # exporting a corner value that is physically a different quantity.
    good = np.array(mesh.interior_j, dtype=int)
    good = good[ok[good]]
    if len(good):
        for j in range(len(y_if)):
            if j not in good:
                Q[j] = Q[good[np.argmin(np.abs(good - j))]]
else:
    # NEUMANN SIDE: the reaction formula MUST NOT be used here. These interface
    # dofs are FREE, so FEBio reports Rx = Ry = 0 on them and the expression
    # would silently export ZERO traction with no error raised.
    #
    # It must not be an ECHO of the imported traction either, tempting as that
    # is (it would be the discretely exact consistent traction of this side —
    # a Neumann condition applied through the consistent nodal force vector is
    # satisfied EXACTLY in the variational sense, so -(A u - b_vol)_i / w_i is
    # algebraically the number that was handed in). Echoing makes the interface
    # BALANCE CHECK an identity: it would then report ~1e-16 on any coupling,
    # including a broken one, and stop being evidence.
    #
    # So this side recovers its OWN traction from its OWN stress field: FEBio's
    # element sx/sxy (the integration-point average of the interface-adjacent
    # element column) averaged onto the interface nodes, with the sign
    # convention applied. That is a stress recovered from the gradient of a
    # trilinear solution and evaluated on a boundary, so it is O(h) — FIRST
    # order, not second, exactly like the L2 stress projection the FEniCSx and
    # scikit-fem siblings use on THEIR Neumann side. It is not what the partner
    # consumes (the Dirichlet partner reads this side's `values`); it exists so
    # the balance check is an independent statement.
    elog = parse_log(LOG_E, 2)
    if not elog:
        sys.stderr.write(f"empty FEBio element logfile {LOG_E}\n")
        sys.exit(2)
    se = np.array([elog[e] for e in mesh.iface_elems], float)   # (NY, 2)
    Q = np.zeros((len(y_if), 2))
    cnt = np.zeros(len(y_if))
    for j in range(len(mesh.iface_elems)):
        Q[j] += -S * se[j]
        Q[j + 1] += -S * se[j]
        cnt[j] += 1.0
        cnt[j + 1] += 1.0
    Q /= cnt[:, None]

print(f"[febio {SIDE}] interface n={len(U)} "
      f"ux=[{U[:,0].min():.6g},{U[:,0].max():.6g}] "
      f"uy=[{U[:,1].min():.6g},{U[:,1].max():.6g}] "
      f"tx=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}] "
      f"ty=[{Q[:,1].min():.6g},{Q[:,1].max():.6g}]")

# exports.json LAST: the driver takes its existence as proof of success.
Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(y_if)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [[float(a_), float(b_)] for a_, b_ in U],
    "normal_fluxes": [[float(a_), float(b_)] for a_, b_ in Q],
}, indent=2))
