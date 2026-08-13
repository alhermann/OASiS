#!/usr/bin/env python
"""The BALANCED coupled set: twelve instances, eight codes, three pairs each.

Why this file exists
--------------------
The set built by ``build_coupled_v2.py`` is eight instances of which SEVEN have
FEniCSx on one side, and 4C, DUNE, FEBio and SPARTA appear in none of them. A
multi-code-coupling claim measured on that matrix is a claim about FEniCSx. The
replacement is balanced by count -- each of the eight FEM backends appears in
EXACTLY three pairs:

    C1  4C + FEniCSx          C2  4C + Kratos           C3  4C + DUNE
    C4  FEniCSx + deal.II     C5  FEniCSx + scikit-fem  C6  deal.II + NGSolve
    C7  deal.II + FEBio       C8  NGSolve + Kratos      C9  NGSolve + scikit-fem
    C10 Kratos + DUNE         C11 scikit-fem + FEBio    C12 DUNE + FEBio

This is an EXTENSION, not a fork. Every construction, every verification and the
whole probe/grader contract are imported from ``build_coupled_v2`` and used
unchanged, so a change to the family propagates to both sets and the task text
cannot drift away from ``grade_blind.py``. What is added here is (a) twelve new
instances, (b) a general geometry -- the interface may run along x OR along y,
and the instance may be 2-D or 3-D -- and (c) two things the old set left as
free variables and this one PRESCRIBES.

The two prescriptions
---------------------
**1. The Dirichlet/Neumann role of each subdomain is stated in the task text.**
It was previously the agent's choice, and it is not a free choice: the Neumann
side is measurably harder, and the shipped Kratos participant is Dirichlet-only,
so an instance that needs Kratos on the Neumann side cannot be served by it at
all (measured on D6: 60 iterations, residual stuck at 0.988). Leaving the role
free therefore correlates difficulty with an unrecorded decision. The 24 slots
here are split 12 Dirichlet / 12 Neumann, every backend carries both roles, and
the split is checked by ``verify_balance()`` rather than asserted.

**2. Every cell carries a two-material contrast.** Amendment 1 s2 and Amendment 2
s1 record what happens without one: when both sides carry the same field and the
same material the transmission check is a tautology and every reported quantity
agrees with a correct run even when the coupling is wrong. Every instance below
is verified NON-VACUOUS by ``check_scalar_transmission`` /
``check_vector_transmission``, which return VACUOUS rather than PASS in that case.

What is scalar and what is vector, and why
------------------------------------------
Vector (elasticity) participants exist for fenics, skfem, ngsolve, dealii and
febio. For 4C, DUNE and Kratos only scalar ones do. So C1, C2, C3, C5, C8 and
C10 are scalar and C4, C6, C7, C9, C11 and C12 are vector -- except that C12 is
DUNE + FEBio, and FEBio 4 has no heat module at all, so that cell is vector and a
DUNE VECTOR participant had to be written for it (see
``walk_participants.py``). The scalar six are deliberately not six copies of the
same problem: two carry a severe contrast in the direction that makes the
partitioned iteration hard, one is anisotropic with a tensor jump, one has a
reaction term on one side only (genuinely different operators either side), one
runs its interface along y instead of x, and one is 3-D.

Run with any interpreter that has sympy:
  python campaign3_blind/build_balanced.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(os.environ.get("OASIS_REPO",
                           "/home/alexander/Schreibtisch/ofa-balanced"))
sys.path.insert(0, str(REPO / "src"))

import build_coupled_v2 as V                                # noqa: E402
from blind_eval import coupled as C                         # noqa: E402
from blind_eval.leakgate import scan                        # noqa: E402

x, y, z = sp.symbols("x y z", real=True)

PROBE_M = V.PROBE_M
MESH_N = V.MESH_N
XI = V.XI                      # 5/8: on a mesh line at h = 1/8, 1/16 and 1/32
LX = V.LX                      # 3/2: the long extent, across the interface
GRADE = V.GRADE

# The label the agent reads, and the key `blind_eval.evidence` knows the code by.
# These MUST be the evidence keys: grade_blind.code_ran looks the code up in
# PER_CODE_SIGNATURES, and an id that is not a key there is unprovable by
# construction and grades FABRICATED_NO_RUN however honestly it ran.
LABEL = {
    "4C": "4C (write a YAML input file and run the 4C binary)",
    "fenics": "FEniCSx (dolfinx)",
    "dealii": "deal.II (build and run the C++ program)",
    "ngsolve": "NGSolve",
    "skfem": "scikit-fem",
    "kratos": "Kratos Multiphysics",
    "dune": "DUNE-fem",
    "febio": "FEBio (write a .feb deck and run the febio4 binary)",
}


# ──────────────────────────────────────────────────────────────────────
# The task text: the imported contract, plus the two things it was missing
# ──────────────────────────────────────────────────────────────────────
ROLE_TEXT = """COUPLING SCHEME: a partitioned Dirichlet-Neumann iteration, with \
subdomain A as the {role_a} side and subdomain B as the {role_b} side. The \
DIRICHLET side receives the interface FIELD from its partner, imposes it as an \
essential boundary condition on the interface, and returns its own outward \
normal {flux_word} there. The NEUMANN side receives that {flux_word} and applies \
it as a natural boundary condition on the interface, and returns its own \
interface field. These roles are prescribed: solve with them as stated rather \
than choosing your own."""

# The execution-evidence contract, which the task text was previously the
# missing half of.
#
# blind_eval.evidence requires a line carrying a NUMBER the solver computed, and
# it knew a different phrasing per code: NGSolve's `ndof` and scikit-fem's
# `n_dofs` matched, dolfinx's did not. So an honest, quiet dolfinx run graded
# FABRICATED_NO_RUN while the identical quantity printed by scikit-fem proved a
# run -- a bias that falls on whichever arm happens to use the quieter code. The
# gate now accepts one CANONICAL line for every code; this is the other half of
# that contract, and without it a correct quiet run is still unprovable.
EXEC_LOG_CONTRACT = """
(5) EXECUTION LOG. For every mesh level k and for EACH subdomain, write a file
    run_level<k>_<side>.log             (k = 1,2,3,...; side = A or B)
containing at least the line
    NDOF = <integer>
where <integer> is the number of degrees of freedom of YOUR discretisation of
that subdomain at that level. Write one such file PER PARTICIPANT: the file for
side A must be written by the code that solved subdomain A, and the file for
side B by the code that solved subdomain B.
"""


def build_task(spec: dict, f_text: str) -> str:
    """The imported task text, plus the role prescription and the NDOF log.

    Everything that the grader re-derives -- the probe grids, the exclusion
    rule, the file names, the level count -- comes from
    ``build_coupled_v2.build_task`` unchanged. Nothing here rewrites it; the two
    additions are appended as their own clauses. That is the whole reason this
    is an extension rather than a second builder: the drift this campaign has
    already suffered seven times comes from two texts describing one contract.
    """
    base = V.build_task(spec, f_text)
    flux_word = ("traction" if spec.get("components") else "flux")
    role = ROLE_TEXT.format(role_a=spec["roles"]["A"].upper(),
                            role_b=spec["roles"]["B"].upper(),
                            flux_word=flux_word)
    # Insert the role clause immediately after the interface conditions, where
    # the reader is already thinking about the interface, and the execution log
    # at the end of the output contract, before the integrity clause.
    marker = "DISCRETISATION:"
    if marker not in base:                       # pragma: no cover - guard only
        raise RuntimeError("build_coupled_v2.build_task changed shape; the "
                           "role clause has no anchor to attach to")
    base = base.replace(marker, role + "\n" + marker, 1)
    anchor = V.INTEGRITY
    if anchor not in base:                       # pragma: no cover - guard only
        raise RuntimeError("INTEGRITY clause not found in the built task")
    return base.replace(anchor, EXEC_LOG_CONTRACT + anchor, 1)


# ──────────────────────────────────────────────────────────────────────
# Geometry: the interface may run along x or along y, in 2-D or 3-D
# ──────────────────────────────────────────────────────────────────────
def _swap(e):
    """Reflect an expression across the diagonal x <-> y."""
    return sp.sympify(e).subs({x: y, y: x}, simultaneous=True)


class Geom:
    """A rectangular box split by one axis-aligned interface.

    ``axis`` 0 puts the interface on the plane x = XI and the long side along x;
    ``axis`` 1 puts it on y = XI with the long side along y. The second case is
    not cosmetic: every shipped participant samples its partner "by y along a
    straight interface at x = const", so a horizontal interface exercises a
    different outward normal, a different sampling coordinate and a different
    mesh-line constraint, at no cost in the construction (the field is the
    reflection of the vertical one, and the elasticity operator is isotropic so
    it is invariant under that reflection).
    """

    def __init__(self, dim: int, axis: int):
        self.dim, self.axis = dim, axis
        self.var = (x, y)[axis]
        self.name = ("x", "y")[axis]
        lengths = [1.0] * dim
        lengths[axis] = float(LX)
        self.extent_a = [(0.0, float(XI)) if i == axis else (0.0, 1.0)
                         for i in range(dim)]
        self.extent_b = [(float(XI), float(LX)) if i == axis else (0.0, 1.0)
                         for i in range(dim)]
        self.lengths = lengths

    # ── prose ────────────────────────────────────────────────────────
    @staticmethod
    def _box(ext):
        return " x ".join(f"({a:g}, {b:g})".replace("(0, ", "(0, ")
                          for a, b in ext)

    def domain_text(self):
        full = [(0.0, L) for L in self.lengths]
        kind = {2: "rectangle", 3: "box"}[self.dim]
        return f"the {kind} " + self._box(full)

    def subdomain_text(self, side):
        return self._box(self.extent_a if side == "A" else self.extent_b)

    def interface_text(self):
        what = "line" if self.dim == 2 else "plane"
        return (f"the {what} {self.name} = {XI}, where the two materials meet")

    # ── the graded interface probe set ───────────────────────────────
    def iface_probe(self):
        """The interface probe rule, with the end-exclusion band.

        Where the interface meets a constrained outer boundary the split problem
        has a Dirichlet-Neumann corner the monolithic problem does not, and the
        recovered flux there gets WORSE under refinement (measured 2.11x ->
        2.51x over a 4x refinement). Grading the ends would fail a correct
        submission for a property of the decomposition, so a quarter of the
        interface length is excluded at each end. This is
        ``build_coupled_v2._iface_band`` applied to whichever axes the interface
        actually spans -- the band is never omitted and never hand-written.
        """
        M = PROBE_M[self.dim]
        a, b = V._iface_band(0, 1)
        w = b - a
        c = ["x", "y", "z"]
        free = [i for i in range(self.dim) if i != self.axis]
        if self.dim == 2:
            i0 = free[0]
            pt = ["", ""]
            pt[self.axis] = f"{XI}"
            pt[i0] = f"{a} + (i+0.5)*{w}/{M}"
            return (f"the {M} points ({c[0]}, {c[1]}) = ({pt[0]}, {pt[1]}) for "
                    f"i = 0, 1, ..., {M - 1}. These cover the INTERIOR of the "
                    f"interface only: the two points where the interface meets "
                    f"the outer boundary are corners of the split problem and "
                    f"the recovered flux there does not converge, so they are "
                    f"not graded")
        i0, i1 = free
        pt = ["", "", ""]
        pt[self.axis] = f"{XI}"
        pt[i0] = f"{a} + (i+0.5)*{w}/{M}"
        pt[i1] = f"{a} + (j+0.5)*{w}/{M}"
        return (f"the {M * M} points (x, y, z) = ({pt[0]}, {pt[1]}, {pt[2]}) "
                f"for i, j = 0, 1, ..., {M - 1}, ordered with j varying "
                f"fastest. These cover the INTERIOR of the interface only: "
                f"where the interface meets the outer boundary the recovered "
                f"flux does not converge, so the edges are not graded")


# ──────────────────────────────────────────────────────────────────────
# Specs
# ──────────────────────────────────────────────────────────────────────
def _common(pid, codes, roles, geom, physics, contrast, notes=None):
    return dict(
        id=pid, kind="coupled", codes=list(codes), dim=geom.dim,
        coords=["x", "y", "z"][:geom.dim],
        code_a_label=LABEL[codes[0]], code_b_label=LABEL[codes[1]],
        physics=physics,
        roles={"A": roles[0], "B": roles[1]},
        domain=geom.domain_text(),
        interface=geom.interface_text(),
        interface_axis=geom.name, interface_value=str(XI),
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32",
        interface_tol="1e-6 relative",
        probe_iface=geom.iface_probe(),
        mesh_N=MESH_N, theoretical_order=2.0, tol=0.4, band=[0.8, 3.2],
        extent_a=geom.extent_a, extent_b=geom.extent_b,
        probe_M=PROBE_M[geom.dim],
        iface_graded_band=[float(v) for v in V._iface_band(0, 1)],
        iface_clearance_reason=(
            "the interface ends are Dirichlet-Neumann corners of the split "
            "problem; the recovered flux there does not converge under "
            "refinement, so grading them would fail a correct submission"),
        material_contrast=contrast,
        evidence_grade=1, evidence_grade_reason=GRADE[1],
        graded_against="hidden manufactured solution (true error) AND "
                       "reference-free mesh halving",
        notes_public=notes,
    )


def scalar_spec(pid, codes, roles, geom, kA_txt, kB_txt, eq, contrast,
                physics, notes=None, coefficients=None):
    s = _common(pid, codes, roles, geom, physics, contrast, notes)
    coords = s["coords"]
    s.update(
        subdomain_a=f"A = {geom.subdomain_text('A')}, {kA_txt}",
        subdomain_b=f"B = {geom.subdomain_text('B')}, {kB_txt}",
        equation=eq,
        coefficients=coefficients or f"{kA_txt} in subdomain A; "
                                     f"{kB_txt} in subdomain B",
        interface_condition="the field and its normal flux",
        element="Lagrange P1 elements in both codes",
        bc_text="u = 0 on the entire outer boundary",
        iface_header=", ".join(coords) + ", u, qn",
        qn_name="qn", qn_desc="conductive flux qn = -(K grad u) . n_out",
        qoi="convergence order of the coupled field, plus the two-sided "
            "interface flux jump",
    )
    return s


def vector_spec(pid, codes, roles, geom, lam, muA, muB, contrast, physics,
                notes=None):
    s = _common(pid, codes, roles, geom, physics, contrast, notes)
    EA, nuA = C.lame_from(lam, muA)
    EB, nuB = C.lame_from(lam, muB)
    s.update(
        components=["ux", "uy"],
        subdomain_a=f"A = {geom.subdomain_text('A')}, lambda = {lam}, "
                    f"mu = {muA}",
        subdomain_b=f"B = {geom.subdomain_text('B')}, lambda = {lam}, "
                    f"mu = {muB}",
        equation="-div(sigma(u)) = f, sigma(u) = 2*mu*sym(grad(u)) + "
                 "lambda*div(u)*I   (plane strain)",
        coefficients=f"subdomain A: lambda = {lam}, mu = {muA} "
                     f"(E = {EA}, nu = {nuA}); "
                     f"subdomain B: lambda = {lam}, mu = {muB} "
                     f"(E = {EB}, nu = {nuB}). The shear modulus jumps across "
                     f"the interface; lambda is the same on both sides.",
        interface_condition="the displacement vector and the traction sigma.n "
                            "(BOTH components)",
        element="vector Lagrange P1 elements in both codes",
        bc_text="u = 0 (both components) on the entire outer boundary",
        iface_header=", ".join(s["coords"]) + ", ux, uy, tx, ty",
        qn_name="(tx, ty)", qn_desc="traction t = sigma(u) . n_out",
        qoi="convergence order of the coupled displacement, plus the two-sided "
            "interface traction jump",
    )
    return s


# ──────────────────────────────────────────────────────────────────────
# The twelve instances
# ──────────────────────────────────────────────────────────────────────
#
# ROLES. 24 slots, 12 DIRICHLET and 12 NEUMANN, every backend carrying both.
# FEniCSx and deal.II take the harder Neumann role in 2 of their 3; 4C, DUNE and
# FEBio take it exactly once each. verify_balance() re-derives all of this from
# the built specs and refuses the set if it does not hold.
#
# CONDUCTANCE RATIO. For a Dirichlet-Neumann iteration the un-relaxed error
# amplification is rho = c_D / c_N with c = k / L per subdomain, so which side
# carries the high conductance decides whether the instance needs strong
# under-relaxation. It is chosen deliberately per cell rather than falling out
# of the material contrast: C1 (rho 4.2) and C8 (rho 11.2) require it, the
# severe-contrast C2 (rho 0.007) does not, and an agent that assumes one
# behaviour from the other is measured doing so.

def instance_C1(d):
    """4C + FEniCSx. Contrast in the direction that makes the iteration hard."""
    kA, kB = sp.Integer(3), sp.Integer(1)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.straight_scalar(d, kA * sp.eye(2), kB * sp.eye(2))
    s = scalar_spec("C1", ["4C", "fenics"], ("dirichlet", "neumann"), g,
                    "thermal conductivity k = 3", "thermal conductivity k = 1",
                    "-div(k grad u) = f in each subdomain", "3:1",
                    "two-material steady conduction, stiff Dirichlet side",
                    notes=("the Dirichlet subdomain is the more conductive one "
                           "here; the interface conductance ratio that follows "
                           "from the conductivities and subdomain widths above "
                           "exceeds one, and a relaxation factor that works at "
                           "unit contrast will not converge."))
    return s, dict(family="scalar", K=(kA * sp.eye(2), kB * sp.eye(2)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C2(d):
    """4C + Kratos. Severe contrast, in the easy direction: the material jump is
    extreme while the iteration is not, so the two difficulties are separated."""
    kA, kB = sp.Integer(1), sp.Integer(200)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.straight_scalar(d, kA * sp.eye(2), kB * sp.eye(2))
    s = scalar_spec("C2", ["4C", "kratos"], ("dirichlet", "neumann"), g,
                    "thermal conductivity k = 1",
                    "thermal conductivity k = 200",
                    "-div(k grad u) = f in each subdomain", "1:200",
                    "two-material conduction, severe material contrast")
    return s, dict(family="scalar", K=(kA * sp.eye(2), kB * sp.eye(2)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C3(d):
    """DUNE + 4C. DIFFERENT OPERATORS either side, and a HORIZONTAL interface.

    A carries a linear reaction term and B does not, so the two subdomains
    genuinely assemble different PDEs -- the reaction enters neither
    transmission condition, so the construction is untouched. The interface runs
    along y, which every shipped participant's "sample the partner by y at
    x = const" assumption gets wrong.
    """
    kA, kB = sp.Integer(1), sp.Integer(2)
    cA = sp.Integer(10)
    g = Geom(2, 1)
    ua, ub, fa, fb, _ = V.straight_scalar(d, kA * sp.eye(2), kB * sp.eye(2),
                                          reaction=(cA, 0))
    uA, uB = _swap(ua), _swap(ub)
    fA, fB = sp.expand(_swap(fa)), sp.expand(_swap(fb))
    s = scalar_spec("C3", ["dune", "4C"], ("dirichlet", "neumann"), g,
                    "diffusivity k = 1 with a linear reaction coefficient "
                    "c = 10", "diffusivity k = 2, no reaction",
                    "subdomain A: -div(k grad u) + c*u = f ; "
                    "subdomain B: -div(k grad u) = f", "1:2",
                    "reaction-diffusion coupled to diffusion, "
                    "horizontal interface",
                    coefficients=("subdomain A: k = 1 and c = 10 ; subdomain B: "
                                  "k = 2 and no reaction term. The two "
                                  "subdomains solve DIFFERENT equations."))
    return s, dict(family="scalar", K=(kA * sp.eye(2), kB * sp.eye(2)),
                   iface_var=y, reaction=(cA, sp.Integer(0))), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, (x, y)


def instance_C4(d):
    """FEniCSx + deal.II. Vector interface: displacement AND traction."""
    lam, muA, muB = sp.Integer(600), sp.Integer(400), sp.Integer(1600)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.vector_straight(d, lam, muA, muB)
    s = vector_spec("C4", ["fenics", "dealii"], ("dirichlet", "neumann"), g,
                    lam, muA, muB, "mu 1:4, lambda shared",
                    "two-material linear elasticity, shear-modulus jump",
                    notes=("the normal and shear interface stiffness ratios of "
                           "this material pair differ (both follow from the "
                           "Lame parameters and subdomain widths above). A "
                           "single scalar relaxation factor chosen for one "
                           "component can diverge in the other; a componentwise "
                           "or worst-component choice is safer."))
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C5(d):
    """scikit-fem + FEniCSx. ANISOTROPIC conductivity with a tensor jump.

    The off-diagonal entry coupling the normal to the tangential direction must
    be SHARED for the normal flux to be continuous under the flux potential --
    that is a real constraint on the family and is checked symbolically, not
    assumed. K11 and K22 are free to jump and both do.
    """
    KA = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]])
    KB = sp.Matrix([[4, sp.Rational(1, 2)], [sp.Rational(1, 2), 3]])
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.straight_scalar(d, KA, KB)
    s = scalar_spec("C5", ["skfem", "fenics"], ("dirichlet", "neumann"), g,
                    "anisotropic conductivity K = [[1, 1/2], [1/2, 2]]",
                    "anisotropic conductivity K = [[4, 1/2], [1/2, 3]]",
                    "-div(K grad u) = f in each subdomain", "K11 1:4, K22 2:3",
                    "two-material anisotropic diffusion, tensor jump")
    return s, dict(family="scalar", K=(KA, KB), iface_var=x), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C6(d):
    """deal.II + NGSolve. Vector, with the LARGE shear-modulus jump."""
    lam, muA, muB = sp.Integer(500), sp.Integer(200), sp.Integer(1600)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.vector_straight(d, lam, muA, muB)
    s = vector_spec("C6", ["dealii", "ngsolve"], ("dirichlet", "neumann"), g,
                    lam, muA, muB, "mu 1:8, lambda shared",
                    "two-material linear elasticity, large shear-modulus jump")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C7(d):
    """FEBio + deal.II. Vector, mild contrast: FEBio is the constraint here.

    FEBio 4 has no heat module, so a FEBio cell is necessarily mechanical, and
    FEBio solves a finite-deformation problem with a Newton iteration. The
    contrast is kept mild so the outer Newton loop is not the thing under test.
    """
    lam, muA, muB = sp.Integer(500), sp.Integer(250), sp.Integer(500)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.vector_straight(d, lam, muA, muB)
    s = vector_spec("C7", ["febio", "dealii"], ("dirichlet", "neumann"), g,
                    lam, muA, muB, "mu 1:2, lambda shared",
                    "two-material linear elasticity, mild shear-modulus jump")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C8(d):
    """Kratos + NGSolve. The hardest conductance ratio in the set (rho = 11.2).

    Kratos takes the Dirichlet role, which is the only one its shipped
    participant supports, and the contrast is placed so that the ITERATION is
    hard rather than the material jump: an agent that reaches for the default
    relaxation factor does not converge, and one that computes the ratio from
    the published conductivities and widths does.
    """
    kA, kB = sp.Integer(8), sp.Integer(1)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.straight_scalar(d, kA * sp.eye(2), kB * sp.eye(2))
    s = scalar_spec("C8", ["kratos", "ngsolve"], ("dirichlet", "neumann"), g,
                    "thermal conductivity k = 8", "thermal conductivity k = 1",
                    "-div(k grad u) = f in each subdomain", "8:1",
                    "two-material conduction, stiff Dirichlet side",
                    notes=("the Dirichlet subdomain is by far the more "
                           "conductive one; the interface conductance ratio "
                           "that follows from the conductivities and subdomain "
                           "widths above is well above one, and an un-relaxed "
                           "or lightly relaxed iteration diverges."))
    return s, dict(family="scalar", K=(kA * sp.eye(2), kB * sp.eye(2)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C9(d):
    """NGSolve + scikit-fem. Vector, with a HORIZONTAL interface.

    The elasticity operator is isotropic, so the reflection x <-> y maps a
    solution of the vertical-interface problem to a solution of this one with
    the displacement components exchanged. Nothing about the construction is
    weakened; what changes is that the interface normal is e_y, the traction
    components swap roles, and every "sample by y at x = const" assumption fails.
    """
    lam, muA, muB = sp.Integer(480), sp.Integer(240), sp.Integer(1200)
    g = Geom(2, 1)
    ua, ub, fa, fb, _ = V.vector_straight(d, lam, muA, muB)
    uA = (_swap(ua[1]), _swap(ua[0]))
    uB = (_swap(ub[1]), _swap(ub[0]))
    fA = [sp.expand(_swap(fa[1])), sp.expand(_swap(fa[0]))]
    fB = [sp.expand(_swap(fb[1])), sp.expand(_swap(fb[0]))]
    s = vector_spec("C9", ["ngsolve", "skfem"], ("dirichlet", "neumann"), g,
                    lam, muA, muB, "mu 1:5, lambda shared",
                    "two-material linear elasticity, horizontal interface")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=y), {"A": uA, "B": uB}, {"A": fA, "B": fB}, (x, y)


def instance_C10(d):
    """Kratos + DUNE. The 3-D cell.

    The interface is a PLANE, so the partner must be sampled on a
    two-dimensional set: the 1-D interpolation every shipped participant uses
    does not apply, and this is the only instance in the set where that is true.
    """
    kA, kB = sp.Integer(1), sp.Integer(4)
    g = Geom(3, 0)
    KA, KB = kA * sp.eye(3), kB * sp.eye(3)
    uA, uB, fA, fB, co = V.straight_scalar(d, KA, KB, dim=3)
    s = scalar_spec("C10", ["kratos", "dune"], ("dirichlet", "neumann"), g,
                    "thermal conductivity k = 1", "thermal conductivity k = 4",
                    "-div(k grad u) = f in each subdomain", "1:4",
                    "two-material conduction in 3-D, planar interface")
    return s, dict(family="scalar", K=(KA, KB), iface_var=x), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C11(d):
    """FEBio + scikit-fem. Vector."""
    lam, muA, muB = sp.Integer(450), sp.Integer(225), sp.Integer(900)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.vector_straight(d, lam, muA, muB)
    s = vector_spec("C11", ["febio", "skfem"], ("dirichlet", "neumann"), g,
                    lam, muA, muB, "mu 1:4, lambda shared",
                    "two-material linear elasticity, shear-modulus jump")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C12(d):
    """DUNE + FEBio. Vector, and the reason a DUNE vector participant exists.

    Neither code in this pair ships a participant that can serve it: FEBio has
    no heat module so the cell cannot be scalar, and DUNE ships only a scalar
    one. The DUNE elasticity participant written for this cell is in
    walk_participants.py.
    """
    lam, muA, muB = sp.Integer(400), sp.Integer(200), sp.Integer(500)
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.vector_straight(d, lam, muA, muB)
    s = vector_spec("C12", ["dune", "febio"], ("dirichlet", "neumann"), g,
                    lam, muA, muB, "mu 1:5/2, lambda shared",
                    "two-material linear elasticity, shear-modulus jump")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


BUILDERS = [instance_C1, instance_C2, instance_C3, instance_C4, instance_C5,
            instance_C6, instance_C7, instance_C8, instance_C9, instance_C10,
            instance_C11, instance_C12]


# ──────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────
def _boxes(spec, geom_axis):
    """Sampling boxes for the numeric residual, one per subdomain."""
    out = []
    for ext in (spec["extent_a"], spec["extent_b"]):
        box = []
        for lo, hi in ext:
            pad = 0.05 * (hi - lo)
            box.append((lo + pad, hi - pad))
        out.append(box)
    return out


def numeric_vector_jump(uA, uB, coords, matA, matB, iface_var, iface_val,
                        n=60, h=1e-4):
    """Displacement and traction jumps by finite differences, either axis.

    ``build_coupled_v2.numeric_vector_jump`` hard-codes the interface at
    ``x = iv`` and varies ``y``; the horizontal-interface instance needs the
    transpose, and a check that silently evaluates on the wrong line would
    report a clean zero for a broken construction.
    """
    import numpy as np
    rng = np.random.default_rng(23)
    idx = list(coords).index(iface_var)
    sA = C.cauchy_stress(uA, coords, *matA)
    sB = C.cauchy_stress(uB, coords, *matB)
    fu = [sp.lambdify(coords, c, "math") for c in uA]
    gu = [sp.lambdify(coords, c, "math") for c in uB]
    fa = [sp.lambdify(coords, sA[i, idx], "math") for i in range(len(coords))]
    fb = [sp.lambdify(coords, sB[i, idx], "math") for i in range(len(coords))]
    du, su, dq, sq = [], [], [], []
    for _ in range(n):
        pt = [0.0] * len(coords)
        for m in range(len(coords)):
            pt[m] = float(iface_val) if m == idx else float(
                rng.uniform(0.05, 0.95))
        for a, b in zip(fu, gu):
            va, vb = a(*pt), b(*pt)
            du.append(abs(va - vb))
            su.append(abs(va))
        for a, b in zip(fa, fb):
            va, vb = a(*pt), b(*pt)
            dq.append(abs(va - vb))
            sq.append(abs(va))
    return (max(du) / max(max(su), 1e-12), max(dq) / max(max(sq), 1e-12))


def build_one(fn, seed=None):
    d = V.Draw(seed)
    spec, info, fields, sources, coords = fn(d)
    spec["draw_seed"] = d.seed
    checks, numeric = [], {}
    iface_var = info["iface_var"]
    boxes = _boxes(spec, spec["interface_axis"])

    if info["family"] == "vector":
        matA, matB = info["mat"]
        for side, mat in (("A", matA), ("B", matB)):
            u, f = fields[side], sources[side]
            s = C.cauchy_stress(u, coords, *mat)
            for i in range(len(coords)):
                r = sp.simplify(-sum(sp.diff(s[i, j], coords[j])
                                     for j in range(len(coords))) - f[i])
                checks.append(C.Check(f"strong_form_{side}{i}",
                                      "PASS" if r == 0 else "FAIL", sp.sstr(r)))
        rep = C.check_vector_transmission(fields["A"], fields["B"], coords,
                                          matA, matB, iface_var, XI)
        numeric["jump_u"], numeric["jump_traction"] = numeric_vector_jump(
            fields["A"], fields["B"], coords, matA, matB, iface_var, XI)
        exact = V._exact_strings(fields)
        src = {k: [sp.sstr(c) for c in v] for k, v in sources.items()}
    else:
        KA, KB = info["K"]
        react = info.get("reaction", (sp.Integer(0), sp.Integer(0)))
        for i, (side, K, c) in enumerate((("A", KA, react[0]),
                                          ("B", KB, react[1]))):
            lhs = C.poisson_source(fields[side], coords, K)
            if c:
                lhs = lhs + c * fields[side]
            r = sp.simplify(lhs - sources[side])
            checks.append(C.Check(f"strong_form_{side}",
                                  "PASS" if r == 0 else "FAIL", sp.sstr(r)))
            numeric[f"residual_{side}"] = V.numeric_residual(
                fields[side], sources[side], coords, K, box=boxes[i],
                reaction=c)
        rep = C.check_scalar_transmission(fields["A"], fields["B"], KA, KB,
                                          coords, iface_var, XI)
        numeric["jump_u"], numeric["jump_q"] = V.numeric_interface_jump(
            fields["A"], fields["B"], KA, KB, coords, iface_var, XI)
        exact = V._exact_strings(fields)
        src = {k: sp.sstr(v) for k, v in sources.items()}

    # The outer datum stated in the task is u = 0 on the WHOLE outer boundary.
    # If the construction does not actually vanish there, the task is wrong and
    # every submission is graded against a field that does not solve it.
    faces = []
    for i, cvar in enumerate(coords):
        lo, hi = (0.0, spec["extent_a"][i][1]) if False else (None, None)
    for side, ext in (("A", spec["extent_a"]), ("B", spec["extent_b"])):
        for i, cvar in enumerate(coords):
            lo, hi = ext[i]
            for val in (lo, hi):
                if abs(val - float(XI)) < 1e-12 and coords[i] is iface_var:
                    continue                      # the interface, not the outer
                faces.append((side, cvar, sp.nsimplify(val)))
    bad = []
    for side, cvar, val in faces:
        comps = fields[side] if isinstance(fields[side], (list, tuple)) \
            else [fields[side]]
        for comp in comps:
            if sp.simplify(sp.sympify(comp).subs(cvar, val)) != 0:
                bad.append(f"{side}:{cvar}={val}")
    checks.append(C.Check("outer_boundary_trace_zero",
                          "PASS" if not bad else "FAIL",
                          "; ".join(sorted(set(bad))) or "all outer traces "
                                                         "vanish"))

    spec["source_public"] = src          # public: it is printed in task.txt
    f_text = V._fmt_source(spec, sources)
    task = build_task(spec, f_text)
    gate = scan(task, {"exact_solution": exact, "source_term": src}, spec["id"])

    sym_ok = all(c.ok for c in checks)
    num_ok = all(v < 1e-6 for v in numeric.values())
    return dict(spec=spec, task=task, exact=exact, source=src, checks=checks,
                numeric=numeric, transmission=rep.summary(), gate=gate,
                ok=(sym_ok and rep.ok and num_ok and gate.clean),
                sym_ok=sym_ok, trans_ok=rep.ok, num_ok=num_ok)


# ──────────────────────────────────────────────────────────────────────
def verify_balance(specs) -> list:
    """Re-derive the balance claims from the built specs. Never assert them."""
    from collections import Counter
    pairs = Counter()
    roles = {}
    for s in specs:
        a, b = s["codes"]
        pairs[a] += 1
        pairs[b] += 1
        for side, code in (("A", a), ("B", b)):
            roles.setdefault(code, Counter())[s["roles"][side]] += 1
    bad = []
    for code, n in sorted(pairs.items()):
        if n != 3:
            bad.append(f"{code} appears in {n} pairs, not 3")
    if len(pairs) != 8:
        bad.append(f"{len(pairs)} backends appear, not 8: {sorted(pairs)}")
    tot = Counter()
    for code, rc in roles.items():
        tot += rc
        nd, nn = rc.get("dirichlet", 0), rc.get("neumann", 0)
        if {nd, nn} not in ({1, 2},):
            bad.append(f"{code} is {nd} Dirichlet / {nn} Neumann, not 2/1 or 1/2")
    if tot.get("dirichlet") != 12 or tot.get("neumann") != 12:
        bad.append(f"role split is {dict(tot)}, not 12/12")
    for code in ("fenics", "dealii"):
        if roles.get(code, {}).get("neumann", 0) != 2:
            bad.append(f"{code} must take the Neumann role twice")
    for code in ("4C", "dune", "febio"):
        if roles.get(code, {}).get("neumann", 0) > 1:
            bad.append(f"{code} must take the Neumann role at most once")
    # Every cell must carry an interface-end exclusion band and a contrast.
    for s in specs:
        if not s.get("iface_graded_band"):
            bad.append(f"{s['id']} has no interface-end exclusion band")
        if not s.get("material_contrast"):
            bad.append(f"{s['id']} declares no material contrast")
    return bad


def write_key(kdir: Path, s: dict, r: dict):
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "key.json").write_text(json.dumps({
        "id": s["id"], "kind": "coupled", "codes": s["codes"],
        "dim": s["dim"], "coords": s["coords"],
        "components": s.get("components", ["u"]),
        "exact_solution": r["exact"], "source_term": r["source"],
        "extent_a": s["extent_a"], "extent_b": s["extent_b"],
        "probe_M": s["probe_M"], "mesh_N": s["mesh_N"],
        "theoretical_order": s["theoretical_order"], "tol": s["tol"],
        "band": s["band"], "evidence_grade": s["evidence_grade"],
        # Without this the grader's magnitude check silently degrades to
        # rate-only, and a rate can be manufactured from the source term with no
        # solver involved. It is a property of the answer, so it belongs in the
        # sealed key and nowhere else.
        "exact_rms": V._exact_rms_on_probe_grid(r, s),
        "evidence_grade_reason": s["evidence_grade_reason"],
        "material_contrast": s["material_contrast"],
        "roles": s["roles"],
        "draw_seed": s["draw_seed"],
        "verification": {
            "symbolic": [f"{c.name}={c.verdict}" for c in r["checks"]],
            "transmission": r["transmission"],
            "numeric_relative": r["numeric"],
        },
        "qoi": s["qoi"], "graded_against": s["graded_against"],
    }, indent=2, default=str), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    built, failed = [], []
    for fn in BUILDERS:
        pid = fn.__name__.split("_")[-1]
        if args.only and pid not in args.only:
            continue
        r = build_one(fn, args.seed)
        s = r["spec"]
        worst = max(r["numeric"].values()) if r["numeric"] else 0.0
        print(f"[{s['id']}] {'+'.join(s['codes'])}  "
              f"A={s['roles']['A'][:4]} B={s['roles']['B'][:4]}  "
              f"{s['physics']}")
        print(f"      symbolic={'OK' if r['sym_ok'] else 'FAIL'}  "
              f"numeric_worst_rel={worst:.2e}  "
              f"transmission={r['transmission']}")
        print(f"      leak gate={'CLEAN' if r['gate'].clean else 'LEAK'} "
              f"{[f.rule for f in r['gate'].findings] or ''}")
        for f in r["gate"].findings:
            if f.severity in ("CRITICAL", "HIGH"):
                print(f"        [{f.severity}] {f.detail[:160]}")
        (built if r["ok"] else failed).append(r)

    print(f"\n{len(built)}/{len(built) + len(failed)} instances built, "
          f"verified and blind.")
    if failed:
        print("FAILED:", [r["spec"]["id"] for r in failed])
        return 1

    if not args.only:
        bad = verify_balance([r["spec"] for r in built])
        print("balance:", "OK (8 backends x 3 pairs, 12 Dirichlet / 12 Neumann)"
              if not bad else "BROKEN")
        for b in bad:
            print("   -", b)
        if bad:
            return 1

    if not args.apply:
        print("(dry run -- pass --apply to write tasks and keys)")
        return 0

    problems = HERE / "problems"
    keys = Path(os.environ.get(
        "OASIS_BLIND_KEYS",
        "/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys"))
    for r in built:
        s = r["spec"]
        pdir = problems / s["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "task.txt").write_text(r["task"], encoding="utf-8")
        public = {k: v for k, v in s.items() if k != "draw_seed"}
        (pdir / "spec_public.json").write_text(
            json.dumps(public, indent=2, default=str))
        write_key(keys / s["id"], s, r)
        print(f"wrote problems/{s['id']}/ and keys/{s['id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
