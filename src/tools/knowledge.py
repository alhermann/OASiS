"""
MCP tools for accessing physics knowledge and input generation.
"""

import json
from mcp.server.fastmcp import FastMCP
from core.registry import get_backend, available_backends


def discover_test_dirs() -> dict:
    """Return a {solver_key: Path} mapping of locally-present test/
    demo directories.

    The same lookup is needed by prepare_simulation (via
    _find_reference_test_files) and the `examples` MCP tool. Before
    2026-06-01 these were two separate hardcoded fourc+dealii-only
    dicts — meaning the examples tool returned 0 results for fenics
    / ngsolve / kratos / dune / febio even though demo trees existed
    locally. Centralising here keeps the two surfaces in sync.

    Probes (each gated on directory existence):
      fourc / 4c   -> $FOURC_ROOT/tests/input_files
      dealii       -> /usr/share/doc/libdeal.ii-doc/examples
      fenics(x)    -> any *fenics* conda env's
                      share/dolfinx/demo OR
                      etc/conda/test-files/fenics-dolfinx/0/python/demo
      ngsolve      -> .venv .../ngsolve/demos OR
                      conda *fenics* envs ../ngsolve/demos
      kratos       -> .venv .../KratosMultiphysics
    """
    import os
    from pathlib import Path

    test_dirs = {
        "fourc": Path(os.environ.get("FOURC_ROOT", "")) / "tests" / "input_files",
        "4c": Path(os.environ.get("FOURC_ROOT", "")) / "tests" / "input_files",
        "dealii": Path("/usr/share/doc/libdeal.ii-doc/examples"),
    }

    # FEniCS demos — probe several known conda-forge layouts.
    # Without this, prepare_simulation('fenics', 'poisson')
    # silently emits no reference test file because the
    # hardcoded share/dolfinx/demo path does not exist on
    # current ofa-fenicsx installs (conda-forge moved demos
    # under etc/conda/test-files/fenics-dolfinx/0/python/demo).
    candidates = [
        # Legacy path (older conda-forge layout)
        Path.home() / "miniconda3" / "envs" / "fenics"
        / "share" / "dolfinx" / "demo",
        # Current conda-forge ofa-fenicsx layout (probed 2026-06-01)
        Path.home() / "miniconda3" / "envs" / "ofa-fenicsx"
        / "etc" / "conda" / "test-files" / "fenics-dolfinx"
        / "0" / "python" / "demo",
    ]
    # Also probe any *fenics* conda env present locally
    conda_envs = Path.home() / "miniconda3" / "envs"
    if conda_envs.is_dir():
        for env in conda_envs.iterdir():
            if "fenics" in env.name.lower():
                candidates.extend([
                    env / "share" / "dolfinx" / "demo",
                    env / "etc" / "conda" / "test-files"
                    / "fenics-dolfinx" / "0" / "python" / "demo",
                ])
    for fenics_demo in candidates:
        if fenics_demo.is_dir():
            test_dirs["fenics"] = fenics_demo
            test_dirs["fenicsx"] = fenics_demo
            break

    # NGSolve ships demos inside the installed wheel:
    # site-packages/ngsolve/demos/{intro,howto,mpi,
    # TensorProduct,...}. Probe the active .venv plus
    # any conda env that includes ngsolve.
    ngsolve_candidates = [
        Path(__file__).resolve().parents[2] / ".venv" / "lib"
        / "python3.12" / "site-packages" / "ngsolve" / "demos",
    ]
    for envname in ("ofa-fenicsx",):  # other envs may ship ngsolve too
        ngsolve_candidates.append(
            Path.home() / "miniconda3" / "envs" / envname / "lib"
            / "python3.12" / "site-packages" / "ngsolve" / "demos")
    for ng_demo in ngsolve_candidates:
        if ng_demo.is_dir():
            test_dirs["ngsolve"] = ng_demo
            break

    # Kratos ships Python tests inside each Application:
    # site-packages/KratosMultiphysics/<App>/tests/*.py. The
    # tree is broad (many Applications) so we point at the
    # top KratosMultiphysics dir and let the rglob walk find
    # *.py matching the keyword.
    kratos_candidates = [
        Path(__file__).resolve().parents[2] / ".venv" / "lib"
        / "python3.12" / "site-packages" / "KratosMultiphysics",
    ]
    for kr_dir in kratos_candidates:
        if kr_dir.is_dir():
            test_dirs["kratos"] = kr_dir
            break

    return test_dirs


def resolve_search_keywords(solver: str, physics: str) -> list[str]:
    """Return a prioritised list of filename-substring keywords to
    probe when looking for upstream test/demo files for the given
    (solver, physics) pair.

    Audit 2026-06-01: this logic was previously baked into
    _find_reference_test_files, so the MCP `examples` tool (which
    has its own file walk) couldn't reach NGSolve demos via aliases
    — examples('hyperelasticity', solver='ngsolve') walked for the
    literal substring 'hyperelasticity' and missed nonlin.py.
    Factoring this out keeps the two LLM-facing surfaces in sync.
    """
    # Map physics to search keywords for ALL physics types
    search_terms = {
        "particle_pd": "pdbody",
        "particle_sph": "sph",
        "fsi": "fsi",
        "tsi": "tsi",
        "ssi": "ssi",
        "ssti": "ssti",
        "sti": "sti",
        "fluid": "fluid",
        "contact": "contact",
        "beams": "beam",
        "poisson": "scatra",
        "heat": "thermo",
        "linear_elasticity": "solid",
        "structural_dynamics": "genalpha",
        "ale": "ale",
        "electrochemistry": "elch",
        "level_set": "level_set",
        "low_mach": "loma",
        "lubrication": "lubrication",
        "cardiac_monodomain": "cardiac",
        "arterial_network": "art_",
        "ehl": "ehl",
        "fpsi": "fpsi",
        "fbi": "fbi",
        "pasi": "pasi",
        "beam_interaction": "beam_contact",
        "multiscale": "multi_scale",
        "reduced_airways": "red_airway",
        # deal.II step tutorials
        "stokes": "step-22",
        "helmholtz": "step-29",
        "eigenvalue": "step-36",
        "wave": "step-23",
        "hyperelasticity": "step-44",
        "nonlinear": "step-15",
        "convection_diffusion": "step-9",
        "hp_adaptive": "step-27",
        "dg_transport": "step-12",
        "parallel": "step-40",
        # FEniCS demos
        "navier_stokes": "navier",
        "mixed_poisson": "mixed",
        "biharmonic": "biharmonic",
        "reaction_diffusion": "reaction",
    }

    solver_key = solver.lower()

    # NGSolve demo filenames don't match the catalog or 4C /
    # deal.II keys. Demos: poisson.py / navierstokes.py /
    # elasticity.py / cmagnet.py (magnetostatics) / pml.py
    # (helmholtz / PML) / hhj.py (Hellan-Herrmann-Johnson
    # biharmonic) / hybrid_dg.py (DG methods) / nonlin.py
    # (nonlinear elasticity) / mixed.py (mixed_poisson) /
    # timeDG.py (time-dependent DG) / tdnns.py.
    ngsolve_aliases = {
        "navier_stokes": "navierstokes",
        "maxwell": "cmagnet",
        "magnetostatics": "cmagnet",
        "helmholtz": "pml",
        "biharmonic": "hhj",
        "hdivdiv": "hhj",
        "dg_methods": "hybrid_dg",
        "hyperelasticity": "nonlin",
        "nonlinear_elasticity": "nonlin",
        "mixed_poisson": "mixed",
        "time_dependent_heat": "timeDG",
        "time_dependent_ns": "timeDG",
    }

    keywords = [physics]
    if physics in search_terms:
        keywords.insert(0, search_terms[physics])
    if solver_key == "ngsolve" and physics in ngsolve_aliases:
        keywords.insert(0, ngsolve_aliases[physics])
    if "_" in physics:
        keywords.append(physics.replace("_", "-"))
    # Common substring trims so the FEniCS demo naming
    # convention (demo_elasticity.py, no "linear_" prefix)
    # is reachable from the catalog name (linear_elasticity).
    # Audit 2026-06-01.
    for prefix in ("linear_", "nonlinear_", "time_dependent_"):
        if physics.startswith(prefix):
            keywords.append(physics[len(prefix):])

    return keywords


def _find_reference_test_files(solver: str, physics: str) -> str:
    """Find real test files from a solver's test suite as reference.

    Returns a formatted block with file paths and content previews,
    or empty string when no local demos are available.
    """
    # Empty / whitespace-only physics would match every filename
    # via the substring-of-everything pattern that bit
    # prepare_simulation, examples('search'), and discover(
    # 'recommend'). Callers in this module already guard before
    # reaching here, but the helper is publicly importable
    # from src/tools/knowledge.py — guard it here too so a
    # future caller can't reintroduce the bug. (Audit
    # 2026-06-01.)
    if not physics or not physics.strip():
        return ""

    test_dirs = discover_test_dirs()
    solver_key = solver.lower()
    test_dir = test_dirs.get(solver_key)
    if not test_dir or not test_dir.is_dir():
        return ""

    ext = ("*.4C.yaml" if solver_key in ("fourc", "4c")
           else "*.cc" if solver_key == "dealii"
           else "*.py")

    keywords = resolve_search_keywords(solver, physics)
    # Drop any empty / whitespace-only keyword the resolver
    # produced (a defensive de-dup against the same class of
    # bug in resolve_search_keywords' alias map).
    keywords = [k for k in keywords if k and k.strip()]
    matches = []
    for kw in keywords:
        for f in sorted(test_dir.rglob(ext)):
            if kw.lower() in f.name.lower() and f not in matches:
                matches.append(f)
                if len(matches) >= 2:
                    break
        if len(matches) >= 2:
            break

    if not matches:
        return ""

    parts = ["## Reference: Real test files from the solver's own test suite\n"]
    for f in matches:
        rel = f.relative_to(test_dir)
        parts.append(f"### `{rel}`")
        try:
            content = f.read_text()[:2000]
            parts.append(f"```\n{content}\n```\n")
        except Exception:
            pass

    return "\n".join(parts)


# Behaviour that is true of EVERY backend, appended at the single point where
# every physics payload leaves the server. Filed here rather than per backend
# because I have now made the same filing error three times: a rule written
# into one backend's table is invisible to the other eight, and each time it
# took a lost run to notice. There is exactly one consumer of
# backend.get_knowledge(), so this is the one place a cross-backend rule
# cannot be missed.
# ONE COPY, SERVED ON BOTH PATHS THAT NEED IT.
#
# This first went into the generic capture recipe, which lives in the full
# block on the topic="physics" path — and coupled agents call topic="coupling".
# Measured right after writing it: the physics reply carried it, the coupling
# reply did not, which is the exact defect class this campaign keeps hitting.
_PER_SIDE_NAMING = """ON A COUPLED TASK THE NAME CARRIES THE SIDE: run_level<k>_<side>.log, one per
participant per level, each holding THAT participant's own solver output. Two
codes writing into one file cannot be told apart, and a submission whose logs
are named the single-code way is read as one code having produced everything.
The same applies to solution_level<k>_<side>.csv and interface_level<k>_<side>.csv:
copy the exact names out of the task's own output clause rather than the shape
you saw in an example."""


_UNIVERSAL = """
BEFORE YOU CONCLUDE A SOLVER IS BROKEN ON THIS MACHINE
──────────────────────────────────────────────────────
Two runs of this campaign gave up entirely -- zero output files, at a quarter of
their time budget -- after deciding the 4C binary did not work. It worked. The
same binary produced a complete three-level submission in the same minutes for
another agent. What they saw was this:

  * A WARNING BEFORE THE BANNER IS THE ENVIRONMENT, NOT THE SOLVER.
    `Invalid MIT-MAGIC-COOKIE-1 key` is an X11 display-authority warning. 4C
    prints it on EVERY run on this machine, including successful ones: run
    `4C -p` and you get the cookie line followed by the entire input grammar.
    It never explains a failure. Test any suspicious line this way -- run the
    binary's own `-p`/`--help`/`--version` and see whether the line appears
    there too. If it does, it is noise.

  * THE REAL DIAGNOSTIC COMES BEFORE THE ABORT BOILERPLATE. An MPI solver
    prints its own error and THEN ~40 lines of "MPI_ABORT was invoked on rank
    0 ... Open MPI will now kill all processes". So `2>&1 | tail` shows you the
    boilerplate and hides the cause. READ THE LOG FROM THE TOP:
        <run command> > run.log 2>&1 ; head -40 run.log
        grep -n "ERROR\\|Could not match\\|exception" run.log
    4C's real message looks like
        PROC 0 ERROR in 4C_io_input_spec_builders.cpp, line 633:
        Could not match this input
        STRUCTURAL DYNAMIC:
          NOT_A_REAL_KEY: 42
    and it names the offending key. That is a five-second fix; "the binary is
    broken" is a dead end.

  * A NON-ZERO EXIT WITH NO MESSAGE IS NOT EVIDENCE THE TOOL IS BROKEN. It is
    evidence you have not found the message yet. Before writing
    COULD_NOT_COMPLETE for an infrastructure reason, run the binary on its own
    trivial self-test (`-p`, `--help`) and report THAT result: if the self-test
    passes, the defect is in your input.

A SOLVER'S INPUT LANGUAGE IS NOT PYTHON
───────────────────────────────────────
The task states its source term and boundary data in Python notation, e.g.
`f = 36*x**3*y - 54*x**2*y**2 + ...`. Copying that verbatim into a solver's
expression field fails, and the two failures below were both measured here.

  * `**` IS NOT EXPONENTIATION in these expression parsers. USE `^`.
    - 4C `SYMBOLIC_FUNCTION_OF_SPACE_TIME`: `-12*x**3*y/5` gives
        PROC 0 ERROR ... 4C_utils_symbolic_expression.cpp
        Error while parsing: -12*x**3*y/5 + ...
      and the identical expression with `^` parses. Measured: converting `**`
      to `^` moved a failing deck past this error entirely.
    - FEBio `type="math"` loads: `-1*X**2` gives `Token expected (position 6)`;
      `-1*X^2` is accepted.
    Rewrite the whole expression, not the first term: one surviving `**` fails
    the parse just as completely.

  * A 4C LEGACY BLOCK THAT LISTS STRINGS IS A YAML SEQUENCE. Each entry needs
    a `- `:
        DLINE-NODE TOPOLOGY:
          - "NODE 1 DLINE 1"
          - "NODE 7 DLINE 1"
    Writing the strings bare, as
        DLINE-NODE TOPOLOGY:
        "NODE 1 DLINE 1"
    makes YAML read them as mapping keys and 4C dies during input parsing with
        ERROR: could not find ':' colon after key
        53:17: "NODE 7 DLINE 1"
    BEFORE PRINTING ITS OWN BANNER. Without line-buffered output that abort
    shows only `MPI_ABORT was invoked on rank 0` and nothing else, which reads
    as a broken binary and is not one. The same rule applies to NODE COORDS and
    every `*_ELEMENTS` block.

IF YOU DID SUBMIT AND WANT TO KNOW WHETHER IT IS RIGHT
──────────────────────────────────────────────────────────────────────
   GUESS. Among submissions with a complete level set the self-convergence
   order is a median 1.96 to 1.99 in both arms: the discretisation converges
   cleanly. Two failures survive that, and neither shows up in a refinement
   study — a solution converging beautifully TO THE WRONG FUNCTION (7% of all
   runs), and a field off by orders of magnitude in its overall size.

   YOUR OWN CONVERGENCE VERDICT DOES NOT SEPARATE THEM. Measured, a run's
   MESH_INDEPENDENCE = NOT_CONVERGED catches about three quarters of the wrong
   runs, but it also fires on HALF the correct ones, so on its own it says
   almost nothing. What is falsifiable, with no reference answer:
     a. DOES YOUR FIELD SATISFY THE EQUATION YOU WERE GIVEN? Pick a smooth
        function v that vanishes on the boundary, and check the identity
            integral of u * (L* v)  ==  integral of f * v
        by quadrature on the probe values you already have, with the L and f
        the task states (L* is the adjoint; for -div(K grad u) with symmetric
        K, L* = L). Refine and watch it. Measured: fields that solve the stated
        problem give 6.7e-2 -> 1.5e-2 -> 4.3e-3, falling at order 2; fields
        that do not give 5.49 -> 5.62 -> 5.64, flat. It costs no solver run and
        separated a third of all submissions.
     b. IS YOUR EVALUATOR ITSELF SECOND ORDER? Push a function you KNOW (say
        x(1-x)y(1-y)) through the SAME code that produces your probe values.
        Measured on a 44x44 grid at N = 8, 16, 32:
            nearest-node lookup    5.39e-3 -> 2.66e-3 -> 1.34e-3   order ~1.0
            linear/shape function  1.09e-3 -> 2.73e-4 -> 6.85e-5   order ~2.0
        Your evaluator's own order BOUNDS the order you can report.
     c. DO THE BOUNDARY VALUES COME BACK? Probe nearest a boundary where the
        value is prescribed. Values that are not right there mean the condition
        landed elsewhere, or your probe coordinates are in a different frame.
        A common form of this: a source expression evaluated in ELEMENT-LOCAL
        coordinates instead of global ones, which is silent and wrong
        everywhere.
     d. IS THE SIZE PLAUSIBLE? Compare the magnitude of your field against what
        the source term and the domain size imply. Measured, 11 to 30% of
        submissions are off by more than a factor of ten, including fields that
        are all zero and fields of order 1e11.
     e. DID THE MESH CHANGE? The degree-of-freedom count must GROW per level.

   Report what you measured either way: a run that states NOT_CONVERGED with
   its largest relative change scores better than one claiming a convergence
   it cannot show.

5. IF YOUR SOLVER IS A BINARY, ITS INPUT FILE IS THE RUN INTERFACE and you
   cannot guess it. For 4C, FEBio and SPARTA the full deck grammar — every
   required section, in order, with the silent failure modes — is served by
       knowledge(topic="physics", solver=<name>, physics=<name>)
   and by NO other topic. Ask for it before deciding a deck cannot be written.
   Two measured corrections, because runs have concluded the opposite and
   stopped:
     * 4C DOES ACCEPT PER-NODE DIRICHLET VALUES. `DESIGN LINE DIRICH
       CONDITIONS` takes one scalar per line, which is why a per-node list
       there aborts in "Read/generate conditions" — but `DESIGN POINT DIRICH
       CONDITIONS` with a `DNODE-NODE TOPOLOGY` block sets a different value at
       every named node, which is exactly what a Dirichlet-Neumann interface
       needs. A run that reports this as a 4C limitation is wrong.
     * FEBio DOES ACCEPT A POSITION-DEPENDENT BODY FORCE, as a `<body_load>`
       component carrying `type="math"`. Without that attribute the expression
       is silently truncated to its numeric prefix.

WHERE THE DELIVERABLE HAS TO END UP
──────────────────────────────────
OASiS's run tools write their results into a TIMESTAMPED directory of their own,
e.g. work/simulation_outputs/ngsolve_20260821_113326/. That is convenient for
you and invisible to the assessment, which looks for the files the task names.

  * COPY (or move) each deliverable to the TOP of your sandbox when you are
    done: work/solution_level<k>.csv, work/RESULT.txt, and so on. A file that
    exists only inside a tool's output directory is an intermediate artefact.
  * IF YOU RUN THE SAME LEVEL TWICE, the second run gets a NEW timestamped
    directory and the first one stays. Two differing copies of
    solution_level<k>.csv with no copy at the top of the sandbox is an
    AMBIGUOUS submission -- nothing in the task says which attempt is your
    answer -- and it is rejected as malformed rather than guessed at. Overwrite
    the copy at the top, or delete the superseded directory.

Measured on this campaign: 16 runs were rejected for exactly this, every one of
them in the tool-using arm and none in the arm that wrote straight to the
sandbox. It costs you the whole cell for a file-copy.

CAPTURING YOUR SOLVER'S OWN OUTPUT (asked for by every task's run-log clause)
────────────────────────────────────────────────────────────────────────────
The run log must carry the text YOUR SOLVER printed, not a line you wrote about
it, because that text is what shows WHICH code ran. Redirect the run:

    <your run command>  > run_level<k>.log 2>&1        # keep 2>&1

""" + _PER_SIDE_NAMING + """

Four codes print nothing useful at their defaults, so this needs one extra line.
Measured on this machine, per code:

  * FEniCSx / dolfinx  SILENT by default -- a redirected run gives a 0-byte
    file. Before creating the mesh:
        import dolfinx
        dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    The DOLFINX_LOGLEVEL environment variable is NOT honoured.
  * deal.II            needs BOTH, and depth_console alone prints nothing:
        deallog.depth_console(2);
        SolverControl ctl(max_it, tol, true /*log_history*/,
                          true /*log_result*/);
    deal.II's LIBRARY emits no mesh count at any verbosity -- the
    "Number of active cells" line in the tutorials is the tutorial's own print.
    Its solver iteration count is the number to log.
  * NGSolve            ngsolve.ngsglobals.msg_level = 3 (level 0 gives 0 lines,
    levels 1-2 give 1). Or ngsolve.solvers.CG(..., printrates=True), which is
    pure Python and works at the default level.
  * DUNE-fem           parameters={"linear.verbose": True} on galerkin(...).
    `fem.solver.verbose` does NOT work. Ignore the DUNE-INFO "Compiling Space"
    lines: they appear only on a cold JIT cache and vanish on a second run.
  * scikit-fem         logging.basicConfig(level=logging.INFO), or print(basis).
    Its log goes to STDERR, so `> log` without `2>&1` LOSES it.
  * Kratos, 4C, FEBio, SPARTA   nothing to set; all print at default settings.
    4C also writes <name>.control next to its results, carrying its own git sha
    and a num_dof field -- keep it, it is strong evidence the binary ran.

EVALUATING YOUR SOLUTION AT THE GRADED PROBE POINTS
───────────────────────────────────────────────────
Every cell is graded at a FIXED set of points that does not move with your
mesh, so the points sit INSIDE elements, not on nodes.

READING THE NEAREST NODE'S VALUE CAPS YOUR MEASURED ORDER AT 1, whatever your
solver did. It is the most common scoring defect in this campaign -- 144 runs
did it, and their observed orders cluster at 0 and 1. Nearest-node lookup is a
piecewise-CONSTANT reconstruction with O(h) error, which swamps the O(h^2) or
O(h^3) error of the solve, so you measure the reconstruction. Measured on an
exactly-known field with NO solver involved, over N = 8, 16, 32, 64:

    nearest node      err 1.99e-01 -> 2.41e-02   order 1.12, 1.00, 0.93
    shape functions   err 2.08e-02 -> 3.61e-04   order 1.80, 2.03, 2.01

Same trap: scipy.griddata(method="nearest"), pyvista interpolate(n_points=1),
any scattered-data interpolation of nodal values. And pyvista's
mesh.sample(cloud) is backwards -- use pv.PolyData(points).sample(mesh).

The call, per backend, verified on the installed versions to machine
precision against u = 3x + 2y:

    scikit-fem   values = basis.probes(P) @ sol        # P shaped (2, N)
    FEniCSx      tree  = geometry.bb_tree(domain, domain.topology.dim)
                 cand  = geometry.compute_collisions_points(tree, pts)
                 coll  = geometry.compute_colliding_cells(domain, cand, pts)
                 cells = [coll.links(i)[0] for i in range(len(pts))]
                 values = uh.eval(pts, cells).reshape(-1)   # pts: 3 columns
    NGSolve      values = [gfu(mesh(px, py)) for px, py in pts]
    VTK/VTU      values = pv.PolyData(pts).sample(mesh).point_data[name]

VERIFY THE SAMPLER AGAINST THE IDENTITY MAP FIRST. An MMS submission is TWO
independent programs -- the deck and the sampler -- and the solver's own
convergence checks cover only the first. Nothing in a blind run checks the
second, so check it yourself, with the one field whose answer you know without
any key: the COORDINATES.

    sample the field u(x,y) = x at your probe points
    -> it must return the probe points' x, to machine precision

If it does not, the sampler is broken and every physical number it reports is
broken. This costs one line and needs no reference solution, which is what a
blind evaluation demands. It catches the whole family: a wrong shape-function
normalisation, a wrong node ordering in the connectivity, wrong natural
coordinates, the wrong element selected, a z-layer mix-up.

MEASURED, on a real submission: an extractor wrote the QUAD4 factor 0.25 into
a HEX8 shape function instead of 0.125, so sum(N) = 2 everywhere. The same
doubled N was used inside the Newton inversion of the geometry, so the point
located was (x/2, y/2). The submitted field was exactly 2*u(x/2, y/2): a FIXED
wrong function, converged to beautifully, at order -0.035 -- refinement cannot
help because the map is mesh-independent. The deck was correct and its own
solution converged at order 1.98. Fixing that one character in the extractor,
with the solver output untouched, turns the run from unphysical into correct.

The identity check would have caught it instantly: sampling x would have
returned 2*(x/2) = x for the value but from the wrong element, and sampling
sum(N) returns 2.

SELF-CHECK, ONE MINUTE, BEFORE REPORTING AN ORDER. Push a known analytic field
through your OWN extraction path -- set it at the nodes, extract at the probe
points, fit the order. If that path does not converge at the rate you are
about to claim, the extraction is the defect, not the solver.

## BEFORE YOU REFINE ANYTHING: WRITE THE ANSWER FILE

Agents stop VOLUNTARILY at a median of about half their wall budget, and
roughly one in six is stopped by the clock mid-thought. Both leave the same wreckage: a solver that ran correctly, a
result understood, and nothing written where a grader can read it. One run
solved its first mesh level cleanly and ended without writing a single
deliverable — that scores exactly what doing nothing scores.

So invert the order of work:

  1. The moment your FIRST level or configuration produces numbers, write the
     complete deliverable to disk in its final requested format, with the
     levels you have and an honest marker for the ones you do not.
  2. Then compute the next level, and REWRITE the whole file.
  3. Repeat. Rewriting a small text file costs nothing next to a solver run.

The same rule applies when a run looks like it is going badly: write what you
have BEFORE you investigate why, because the investigation is what runs out of
clock. A partial result on disk is a partial result and is scored as one; a
finished result that exists only in your reasoning is not a result at all.

NOT VERIFIED and NOT A RESULT are different outcomes. If a check you ran
complains about an answer you computed, report the answer AND the complaint —
do not withhold the answer.

## IF THE TASK NAMES AN ELEMENT, THE ELEMENT WINS. OTHERWISE DEGREE SETS ORDER

Read this in that order, because the second rule has a carve-out that the
first one settles.

FIRST: if the task prescribes a discretisation — "use exactly this element" —
that is part of the problem, not a suggestion, and no convergence argument
overrides it. The degree-to-order rule of thumb fails for whole families of
elements: a nonconforming or mixed element can converge well below degree + 1,
and a prescribed element carries its own rate. Where a task states both an
element and an order, they are consistent with each other and both are data.
An agent that "corrects" the prescribed element is solving a different
problem.

SECOND, where the element is yours to choose and you are running a CONFORMING
Lagrange method on a second-order problem with a smooth solution and adequate
quadrature: the L2 error of the primal field converges at order p+1 for degree
p.

    order 2 wanted  ->  degree 1 (P1/Q1)
    order 3 wanted  ->  degree 2 (P2/Q2)
    order 4 wanted  ->  degree 3

Run degree 1 where order 3 is wanted and you get a clean, monotone study that
converges at 2 — the wrong answer to the wrong problem, and expensive
precisely because nothing looks broken.

The p+1 rate is NOT unconditional. It needs the elliptic-regularity /
duality argument behind it, so a re-entrant corner, a crack, or a jumping
coefficient can cap it below p+1 at any degree; nonconforming, mixed H(div),
and reduced-integration elements follow their own rates entirely.

Three things that cap the order even when the degree is right:
  * THE NORM. p+1 is the L2 norm of the FIELD. A gradient or flux converges
    one order lower, and a probe value can superconverge. Match what is asked.
  * QUADRATURE. A rule that was exact for your previous degree will not
    integrate degree-2 bases against a polynomial source; the load error then
    sets the rate.
  * TIME. A first-order integrator behind a spatial study caps the result
    whenever the steps are refined together — with dt proportional to h,
    backward Euler holds the whole study at 1.

If your own levels improve at p while you are about to claim p+1, check the
element first — but check the task's prescribed element before you change it.

## AN INGREDIENT YOU DEFINE IS INERT UNTIL IT IS WIRED IN

The most expensive failure in this kind of work is not a wrong method. It
is a right ingredient that never reached the solve: a source term derived
correctly and never referenced, a formulation built correctly and never
assembled, a tolerance chosen correctly and never applied. Nothing errors. The
solver runs, converges, and returns the answer to the problem you accidentally
posed — usually a field that is identically zero, or identically your boundary
value.

It is common to write a complete manufactured source into a deck, leave the
condition that references it switched off, and submit a field of exactly 0.0 at
every probe point — with no error raised anywhere.

So after you build the ingredient, CHECK THE WIRE. Each code has its own, and
being fluent in one is no help in another:

  FEniCSx      the term must appear in the linear form L and that form must be
               re-assembled: `L += f * v * ufl.dx`, then
               `assemble_vector(fem.form(L))`. A Function you interpolate and
               never place in L is inert.
  NGSolve      `f += source * v * dx` AND `f.Assemble()`. A CoefficientFunction
               that never enters the LinearForm does nothing.
  scikit-fem   the @LinearForm must be assembled AND its result used:
               `b = my_load.assemble(basis)`. Leaving `b = basis.zeros()` is a
               zero load, and it looks deliberate.
  DUNE-fem     the source must be in the UFL form the scheme receives
               (`b = ffun * v * dx`); filling a discrete function you never
               reference changes nothing.
  deal.II      it must be added to `cell_rhs` inside the assembly loop AND the
               cell vector distributed into `system_rhs`. Assembling into a
               local vector you never distribute is silent.
  Kratos       a process declared in the JSON runs only if it is IN the right
               list — `loads_process_list`, `constraints_process_list`. A
               declared-but-unlisted process is never executed.
  FEBio        a load applies only through an ACTIVE load controller: the
               `<nodal_load>`/`<body_load>` value needs `lc="<id>"` and that
               `<load_controller>` must exist in `<LoadData>` with points that
               are non-zero over your step.
  4C           `VAL` MULTIPLIES `FUNCT`. `FUNCT: [0]` means NO function and
               `VAL: [0.0]` scales any function to nothing. For a manufactured
               source you almost always want `VAL: [1.0], FUNCT: [1]`.
  SPARTA       a `compute` produces no output by itself. A `fix ave/time`
               (or dump/print) must reference it as `c_<id>` for any number to
               be written at all.

THE CHECK COSTS ONE COMMAND. Before you believe a result, grep your own input
for the ingredient's name and confirm something CONSUMES it. A driven problem
whose field is identically zero is this bug until you have proven otherwise.

## BEFORE YOU SUBMIT: RUN `audit_results` ON YOUR OWN OUTPUT

One tool call: `audit_results(work_dir=<your results directory>,
claimed_order=<the order you are about to claim>)`. It reads only files YOU
produced and catches, in seconds, the failures that most often sink an
otherwise complete submission — a field that is numerically zero because the
source was defined but never referenced by any condition; error levels sitting
at a solver-tolerance floor so refinement changes nothing; a convergence rate
your own numbers contradict. Calibrated against 94 independently-checked
correct submissions it raised no false alarm on any of them, and it catches
about four in ten submissions that are complete but wrong. It catches many
more when your summary states the convergence order you are claiming — the
order check has nothing to compare against otherwise.

A finding is not a verdict — it is a pointer at the exact place to look while
you still have budget to fix it. The single most common root cause it finds:
an ingredient you correctly BUILT (a source function, a mixed formulation, a
tightened tolerance) that the solve never actually USED. Check the wiring, not
the ingredient.
"""





# ── THE CORE THAT MUST REACH EVERY knowledge() CALL ────────────────────────────
# MEASURED, and this is why this constant exists. `_UNIVERSAL` was appended on
# exactly ONE of the 31 return paths of tools.consolidated.knowledge() — the
# topic="physics" path. Across the campaign's 995 measured knowledge calls from
# 193 OASiS-arm runs, topic="pitfalls" was 66.3% and topic="physics" only 11.5%,
# so 75.6% OF OASiS-ARM RUNS RECEIVED NONE OF IT. Every universal rule added
# during development — the deliverable's location, the input-language warning,
# the do-not-declare-the-solver-broken rule, the refinement ladder — reached at
# most a quarter of the runs it was written for.
#
# Why a COMPACT core and not the whole 20k block everywhere: it is appended to
# every call, and repetition consumes the same 262144-token window the history
# lives in. Why not "serve the full block once per session": the runner executes
# several run_one() calls in ONE process and nothing in the tool process
# identifies the current run, so a module-level flag would serve run 1 and
# starve runs 2..N — order-dependent, and asymmetric between arms and cells.
# A fixed core on every path is deterministic and arm-symmetric.
_UNIVERSAL_CORE = """

────────────────────────────────────────────────────────────────────────────────
FOUR RULES THAT APPLY WHATEVER YOU ASKED FOR
────────────────────────────────────────────────────────────────────────────────

1. THE DELIVERABLE GOES IN THE DIRECTORY YOU WERE GIVEN. Write your results
   file, your scripts and your solver output under the working directory named
   in your task, not in a temporary directory, not in the tool's installation
   tree, and not in $HOME. Work that cannot be found is scored as absent.

2. A SOLVER'S INPUT LANGUAGE IS NOT PYTHON. In decks and expression strings,
   powers are `^` and not `**` (`-1*X^2`, never `-1*X**2`), and `pi`, `sin`
   and `exp` may not exist. A wrong operator is often SILENT: the numeric
   prefix is taken and the rest discarded, so the run succeeds with the wrong
   load. Rewrite every term of a source you copied out of the task text.

3. DO NOT CONCLUDE A SOLVER IS BROKEN. Almost every "broken solver" in this
   campaign was a missing capture or an unread log. Redirect BOTH streams
   (`cmd > out.log 2>&1`), read the log rather than the exit code — several
   codes print their fatal error and still exit 0, and several print success
   letter-spaced so a grep for the contiguous word never matches — and re-run
   with the backend's verbose flag before reporting a failure.

4. THE MOST COMMON WAY A RUN FAILS IS THAT IT NEVER PRODUCES THE NUMBERS.
   Measured over 464 single-code runs: 36% wrote NO probe output at all, and
   the largest slice of that is a solve that SUCCEEDED and was then never read
   back at the required points (13%) — the solver ran, wrote its native output,
   and the values were never extracted. Next comes not managing to impose a
   spatially varying source in the input language (10%, almost all in the two
   deck-driven codes), then a toolchain that will not build or import (8%).
   Getting values out is not the last step to leave until the end; it is the
   step most likely to end the run. Do one coarse level end to end — solve,
   extract at the prescribed points, write the file — before refining anything.

5. IF YOU DID SUBMIT AND WANT TO KNOW WHETHER IT IS RIGHT, MEASURE, DO NOT
   GUESS. Among submissions with a complete level set the self-convergence
   order is a median 1.96 to 1.99 in both arms — the discretisation converges
   cleanly — and the failures that survive that do not show up in a refinement
   study at all: a solution converging beautifully TO THE WRONG FUNCTION, and a
   field off by orders of magnitude in size. Your own MESH_INDEPENDENCE verdict
   does not separate them: it catches three quarters of the wrong runs and also
   fires on HALF the correct ones.

   What is falsifiable, with no reference answer, is whether your field
   satisfies the equation you were given:

       verify_pde_consistency(solution_files=..., source_term=...,
                              coefficient=..., domain=...)

   Measured: fields that solve the stated problem give 2.32e-02 -> 5.27e-03 ->
   1.23e-03 -> 2.56e-04, falling at order ~2.2; fields that do not give 6.25 ->
   6.37 -> 6.40, flat. On a coupled task run it on EACH side separately, with
   that side's own source and coefficient. For the other cheap checks — is your
   evaluator itself second order, do the boundary values come back, is the size
   plausible, did the mesh change — ask
   knowledge(topic="physics", solver=..., physics=...).

   Report what you measured either way: a run that states NOT_CONVERGED with
   its largest relative change scores better than one claiming a convergence
   it cannot show.

6. IF YOUR SOLVER IS A BINARY, ITS INPUT FILE IS THE RUN INTERFACE and you
   cannot guess it. For 4C, FEBio and SPARTA the full deck grammar — every
   required section, in order, with the silent failure modes — is served by
       knowledge(topic="physics", solver=<name>, physics=<name>)
   and by NO other topic. Ask for it before deciding a deck cannot be written.
   Three measured corrections, because runs have concluded the opposite and
   stopped:
     * 4C DOES ACCEPT PER-NODE DIRICHLET VALUES. `DESIGN LINE DIRICH
       CONDITIONS` takes one scalar per line, which is why a per-node list
       there aborts — but `DESIGN POINT DIRICH CONDITIONS` with a `DNODE-NODE
       TOPOLOGY` block sets a different value at every named node, which is
       what a Dirichlet-Neumann interface needs.
     * 4C DESIGN ENTITY IDS ARE ONE-BASED, AND A 0 SEGFAULTS WITH NO MESSAGE.
       `E: 0` with `NODE n DLINE 0` dies with "Signal: Segmentation fault (11)"
       and no diagnostic, during "Read/generate conditions" so it reads as a
       problem with the condition's content. Measured on one deck: E:0 -> exit
       139, the SAME deck with E:1 -> "processor 0 finished normally", exit 0.
       Check the digit before rewriting section names or element types.
     * FEBio DOES ACCEPT A POSITION-DEPENDENT BODY FORCE, as a `<body_load>`
       component carrying `type="math"`. Without that attribute the expression
       is silently truncated to its numeric prefix.

Full detail, per backend: knowledge(topic="physics", solver=..., physics=...)
"""

def register_knowledge_tools(mcp: FastMCP):

    @mcp.tool()
    def get_physics_knowledge(solver: str, physics: str) -> str:
        """Get domain knowledge for a physics module from a specific solver backend.

        Returns materials, solver recommendations, pitfalls, and best practices.

        Args:
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'febio')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'heat')
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        knowledge = backend.get_knowledge(physics)
        if not knowledge:
            return f"No knowledge available for '{physics}' in {backend.display_name()}"

        result = json.dumps(knowledge, indent=2, default=str) + _UNIVERSAL

        # Automatically append real test file examples for ALL solvers
        ref = _find_reference_test_files(solver, physics)
        if ref:
            result += f"\n\n{ref}"

        return result

    @mcp.tool()
    def generate_input(solver: str, physics: str, variant: str = "2d",
                       params: str = "{}") -> str:
        """Generate a complete, runnable input for a solver backend.

        The generated input is solver-specific:
        - 4C: YAML input file (.4C.yaml)
        - FEniCS: Python script using dolfinx
        - deal.II: C++ source code
        - FEBio: XML input file (.feb)

        Args:
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'febio')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity')
            variant: Template variant (e.g. '2d', '3d', '2d_steady')
            params: JSON string of parameters to override defaults,
                    e.g. '{"kappa": 2.5, "nx": 64}'
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        import json as _json
        try:
            param_dict = _json.loads(params)
        except _json.JSONDecodeError as e:
            return f"Invalid params JSON: {e}"

        try:
            content = backend.generate_input(physics, variant, param_dict)
            format_name = backend.input_format().value
            result = f"```{format_name}\n{content}\n```"

            # Include real test file references so the agent can see
            # validated parameter values from the solver's own test suite
            ref_note = _find_reference_test_files(solver, physics)
            if ref_note:
                result += f"\n\n{ref_note}"

            return result
        except ValueError as e:
            return str(e)

    @mcp.tool()
    def validate_input(solver: str, content: str) -> str:
        """Validate solver-specific input content before running.

        Args:
            solver: Backend name
            content: The input content (YAML / Python / C++ / XML)
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        errors = backend.validate_input(content)
        if not errors:
            return "Input is valid."
        return "Validation errors:\n" + "\n".join(f"- {e}" for e in errors)

    @mcp.tool()
    def get_coupling_knowledge(solver: str = "", signal: str = "") -> str:
        """Complete knowledge for partitioned multi-code coupling via `couple`.

        With no solver: the participant contract, the InterfaceData shapes, how
        the driver iterates and relaxes, the interface-flux sign convention,
        which side each backend can take, and the failure modes.
        With a solver name: a COMPLETE runnable participant script for that
        backend plus the traps specific to it.
        With a signal: only the failure entries matching that symptom. Describe
        what you SAW, in your own words, with no mechanism in it — "it converged
        but the answer is wrong", "the residual stops falling and stays there".
        """
        from tools.coupling_knowledge import coupling_knowledge
        return coupling_knowledge(solver, signal)

    @mcp.tool()
    def get_tsi_knowledge() -> str:
        """Get complete knowledge for thermo-structural interaction (TSI) coupling.

        Returns 4C TSI patterns, material types, CLONING MAP, coupling algorithms,
        and cross-solver TSI workflow. Essential for thermal-structural simulations.
        """
        return '''\
# Thermo-Structural Interaction (TSI) Knowledge

## 4C Native TSI

4C has built-in thermo-structural coupling via `PROBLEMTYPE: "Thermo_Structure_Interaction"`.

### Required Components

1. **Element type:** `SOLIDSCATRA HEX8` (3D) — TSI needs the
   combined eletype, NOT plain `SOLID HEX8` (structure-only) or
   the legacy `WALL` 2D eletype.
   - SOLIDSCATRA combines structural + scalar transport capabilities
   - Must be 3D (no 2D TSI elements in 4C)

2. **Material:** `MAT_Struct_ThermoStVenantK`
   ```yaml
   MATERIALS:
     - MAT: 1
       MAT_Struct_ThermoStVenantK:
         YOUNGNUM: 1
         YOUNG: [200000]      # Young's modulus (Pa or MPa)
         NUE: 0.3             # Poisson's ratio
         DENS: 1.0            # Density
         THEXPANS: 1.2e-5     # Thermal expansion coefficient (1/K)
         INITTEMP: 0.0        # Reference temperature
         THERMOMAT: 2         # Links to thermal material ID
     - MAT: 2
       MAT_Fourier:
         CAPA: 1.0            # Heat capacity
         CONDUCT:
           constant: [1.0]    # Thermal conductivity
   ```

3. **Cloning material map** (required for multi-field coupling):
   ```yaml
   CLONING MATERIAL MAP:
     - SRC_FIELD: "structure"
       SRC_MAT: 1
       TAR_FIELD: "thermo"
       TAR_MAT: 2
   ```

4. **Three dynamics sections:**
   - `STRUCTURAL DYNAMIC`: structural solver parameters
   - `THERMAL DYNAMIC`: thermal solver parameters + INITIALFIELD
   - `TSI DYNAMIC`: coupling algorithm control

### TSI Coupling Algorithms

| Algorithm | COUPALGO value | Use case |
|-----------|---------------|----------|
| One-way | `tsi_oneway` | Thermal → structural (no feedback) |
| Iterative staggered | `tsi_iterstagg` | Two-way, sequential |
| Aitken staggered | `tsi_iterstaggaitken` | Two-way with Aitken acceleration |
| Monolithic | (use `TSI DYNAMIC/MONOLITHIC`) | Simultaneous, tight coupling |

### Thermal Boundary Conditions

- `DESIGN SURF THERMO DIRICH CONDITIONS`: prescribed temperature on surfaces
- `DESIGN SURF THERMO NEUMANN CONDITIONS`: prescribed heat flux on surfaces
- `DESIGN VOL THERMO DIRICH CONDITIONS`: prescribed temperature on volumes
- **Note:** use "THERMO" not "THERMAL" in the section name

### Initial Temperature Field

```yaml
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "100.0 * (1.0 - x)"
```

## TWO-WAY TSI ACROSS TWO CODES, through `couple`

This is a FIELD coupling, not a domain decomposition, and that changes
everything about how it is set up and checked. Both participants own the WHOLE
body. There is no interface, no outward normal, and no flux to balance — so
`couple`'s conservation checks report themselves as NOT RUN, and the only things
that can catch a wrong answer are the `monolithic=` comparison and the direction
controls below.

### The two equations, and which term is which direction

Mechanical (quasi-static), THE THERMAL -> MECHANICAL DIRECTION:

    div(sigma) = 0,  sigma = 2 mu eps(u) + lam tr(eps(u)) I - beta (T - T_ref) I

Energy (one implicit step), THE MECHANICAL -> THERMAL DIRECTION is the LAST term:

    rho_c (T - T_old)/dt - div(k grad T) + T_ref*beta*(tr eps(u) - tr eps(u_old))/dt = 0

with `beta = (3 lam + 2 mu) * alpha` the thermal stress modulus. Drop that last
term and you have a ONE-WAY coupling — a different and much weaker capability.
Most published "TSI couplings" are one-way and do not say so.

### What each participant exchanges

  thermal  imports the volumetric strain `e = tr(eps(u))`, exports the
           temperature CHANGE `theta = T - T_ref` at its own nodes;
  mech     imports `theta`, exports `e` at its own nodes.

EXPORT THE TEMPERATURE CHANGE, NOT THE ABSOLUTE TEMPERATURE. The driver's
convergence test is a RELATIVE norm, so a quantity carrying a large constant
offset makes that norm small for free — the same coupling exchanging T in
kelvin and in celsius reports residuals a factor of ~20 apart. Worse, the offset
makes the temperature block dominate the global norm, so the STRAIN block hides
behind it and the run stops while the strain is still moving. Measured on a
converged pair: exporting absolute T, the global residual read 6e-11 while the
strain block was still changing by 3e-09 per iteration.

EXPORT THE STRAIN, NOT THE DISPLACEMENT. The energy equation couples to
d/dt tr(eps), not to u. Exporting u makes the thermal side differentiate a field
it interpolated off a foreign mesh — the derivative of an interpolant, one order
of accuracy down. Use a QUADRATIC displacement space with a LINEAR temperature
space, so tr(eps(u)) lands in the same space the temperature lives in; with
linear displacement the strain is piecewise constant, one order below the
temperature, and the coupled answer settles on a different fixed point for that
reason alone.

### How strong the coupling is, and what theta to use

    delta = T_ref * beta^2 / (rho_c * (lam + 2 mu))

is the classical thermoelastic coupling parameter and it IS the size of the
reverse direction: in uniaxial strain the reverse coupling multiplies the
effective heat capacity by (1 + delta). For a real metal delta ~ 1e-2 — small,
but not zero, and four orders of magnitude above a coupling tolerance of 1e-12.

delta plays exactly the role rho plays for a Dirichlet-Neumann split, so the
same theta rule applies: the relaxed iteration's amplification is
sqrt((1-theta)^2 + delta*theta^2), minimised at `theta = 1/(1+delta)`. At
delta > 1 the UN-RELAXED iteration diverges (amplification sqrt(delta) > 1), so
relaxation is not optional there.

USE `accelerator="constant"` FOR A STRONGLY COUPLED TSI. This contradicts the
default and the measurement is the reason. On the same pair at delta ~ 1.25:

    theta=1.0 (un-relaxed)   did NOT converge in 300 iterations; the residual
                             oscillates rather than falling, which is what the
                             amplification formula predicts at 1.117
    theta=0.5                converged,  60 iterations
    theta=1/(1+delta)        converged,  70 iterations
    accelerator="aitken"     did NOT converge in 300 iterations, from either
                             starting theta

At delta ~ 0.012 Aitken converges in 14-15 iterations, so this is not a claim
that Aitken is broken — it is that a two-way TSI's composite map has eigenvalues
+-i*sqrt(delta), PURELY IMAGINARY, so the residual turns by about a right angle
each iteration instead of shrinking along a fixed direction. Aitken extrapolates
along that direction and there is not one. When delta is small the iteration
converges anyway because every admissible theta does; when delta exceeds one it
does not, and the constant relaxation the theta rule gives is what works.

### Convergence tolerance: watch the per-block check

`couple` checks each exchanged block separately against tol*10, because a
global relative norm is set by the largest-magnitude block. The strain block's
worst ENTRY-WISE relative change runs several times the global residual and has
its own roundoff floor. Measured on a converged pair: tol=1e-10 left the strain
block at 1.4e-09 against a limit of 1e-09 (a finding); tol=1e-12 was clean;
tol=1e-13 put the block on its own floor and produced a finding again. Choose
tol so the BLOCKS clear, not so the global norm looks small.

AND KEEP THE EXCHANGED FIELDS AWAY FROM ZERO. If the initial temperature equals
the reference temperature, both exchanged fields are ~0 over most of the body
after one step and the per-block check — an entry-wise relative change — reports
blocks "still changing" at 1e-06 on a run whose global residual is 8e-13. Those
entries carry no information. Offset the initial state instead of loosening the
check.

### Proving it is actually TWO-way — do not skip this

A "two-way" coupling whose reverse direction changes nothing is one-way with
extra steps, and nothing in the iteration can tell you which you have. Two
controls, and the second is the real one:

  1. SUPPRESS the reverse direction and check the answer MOVES. You do not need
     to edit a participant: give the thermal participant `imports_from: []` and
     it falls back to its initial strain, which makes the (e - e_old) source
     term identically zero. `couple` will report ONE-WAY in `validation` — that
     is the tool confirming the control did what you asked. Both runs use the
     SAME meshes, so discretisation error is common mode and the only floor is
     the coupling residual.
  2. Check it moves BY THE RIGHT AMOUNT. A participant exchanging the wrong
     quantity, sign or unit would also move when switched off, and would land
     somewhere else. Run the monolithic reference twice, with and without the
     reverse term, and compare the coupled two-way-minus-one-way difference
     against the monolithic one. That is a difference of differences, so it is
     insensitive to the discretisation error that limits a plain agreement
     check.

THE SIGN OF THE REVERSE TERM IS THE SILENT-WRONG. Compressing a body heats it
and expanding it cools it, so the term enters with a PLUS as written above.
Flipping it converges just as prettily onto a temperature field wrong by twice
the coupling effect, and no convergence, balance, finiteness or responsiveness
check can see it. Only a monolithic or native-TSI comparison can.

### Ready-made participants

`data/coupling_participants/participant_tsi_thermal_{skfem,fenics,ngsolve}.py`
and `participant_tsi_mech_{skfem,fenics,ngsolve}.py`. Each is self-contained,
has an EDIT THIS BLOCK of placeholders, and follows the standard participant
contract (reads imports.json, writes exports.json last). Any thermal script
pairs with any structural script.

## 4C-NATIVE TSI: FOUR THINGS THAT COST A RUN

Each of these was found by running, not by reading, and each produces an error
whose message points somewhere other than the cause.

1. **THE MESH MUST BE COARSER THAN 1e-3 IN ABSOLUTE UNITS.** 4C matches the
   structure and thermo discretisations with a geometric octree whose default
   tolerance is an ABSOLUTE 1e-3 (Coupling::Adapter::Coupling::match_nodes).
   Below that spacing distinct nodes collapse into one match and the run aborts
   with `Did not get 1:1 correspondence. masternodes.size()=324 (structure),
   coupling.size()=320 (thermo)` — a message about node COUNTS whose cause is
   geometric SCALE. It does not change when you fix the mesh, the conditions or
   the physics. Pose the problem at metre scale, or coarsen.

2. **TOLTEMP AND TOLDISP ARE ABSOLUTE INCREMENT NORMS.** On a temperature near
   293 K the increment norm cannot go below ~1e-12, so `TOLTEMP: 1e-14` is under
   its own roundoff floor. 4C's default `NORMCOMBI_RESFINC: Coupl_And_Single`
   requires the coupled residual, the coupled increment AND every field's own
   residual and increment at once, so one unreachable tolerance is enough: a
   Newton fully converged at iteration 3 grinds to ITEMAX and aborts with
   `Newton unconverged in 50 iterations`, which reads as a physics failure.

3. **A `Statics` STRUCTURE SILENTLY KILLS THE REVERSE DIRECTION.** 4C's coupling
   term is assembled from the structural VELOCITY. Run the structure as
   `DYNAMICTYPE: Statics` and there is no velocity, so the mechanical -> thermal
   direction vanishes with no message — the run succeeds and is one-way. Use
   `OneStepTheta` with `THETA: 1.0`, which makes v_{n+1} exactly
   (d_{n+1} - d_n)/dt, and set a tiny `DENS` if you want the mechanics
   quasi-static anyway.

4. **ONE LINEAR SOLVER IS ENOUGH for monolithic TSI**, contrary to what the
   upstream decks suggest. They configure a second Belos/Teko block
   preconditioner, but `TSI DYNAMIC/MONOLITHIC: MERGE_TSI_BLOCK_MATRIX: true`
   with a single `SOLVER 1: {SOLVER: UMFPACK}` solves the merged system
   directly. Verified on a 320-element TSI deck.

### 4C's own reverse term, and how to identify it without reading its source

4C assembles `- N^T . ctemp : (B_L . d') . N . T`, i.e. `+ beta * T * d/dt
tr(eps)` with the CURRENT temperature where the classical linear theory uses
T_ref. If you are comparing against a linear-theory implementation, keep the
temperature EXCURSION small relative to T_ref — everything is linear in the
excursion, so delta, the relaxation and the relative size of the reverse
direction are unchanged, while the model difference shrinks with it.

You can confirm 4C's reverse term BLACK BOX, with no source reading: in uniaxial
strain, 4C's two-way solve at `CAPA` must equal its `tsi_oneway` solve at
`CAPA*(1+delta)`. If it does, the term is the classical one with the same beta.

### One-way on purpose

`TSI DYNAMIC: COUPALGO: tsi_oneway` with
`TSI DYNAMIC/PARTITIONED: COUPVARIABLE: Temperature` gives thermal -> structural
with no feedback. `COUPVARIABLE: Displacement` (the DEFAULT) gives the other
direction, which is not what "one-way TSI" usually means.

### Pitfalls

1. **Must use SOLIDSCATRA elements** — standard SOLID or WALL elements cannot couple
2. **CLONING MAP is mandatory** — without it, 4C crashes at initialization
3. **THEXPANS units** — must be consistent with temperature units (1/K or 1/°C)
4. **INITTEMP** — the reference temperature for zero thermal strain
5. **3D only** — no 2D TSI elements available in 4C
'''

    @mcp.tool()
    def get_precice_knowledge(solver: str = "") -> str:
        """Complete knowledge for preCICE coupling via `couple_precice`.

        With no solver: when to use preCICE instead of `couple`, what you supply
        versus what OASiS generates, the HARD LIMITS of the generated config,
        the participant loop, and the launch traps.
        With a solver name: whether that backend CAN be a preCICE participant on
        this install, and its backend-specific traps.

        `solver` must be accepted here. This function is reached through
        `knowledge(topic='precice', solver=...)`, whose wrapper passes the
        argument positionally; a zero-argument signature made every such call
        return the string "⚠ `get_precice_knowledge()` raised: `TypeError: ...
        takes 0 positional arguments but 1 was given`" instead of any payload —
        the whole preCICE surface, core included, served as an error message.
        """
        from tools.coupling_knowledge import precice_knowledge
        return precice_knowledge(solver)

    @mcp.tool()
    def list_physics(solver: str = "") -> str:
        """List all physics problems solvable by available backends.

        Args:
            solver: Optional — filter by backend name. If empty, shows all.
        """
        if solver:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            backends = [backend]
        else:
            backends = available_backends()

        if not backends:
            return "No backends available."

        lines = []
        for b in backends:
            lines.append(f"## {b.display_name()}")
            for p in b.supported_physics():
                lines.append(f"- **{p.name}**: {p.description}")
                lines.append(f"  Dims: {p.spatial_dims}, Variants: {', '.join(p.template_variants)}")
            lines.append("")

        return "\n".join(lines)
