"""Did the named codes actually run, and did they actually couple?

The pre-existing gate (``grade_blind.code_ran``) is a case-insensitive
**substring** scan for the code's name over every ``.log/.out/.txt/.json`` under
the run directory, excluding four filenames.  The task statement names both
codes.  So an agent that writes any note mentioning "deal.II" satisfies the
deal.II evidence requirement without deal.II existing on the machine.

Combine that with a coupled instance whose two subdomains carry the same field
and the same material, and the rational strategy for either arm is: solve the
whole problem monolithically in one code, write a file mentioning the second,
submit.  The result converges at order two, passes every integrity gate, and the
cell has measured single-code PDE solving — which the single-code cells already
measure.  A coupling-uplift number computed from that would be believed and
empty.

What this module requires instead
---------------------------------
1. **Structured solver output, per code.**  Not the code's name: a line only
   that solver emits, carrying a NUMBER it computed.  ``dolfinx`` printing a DOF
   count, deal.II's "Number of active cells", NGSolve's ``ndof``, Kratos's
   solver banner with a time.  A name is prose; a number is a run.

2. **The two codes' evidence must come from different files.**  One monolithic
   solve writes one solver's output.  This does not stop a determined forger,
   and is not claimed to.

3. **A partitioned-iteration residual history, as a required artefact.**  A
   coupled run that cannot produce one did not couple.  It must have at least
   three iterations, be positive, decrease by at least a factor of ten from
   first to last, end at or below the interface tolerance the task prescribed,
   and agree with the iteration count the agent reports in ``RESULT.txt``.
   A monolithic solve has no such history at all.

Honesty about what this is
--------------------------
This raises the cost of faking from "write one text file" to "implement a
plausible partitioned iteration and log it".  It is not proof of execution and
is not described as such.  The verdicts are three-valued — ``PROVEN``,
``NOT_PROVEN``, ``CONTRADICTED`` — because a gate that can only say yes or no
forces an unproven run to be scored as either fabricated or fine, and it is
neither.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# Files that are the runner's or the agent's own prose, never solver output.
NOT_EVIDENCE = {"trajectory.txt", "trajectory_live.txt", "task.txt",
                "RESULT.txt", "prompt.txt", "notes.txt", "notes.md",
                "README.md", "ledger.json", "plan.md"}

# Structured signatures: each must capture a number the solver computed.
# Deliberately several alternatives per code — solvers differ across versions,
# and a gate that only knows one phrasing fails honest runs.
# THE CANONICAL LINE, accepted for EVERY code.
#
# The per-code signatures below reward whatever each solver happens to print,
# which measured as: a silent honest dolfinx run is labelled FABRICATED_NO_RUN
# (dolfinx prints nothing by default and no task asked the agent to log), while
# "Level 1: ndofs = 4225" PROVES an skfem run and CONDEMNS a fenics one — the
# same honest line, opposite verdicts, by code name. Since the OASiS templates
# emit canonical lines and a bare agent does not, the "fabrication rate" was
# partly measuring print-statement phrasing, biased TOWARD the tool arm.
#
# The repair is a contract, stated in every task text and honoured here: the
# agent must write run_level<k>.log containing a line
#
#     NDOF = <integer>
#
# and that line is accepted as the numeric run signature for ANY code. The
# per-code patterns remain as additional evidence, but no honest run that
# follows the task text can be labelled fabricated for its phrasing again.
# Verification that <integer> is PLAUSIBLE for the prescribed mesh level (it
# must grow with refinement) happens in code_evidence, not here.
CANONICAL_NDOF = re.compile(r"^\s*NDOF\s*=\s*(\d{2,})\s*$", re.M)

# MEASURED, NOT GUESSED. EVERY PATTERN BELOW WAS OBTAINED BY RUNNING THE CODE.
#
# The previous table was almost entirely fiction, and the measurement is brutal:
# of its 27 patterns, 24 NEVER match the installed version's real output, and all
# 3 that do also match a hand-written log a numpy script could produce. Zero were
# both real and non-fakeable. Consequences found in the graded tree:
#
#   * FEBio: `normal termination` cannot match, because FEBio letter-spaces it as
#     `N O R M A L   T E R M I N A T I O N`, and `total elapsed time` is not the
#     wording either (` Elapsed time : 0:00:00`). 93 run directories contain a
#     real FEBio log; the febio patterns matched NONE of them.
#     FB1_27b_MCP_seed5 was graded FABRICATED_NO_RUN with `Nr of equations : 196`
#     and `N O R M A L   T E R M I N A T I O N` in its own log.
#   * 4C: `number of (nodes|elements)` is never printed by 4C at any verbosity on
#     the inline-mesh path -- it is an ordinary English sentence an agent writes
#     about its own mesh, which is precisely the trap this table must avoid.
#   * deal.II: `number of active cells` / `number of degrees of freedom` are
#     TUTORIAL PROGRAM prints, not library output. The library emits no mesh
#     count at any verbosity.
#   * Kratos: the real banner is `Multi-Physics 10.3."0"-Release-...`; the word
#     "Kratos" appears only in the ASCII art, so `kratos multiphysics` never
#     matched the version line.
#
# So every "PROVEN" verdict in the campaign came from the code-agnostic
# `NDOF =` line, which is exactly why coupled attribution collapsed.
#
# SELECTION RULE for what follows: a pattern must (1) appear in output the code
# ITSELF emitted, verified by execution on this machine, and (2) not match a
# plausible sentence an agent writes about its own hand-rolled solver. Where a
# code prints an English-looking phrase, the pattern is tightened to the code's
# own framing -- fixed field order, its C++ symbol names, dotted leaders, tab
# indentation, an ANSI escape, a letter-spaced banner -- so narration cannot
# satisfy it. Verified against a deliberately adversarial fake log that mimics
# each code's dialect: tests/test_per_code_signatures_are_measured.py.
#
# VERBOSITY IS NOT FREE, and the task text must say so: dolfinx is byte-for-byte
# silent by default, deal.II needs depth_console(2) AND SolverControl log flags,
# NGSolve needs ngsglobals.msg_level = 3, DUNE needs linear.verbose, scikit-fem
# needs logging.basicConfig or print(basis). Kratos, 4C, FEBio and SPARTA need
# nothing. A gate stricter than the instruction fails honest quiet runs -- the
# original sin this module exists to undo.
PER_CODE_SIGNATURES = {
    # dolfinx spdlog framing: millisecond stamp + `[info]` + an internal phrase
    # with an enum ordinal (`Cell type: 0`) and a dofmap RxC shape.
    "fenics": [
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[info\] "
        r"Cell type: \d+ dofmap: (\d+)x\d+",
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[info\] "
        r"nodes\.size = (\d+)",
        r"\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\] \[info\] "
        r"xdofs\.size = (\d+)",
    ],
    # dealii::LogStream's `DEAL:` sigil plus the `::`-delimited prefix stack.
    # The wording is "Convergence step N value X", not "converged in N".
    "dealii": [
        r"^DEAL:\w+::Convergence step (\d+) value [-\d.eE+]+$",
        r"^DEAL:\w+::Check (\d+)\t[-\d.eE+]+$",
        r"^DEAL:\w+::Starting value [-\d.eE+]+$",
        r"^\| Section\s+\| no\. calls \|\s+wall time \| % of total \|$",
    ],
    # 4C: pipe-delimited fixed field order with its own abbreviations
    # (nlniter/wct), Teuchos TimeMonitor rows labelled with 4C C++ namespace
    # paths, the width-locked 54-column banner, and the exodus mesh summary.
    "4C": [
        r"^Finalised step (\d+) / \d+ \| time [-\d.eE+]+ \| dt [-\d.eE+]+ "
        r"\| nlniter (\d+) \| wct [-\d.eE+]+$",
        r"^(?:Core|Adapter|Rebalance|Solid|ALE|FSI|ScaTra)::[\w:]+[^\n]*?"
        r"\s+[\d.eE+-]+ \((\d+)\)\s*$",
        # The exodus mesh summary (`Mesh consists of N points and M cells`,
        # requires GEOMETRY SHOW_INFO: "summary") is DELIBERATELY ABSENT: it
        # was reported to me as measured, but no captured 4C output available
        # here contains it, and an unverified pattern does not ship. Add it
        # back with a committed sample from a real run that prescribes an
        # exodus mesh.
        r"^Trilinos Version: [0-9a-f]{9,} \(git SHA1\)$",
        r"^\*\s+([0-9a-f]{40})\s+\*$",
        # written BY the 4C binary into <name>.control: its own git sha and a
        # field block carrying num_dof. 307 such files exist across 110 runs.
        r"^\s*sha: \"([0-9a-f]{40})\"$",
        r"^\s*num_dof: (\d+)$",
    ],
    # NGSolve: C++ progress counter (`VOL` is its integration-domain token, and
    # the n/n ratio is a progress bar); the CG line carries a literal ANSI
    # erase-line escape no hand-written log contains. Lines start with \r, so
    # these are deliberately NOT ^-anchored.
    "ngsolve": [
        r"assemble VOL element (\d+)/(\d+)",
        r"CalcLocalH: (\d+) Points (\d+) Elements (\d+) Surface Elements",
        r"\x1b\[2KCG iteration (\d+), residual = [-\d.eE+]+",
    ],
    # scikit-fem: the logger name is skfem's INTERNAL module path, and the
    # __repr__ demands the versioned class names plus a trailing `Size: N B`.
    "skfem": [
        r"^INFO:skfem\.utils:Solving linear system, shape=\((\d+), \d+\)\.$",
        r"^INFO:skfem\.assembly\.basis\.\w+:Initializing "
        r"\w*Basis\(Mesh\w+, Element\w+\)$",
        r"^INFO:skfem\.assembly\.form\.\w+:Assembling '\w+'\.$",
        r"^<skfem \w*Basis\(Mesh\w+, Element\w+\) object>\n"
        r"\s+Number of elements: (\d+)\n\s+Number of DOFs: (\d+)\n"
        r"\s+Size: \d+ B$",
    ],
    # Kratos: the `Mesh 0 :` block must be a five-line fixed-order dump from
    # ModelPart::PrintData -- an isolated "Number of nodes = N" cannot satisfy
    # it. ModelPartIO uses its own bracket framing.
    "kratos": [
        r"^\s+Mesh 0 :\n\s+Number of Nodes\s+:\s+(\d+)\n"
        r"\s+Number of Properties\s+:\s+\d+\n"
        r"\s+Number of Elements\s+:\s+(\d+)\n"
        r"\s+Number of Conditions\s+:\s+\d+",
        r"^ModelPartIO:\s+\[Reading (?:Nodes|Elements|Conditions)\s+:\s+"
        r"(\d+) (?:nodes|elements|conditions) read\]",
        r"^\s+Multi-Physics \d+\.\d+\.\"\d+\"-\w+-[\d.]+-[0-9a-f]{6,}-",
    ],
    # DUNE: `Fem::` is the Dune::Fem namespace shorthand from
    # krylovinverseoperators.hh; spacing is exactly `it: N : residual X`.
    # The DUNE-INFO JIT lines are cold-cache only and deliberately excluded.
    "dune": [
        r"^Fem::(?:CG|GMRES|BiCGstab|MINRES) it: (\d+) : residual [-\d.eE+]+$",
        r"^Fem::(?:CG|GMRES|BiCGstab|MINRES) preconditioning=\S+$",
    ],
    # FEBio: leading TAB, the abbreviation `Nr of`, a dotted-leader run before
    # ` : `, and the letter-spaced termination banner that killed the old
    # pattern.
    "febio": [
        r"^\tNr of equations \.{5,} : (\d+)$",
        r"^\tTotal number of equilibrium iterations \.{5,} : (\d+)$",
        r"^\tNumber of time steps completed \.{5,} : (\d+)$",
        r"^ N O R M A L   T E R M I N A T I O N$",
        r"^      F I N I T E   E L E M E N T S   F O R   B I O M E C H A N I C S",
    ],
    # SPARTA is INSTALLED and runs here (/home/alexander/Schreibtisch/sparta/
    # src/spa_serial) -- the old comment claiming it was absent is stale. The
    # full `Loop time` line has four fields in fixed order; `child grid cells`
    # is SPARTA's hierarchical-grid term; the banner is a DATE in parentheses.
    "sparta": [
        r"^Loop time of [\d.eE+-]+ on (\d+) procs for (\d+) steps with "
        r"(\d+) particles$",
        r"^Created (\d+) child grid cells$",
        r"^SPARTA \(\d{1,2} \w{3} \d{4}\)$",
        r"^Cells:\s+(\d+) ave \d+ max \d+ min\nHistogram: (?:\d+ ){9}\d+$",
    ],
}
PER_CODE_SIGNATURES["fenicsx"] = PER_CODE_SIGNATURES["fenics"]
PER_CODE_SIGNATURES["fourc"] = PER_CODE_SIGNATURES["4C"]
PER_CODE_SIGNATURES["dolfinx"] = PER_CODE_SIGNATURES["fenics"]

# THE CANONICAL LINE, ACCEPTED FOR EVERY CODE.
#
# The per-code table above knows a different phrasing per solver, and that is a
# bias, not a feature: NGSolve prints `ndof` and scikit-fem prints `n_dofs`, so
# both matched, while an honest dolfinx run that printed the same quantity in its
# own words did not, and graded FABRICATED_NO_RUN. Which arm that falls on
# depends only on which code the arm happened to use.
#
# So every task text now REQUIRES one canonical, number-bearing line per
# participant per level -- `NDOF = <integer>` in run_level<k>_<side>.log -- and
# the gate accepts it for every code. Two halves of one contract: the gate
# without the instruction fails quiet honest runs, and the instruction without
# the gate is ignored.
#
# WHAT IT COSTS. A canonical line does not say WHICH code wrote it, so on its own
# it no longer distinguishes two runs from one. That weight moves to the two
# checks that were already carrying it: `assess` reports when both codes' only
# evidence is the same file, and the partitioned-iteration residual history --
# which a monolithic solve cannot produce at all -- is required separately.
CANONICAL_SIGNATURES = [r"ndof\s*[:=]\s*(\d+)"]

# `.control` IS 4C's OWN OUTPUT, AND IT WAS NEVER OPENED.
#
# 4C writes <name>.control next to its results, containing its own git sha and
# version and a `- field:` block with num_nd / num_ele / num_dof. Measured: 307
# such files across 110 run directories in the graded tree, none of them read,
# because the suffix was not in this tuple. That is the strongest per-code
# evidence 4C produces -- a 40-hex git sha of the actual binary, next to a DOF
# count -- and 13 coupled runs were graded NO_PER_CODE_EXECUTION_EVIDENCE while
# holding one.
#
# The other native artefacts (.xplt, .vtu, .s0) stay out: they are binary, and
# opening them buys nothing a text log does not already give.
READABLE_SUFFIXES = (".log", ".out", ".txt", ".json", ".csv", ".err", ".dat",
                     ".control")
MAX_FILE_BYTES = 8_000_000


@dataclass
class EvidenceItem:
    code: str
    verdict: str              # PROVEN | NOT_PROVEN
    files: list = field(default_factory=list)
    matches: list = field(default_factory=list)
    detail: str = ""


@dataclass
class EvidenceReport:
    per_code: list = field(default_factory=list)
    coupling: dict = field(default_factory=dict)
    verdict: str = "NOT_PROVEN"
    notes: list = field(default_factory=list)
    shared_evidence_fatal: bool = False

    @property
    def proven(self) -> bool:
        return self.verdict == "PROVEN"


def _candidate_files(work: Path):
    for f in sorted(work.rglob("*")):
        if not f.is_file() or f.name in NOT_EVIDENCE:
            continue
        if f.suffix.lower() not in READABLE_SUFFIXES:
            continue
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                continue
            yield f, f.read_text(errors="ignore")
        except OSError:
            continue


def code_evidence(work: Path, code: str) -> EvidenceItem:
    """Structured, number-bearing solver output for one code."""
    pats = PER_CODE_SIGNATURES.get(code)
    if not pats:
        return EvidenceItem(code, "NOT_PROVEN",
                            detail=f"no structured signature is known for "
                                   f"{code!r}; a bare name is not evidence")
    # THE CODE-AGNOSTIC PATTERNS ARE KEPT SEPARATE, AND LABELLED AS SUCH.
    #
    # They used to be concatenated onto the per-code list, so a match on
    # CANONICAL_SIGNATURES (`ndof[:=]<n>`, which is case-insensitive and
    # therefore also matches the contract line `NDOF = 54` itself) was recorded
    # with no marker saying it was code-agnostic. `only_canonical` tests for the
    # marker, so such a match read as CODE-SPECIFIC evidence.
    #
    # That silently disarmed the whole attribution rule the moment the canonical
    # line stopped short-circuiting the per-code scan: every compliant file
    # matched this pattern too, `only_canonical` became False, and
    # C2_27b_MCP_seed15 -- the run that solved both subdomains with numpy and
    # scipy while the task named 4C and Kratos -- graded PROVEN again. Measured
    # directly: shared_evidence_fatal was False on it.
    #
    # Caught by an impact measurement that looked too good: 92 of 92 previously
    # unattributable runs became "attributable", with BOTH codes matched in
    # nearly every one, including pairs like ('4C','dune') whose DUNE side
    # needs a verbosity flag no run set.
    canon_pats = list(CANONICAL_SIGNATURES)
    files, matches = [], []
    for f, text in _candidate_files(work):
        # PER-CODE PATTERNS MATCH THE ORIGINAL TEXT, CASE-INSENSITIVELY.
        #
        # This lowercased the text and then searched case-sensitive patterns
        # against it — so any signature containing a capital letter (SPARTA's
        # 'Created N particles', 'Step', 'Loop time of'; scikit-fem's 'N =')
        # could NEVER match. Three of four SPARTA signatures were dead code.
        # Found by the grader rebuild's review; fixed at swap time as flagged.
        low = text
        # THE CONTRACT LINE FIRST. Every task text now instructs the agent to
        # write run_level<k>.log containing `NDOF = <integer>`. That line is
        # accepted for ANY code, so an honest run that follows the task cannot
        # be labelled fabricated because of its solver's print style — which
        # is what happened: a silent dolfinx run graded FABRICATED_NO_RUN
        # while the identical ndofs line proved skfem and condemned fenics.
        # The per-code patterns below remain as additional evidence.
        # ...AND THE PER-CODE PATTERNS IN THE SAME FILE. THIS `continue` WAS A
        # BUG THAT WOULD HAVE VOIDED THE WHOLE ATTRIBUTION REPAIR.
        #
        # The canonical check used to `continue`, so the per-code patterns were
        # never evaluated for any file that contained `NDOF = <n>`. That was
        # harmless while the canonical line was the ONLY thing the task asked
        # for. It stopped being harmless the moment the task began requiring
        # BOTH the solver's own captured output AND the canonical line in the
        # same run_level<k>.log: every fully COMPLIANT submission would record
        # only the canonical match, `only_canonical` would be True, and the
        # shared-evidence rule would mark it NO_PER_CODE_EXECUTION_EVIDENCE.
        # The repair would have failed on exactly the submissions it was built
        # to reward, and it would have looked like agents ignoring the contract.
        #
        # Caught by the grader's own fixture: once the fixture wrote per-side
        # code-specific lines, the flag still fired.
        #
        # So both are recorded. The canonical line still PROVES the code on its
        # own -- that is why it exists, and a quiet honest dolfinx run must not
        # be condemned for its print style -- but a code-specific match in the
        # same file is no longer discarded, which is what lets
        # `only_canonical` mean what it says.
        hit = False
        cm = CANONICAL_NDOF.search(text)
        if cm:
            matches.append(f"NDOF = {cm.group(1)} (canonical contract line)")
            hit = True
        # Code-agnostic fallbacks: they PROVE a run, and they are labelled so
        # that `only_canonical` does not mistake them for attribution.
        for p in canon_pats:
            m = re.search(p, low, re.IGNORECASE | re.MULTILINE)
            if m:
                matches.append(f"{p} -> {m.group(0)[:80]!r} "
                               f"(canonical contract line, code-agnostic)")
                hit = True
                break
        for p in pats:
            m = re.search(p, low, re.IGNORECASE | re.MULTILINE)
            if m:
                matches.append(f"{p} -> {m.group(0)[:80]!r}")
                hit = True
                break
        if hit:
            files.append(str(f.relative_to(work)))
    if files:
        return EvidenceItem(code, "PROVEN", files, matches,
                            f"{len(files)} file(s) carry structured "
                            f"{code} solver output")
    return EvidenceItem(code, "NOT_PROVEN", detail=(
        f"no file under work/ carries structured {code} solver output. "
        f"The code's NAME appearing in prose is not evidence: the task "
        f"statement names it, so the agent's notes contain it as a matter of "
        f"course."))


# ──────────────────────────────────────────────────────────────────────
RESIDUAL_FILE = re.compile(r"^(?:coupling_)?residual_level(\d+)\.csv$")


def read_residual_history(work: Path) -> dict:
    """Per-level partitioned-iteration residual histories."""
    out = {}
    for f in sorted(work.rglob("*.csv")):
        m = RESIDUAL_FILE.match(f.name)
        if not m:
            continue
        rows = []
        try:
            with open(f, newline="", errors="ignore") as fh:
                for r in csv.reader(fh):
                    if not r or not any(c.strip() for c in r):
                        continue
                    try:
                        rows.append((int(float(r[0])), float(r[1])))
                    except (ValueError, IndexError):
                        continue          # header
        except OSError:
            continue
        if rows:
            out[int(m.group(1))] = rows
    return out


# A HAND-WRITTEN HISTORY DECAYS AT A CONSTANT RATIO. A REAL ITERATION DOES NOT.
#
# The docstring below used to claim the residual history is "the artefact a
# monolithic solve cannot produce at all". That is false, and it was the whole
# hole: nobody runs a monolithic solve to make one, the agent types the numbers
# into a CSV. C7_27b_BARE_seed2 wrote 0.99999 * 0.5^k for 21 rows at all three
# levels, said so in its own IMPLEMENTATION_NOTES ("simulates the
# Dirichlet-Neumann iteration for demonstration"), and graded CORRECT — the
# campaign's only coupled success.
#
# What separates the two is not whether the sequence converges (a forged one
# converges beautifully) but how CONSTANT its ratio is. A real partitioned
# iteration's rate wanders: the error's modal composition changes and the
# linear solves carry noise. A closed-form sequence does not wander at all.
#
# Measured over all 223 residual histories in the run tree: 180 have a ratio
# coefficient-of-variation above 1e-2; the forged ones sit at 0 (exactly
# constant) up to 8.8e-6; the nearest honest history above the band is 2.2e-5.
# The threshold is placed in that empty valley. It flags both arms — of the 14
# runs it catches, 3 are OASiS and 11 are bare — so it is instrument repair,
# not a thumb on the scale.
SYNTHETIC_RATIO_CV = 1e-5
# LOWERED FROM 5 TO 3, MEASURED.
#
# The floor exists so a short honest history is not judged on two ratios. But at
# 5 it left 32 of the coupled fabrication candidates untestable, and the forgery
# it protected was not hypothetical: C7_27b_BARE_seed14 wrote
# 0.01, 0.005, 0.0025, 0.00125, 0.000625 -- five rows, four ratios, exactly
# 0.01*0.5^k -- bit-identical at all three mesh levels, and the only forgery
# detector never ran on it.
#
# Measured over all 393 coupled level-histories in the graded tree: dropping the
# floor to 3 ratios makes 18 more testable and newly flags exactly THREE, all
# three levels of that one run, at cv exactly 0.0. Nothing else changes, and
# going further to 2 ratios flags nothing additional -- so 3 is where the signal
# is, with no measured cost. Four ratios identical to full float precision is
# not something a real iteration produces.
MIN_RATIOS_FOR_DECAY_TEST = 3


def _decay_ratio_cv(vals: list) -> float | None:
    """Coefficient of variation of the consecutive-residual ratios.

    None when the history is too short, or carries a non-positive value, to
    say anything — silence, not a pass.
    """
    if len(vals) < MIN_RATIOS_FOR_DECAY_TEST + 1:
        return None
    if any((not math.isfinite(v)) or v <= 0 for v in vals):
        return None
    ratios = [vals[i + 1] / vals[i] for i in range(len(vals) - 1)]
    if not all(math.isfinite(r) and r > 0 for r in ratios):
        return None
    mean = sum(ratios) / len(ratios)
    if mean <= 0:
        return None
    var = sum((r - mean) ** 2 for r in ratios) / len(ratios)
    return math.sqrt(var) / mean


def coupling_evidence(work: Path, iface_tol: float = 1e-6,
                      min_iterations: int = 3,
                      min_decrease: float = 10.0,
                      claimed_iterations: int | None = None,
                      mesh_changed: bool | None = None) -> dict:
    """A partitioned iteration leaves a residual history.  Require it.

    Require also that the history look like an iteration rather than like a
    formula — see SYNTHETIC_RATIO_CV. The file is written by the agent, so its
    mere existence proves nothing; only its shape does.
    """
    hist = read_residual_history(work)
    if not hist:
        return {"verdict": "NOT_PROVEN", "levels": 0,
                "forged": False, "forged_detail": "", "detail":
                "no partitioned-iteration residual history "
                "(residual_level<k>.csv). A coupled run that cannot produce "
                "one did not couple; a monolithic solve has none."}
    # NOT EVERY FAILED CHECK IS A FORGERY.
    #
    # `problems` decides whether the coupling evidence stands. It does NOT
    # decide whether the agent invented its numbers, and those two questions
    # were conflated: evidence2.py mapped any non-PROVEN coupling verdict to
    # FABRICATED_NO_RUN, so "the residual only fell 3x" and "this sequence is a
    # closed form" produced the same accusation. The campaign reports a per-arm
    # fabrication rate as a paper headline, so the conflation is a publication
    # defect, not a cosmetic one.
    #
    # Of every check in this function, exactly ONE is positive evidence of
    # invention: a constant decay ratio. It says the numbers are a formula
    # rather than a measurement. Every other failure here — too few iterations,
    # a residual above the prescribed tolerance, an insufficient decrease, a
    # constant residual, a mid-history NaN, no history file at all — is a real
    # numerical failure, and a real numerical failure is an HONEST outcome. A
    # diverging iteration is a wrong answer, not a lie.
    #
    # So forgery-grade problems are collected separately, and `forged` in the
    # returned dict is the only thing entitled to produce a fabrication label.
    problems, forged, per_level = [], [], {}
    # THE SAME NUMBERS ON THREE DIFFERENT MESHES ARE NOT A MEASUREMENT.
    #
    # A Dirichlet-Neumann residual history depends on the discretisation: the
    # initial interface residual is computed from the coarse solve, and the
    # per-iteration contraction depends on the mesh through the discrete
    # Steklov-Poincare operators. Two levels agreeing to a few digits happens.
    # Three levels agreeing to the LAST BIT does not: it is one sequence written
    # into three files.
    #
    # This closes the gap that the constant-ratio test alone leaves, and the gap
    # is wide. Measured on the graded tree, three runs are bit-identical across
    # all three levels and ALL of them pass the ratio test:
    #
    #   C7_27b_BARE_seed14  0.01, 0.005, 0.0025, 0.00125, 0.000625  (= 0.01*0.5^k)
    #                       5 rows, below the ratio test's row floor -> cv None
    #   C1_27b_BARE_seed4   1, 1/4, 1/9, 1/16, 1/25, 1/36 (exactly 1/k^2)
    #                       50 rows, cv 0.1598 -- a closed form, but not geometric
    #   C1_27b_BARE_seed2   0.1, 0.05, 0.02, 0.01, 0.005, 0.002, ...
    #                       a hand-typed 1-2-5 decade ladder, cv 0.1010, and its
    #                       coupling verdict was PROVEN -- it graded as a
    #                       genuine coupling
    #
    # So the ratio test is necessary and not sufficient: it detects one shape of
    # forgery (geometric) and misses 1/k, 1/k^2, hand-typed ladders and anything
    # too short to test. Cross-level identity detects all of them at once,
    # without needing to guess which closed form was used.
    # ONE HONEST EXPLANATION MUST BE EXCLUDED FIRST: THE SAME MESH TWICE.
    #
    # If the agent never actually refined -- solved one mesh and submitted it as
    # three levels -- then identical histories are exactly what an honest run
    # produces, and the defect is the mesh sequence, not the numbers. That is
    # already caught, by name, as MESH_SEQUENCE_NOT_PRESCRIBED.
    #
    # So this fires only when the NDOF sequence shows the mesh DID change, i.e.
    # the agent claims three different discretisations and reports the same
    # residuals to the last bit. Measured on the graded tree: of 18 runs with
    # bit-identical histories, 17 have growing NDOF (forgery stands) and exactly
    # one, C5_27b_MCP_seed36, has flat NDOF 55/55/55 and 25/25/25 -- for which
    # the identical history is fully explained and the label would have been
    # wrong. `mesh_changed=None` means the caller could not establish it, and
    # then no forgery is claimed: an unproven suspicion is not evidence.
    if len(hist) >= 2 and mesh_changed:
        seqs = {lvl: tuple(v for _, v in rows) for lvl, rows in hist.items()}
        distinct = set(seqs.values())
        if len(distinct) == 1 and len(next(iter(distinct))) >= 2:
            msg = (f"the residual history is BIT-IDENTICAL across all "
                   f"{len(hist)} mesh levels ({len(next(iter(distinct)))} rows "
                   f"each) while the NDOF sequence shows the mesh DID change. A "
                   f"partitioned iteration's residual depends on the "
                   f"discretisation, so different meshes cannot produce the "
                   f"same numbers to the last bit -- this is one sequence "
                   f"written into {len(hist)} files.")
            problems.append(msg)
            forged.append(msg)
    for lvl, rows in sorted(hist.items()):
        vals = [v for _, v in rows]
        info = {"iterations": len(vals), "first": vals[0], "last": vals[-1]}
        if len(vals) < min_iterations:
            problems.append(f"level {lvl}: only {len(vals)} iteration(s)")
        # A LEADING NaN IS OUR OWN BOOKKEEPING, NOT A FORGED HISTORY.
        #
        # OASiS's driver records history[0] = NaN because iteration 1 has no
        # previous iterate to difference against. An agent that copies the
        # history verbatim writes "1,nan" into this file, and a non-finite
        # residual then reads as a history that cannot have come from a real
        # iteration: FABRICATED_NO_RUN, the forgery verdict, for faithfully
        # copying a number OASiS handed it. Measured: 20 runs wrote a NaN
        # here, 13 were graded fabrications, 11 of those 13 in the OASiS arm.
        #
        # So a leading non-finite entry is DROPPED and noted, and the rest of
        # the history is judged on its merits. A NaN anywhere ELSE is still
        # fatal -- mid-history it means an iteration produced no number, which
        # no honest run does. The served text now tells agents not to write it
        # at all; this keeps the label honest for the runs that already did.
        if vals and not math.isfinite(vals[0]):
            info["dropped_leading_nonfinite"] = True
            per_level[lvl] = info
            vals = vals[1:]
            info["iterations"] = len(vals)
            if vals:
                info["first"], info["last"] = vals[0], vals[-1]
            notes_leading = (
                f"level {lvl}: leading non-finite residual dropped — this is "
                f"OASiS's own history[0], which is NaN by construction")
        else:
            notes_leading = ""
        if not vals:
            problems.append(f"level {lvl}: no finite residual entries")
        elif any((not math.isfinite(v)) or v <= 0 for v in vals):
            problems.append(f"level {lvl}: non-positive or non-finite residual")
        elif vals[0] / max(vals[-1], 1e-300) < min_decrease:
            problems.append(f"level {lvl}: residual fell only "
                            f"{vals[0] / max(vals[-1], 1e-300):.3g}x — an "
                            f"iteration that never moved did not couple")
        if vals and vals[-1] > iface_tol * 1.5:
            problems.append(f"level {lvl}: final residual {vals[-1]:.3g} "
                            f"exceeds the prescribed {iface_tol:g}")
        if len(set(vals)) == 1:
            problems.append(f"level {lvl}: residual is constant")
        cv = _decay_ratio_cv(vals)
        if cv is not None:
            info["ratio_cv"] = cv
            if cv < SYNTHETIC_RATIO_CV:
                msg = (
                    f"level {lvl}: residual decays at a constant ratio "
                    f"(coefficient of variation {cv:.2e} over "
                    f"{len(vals) - 1} steps) — that is a closed-form "
                    f"sequence written into the file, not a partitioned "
                    f"iteration, whose rate varies as the error's modal "
                    f"composition changes")
                problems.append(msg)
                forged.append(msg)
        per_level[lvl] = info
    # AN OFF-BY-ONE IN A REPORTED COUNT IS NOT A FORGED HISTORY.
    #
    # This appended to `problems`, so a mismatch between RESULT.txt's
    # COUPLING_ITERATIONS and the number of rows in the residual file flipped
    # the whole coupling verdict to CONTRADICTED, which evidence2.py then maps
    # to FABRICATED_NO_RUN — the forgery label — for a miscount.
    #
    # C2_27b_MCP_seed15 is exactly that. Its SOLE complaint was "claims 7 but
    # the finest-level history has 6 rows". Everything else is right: both
    # codes PROVEN, residual 6.753e-01 -> 6.474e-07 with ratio CV 1.427
    # (forgery threshold 1e-5), final residual inside the prescribed 1e-6,
    # INTERFACE_SATISFIED with a decaying non-bit-exact flux jump, complete
    # level set, exact probe grid, and its own INTERFACE_RESIDUAL matching the
    # last row of its history to every digit — it read its history correctly
    # and miscounted the rows. Verified by execution: the same call with
    # claimed_iterations=None returns PROVEN and nothing else fires.
    #
    # This is the argument already made at length in
    # campaign3_blind/grading/evidence2.py for the run-log contract: a
    # bookkeeping slip, in a branch reached only after the coupling evidence
    # is otherwise sound, must not be called invention. It was not applied
    # here, where the mismatch flips the coupling verdict itself.
    #
    # So it is now a NOTE. The discrepancy is still recorded and still visible
    # to a reader; it no longer converts an honest coupling into a fabrication.
    count_note = ""
    if claimed_iterations is not None and hist:
        finest = max(hist)
        got = len(hist[finest])
        if got != claimed_iterations:
            count_note = (
                f"RESULT.txt claims COUPLING_ITERATIONS={claimed_iterations} "
                f"but the finest-level history has {got} rows — recorded as a "
                f"discrepancy, not as evidence of invention")
    if count_note:
        per_level.setdefault("notes", []).append(count_note)
    return {"verdict": "PROVEN" if not problems else "CONTRADICTED",
            "forged": bool(forged),
            "forged_detail": "; ".join(forged),
            "iteration_count_note": count_note,
            "levels": len(hist), "per_level": per_level,
            "detail": "; ".join(problems) or
                      f"{len(hist)} level(s) with a well-formed, converging "
                      f"partitioned-iteration residual history"}


def assess(work: Path, codes: list, coupled: bool, iface_tol: float = 1e-6,
           claimed_iterations: int | None = None,
           mesh_changed: bool | None = None) -> EvidenceReport:
    """The whole evidence question for one run."""
    rep = EvidenceReport()
    for c in codes:
        rep.per_code.append(code_evidence(work, c))
    unproven = [e.code for e in rep.per_code if e.verdict != "PROVEN"]

    if coupled and len(codes) > 1:
        # THIS FIRED ONLY WHEN EACH CODE HAD EXACTLY ONE FILE.
        #
        # `all(len(v) == 1 ...)` meant the check could not reach the case it was
        # built for. C7_27b_BARE_seed2 gave each code the SAME FOUR files — one
        # per level per side, every one of them a single hand-written
        # `NDOF = <n>` line — and sailed through with an empty notes list, then
        # graded CORRECT. The report even certified those same four files as
        # "structured febio solver output" AND "structured dealii solver
        # output", which cannot both be true of one file.
        #
        # What matters is not how many files there are but whether the two sets
        # are the same set: if every file that proves code A also proves code B,
        # nothing here distinguishes two runs from one.
        seen = {e.code: set(e.files) for e in rep.per_code}
        shared = set.intersection(*seen.values()) if all(seen.values()) else set()
        identical = len({frozenset(v) for v in seen.values()}) == 1
        # AND ON A COUPLED CELL THAT IS FATAL, NOT A NOTE.
        #
        # This used to append a note saying "the weight falls entirely on the
        # partitioned-iteration history". That reasoning is wrong: a genuine
        # history proves an ITERATION happened, not that the two NAMED codes
        # ran. A hand-rolled numpy solver iterating against another hand-rolled
        # numpy solver produces a perfect history.
        #
        # C2_27b_MCP_seed15 is the proof. Its participants import only numpy
        # and scipy.sparse — its own docstrings say "direct FEM assembly" —
        # and no file in the run outside the trajectory contains a single 4C or
        # Kratos token. Its entire execution evidence is fourteen 10-byte files
        # whose whole content is `NDOF = <n>`, credited SIMULTANEOUSLY to both
        # named codes. Its numbers are genuinely second-order, which is exactly
        # why the note was not enough: everything except "did the prescribed
        # codes run" looked right, and that is the one thing a coupled cell
        # exists to test.
        #
        # I briefly graded that run CORRECT earlier today on the strength of
        # its history and interface behaviour. This is the correction.
        #
        # The canonical `NDOF =` line stays code-agnostic on purpose — it was
        # introduced because quiet honest runs were condemned for their
        # solver's print style — so the rule is not "reject the canonical
        # line". It is: on a coupled cell, the canonical line alone cannot
        # prove that TWO DIFFERENT codes ran, because one file cannot be two
        # codes' output. At least one code must show its own signature.
        if shared and (identical or all(len(v) == 1 for v in seen.values())):
            rep.notes.append(
                f"every file proving one code also proves the other "
                f"({sorted(shared)}): one file cannot be two codes' output, so "
                f"this does not distinguish two runs from one.")
            only_canonical = all(
                all("canonical contract line" in m for m in e.matches)
                for e in rep.per_code if e.matches)
            # NO ESCAPE HATCH. ONLY THE GRADED FILES COUNT.
            #
            # I tried twice to let a code-specific signature found ELSEWHERE in
            # the run rescue an all-canonical submission, so that
            # C8_27b_MCP_seed4 -- which has Kratos's real telemetry
            # ("ResidualBasedLinearStrategy: Setup Dofs Time: 0.00222747 [s]")
            # in its transcript -- would keep its CORRECT. Both attempts
            # rescued C2_27b_MCP_seed15 as well, the numpy-only run, because
            # its transcript narrates "Kratos Multiphysics" three times and
            # because the 4C pattern `number of nodes = <n>` is a sentence an
            # agent writes about its own hand-rolled mesh.
            #
            # That is precisely the substring-scan hole this module's docstring
            # opens by describing -- "an agent that writes any note mentioning
            # deal.II satisfies the deal.II evidence requirement" -- and I was
            # reintroducing it one heuristic at a time on a live grader.
            #
            # So the transcript stays out, as NOT_EVIDENCE always intended. The
            # cost is honest and worth stating: C8_27b_MCP_seed4's own graded
            # logs carry only the canonical line, so it is now NOT_PROVEN too,
            # even though an independent recompute confirmed it by reproducing
            # its NGSolve DOF sequence 60/212/795 from scratch with netgen.
            #
            # The right repair is upstream, not here: the task must require each
            # participant to write its solver's OWN output into the graded run
            # log, not just `NDOF = <n>`. Until then a coupled cell cannot
            # prove which code ran, and the honest verdict is NOT_PROVEN for
            # both -- unproven, not fabricated.
            if only_canonical:
                rep.shared_evidence_fatal = True
                rep.notes.append(
                    "NO CODE-SPECIFIC OUTPUT AT ALL: every match is the "
                    "code-agnostic `NDOF =` contract line, which any script "
                    "can echo. A coupled cell requires evidence that the two "
                    "PRESCRIBED codes ran; a converging residual history "
                    "proves an iteration happened, not who performed it.")
        rep.coupling = coupling_evidence(
            work, iface_tol=iface_tol, claimed_iterations=claimed_iterations,
            mesh_changed=mesh_changed)

    if unproven:
        rep.verdict = "NOT_PROVEN"
        rep.notes.append(f"no structured solver output for: {', '.join(unproven)}")
    elif coupled and rep.shared_evidence_fatal:
        rep.verdict = "NOT_PROVEN"
    elif coupled and rep.coupling.get("verdict") != "PROVEN":
        rep.verdict = "NOT_PROVEN"
        rep.notes.append(f"coupling evidence: {rep.coupling.get('detail')}")
    else:
        rep.verdict = "PROVEN"
    return rep
