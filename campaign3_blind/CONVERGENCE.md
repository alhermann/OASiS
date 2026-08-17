# Development phase: what "converged" means, and how the ladder runs

Written 2026-08-15, after development round 1 on the 27B. This file is the
rule the campaign is held to. It is deliberately numeric: "it looks better
now" is not a criterion, and neither is my opinion.

## Why a development phase exists at all

The evaluation numbers in the paper must measure ONE thing: what the model
can do alone versus with OASiS. Every other cause of failure is noise we are
obliged to remove first — a grader that rejects correct data, a harness that
kills a run, a task that cannot be solved, knowledge that omits the one fact
the intended path needs. Round 1 found four such defects. Until a full round
finds none, the campaign is measuring our own mistakes.

The development problems are BURNT by this process: we look at them, diagnose
against them, and fix knowledge because of them. They can never appear in the
paper's numbers. That is the whole reason the evaluation phase draws FRESH
problems after a freeze — the honest test of whether knowledge fixed on
development problems generalises to problems nobody tuned against. A reviewer
who asks "did you tune on your test set?" must get: no, we tuned on the
development set, it is burnt and listed, and here are the frozen fresh draws.

## The convergence criterion — THE PAPER'S, not one I invented

Section 3.2 of the paper states it, and it is the only definition that counts:

    "During development, each agent failure was root-caused and distilled
     into a general primitive ... a reusable piece of knowledge, never the
     answer to one specific problem, then verified by running and merged
     into the code knowledge. WHEN THIS FAILURE-TO-PRIMITIVE CYCLE STOPPED
     YIELDING NEW ENTRIES, OASiS was frozen."

So the question each round asks is NOT "was the harness clean". It is:

    DID THIS ROUND TEACH OASiS ANYTHING NEW AND REUSABLE?

Freeze when a round answers no. Alexander's phrasing: we run it a few times,
and when OASiS provides everything necessary to solve problems from the
distribution, we go to evaluation.

A primitive qualifies only under the paper's two rules for knowledge
(Section 3.1): it must be GENERAL — a pattern, a calling convention, a
failure mechanism — never the answer to a benchmark problem; and it may not
be merged until the documented pattern has been EXECUTED end to end on the
installed code and shown to behave as described.

An earlier version of this file substituted my own criterion — "a round with
zero findings in four defect classes" — and reported against it for two
rounds. That was inventing a standard while the real one was written down.
The defect classes below are still worth tracking, but as a PRECONDITION,
not the criterion: a failure caused by our own broken instrument teaches
nothing about which knowledge is missing, so the instrument must be clean
before a round's primitive count means anything at all.

  D1 INSTRUMENT  grader or harness behaviour that changes an outcome
  D2 TASK        a cell that is impossible or self-contradictory
  D3 KNOWLEDGE   a path we have walked that the served knowledge does not
                 convey — THIS is the class that produces primitives
  D4 CUSTODY     leak, contamination, or bookkeeping error

## Primitives produced, per round (the number that decides the freeze)

  Round 1 — many. Vector participants for four backends served to nobody;
    FEBio exporting one averaged traction (measured order 0.001); the 4C
    plane-strain route uncallable; CALCFLUX_BOUNDARY vs CALCFLUX_DOMAIN;
    the anisotropic transmission rule; deal.II C++ promised but not served.
  Round 2 — the four built participants (FEBio traction, DUNE vector,
    transient pair, 3-D pair), each verified by running to order 2.
  Round 3 — in progress.

The rate is not yet falling. We are not close to a freeze.

## The ladder

27B first (weakest — it surfaces knowledge gaps a strong model papers over),
then 122B, then 397B. Advance only from a converged tier.

Because knowledge is shared across tiers, a knowledge change after a tier has
converged INVALIDATES that tier's claim for the cells it touches: re-run those
cells at every tier already passed. Harness and grader fixes invalidate every
cell they could change, at every tier — cheap to re-grade, so re-grade all.

The whole ladder must be converged before the freeze. A defect found at 397B
that implicates knowledge sends the affected cells back down the ladder; it
does not send the campaign back to zero.

## What may change between rounds, and what may not

MAY change, and must be committed with the defect that justifies it:
  - served knowledge (that is the system under development)
  - harness and grader defects, with a firing test per fix
  - a task text ONLY to remove an impossibility or a contradiction, never to
    make a cell easier, and the cell is then re-walked before reuse

MAY NOT change:
  - grading thresholds, tolerances, or verdict rules in a direction that
    admits more successes. A threshold moves only if it is WRONG on its own
    terms, proven by a firing test, and it moves for both arms and all cells
  - the two-arm design, the model pins, the temperature, the budgets
  - anything about a cell after the freeze marker exists

The asymmetry is deliberate and must be stated in the paper: knowledge fixes
help the OASiS arm, because the OASiS arm is the thing being built. The bare
arm gets the same tasks, the same budget, the same harness fixes, and the same
grader. What makes the comparison fair is not that both arms improve, but that
the evaluation problems are fresh and drawn after everything is frozen.

## Anti-gaming rules (standing)

  1. A number that improves without a named, committed defect fix is a bug
     until proven otherwise. Diagnose before re-running.
  2. Never re-run a cell "to see if it passes this time". Re-runs happen only
     after a committed fix, and the fix's reason is recorded first.
  3. A fix must be justified by evidence from the run, not by the fact that a
     cell failed. "The agent could not do X" is not a defect; "the agent could
     not do X because the knowledge omits Y, and here is the walk that shows Y
     is required" is.
  4. Results that get better must be scrutinised as hard as results that get
     worse. Round 1's probe-grid repair flipped cells in BOTH arms — that is
     what an honest instrument fix looks like.
  5. Every round's findings land in DEV_FINDINGS.md before the next round
     starts, with the commit that fixed each one.

## THE MEASUREMENT IS NOT REPEATABLE AT n=1 (2026-08-17)

Round 2 graded worse than round 1 in both arms — single-code solved 4→2 bare
and 7→4 OASiS, asserted-wrong 11→15 and 6→16. The campaign's own rule says a
number that gets worse is our bug until proven otherwise, so before hunting a
cause I measured whether a single run per cell can detect a change at all.

Re-ran cells with the SAME seed, the SAME code and the SAME knowledge as
round 2. Nothing differed. Result:

    REPRODUCED 1 OF 6.

    DU1 BARE   round1 CORRECT   round2 COMPLETED_UNPHYSICAL  repeat HONEST_INCOMPLETE
    KR2 BARE   round1 CORRECT   round2 CORRECT               repeat COMPLETED_UNPHYSICAL
    KR2 MCP    round1 CORRECT   round2 COMPLETED_UNPHYSICAL  repeat CORRECT
    FE1 BARE   round1 CORRECT   round2 CONFIDENTLY_WRONG     repeat HONEST_INCOMPLETE
    FC1 BARE   round1 FAILED    round2 FAILED                repeat MALFORMED_SUBMISSION
    DU1 MCP    round1 CORRECT   round2 CORRECT               repeat CORRECT

Three different verdicts from three runs of the same cell is normal here. The
`seed` argument is advisory on a hosted endpoint — the provider batches and
shards, and temperature is 0.2, not 0 — so it does not pin the trajectory.

WHAT THIS MEANS

  * Round 2 being "worse" than round 1 is NOT evidence of a defect. Both
    rounds are single draws from a wide distribution. The honest statement is
    that we cannot distinguish them.
  * No cell-level claim survives at n=1, in either direction. Neither "OASiS
    solved this cell" nor "the bare arm fabricated here" is reproducible on
    its own.
  * The aggregate over 32 cells is better than any single cell, but its error
    bar is unknown and has to be measured, not assumed.
  * The same weakness applies to the paper's existing 39%/70% numbers if they
    came from one run per problem. A referee asking "how many times did you
    run it?" currently has no good answer, and that question ends papers.

WHAT THE DESIGN HAS TO BECOME

  Replicates per cell per arm, reported as a distribution rather than a
  verdict, with the arms compared by a statistic that accounts for the spread.
  The cost is real — N replicates multiplies a 64-run round by N — so the
  number of replicates is a decision for Alexander, and the honest floor is
  the smallest N whose confidence interval separates the arms.

  This is a finding about the EXPERIMENT, not about OASiS, and it is worth
  more than either round's score: it is the difference between a result that
  survives review and one that does not.

## Anchoring audit of the knowledge added between rounds 1 and 2

Seven participants were written or repaired between the rounds. If any of
them encoded this campaign's specifics, the evaluation phase would be
measuring recall of our own problem set rather than transferable knowledge,
and a reviewer would be right to say so. Checked 2026-08-16 against the four
markers that identify a campaign cell — extent 1.5 x 1, interface at x = 5/8,
mesh sequence 8/16/32, probe grid 44 per side:

  * NONE of the seven contains any of those. The interface defaults are 0.5,
    0.55, 0.6 — deliberately not 0.625 — and the mesh and probe values appear
    nowhere.
  * Every literal that pattern-matched turned out to be a MEASURED RESULT in
    a header comment (`1.5e-5`, `O(h^1.5)`, `45.4%`), not geometry.
  * One off-convention default: participant_febio_elastic.py ships
    E_MOD = 1040 where the four pre-existing elastic participants all use
    1000. It comes from the builder's own manufactured test (lambda 600,
    mu 400). It is NOT a FEBio cell's material — the FEBio cells use
    lambda 500 with mu 250 and 1250 — so it gives no head start on anything
    in the set. Left as-is because the measured numbers in that file's header
    were taken at those values, and a default nobody can reproduce is worse
    than one that is merely unusual.

The general rule this enforces, from the project's standing instructions:
templates carry PLACEHOLDERS and an EDIT block, never a problem's dimensions.

## A trap this campaign keeps falling into: measuring the wrong checkout

Two OASiS checkouts exist on this machine and they are NOT the same code:
`ofa-v2/src/tools/consolidated.py` is 280713 bytes, `open-fem-agent`'s is
153388; the coupling knowledge document is 10316 bytes against 6059, and only
the former mentions deal.II at all.

The campaign serves **ofa-v2**: `langgraph_eval/agent.py` launches the MCP
server with `cwd = REPO/"src"`, `args ["-m","server"]`, `PYTHONPATH =
REPO/"src"`, and the driver exports `OASIS_REPO=ofa-v2`. The interpreter
comes from the other checkout's venv, which supplies site-packages only —
cwd and PYTHONPATH decide which CODE answers a tool call.

Anything that measures knowledge, coverage or gaps MUST state the sha256 and
byte length of the tree it measured. The first knowledge audit of round 1
measured `open-fem-agent` because its brief named that path, and its
conclusions described knowledge the campaign never served. Same defect class
as the runner's old `OASIS_REPO` default and the worktree venv: an instrument
pointed at the wrong tree reports a confident, wrong negative.

## Round log

  D2 IS CLOSED. An independent re-derivation of all 28 grade-1 cells (never
  touching the vault — the solutions were exported to a scratch file first)
  reports 28/28 PASS: every strong-form residual EXACTLY zero symbolically,
  boundary data satisfied, and for the coupled cells both transmission
  conditions — continuity of the primal variable AND of the normal
  flux/traction with each side's own coefficient — verified. Orders were not
  argued but MEASURED by solving each cell on its prescribed meshes and probe
  grid: FE1 1.97/1.99/2.00, SK1 3.11/3.06, NG2 3.01/3.02, DU2 2.99/2.96, C5
  1.91/1.98, C11 1.91/1.97, and so on. Zero metadata mismatches between task
  and key across all 28. No cell is broken; every coupled failure is ours or
  the model's, never the problem's.

  Three solver-side dependencies it surfaced, none a cell defect:
    * FC2 explains its own both-arms failure: plain Q1 on that cell measures
      order 0.068 with an error 57% of the solution amplitude, while a
      locking-free Q1/P0 measures 1.984. The task already warns and names
      KINEM linear / TECH eas_full, so the signature says the EAS technology
      was not actually active in the submitted decks.
    * FB2's order depends on the relaxation-integral update, which the task
      does not pin: a second-order update gives 1.99, a first-order one 0.98
      — inside the band but outside tol. Name it, as FC1 names theta.
    * C5's re-entrant corner permits O(h^1.67) in theory; measured 1.91/1.98,
      so it does not bite.
  Also: the coupled tasks omit the PRECISION block the single-code tasks
  carry, and the smallest-amplitude cells in the campaign are coupled
  (C11 max|u| = 1.06e-07). It does not bite at the measured error levels,
  but it costs nothing to add.

  Round 1 findings, second pass (three independent audits, 2026-08-15).
  Ground truth first: path_readiness records ALL 14 coupled cells converged
  non-blind, every level. NO cell is a task defect. The coupled 0/14 is
  therefore not a measurement of the model.

    D1 grader, submission discovery non-recursive while the evidence check
       beside it used rglob — 5 cells mislabelled, including a complete
       converged 3-level set booked as NO_SOLUTION_FILES and an honest
       COULD_NOT_COMPLETE graded a fabrication. Fixed 57917c0b, 5 firing
       tests; regrade moved 5 verdicts in BOTH directions.
    D1 step cap returned a normal state, so 3 bare runs that exhausted 250
       graph steps at 124 tool calls were logged error=null, identical to a
       voluntary stop. Fixed 72c9e7d7.
    D1 trajectory.txt is written only on clean exit — 0 bytes for every
       timed-out run, so the evidence is missing exactly where it is most
       needed. OPEN.
    D3 the vector (elasticity) participants for fenics, ngsolve, skfem and
       dealii shipped on disk and were served to nobody; 4 coupled cells are
       vector problems and their agents rewrote them from scratch. Fixed
       57917c0b. Still ABSENT for fourc, kratos, dune, febio, sparta.
    D3 flux recovery: the shipped 4C participant uses CALCFLUX_DOMAIN
       (volume/L2 recovery, measured field order 1.75 — within 0.15 of
       CONFIDENTLY_WRONG) rather than CALCFLUX_BOUNDARY; the variational
       recovery formula appears nowhere in the coupling knowledge. Blocked
       C9 outright, caps six more cells. OPEN — highest-value remaining fix.
    D3 no route to 4C 2-D thermoelasticity: the plane-strain slab sentence
       IS served and C1's agent recited it, then declared 4C incapable. The
       gap is an executable route (run_with_generator('tsi',
       'plane_strain_2d') exists and was called by 0 of 14 runs), plus the
       walk's Scalar_Transport + one-way-TSI decomposition. OPEN.
    D4 the quarantined /tmp and $HOME scatter was never removed, so later
       runs read earlier runs' files (C2-MCP read a prior attempt's RESULT
       and CSVs; C4-MCP read C6's deal.II source). The exposure sweep only
       covers campaign3_blind, and one run enumerated sibling worktrees
       including a keys worktree. OPEN — custody.
    HARNESS the MANDATORY critic never worked: spawn_subagent raises
       NotImplementedError (StructuredTool does not support sync invocation)
       in 14/14 coupled and 16/18 single-code OASiS runs. OPEN.
    HARNESS no budget signal: 13 of 14 coupled OASiS runs quit voluntarily at
       a mean 37% of budget while inventing deadlines (47 statements). Fixed
       72c9e7d7.

  Round-1 defect closure, 2026-08-16 (all committed, all verified by
  execution rather than by inspection):
    CLOSED  critic never ran (async path; verified live — the critic really
            called knowledge() and returned APPROVED)               f01d13c3
    CLOSED  cross-run contamination: 19 scatter directories quarantined by
            transcript evidence, 24 MB, nothing deleted; preflight now does
            it automatically. The first version of that cleaner would have
            moved 1043 directories — a dry run caught it              c07db06e
    CLOSED  timeouts lost their transcript (0-byte trajectory.txt)    c07db06e
    CLOSED  phantom X11: DISPLAY/XAUTHORITY unset, headless backends  c07db06e
    CLOSED  anisotropic transmission never stated (tensor form + the
            reaction recovery being tensor-agnostic)                  b916d262
    CLOSED  4C plane-strain route uncallable: exact snippet, verified by
            generating the deck AND running 4C to normal exit         b916d262
    CLOSED  4C taught the L2 domain-flux recovery its own corpus condemns;
            CALCFLUX_BOUNDARY (Gresho) now served                     904c8dc9
    CLOSED  deal.II promised C++ sources no tool returned; inlined    904c8dc9
    CLOSED  vector participants shipped but served to nobody          57917c0b
    CLOSED  Kratos had no Neumann participant: built, measured at field
            order 1.999, conservation at round-off, negative control shows a
            sign-flipped flux still CONVERGES while being wrong by 29 K —
            only the self-check catches it                            26b97939
    CLOSED  coupled tasks lacked the precision contract; FB2 never stated
            the temporal order it grades                              fa9c4da7
    CLOSED  FEBio exported ONE domain-averaged stress across the whole
            interface — measured order 0.013/0.003/0.001, i.e. not
            converging at all. Reaction-based recovery now measures
            displacement order 2.02/1.99/1.99 in three configurations f054b4d9
    CLOSED  DUNE vector participant: order 2.014/2.007/2.003, and 2.005/
            2.001/2.000 on a cubic case; both role assignments agree to four
            significant figures                                      9657eb63
    CLOSED  transient participants (FEniCSx + deal.II C++, built not just
            described): field order 2.00 x3, and 2.00 cross-code in the
            failed cell's own role assignment                        73184f5c
    CLOSED  3-D participants (Kratos + DUNE, either role, any interface
            normal): temperature 1.98, interior flux 1.97, divergence
            theorem at 1e-13, area recovered as exactly 1.000000     41abaffc
    CLOSED  the sensitivity probe left PERTURBED solves on disk, including
            the solution files a grader reads; probe defaults to True.
            Round 1 unaffected (only three coupled runs reached that code
            path), and it would have bitten round 2                  f2e007a8
    CLOSED  infrastructure errors matched case-sensitively, so a lowercase
            provider "error code: 504" was booked as a model failure — on a
            BARE run, which inflates the measured uplift             e75c1be8
    CLOSED  a timed-out shell killed the shell but not the solver: three
            orphans found 23-24 h later at 100% CPU, 71 CPU-hours    fa3cf276

  Payload size checked before round 2 rather than assumed: the largest
  single knowledge response is deal.II at ~22.7k tokens, against a measured
  262144-token context on all three campaign models (queried from the
  provider). Under 9% of context, so serving the participants and C++
  sources in one call stands — no chunking, and therefore no new promise the
  corpus would have to keep.

  Round 1 — 27B, seed 1, 2026-08-15. NOT CLEAN.
    D1 x3: grader import dead (645b3eb3); probe-grid row order (645b3eb3);
           tool exceptions killed runs (813625f6)
    D4 x1: 19 runs wrote deliverables outside the sandbox, invisible to the
           grader; write_file now confined, runs re-done
    D3 x1 suspected: C1 4C 2D thermoelasticity decomposition — audit running
    Result after repairs: single-code bare 4/18, OASiS 7/18; coupled 0/14
    both arms. The coupled result is not yet attributable to the model: the
    round was not clean, so it is not evidence about capability.
