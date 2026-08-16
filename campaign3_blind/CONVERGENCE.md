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

## The convergence criterion (per model tier)

A ROUND is a full sweep: every cell, both arms, one model tier, one seed.

A round is CLEAN when every failure in it is attributable to the model — its
capability or its use of the budget — and NONE to us. Concretely, the round
must produce zero findings in all four defect classes:

  D1 INSTRUMENT  a grader or harness behaviour that changes any outcome
                 (round 1: 3 of these — dead grader import, probe-grid row
                 order, escaped tool exceptions killing runs)
  D2 TASK        a cell that is impossible, self-contradictory, or demands
                 what the named codes cannot do
  D3 KNOWLEDGE   a cell whose intended path we know and have walked, which
                 the served knowledge does not convey, so the agent had to
                 rediscover it or gave up (round 1: C1, 4C's 2D
                 thermoelasticity decomposition — suspected, under audit)
  D4 CUSTODY     any leak, contamination, or bookkeeping error (spent lists,
                 seals, commitment drift, out-of-sandbox artifacts)

A tier is CONVERGED when one full round is clean AND an independent critical
audit of that round's runs — a sub-agent that did not run it, told to hunt
for D1-D4 — also finds nothing. One clean round with a second opinion, which
is the same freeze criterion the rest of this project uses: rounds run until
a round finds nothing.

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
    OPEN    FEBio constant traction (C7, C11) — build running
    OPEN    DUNE vector participant (C12) — build running
    OPEN    transient participant (C4) — build running
    OPEN    3-D interface participant (C10) — build running

  Round 1 — 27B, seed 1, 2026-08-15. NOT CLEAN.
    D1 x3: grader import dead (645b3eb3); probe-grid row order (645b3eb3);
           tool exceptions killed runs (813625f6)
    D4 x1: 19 runs wrote deliverables outside the sandbox, invisible to the
           grader; write_file now confined, runs re-done
    D3 x1 suspected: C1 4C 2D thermoelasticity decomposition — audit running
    Result after repairs: single-code bare 4/18, OASiS 7/18; coupled 0/14
    both arms. The coupled result is not yet attributable to the model: the
    round was not clean, so it is not evidence about capability.
