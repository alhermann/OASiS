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
  Round 3 — 128 runs, replicates 2 and 3, zero infrastructure faults. Nine
    primitives, and for the first time the round diagnosed itself: the agents
    named our gaps in their own COULD_NOT_COMPLETE notes rather than failing
    silently.
      1  SPARTA: stop foreclosing the docs; point at $SPARTA_ROOT/doc
      2  SPARTA: the tally sampling rule, filed in _CROSS_CUTTING where a
         flux problem actually reads it (first attempt filed it where the
         physics rows do not serve it — the same defect it was fixing)
      3  Kratos: the interpreter we named cannot import Kratos
      4  Kratos curved_mms: environment entry re-measured, it had ROTTED and
         was inverted on two of three interpreters
      5  Coupling: a refinement study is N calls to `couple`, one per level
      6  Coupling: the volume solution never returns through the driver
      7  Coupling: converged-with-a-failed-check is still a result
      8  Coupling: the points exchanged are not the points reported
      9  The elastic participants we ship cannot express a body force, so
         following our own "edit the marked block only" instruction yields a
         participant whose only solution is u = 0

The rate is not yet falling. We are not close to a freeze.

A new defect class appeared in round 3 and the rules did not cover it.
Entry 4 was TRUE when it was written and FALSE nine days later, because the
machine changed under it. The merge rule ("no entry is merged until it has
been executed and shown to behave as described") is satisfied at merge time
and says nothing about the following week. Entries that describe the
INSTALLED ENVIRONMENT — interpreter paths, which build has which apps —
rot silently, and nothing re-checks them. Entries that describe a code's
SYNTAX or a method's MATHEMATICS do not rot. Until there is a re-measurement
gate, environment entries must tell the reader to probe rather than trust
any recorded path, including their own.

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

  Round 2 — 27B, seed 1, 2026-08-16. CLEAN, and it graded WORSE than round 1
    in both arms. That is not evidence of a defect: at one run per cell the
    measurement cannot detect a change of this size. The repeatability probe
    settled it — re-running eight cells with the same seed, same code and
    same knowledge reproduced the original verdict in 1 of 8. Per-cell
    verdicts are noise; only pooled rates over replicates mean anything.

  Round 3 — 27B, seeds 2 and 3, 2026-08-17. CLEAN: 128 runs, zero
    infrastructure faults, 18 non-null errors all of them model results
    (16 ran out of the 2700 s clock, 1 exhausted the graph step limit, 1
    filled the provider's context window).
    Graded: single-code bare 22.9%, OASiS 33.3% (+10.4 points, p = 0.33 —
    not significant at n = 48 pairs). Coupled 0/36 in BOTH arms.
    SPARTA bare 4/6, OASiS 0/6.

    Neither zero is a physics failure, and both diagnoses produced primitives
    rather than excuses:

      * The coupled 0/36 splits in two. Eight OASiS runs reached the `couple`
        tool and six drove a two-code coupling to convergence (residuals to
        1e-7) — and scored zero on the four steps between a converged
        coupling and a submitted answer, now written down as section 3b of
        the coupling core. The other twenty never called `couple` at all:
        they stopped voluntarily after a median of 36 tool calls and 874 s
        against a 2700 s clock, and wrote COULD_NOT_COMPLETE naming OUR gaps
        — the missing body force above all. The bare arm called `couple`
        zero times (it has no such tool) and reached convergence zero times
        across 2505 shell commands.

      * SPARTA tied on physics: both arms put 4 of 6 values in band. They
        differed on completing the three-level sequence, 6/6 bare against
        1/6 OASiS. The OASiS arm was spending its budget elsewhere.

    Not one of the 128 runs fabricated a result. Every failure was an honest
    incomplete, which is the reliability claim the paper actually needs.

  ORPHAN, SECOND OCCURRENCE — 2026-08-18. A `coupling_main.py` from
    C10_27b_BARE_seed3 was found at 99.9% CPU, 7 h 10 min after its run had
    finished and its ledger had been written. Same mechanism as fa3cf276: the
    agent's shell hit its timeout, the shell died, the solver it had started
    did not. That entry is recorded as CLOSED and it is not — the earlier fix
    did not cover this path, and nothing sweeps for survivors after a round.
    Cost here was about 7 CPU-hours and a core taken from round 4's first
    minutes. It cannot corrupt a result: the ledger was already written and
    the grader reads files, not processes. But it silently taxes every later
    run, which is exactly the kind of drift that makes wall-clock and timeout
    counts incomparable between rounds. A post-round sweep for processes whose
    cwd is under runs/ is the missing piece.

  PENDING FOR ROUND 5 — the critic tool costs the OASiS arm a call it should
    never have to spend. Measured over round 3's 64 OASiS runs: 22 critic
    submissions, of which 11 were rejected outright with "pass exactly one of
    `setup` or `coupling_args`", across 10 runs. Half of every critic call is
    a wasted round trip, and the agent has already written a long review text
    before it learns the call shape was wrong.

    It biases against OASiS by construction: the bare arm has no critic tool,
    so it pays nothing, while the OASiS arm is under a MANDATORY critic
    instruction and must keep trying. It plausibly contributes to the budget
    exhaustion behind SPARTA finishing 1 of 6 mesh sequences against bare's
    6 of 6.

    The docstring describes both parameters and says coupling_args is used
    "for the coupling tools instead of `setup`", but never states that exactly
    one is REQUIRED, so an agent that passes neither — the common failure —
    only finds out by failing.

    NOT FIXED YET, DELIBERATELY. Round 4 is in flight, and changing what the
    agent sees mid-round would split the round into before-and-after
    populations and make its 128 runs uncomparable. This file is campaign
    documentation and is never served, so recording it here is safe. The fix
    goes in after round 4 closes, and round 5 measures it.

    An earlier count in this investigation said "13 of 15 runs never got an
    approval". That was a regex searching for "approved" when the field is
    "accepted", and it described my own pattern rather than the tool's
    behaviour. The numbers above come from the actual JSON.

## The pooling must be declared with every rate, or the numbers are not comparable

Recomputing round 3's single-code rate produced 28.1% vs 37.5% on one pass and
22.9% vs 33.3% on another. Both are correct. They differ only in what was
pooled, and the spread across defensible choices is larger than the effect:

    16 FEM cells, seeds 2+3        bare 28.1%  OASiS 37.5%  +9.4   p=0.55
    18 cells (incl. SPARTA), 2+3   bare 30.6%  OASiS 33.3%  +2.8   p=1.00
    18 cells, seeds 1+2+3          bare 27.8%  OASiS 29.6%  +1.9   p=1.00
    16 FEM cells, seeds 1+2+3      bare 22.9%  OASiS 33.3%  +10.4  p=0.33

The uplift moves from +1.9 to +10.4 depending on whether SPARTA is inside the
single-code pool and whether seed 1 counts as a replicate. Neither choice is
dishonest and both have a rationale — but quoting one without saying which is
how a paper acquires a number it cannot defend, and a reviewer who recomputes
gets a different answer and stops believing the rest.

STANDING RULE for every rate in the paper and in these reports:
  * name the cells (which pool, how many) and the seeds pooled;
  * keep the three evidence grades separate, which means SPARTA is NOT in the
    single-code pool — it is graded band-only, a different grade from the
    order-based FEM cells, and pooling it silently mixes grades 1 and 3;
  * state whether seed 1 (round 2) is included. It measured the same knowledge
    as seeds 2 and 3, so it IS a legitimate third replicate, and the honest
    default is to include it and say so.

Under that rule the round-3 headline single-code figure is 16 FEM cells x 3
seeds: bare 22.9%, OASiS 33.3%, +10.4 points, McNemar p = 0.33 — not
significant at n=48 pairs, which is stated rather than buried. SPARTA is
reported separately as 2/4 vs 0/4 band-only, and coupled separately as 0/28
in both arms.

  Round 4 — 27B, seeds 4 and 5, 2026-08-18. CLEAN: 128 runs, ZERO
    infrastructure faults (the one 504 casualty was re-run), 23 runs out of
    clock, no truncated replies. All fixes it was meant to test landed
    04:34-05:33; the round launched 06:08.

    Graded, with the pooling declared and evidence grades kept apart:

      16 FEM single-code (grade 1)   bare 25.0%  OASiS 37.5%  +12.5  p=0.34
      12 POOLED coupled (grade 1)    bare  0/24  OASiS  0/24
      C13 off-pool (grade 3)         bare  1/2   OASiS  0/2
      C14 off-pool (grade 2)         bare  0/2   OASiS  0/2
      SPARTA (grade 3)               bare  1/4   OASiS  3/4   +50

    Round 3 for comparison: FEM +10.4 (bare 22.9, OASiS 33.3); coupled 0/36
    both; SPARTA bare 4/6, OASiS 0/6.

    WHAT MOVED, and it is behaviour rather than score:
      * coupled OASiS runs reaching `couple`      8/28 -> 13/27
      * coupled OASiS runs achieving convergence     6 -> 10
      * bare runs achieving convergence         0/27 -> 0/28  (0 in 55 tries)
      * SPARTA 3-level sequences completed        1/6 -> 3/4
      * SPARTA score reversed, -66.7 -> +50 points
      * FABRICATION, coupled cells: bare 10 of 28, OASiS 2 of 28

    THE COUPLED ZERO SURVIVED THE FIX, and that is the finding. All 27
    coupled OASiS runs received the core payload, so all 27 were served
    section 3b. Ten drove a coupling to convergence. The dominant grader
    reason is still NO_SOLUTION_FILES: converged, then wrote no field files —
    the exact failure 3b was written to prevent. Being served is not being
    read: 3b sits inside a 53 KB core payload, and a rule buried in a large
    document is a rule the agent may never reach. The universal rule added
    after this round is appended to EVERY knowledge payload instead, which is
    the prominence test round 5 will settle.

  Primitives from round 4 — FIVE, so the rate is NOT falling and we do not
  freeze:
      1  the write-the-answer-file rule reached 1 of 9 backends; a rule true
         of every backend must not live in one backend's table (third
         occurrence of this filing error; now fixed at the single point every
         payload passes through)
      2  DSMC level-to-level change is not automatically a mesh effect —
         statistical scatter does not shrink with refinement, and two runs
         called their own converged sequence NOT_CONVERGED because of it
      3  the critic refusal named the rule but not which half was broken, so
         an agent that passed neither argument read it as "you passed both"
      4  prominence is a property of served knowledge, not just presence:
         section 3b was served to 27 of 27 runs and did not change the
         behaviour it describes
      5  MEASUREMENT, not knowledge: pooling the 14 coupled cells as one rate
         mixed evidence grades 1, 2 and 3, and booked a band-only WITHIN_BAND
         as a coupled "success" for a bare run that never coupled anything.
         The grader keeps the grades apart and refuses mixed aggregation; the
         ad-hoc stats script bypassed that guard by reading the JSON directly.
         It now raises instead.

## What a user of the PUBLIC repo gets today, measured 2026-08-19

The task list recorded "public main emits 24 fabricated 4C keys into decks".
Re-measured by running the current auditor against main's own content in a
throwaway worktree, the number is larger and spread wider:

    backend   unresolved keys on main   in this tree
    fourc                          54              0
    fenics                          3              0
    kratos                          2              0
    dune                            3              0
    febio                           4              0
    TOTAL                          66              0

An unresolved key is one the auditor cannot find anywhere in the installed
code's own grammar or source — so it is a key we tell a user about that the
solver does not have. On main these reach generated decks. Coverage is also
thinner: main's 4C corpus is 432 keys across 241 entries against 733 across
313 here, and main has NO SPARTA entries at all against 103 here.

main is 879 commits behind consolidation/v2 and carries nothing consolidation
does not; it does not even contain the auditor that finds this. So the fix is
the merge, not a patch.

NOT MERGED. Publishing 879 commits to a public repository is an outward-facing
action on a product Alexander has said is for the community, and it is his
call, not something to slip in during a campaign round. Recorded here so the
number is measured rather than remembered when that decision is taken.

## A number moving the WRONG way, and why it is not yet a finding

Watched during round 5 because it points at our own work rather than the
model's. Share of FEM single-code runs that left a deliverable on disk — the
exact behaviour the write-the-answer-file rule targets:

    round 3 (seeds 1,2,3)   bare 27/32 = 84%   OASiS 28/32 = 88%
    round 4 (seeds 4,5)     bare 25/32 = 78%   OASiS 22/32 = 69%
    round 5 (seed 6 only)   bare 14/16 = 88%   OASiS 10/15 = 67%

The OASiS arm was ABOVE bare in round 3 and has been below it since, and the
universal rule added after round 4 has not moved it (69% -> 67%). If that is
real it is a cost of our own knowledge growth, and the obvious mechanism is
payload size: the OASiS arm makes about half the tool calls bare does in every
round (52-56 against 104-114) while carrying ~71-76k tokens of context per
call against bare's ~52k since round 4. Fewer turns, each more expensive.

IT IS NOT YET A FINDING. Fisher exact, two-sided:
    round 3 vs round 4 OASiS deliverable rate   p = 0.129
    OASiS vs bare within round 5 so far         p = 0.220
Detecting an 88% -> 67% drop at 80% power needs about 61 runs per group; we
have 32 and 15. So the honest statement is that the OASiS arm's deliverable
rate has drifted down across three rounds without reaching significance in any
single comparison, and the direction is consistent enough to keep measuring.

Round 5's remaining seed will take the OASiS n to 32, which still does not
reach 61. The decision this forces: if the drift persists at n=32, the next
round must test payload size directly — the same knowledge served narrower —
rather than adding more text to a payload that may already be too large to
finish reading. Adding knowledge is not free, and we have no measurement of
its cost.

## Round 5, first half: the structural primitive landed, and it exposed the real blocker

Coupled cells C1-C8, OASiS arm, like-for-like across rounds (round 5 is seed 6
only, n=8, so every round-5 figure below is provisional):

    metric                        round 3      round 4      round 5*
    per-level directories          7/16 44%    11/16 69%     7/8  88%
    reached `couple`               3/16 19%     6/16 38%     3/8  38%
    achieved convergence           3/16 19%     6/16 38%     1/8  13%
    wrote numeric field files      4/16 25%     8/16 50%     1/8  13%
    timeouts                       0            0            0
    median wall of 2700 s         809 s        933 s        639 s
    median tool calls              34           36           25

Adoption of the per-level directory pattern — the first of the four steps
section 3b describes, with its own level.json — climbs 44 -> 69 -> 88%. The
knowledge IS landing, and prominence plausibly helped.

But the numeric output went the other way, and the reason is not what we
assumed. ZERO coupled runs timed out in any round. Round 5's coupled runs stop
at a median of 639 s out of 2700 — 24% of the clock — after 25 calls, FEWER
than round 4's 36. They are not running out of budget. They are stopping early,
having built the scaffolding and produced no numbers.

THAT DISTINCTION MATTERS AND WE HAD IT WRONG. Every fix aimed at this so far —
"write the answer file at level 1", the universal rule appended to every
payload — is a fix for BUDGET EXHAUSTION: it assumes the agent is interrupted
before it can write. The measurement says the agent is not interrupted. It
decides it is finished, or decides it cannot continue, with three quarters of
its clock unspent. A rule about WHEN to write cannot fix an agent that stops
before there is anything to write.

So the next intervention is not more instruction about ordering. It is whatever
makes an agent with 2000 seconds left keep going: an explicit statement of how
much budget remains and that stopping early scores the same as failing, or a
required self-check before it is allowed to conclude. Round 6 tests that, and
the number to watch is median wall time as a fraction of the clock, not the
solve rate.

Provisional on n=8. The second seed doubles it and the direction is what
decides whether round 6 pivots.

## Correction: there are TWO failure modes, and the previous entry named only one

The entry above concluded from coupled cells at n=8 that the blocker is
premature stopping rather than budget exhaustion. The complete seed-6 slice
shows that was half the picture, and the half that does not apply to
single-code cells.

FEM single-code, OASiS arm, median share of the 2700 s clock actually used:
    round 3  45%      round 4  53%      round 5 (seed 6)  81%

Round 5's OASiS runs use MORE of the clock than any arm in any round, and still
write a deliverable only 69% of the time against bare's 88%. So on single-code
cells they are not stopping early at all.

The five seed-6 FEM OASiS runs with no deliverable split in two:
    DU2 111% of clock, 88 calls, TIMEOUT      budget exhaustion
    NG2 100%           85 calls, TIMEOUT      budget exhaustion
    KR2 100%          107 calls, TIMEOUT      budget exhaustion
    KR1  21%           11 calls, no error     abandoned
    NG1   4%            5 calls, no error     abandoned
Every run that DID write a deliverable used between 32% and 88% of the clock —
the successful band is the middle, and both tails fail for opposite reasons.

Three of five are genuine timeouts, and that is where the write-the-answer-file
rule should have worked and did not: DU2, NG2 and KR2 made 85-107 calls and
wrote nothing. Two are abandonment — NG1 stopped after 5 main-agent calls with
its critic having just returned a REJECTED verdict.

So round 6 needs BOTH interventions, and they are different:
  * for the timeout tail, the write-first rule is right and is not landing;
    the next lever is a hard checkpoint rather than advice — the agent should
    be unable to reach call 40 without a deliverable on disk;
  * for the abandonment tail, a rejected critic review currently reads as a
    stop signal. It is not one: the review is advice to revise, and a rejection
    with 96% of the clock left is a reason to iterate, not to conclude.

MEASUREMENT NOTE, because it nearly misled me: ledger `tool_calls` counts the
MAIN agent (from trajectory.txt) while trajectory_live.txt also captures
subagent activity, so the live log runs 1.4-3.4x higher. NG1's "5 calls" is 5
main-agent calls and 18 including its critic. Both files are correct for
different questions; a mixed comparison is not. Earlier critic-usage counts in
this file come from the live log and therefore include subagent submissions,
which is what "did this run use a critic" should mean.

## Retraction: the critic-rejection explanation for abandonment is refuted

The entry above proposed that a REJECTED critic review is being read as a stop
signal, and named it as one of two interventions for round 6. That came from a
single run (NG1 seed 6, which stopped right after its critic returned REJECTED).
Tested across all 170 OASiS runs in rounds 3-5, grouped by the last critic
signal each run saw:

    last signal      n     median wall        median calls   deliverable
    REJECTED       114   1273 s (47% clock)        48            75%
    accepted        44   1124 s (42% clock)        44            77%
    no critic       12   1132 s (42% clock)        48            50%

Runs that ended on a rejection ran LONGER and produced deliverables at the same
rate. The hypothesis is refuted; NG1 was one run and generalising from it was
wrong. That intervention is withdrawn from the round-6 plan.

What survives: the TIMEOUT tail is real and measured (3 of 5 no-deliverable FEM
OASiS runs in seed 6, 85-107 calls each), and the write-the-answer-file rule is
not landing for it. The ABANDONMENT tail is real but unexplained — n=2 in that
slice, which is not enough to diagnose, and inventing a mechanism for it is how
the last hypothesis got made.

One number worth keeping: runs with NO critic at all produced deliverables 50%
of the time against 75-77% for runs with one. Confounded — a run that gets far
enough to need a critic is already doing better — so it is not evidence that
the critic causes finishing, but it is evidence against the critic being a net
drag, which was a live worry when half of its submissions were being rejected
on call shape.

## The context-window worry was backwards: both events are in the BARE arm

run_blind.py's own comment on CONTEXT_EXHAUSTED says the arms "are not equally
exposed: the OASiS arm receives much larger tool responses (a single knowledge
call can return ~23k tokens)", and flags it as something to watch per arm.
Measured over all 361 runs to date, it has not materialised:

    context-exhaustion events   2 of 361 runs (0.6%)
    both in the BARE arm, both on cell FC2 (4C thermal)
      FC2 BARE seed3   56-58 calls, 11.7M input tokens
      FC2 BARE seed7   56 calls,      8.5M input tokens

Zero in the OASiS arm. The mechanism is the opposite of the one feared: the bare
arm has no knowledge tool, so on a 4C cell it explores by reading the source
tree, and grep output over a large C++ codebase accumulates faster than any
knowledge payload. 8.5M input tokens over 56 calls is ~152k of context per call
against the OASiS arm's ~71-76k.

One event predates the max_output_tokens fix (seed 3, round 3) and one follows
it (seed 7, round 5), so returning 49k tokens of input budget reduced the
pressure without eliminating it for this cell. Both are correctly booked as
MODEL results rather than infrastructure — the harness never trims history, so
filling the window is the agent's own accumulation.

Two consequences. The payload-size concern recorded earlier stands on the
deliverable-rate drift, NOT on context exhaustion, which is a separate mechanism
that is not biting the OASiS arm at all. And FC2 in the bare arm is a cell where
the bare arm is structurally disadvantaged by having to read source — worth
stating in the paper rather than leaving as an unexplained bare failure.

  Round 5 — 27B, seeds 6 and 7, 2026-08-19/20. Seed 6 complete; seed 7 lost 18
    runs to an EXHAUSTED ACCOUNT (HTTP 402, "Insufficient credits"), all in
    coupled cells C5-C14, 10 OASiS and 8 bare, every one with zero tool calls
    and zero wall time — they never reached the model. Quarantined, not graded.
    Model-attributable errors: 18 clock, 1 context.

      16 FEM single-code (grade 1)   bare 18.8%  OASiS 31.2%  +12.5  p=0.42
      12 pooled coupled (grade 1)    bare  0/18  OASiS  0/16
      SPARTA (grade 3)               bare  1/4   OASiS  2/4   +25

    Rounds 3, 4, 5 side by side (FEM uplift): +10.4, +12.5, +12.5.
    Coupled: 0/36, 0/24, 0/16 — three rounds, both arms, no exceptions.

    THE PROMINENCE HYPOTHESIS IS REFUTED. Round 5 existed to test whether the
    coupled zero was caused by section 3b being buried in a 53 KB payload. The
    write-the-deliverable rule was moved to the single point every payload
    passes through, verified served on 9 of 9 backends before launch. The
    structural half was adopted more than ever — per-level directories 50% ->
    61% -> 79% — and the coupled score did not move at all. Placement was not
    the blocker. Round 6 must look elsewhere, and the measured candidate is the
    timeout tail: the FEM runs that fail do so at 85-107 calls having written
    nothing, which advice has now twice failed to fix.

  Primitives from round 5 — TWO, both about the instrument rather than the
  knowledge, and the rate is therefore FALLING but not for a reassuring reason:
      1  HTTP 402 was not in the infrastructure list, so 18 runs that never
         reached the model were booked as model results — 10 of them against
         the OASiS arm. An unpaid invoice would have been published as a
         capability gap. Now classified, with a firing test that also confirms
         429 is still caught and a genuine timeout is still NOT infrastructure.
      2  the prominence hypothesis is refuted, which retires a whole class of
         intervention (move the text somewhere more visible) rather than adding
         one.

    No new KNOWLEDGE gap was found this round. That is the first time, and it
    is what the freeze criterion asks about — but it cannot be read as
    convergence while a quarter of the coupled cells are missing for want of
    credit. Round 6 needs the 18 lost cells re-run first.

## Decision: the 18 lost cells are NOT re-run into round 5

I said in an earlier report that the 18 credit-casualty cells would be re-run
first. That was wrong and is withdrawn.

The just-in-time unsaved-work notice landed AFTER round 5 finished. Re-running
those 18 cells now would run them against knowledge round 5's other 110 runs
never saw, and pooling the two would be exactly the before-and-after mixing this
file forbids elsewhere — the same reason no served text was edited while rounds
4 and 5 were in flight.

So round 5 is reported as it stands, with its n stated:

    12 pooled coupled (grade 1)   bare 0/18   OASiS 0/16
    complete would have been      bare  /24   OASiS  /24

The missing cells cannot rescue the coupled result on any plausible reading —
76 pooled coupled runs across rounds 3, 4 and 5 have produced zero successes in
both arms — but the honest statement is 0 of 16 and 0 of 18, not 0 of 24, and
the gap is named rather than quietly averaged away.

Round 6 then measures the notice on a full, uncontaminated matrix: seeds 8 and
9, all 32 cells, both arms, with the 18 cells present from the start. The metric
the round is FOR is not the solve rate but the share of runs that end with
solver output and no written summary — the failure that three rounds have
measured and two static interventions have failed to move.

## Correction, and it moves the whole target: completion is not the gap. Correctness is.

I have spent three rounds treating "solves correctly, writes nothing" as the
blocker. Measured across all 54 OASiS runs of round 5 rather than off the five
failures in one 16-run slice:

    output + summary        37   69%     submitted properly
    no output + summary      8   15%     honest incomplete, nothing to submit
    no output + NO summary   7   13%     failed before producing anything
    output + NO summary      2    4%     <- the failure I have been fixing

The just-in-time notice targets 4% of runs. The population I described as the
main blocker is two runs. My earlier statement — "three of the five failing
single-code OASiS runs solved and wrote nothing" — was true of five failures in
one slice and I let it stand in for the round, which is the same error as
reading a trend off eight coupled runs.

THE REAL GAP: 69% of OASiS runs submit output AND a summary, and the graded
solve rate is 31%. So roughly 38 points of the distance to the 70% target sits
in runs that submit properly and are WRONG. Completion is not what separates us
from the target; correctness is.

Second defect, found one hour into round 6 rather than at its end: the notice
lives in run_simulation, and OASiS runs do not use it. Round 5, 53 OASiS runs:

    run_bash 51,  couple 10,  run_simulation 8,  run_with_generator 5

Agents shell out. The intervention can reach at most 15% of runs even where it
applies, and 4% x 15% is nothing. It is not wrong, it is irrelevant at this
scale, and it was designed without first measuring which tool executes solvers.

ROUND 6 IS NOT STOPPED, but what it measures is renamed. It cannot test the
notice. It is still 128 runs against unchanged knowledge, which triples the
power behind the uplift estimate (+12.5 at n=32 per round; rounds 4+5+6 give
n=96) and re-measures the deliverable-rate drift at the n=61 that comparison
needs. That is worth the ~60 credits it will cost. Pretending it tests the
notice would not be.

WHAT ROUND 7 MUST TARGET, from this measurement rather than from a story: the
37 runs per round that submit a complete answer and score wrong. Their grader
reasons are the next thing to read, cell by cell, before any further knowledge
is written. If those failures are dominated by one recoverable cause — a wrong
recovery, a misread convention, an off-by-one in a probe grid — that is where
the 38 points are. If they are spread thin across unrelated physics errors,
then 27B has a capability ceiling here and the honest move is to say so and
report the tier for what it is.
