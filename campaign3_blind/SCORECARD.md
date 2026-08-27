# Where the blind re-run actually stands, against the targets

Written 2026-08-20, after five development rounds at 27B (576 runs, of which
366 ledgers survive in the tree plus 18 credit casualties quarantined). This
file exists because the targets were agreed in conversation and written down
NOWHERE, and an unwritten target is one that drifts to meet whatever was
measured.

## The targets, as agreed

    at 27B      OASiS > 70%, bare ~36%, uplift > +33 points, McNemar significant
    at 122B     OASiS > 61%
    at 397B     OASiS > 66%
    coupling    bare completes zero real coupled runs and fabricates;
                OASiS completes them
    substitution  27B + OASiS >= 397B bare (67%)
    cost/energy   better than 1.2x lower per solved task
    reliability   fabrication near zero in the OASiS arm

## What five rounds at 27B measured

    16 FEM single-code cells (grade 1, order-based)
      round 3   bare 22.9%   OASiS 33.3%   +10.4   p = 0.33
      round 4   bare 25.0%   OASiS 37.5%   +12.5   p = 0.34
      round 5   bare 18.8%   OASiS 31.2%   +12.5   p = 0.42

    12 pooled coupled cells (grade 1)
      rounds 3, 4, 5   bare 0/36, 0/24, 0/18   OASiS 0/36, 0/24, 0/16

    SPARTA (grade 3, band-only)
      round 3  bare 4/6  OASiS 0/6      round 4  bare 1/4  OASiS 3/4
      round 5  bare 1/4  OASiS 2/4

## Read honestly

**The headline uplift is about a third of the target and is not significant.**
+10.4, +12.5, +12.5 against a target of +33. The consistency across three
independent rounds is the strongest thing about it — the direction never
reverses — but no single round reaches significance, and OASiS at 31-37% is
nowhere near 70%.

**These are NOT the paper's numbers and must not be compared as if they were.**
The paper's rates come from its own evaluation draw. These 32 cells are the
DEVELOPMENT set: deliberately diagnostic, deliberately burnt, and looked at
repeatedly precisely so that knowledge could be fixed against them. A
development rate is a measure of the instrument and the knowledge, not the
number that goes in a table. The evaluation phase draws fresh problems after a
freeze, and only those numbers are comparable to anything published.

That is a real caveat, not an excuse. The uplift is the quantity least sensitive
to which problems were drawn, and it is +12.5, not +33.

**The coupling claim is supported in behaviour and unsupported in score.**
Across three rounds the bare arm has driven a coupling to convergence ZERO
times in 55 attempts, while the OASiS arm has done it 21 times (7, 11, and 3 in
the measured slices) and has climbed on every intermediate step: reaching the
coupling tool 29% -> 46% -> 57%, building the prescribed per-level structure
50% -> 61% -> 79%. But the graded score is 0 in BOTH arms, so on the paper's own
scoring the coupling claim currently has no numerical support. What we can say
today is a process claim, not an outcome claim.

**The reliability claim IS supported, and it is the cleanest result we have.**
On coupled cells in round 4 the bare arm fabricated 10 of 28 runs; the OASiS arm
fabricated 2. Across all five rounds no OASiS run has been graded
FABRICATED_NO_RUN at a rate approaching bare's. Fabrication near zero in the
OASiS arm is the one target currently met.

**Untested:** the 122B and 397B tiers, substitution, and cost/energy. The ladder
rule requires a converged tier before advancing, and 27B has not converged.

## What has to happen for the targets to be reachable

The blocker is not knowledge coverage. Five rounds have produced 16 primitives
and the last round produced no new knowledge gap at all. The blocker is that
runs which solve correctly do not finish: three of the five failing single-code
OASiS runs in round 5 seed 6 made 85-107 tool calls, produced solver output, and
wrote no summary. Every solved-but-unwritten run is a target point lost for a
reason that has nothing to do with whether OASiS knows enough.

If round 6's just-in-time notice converts even half of those, the single-code
rate moves without a single new fact being added. That is the cheapest available
distance to the target, and it is why round 6 measures the unwritten-summary
rate rather than the solve rate.

---

## CORRECTIONS, 2026-08-27 — three claims above are wrong or unsupported

An independent re-derivation of every number from the raw grades, with code
that shares nothing with the analysis script, found errors I had repeated for
days. Each is verified by me before being written here.

**RETRACTED — the coupling behaviour claim.** I reported "the bare arm has
driven a coupling to convergence ZERO times in 55 attempts, while the OASiS arm
has done it 21 times" as the strongest coupling evidence we had. It is an
artefact of the instrument. I measured it by grepping run transcripts for
`converged: true` — a string emitted by the OASiS `couple` tool's JSON return.
THE BARE ARM HAS NO SUCH TOOL AND CANNOT PRINT IT. The test could only ever
find OASiS runs; it was biased by construction, and the direction of the bias
is the direction of the claim.

The opposite measurement is biased the opposite way. The grader's own
code-agnostic coupling evidence (a falling partitioned residual history in
FILES) gives BARE 20/90 PROVEN against OASiS 5/88 — but 68 of 88 OASiS records
carry NO coupling block at all, because `couple` returns its history in-band
and only 26 of the 61 runs that called it ever wrote a residual file. Each
instrument sees the arm whose idiom it was written for.

So the honest statement today is: WE DO NOT KNOW how the arms compare on
achieving coupling convergence. What survives is grader-adjudicated and
arm-symmetric: the 12 pooled coupled cells are 0/102 (bare) and 0/100 (OASiS)
across rounds 3-6 — neither arm has produced a gradeable coupled answer. A
code-agnostic convergence measure has to be built before any coupling
behaviour claim goes near the paper.

**BROKEN IN ROUND 6 — the reliability claim.** "No OASiS run has been graded
FABRICATED_NO_RUN at a rate approaching bare's" holds for rounds 3-5 (17/36 vs
5/36, 10/24 vs 2/24, 9/18 vs 2/16) and FAILS in round 6: BARE 9/24 and OASiS
9/24, exact parity. The claim is now "held in rounds 3-5, lost in round 6",
and why it regressed is an open question for round 7's grading, not a
footnote.

**OVERSTATED — the +10.4 uplift is one cell.** Per cell over seeds 4-9:
DL1 goes 0/6 -> 6/6 (+6) and NG1 goes 6/6 -> 1/6 (-5); 7 of 16 cells are tied
at 0/6-vs-0/6 or equal. Net +10 paired wins, of which one cell supplies +6.
The instance-level cluster bootstrap interval is [-9.4, +30.2] points and
contains zero comfortably.

**And the significance was computed wrongly.** McNemar over 96 pairs treats six
seeds per cell as independent trials; they are nested in 16 cells. Cluster-
correct (exhaustive enumeration of all 2^16 within-cell arm-label swaps) gives
p = 0.383, and a cell-level sign test gives p = 0.508, against the 0.13 I have
been quoting. p = 0.13 is anti-conservative by roughly a factor of three. Every
future rate must carry the clustered interval, not the paired-binomial one.

**Also fixed:** the fabrication line elsewhere in the campaign log used a
28-run all-14-cell denominator against a grade-1-only numerator (C13/C14
cannot carry FABRICATED_NO_RUN — different verdict vocabulary), and a
campaign-wide fabrication count keyed on the top-level status field misses 24
real fabrications visible only in `evidence.fatal`. True campaign total 106,
not 82. The duplicate `dev_grades_27b_seed1.json` — byte-identical to the
`_round2` file and one `*seed*` glob away from double-counting seed 1 — has
been deleted.

**What this does NOT change:** the FEM uplift arithmetic (+10.4, +12.5, +12.5,
+6.25 per round; pooled 22/96 vs 32/96) reproduced exactly, and the coupled
0-in-both-arms result reproduced exactly. The errors are in what the numbers
were claimed to MEAN, not in the numbers themselves — which is the more
dangerous kind.
