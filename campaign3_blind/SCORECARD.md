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
