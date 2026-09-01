# How much of the coupled corpus can support a coupling claim?

Measured 2026-09-01 over every coupled run directory on disk, using the
grader's own tested per-code signature corpus (`blind_eval.evidence.code_evidence`,
guarded by `tests/test_per_code_signatures_are_measured.py`) — not an ad-hoc
regex. Read-only; no key was opened.

## Both prescribed codes left structured solver output

| arm | coupled runs | both codes proven | one | neither |
|---|---|---|---|---|
| bare | 169 | **132 (78%)** | 4 (2%) | 33 (20%) |
| OASiS | 268 | **172 (64%)** | 17 (6%) | 79 (29%) |

**The bare arm proves both prescribed codes more often than the OASiS arm**, by
14 points. On this axis the SCORECARD statement "bare completes zero real
coupled runs and fabricates" is not merely unproven — it is contradicted.

## What this does NOT show, and the distinction matters

`code_evidence` proves a code ran **somewhere in the workspace**. It does not
prove that code produced **side A specifically**. Those are different claims and
only the second supports "two codes met at the interface".

The grader already records the gap rather than guessing at it: coupled runs
whose only execution evidence is the code-agnostic `NDOF = <integer>` line get
`per_code_attribution: UNPROVEN`, and `grading/evidence2.py` records the reason
in full — **0 of the 47 tasks in the older draw ask each participant to capture
its own solver's output**, against 1 of 1 in each newer draw. So for every seed
from 2 to 96, per-side attribution is impossible by construction and the
harness's gap, not the submission's.

Two consequences, both concrete:

1. A coupling claim in the paper may be built only on the **newer draw**, whose
   tasks demand captured per-participant output. The older draw's coupled runs
   are diagnostic material.
2. The 78% / 64% above is the ceiling on the usable evidence base, not a solve
   rate. Nothing here says a run coupled correctly — only that both named codes
   ran.

## A method note worth keeping

The first version of this measurement used hand-written regexes over
`run_level<k>_<side>.log` and reported that 95% of bare and 90% of OASiS coupled
runs failed to evidence both codes. That was **wrong**: the tasks require those
logs to carry only `NDOF = <integer>`, so an absent signature there proves
nothing. Using the grader's corpus, which searches the whole workspace with
tested per-code dialects, inverted the answer from 5% to 78%.

The lesson is the campaign's own: measure with the instrument the campaign
already trusts and tests, or the number says more about the instrument than
about the runs.
