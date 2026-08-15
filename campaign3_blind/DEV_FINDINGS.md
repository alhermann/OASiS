# Development-phase findings — 27B, seed 1 (live log)

Running notes for the development report. Facts only, verdicts belong to the
grader; items marked POST-MORTEM need a decision or a check before the
evaluation phase. This file is deliberately OUTSIDE the frozen commitment.

## Instrument
- **KR1 BARE is VOID** — killed at 175 s / 37 calls by an escaped
  PermissionError from `write_file` (model wrote to `/solve_kratos.py`;
  the error never reached the model). Fixed in 813625f6: environment errors
  from run_bash/read_file/write_file become tool observations. Re-run queued
  (task #64). Runs started before ~10:35 ran pre-fix code; sweep all errored
  ledgers at fleet end and void only escaped-exception deaths, never
  timeouts.

## Genuine results with context (grade offline as normal)
- **KR1 MCP: TimeoutError at 2700 s, productive to the last minute** —
  54 calls, 3.75 M tokens in, all three mesh levels ran, RESULT.txt written
  at t+38 min; ran out of clock while still working. Artifacts on disk are
  what the grader reads.
- **DU2 BARE: TimeoutError, level 1 of 3 only** — 98 calls, 3.1 M tokens,
  no RESULT.txt. Expected bare-arm shape on a barely-known backend.
- **DU2 MCP: TimeoutError, level 1 + RESULT.txt** — 47 calls, 2.4 M tokens,
  work dir shows many restarts (sipg_advection_diffusion.py, _v2,
  sipg_simple, sipg_test, test_dune_simple).
  - **POST-MORTEM (validity):** the work dir also contains
    `sipg_advection_diffusion_ngsolve.py` — the model appears to have tried
    NGSolve on a DUNE cell. At grading time verify the submitted evidence
    (run_level1.log) is provenance-DUNE, and record whether the grader's
    per-code signal patterns would catch cross-backend evidence on their
    own. This is the exact reviewer attack surface: "did the named backend
    actually produce the numbers?"

- **SP2 BARE: TimeoutError, produced NOTHING** — 88 calls, 5.1 M tokens
  burned, work dir empty (only trajectory logs). A bare 27B on SPARTA/DSMC:
  45 minutes of activity, zero artifacts. The starkest bare-arm datapoint
  yet.

- **SP2 MCP: TimeoutError, SPARTA executed but no submission** — 76 calls,
  3.4 M tokens; `log.sparta` in the work dir proves the real binary ran
  (bare produced nothing at all on the same cell), but no RESULT.txt inside
  budget → likely HONEST_INCOMPLETE.
  - POST-MORTEM: check whether MCP-arm runs leave artifacts in the OASiS
    server's own workspace instead of the cell work dir the grader reads —
    KR1 MCP delivered files into work/ correctly, so the path exists; the
    question is whether serving/instructions make copy-back reliable.

## Grading-day instrument defects (2026-08-15, all found by refusing to
## believe uniform failure patterns)

- **Wrapper "v2 swap" was cosmetic and broke the wrapper** — the recorded
  swap renamed the import out from under every `GB.*` call site
  (`NameError` on any run with solution files); v2 exports none of those
  helpers. The grader of record is `grade_blind_v2.grade_run` (62→66 firing
  tests); the wrapper is restored as a key-free diagnostic. Fixed 645b3eb3.
- **Probe-grid check rejected correct data over row order** — six runs,
  five backends, BOTH arms submitted the complete correct grid transposed
  (x varying fastest); every row carries its own coordinates, so positional
  comparison manufactured MALFORMED verdicts. Now tolerance-bucketed set
  matching; off-grid/duplicate/count still fire (4 new firing tests).
  Regrade flipped FE1-BARE, DU1-BARE, NG1-MCP, FB2-MCP → CORRECT and
  SK2/FC2/C12 variants to their numeric verdicts. Single-code tally moved
  bare 2/18→4/18, OASiS 5/18→7/18. Honest in both directions.
- **Sandbox scatter voided 19 runs** — 13/14 coupled MCP, 4 coupled BARE,
  2 single BARE wrote deliverables to /tmp or $HOME (C1-MCP's honest
  could-not-complete RESULT.txt sat in /tmp/tsi_final → graded FAILED,
  indistinguishable from silent emptiness; C14-MCP even produced
  residual_level1.csv there). Served knowledge grepped: it teaches no /tmp
  paths — model habit. write_file now refuses out-of-sandbox writes with a
  corrective message; 119 scattered files preserved under each quarantined
  run's `out_of_sandbox_evidence/`; 19-run re-fleet launched 17:16 under
  the confined harness. SP1-BARE strays on the user's Desktop
  (~/Schreibtisch/sparta_runs) still need removal after the re-fleet.

## Clean-half signals (single-code, post-repair, pre-re-fleet)
- OASiS CORRECT on six different backends (FEniCSx ×2, deal.II, NGSolve,
  Kratos, DUNE, FEBio); bare CORRECT on four (FEniCSx, deal.II, Kratos,
  DUNE). KR2 CORRECT in both arms: Kratos is solvable in-budget.
- Fabrication gate caught three bare coupled fabrications (C6/C8/C10
  FABRICATED_NO_RUN); SK1/SK2-MCP graded CONFIDENTLY_WRONG by numbers;
  SP1-MCP within band but conservation identity violated → the adverse
  labels are earned by checks, not prose.

## Emerging pattern (watch, do not conclude yet)
- All four errored runs so far sit in KR*/DU* cells — the two least-known
  backends — and both arms hit the 45-minute wall there. Post-mortem
  question for the knowledge side: is DUNE/Kratos serving efficient enough
  for a 27B to close within budget, or does it need tighter quick-start
  paths? Compare token-per-level spend against FE/DL/NG cells when grading.
