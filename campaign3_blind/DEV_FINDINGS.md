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

## Emerging pattern (watch, do not conclude yet)
- All four errored runs so far sit in KR*/DU* cells — the two least-known
  backends — and both arms hit the 45-minute wall there. Post-mortem
  question for the knowledge side: is DUNE/Kratos serving efficient enough
  for a 27B to close within budget, or does it need tighter quick-start
  paths? Compare token-per-level spend against FE/DL/NG cells when grading.
