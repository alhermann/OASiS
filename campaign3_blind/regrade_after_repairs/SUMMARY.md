# Regrade under the repaired evidence gate — DEVELOPMENT numbers

Seeds 2-11, 14, 15 at 27B, both arms, 750 graded rows. These are
DEVELOPMENT instances: burnt, never the paper's numbers.

| slice | grade | bare | OASiS | uplift |
|---|---|---|---|---|
| single-code | 1 | 44/192 = 22.9% | 80/192 = 41.7% | +18.7 |
| single-code | all | 44/216 = 20.4% | 80/216 = 37.0% | +16.7 |
| coupled | 1 | 1/138 = 0.7% | 2/136 = 1.5% | +0.7 |
| coupled | all | 1/160 = 0.6% | 2/158 = 1.3% | +0.6 |

## Fabrication (requires positive evidence of invention)

- bare: 5/376 = 1.3%
- OASiS: 2/374 = 0.5%

## Outcome mix

| outcome | bare | OASiS |
|---|---|---|
| COMPLETED_UNPHYSICAL | 49 | 39 |
| CONFIDENTLY_WRONG | 15 | 10 |
| CORRECT | 45 | 82 |
| FABRICATED_NO_RUN | 5 | 2 |
| FAILED | 50 | 47 |
| HONEST_INCOMPLETE | 111 | 119 |
| MALFORMED_SUBMISSION | 55 | 29 |
| NOT_GRADED(band-only) | 46 | 46 |

## The three coupled CORRECT runs

- `C2_MCP_seed15` order 1.955; coupling PROVEN; 4C=PROVEN, kratos=PROVEN; overall attribution NOT_PROVEN
- `C8_BARE_seed4` order 2.017; coupling PROVEN; ngsolve=PROVEN, kratos=PROVEN; overall attribution NOT_PROVEN
- `C8_MCP_seed4` order 2.106; coupling PROVEN; ngsolve=PROVEN, kratos=PROVEN; overall attribution NOT_PROVEN

One of them is a BARE run (C8, NGSolve+Kratos), which contradicts the
target wording 'coupling: bare completes none and fabricates'. It has six
separate per-side logs, NDOF growing on both sides (54/72 -> 187/255 ->
693/957) and three distinct residual histories, so it is a real solve. It
had been labelled a forgery by the cv rule at 7.9e-6 against a 1e-5
threshold whose calibration turned out to be one run's own scatter.

For all three, per-code attribution is UNPROVEN: the two codes' signatures
are not carried by separate files, so the coupling claim cannot rest on
these runs. Only a draw whose task text demands captured per-code output
can support it, which the current task builder emits.
