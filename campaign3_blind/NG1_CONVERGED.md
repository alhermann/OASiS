# NG1 — converged, and what it cost to get there

Development cell. These numbers are burnt and can never be the paper's.

## The ceiling, established first

Before touching OASiS I solved NG1 by hand in NGSolve and graded it:
**CORRECT, order 2.0687, r² 0.9994, evidence grade 1.** So neither the task nor
the grader was the obstacle. Every later claim is measured against that
reference field (max|u| = 7.196098e-02 at maxh = 1/64).

## Two root causes, each reproduced by execution

**1. A matrix coefficient that silently becomes a scalar.**

| spelling | what NGSolve builds | max\|u\| | graded order |
|---|---|---|---|
| `CoefficientFunction((3,-1,-1,2), dims=(2,2))` | dims `[2,2]` — matrix | 7.196098e-02 | 2.069 |
| `CoefficientFunction([[3,-1],[-1,2]])` | dims `[]` — **scalar** | 5.926314e-02 | 0.07 |
| `CoefficientFunction(3.0)` | scalar | 5.926314e-02 | — bit-identical |

The nested list is what anyone who has used numpy writes. The form assembles,
the solve succeeds, the refinement study looks orderly, and the anisotropy is
gone. Two runs in *different arms* landed on 5.926314e-02 to seven digits.

**2. A source term built from element-local coordinates.**

`specialcf.xref(dim)` is position inside the reference element. One run bound
`x`, `y` to it — its own comment read "reference coordinates which equal
physical coords for unit square" — and built the task's source from them.
Replaying that script reproduces its submitted 6.158955e-02 exactly; deleting
only those two lines gives 7.196098e-02. Ruled out first, by execution: the
source string is character-identical to the task's, K was a true matrix, the
probe list and ordering matched, evaluation used `gfu(mesh(px,py))`, and all
four geometry/Dirichlet spellings give the same answer with the correct K.

## What OASiS was doing wrong

Nothing served said how to write a matrix coefficient, and no query an NG1
agent forms from its own task text reached the one fact that decided the cell:

| query | chars | reached the fact |
|---|---|---|
| `pitfalls, ngsolve` (66% of real calls) | 7,047 | no |
| `pitfalls, physics='anisotropic_diffusion'` | 5,685 | no — **fewer than unfiltered** |
| `pitfalls, signal='anisotropic conductivity tensor'` | 6,949 | no |
| `physics, physics='anisotropic_diffusion'` | 30,819 | no |

The agent was punished for using the task's vocabulary. And "element-local"
appeared **zero** times in everything the failing run was served — that warning
lived only inside `verify_pde_consistency`'s failure text, a tool the run never
called.

## Repairs, all verified against the real runs

- both facts served in the house style with their measured numbers;
- ngsolve aliases the task's vocabulary onto its own bucket names, so
  "anisotropic diffusion" / "diffusion" / "laplace" reach the poisson
  knowledge. Aliases only — an unknown physics still returns `{}`;
- `pitfall_index.narrow()` no longer empties the answer when a physics name
  matches no bucket, which its own docstring had always promised;
- a static `specialcf.xref` check in `audit_results`, the tool runs actually
  call. It **flags** the failing seed 101 and is **silent** on every CORRECT
  run (BARE 102, BARE 104, MCP 103, MCP 104).

Two harness blockers found on the way, both from the new reproducible-build
work: the source snapshots copied `scripts/blind_keys.py` where the leak
scanner flags it, so **every paid run refused at preflight**; and the snapshot
install caught only `FileExistsError` where a directory rename raises
`ENOTEMPTY`, so **five of six parallel cells died before any paid call**.

## Result

Round 2 (seeds 105–107, one build hash, both primitives) taught **nothing new**:
all three OASiS runs were served the xref warning, none used `specialcf.xref`,
all three graded CORRECT. That is the paper's §3.2 criterion met for this cell.

Paired over the six seeds where both arms have a verdict:

| | OASiS | bare |
|---|---|---|
| correct | 5/6 | 3/6 |
| discordant | OASiS-only 2 | bare-only 0 |

n = 6 on one burnt cell. It is a development signal, not a rate, and McNemar on
two discordant pairs is not significant. Historical NG1 for comparison, same
gate: bare 10/14, OASiS 10/19.
