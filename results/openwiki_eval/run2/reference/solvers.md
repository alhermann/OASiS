---
type: input parameter reference
title: Linear solver sections
description: Grammar for `SOLVER 1` through `SOLVER 9` and how physics sections reference solver blocks.
tags: [input-format, solvers]
---

# Linear solver sections

4C registers nine solver blocks, `SOLVER 1` through `SOLVER 9`, in `src/core/linear_solver/src/method/4C_linear_solver_method_input.cpp`. Each block is optional at parser level, but any positive `LINEAR_SOLVER` value in a physics section must point to a configured block in a useful deck.

## Shape

```yaml
STRUCTURAL DYNAMIC:
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
```

`SOLVER` is required inside a present solver block. All other keys below are optional.

## Parameters

| Key | Type | Required | Default | Accepted values or notes |
|---|---:|---:|---:|---|
| `SOLVER` | enum | yes | none | Linear solver backend. Tests commonly use `UMFPACK`; source type is `Core::LinearSolver::SolverType`. |
| `AZSOLVE` | enum | no | `GMRES` | Iterative algorithm: `CG`, `GMRES`, `BiCGSTAB`. |
| `AZPREC` | enum | no | `ILU` | Internal preconditioner: `ILU`, `MueLu`, `AMGnxn`, `Teko`. Used when operator supports Trilinos and no external preconditioner is passed. |
| `IFPACK_XML_FILE` | optional path | no | none | XML file for Trilinos/Ifpack preconditioner settings. |
| `AZITER` | int | no | `1000` | Maximum iterative-solver iterations. |
| `AZTOL` | double | no | `1e-8` | Residual norm convergence tolerance. |
| `AZCONV` | enum | no | `AZ_r0` | Residual scaling: `AZ_r0`, `AZ_noscaled`. |
| `AZOUTPUT` | int | no | `0` | Iteration interval for progress output. |
| `AZREUSE` | int | no | `0` | Recompute preconditioner after this many nonlinear iterations; recomputed at each nonlinear solve start. |
| `REUSE_STALL_ITER` | int | no | `50` | Linear-iteration count that declares nonlinear iteration stalled and forces preconditioner recomputation. |
| `AZSUB` | int | no | `50` | GMRES Krylov subspace size before restart. |
| `THROW_IF_UNCONVERGED` | bool | no | `true` | If true, throw when the iterative solver does not converge; if false, warn. |
| `SOLVER_XML_FILE` | optional path | no | none | XML file defining an iterative solver. |
| `MUELU_XML_FILE` | optional path | no | none | XML file defining a MueLu preconditioner. |
| `TEKO_XML_FILE` | optional path | no | none | XML file defining a Teko preconditioner. |
| `NAME` | string | no | `No_name` | Human-readable solver block name. |
| `AMGNXN_TYPE` | string | no | `AMG(BGS)` | Prebuilt AMGnxn preconditioner name; `XML` means configured by XML. |
| `AMGNXN_XML_FILE` | optional path | no | none | XML file defining AMGnxn preconditioner. |

## Referencing solver blocks

Common references:

- `STRUCTURAL DYNAMIC/LINEAR_SOLVER`
- `FLUID DYNAMIC/LINEAR_SOLVER`
- `SCALAR TRANSPORT DYNAMIC/LINEAR_SOLVER`
- `CONTACT DYNAMIC/LINEAR_SOLVER`
- `FLUID DYNAMIC/VELGRAD_PROJ_SOLVER`
- projection or auxiliary solvers in specialized subsections

`-1` is the common default for “not set”. If the solver code reaches a linear solve with an unset or invalid solver id, it throws a runtime error; examples therefore set `LINEAR_SOLVER: 1` and provide `SOLVER 1`.

## Direct vs iterative first attempts

For small generated-domain tests, direct `UMFPACK` is the least parameterized first attempt:

```yaml
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Direct_test_solver"
```

For larger MPI runs or when direct solver support is unavailable, use an iterative solver and set at least `AZSOLVE`, `AZPREC`, `AZITER`, and `AZTOL`, optionally with XML files for Trilinos preconditioners. Keep XML files in paths resolvable relative to the input deck.
