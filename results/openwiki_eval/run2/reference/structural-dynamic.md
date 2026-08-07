---
type: input parameter reference
title: STRUCTURAL DYNAMIC
description: Required and optional structural mechanics control keys, defaults, enum values, and solver/material/condition relationships for accepted structure decks.
tags: [input-format, structural]
---

# `STRUCTURAL DYNAMIC`

`STRUCTURAL DYNAMIC` and its subsections are registered in `src/inpar/4C_inpar_structure.cpp`. The section is optional to the generic parser, but a `PROBLEMTYPE: "Structure"` deck needs it to choose the structural time integrator, nonlinear solve controls, output cadence, and linear solver. Theory links: [Structural mechanics theory](../theory/structural.md).

## Main section parameters

All keys are optional unless noted; omitted keys use the listed default.

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `INT_STRATEGY` | enum | `Standard` | `Old`, `Standard`. |
| `TIME_ADAPTIVITY` | bool | `false` | Enables adaptive time integration. |
| `DYNAMICTYPE` | enum | `GenAlpha` | `Statics`, `GenAlpha`, `GenAlphaLieGroup`, `OneStepTheta`, `ExplicitEuler`, `CentrDiff`, `AdamsBashforth2`, `AdamsBashforth4`. |
| `PRESTRESS` | enum | `none` | `none`, `None`, `NONE`, `mulf`, `Mulf`, `MULF`, `Material_Iterative`, `MATERIAL_ITERATIVE`, `material_iterative`. |
| `PRESTRESSTIME` | double | `0.0` | Time to switch from pre- to post-stressing. |
| `PRESTRESSTOLDISP` | double | `1e-9` | Displacement-norm tolerance during prestressing. |
| `PRESTRESSMINLOADSTEPS` | int | `0` | Minimum load steps during prestressing. |
| `RESULTSEVERY` | int | `0` | Old HDF5/contact output cadence in steps. |
| `RESEVERYERGY` | int | `0` | Energy output cadence. Name is spelled exactly as source. |
| `RESTARTEVERY` | int | `0` | Restart output cadence. |
| `CALC_ACC_ON_RESTART` | bool | `false` | Compute initial acceleration for restart dynamics. |
| `OUTPUT_STEP_OFFSET` | int | `0` | Offset added to written step ids. |
| `TIMESTEP` | double | `0.05` | Time/load-step size. |
| `NUMSTEP` | int | `200` | Maximum number of steps. |
| `TIMEINIT` | double | `0.0` | Initial time. |
| `MAXTIME` | double | `5.0` | Maximum simulated time. |
| `DAMPING` | enum | `None` | `None`, `Rayleigh`, `Material`. |
| `M_DAMP` | double | `-1.0` | Rayleigh mass damping coefficient when `DAMPING: Rayleigh`. |
| `K_DAMP` | double | `-1.0` | Rayleigh stiffness damping coefficient. |
| `TOLDISP` | double | `1.0E-10` | Newton displacement-norm tolerance. |
| `NORM_DISP` | enum | `Abs` | `Abs`, `Rel`, `Mix`. |
| `TOLRES` | double | `1.0E-08` | Newton residual-force tolerance. |
| `NORM_RESF` | enum | `Abs` | `Abs`, `Rel`, `Mix`. |
| `TOLPRE` | double | `1.0E-08` | Pressure-norm tolerance. |
| `NORM_PRES` | enum | `Abs` | `Abs`. |
| `TOLINCO` | double | `1.0E-08` | Incompressible residual tolerance. |
| `NORM_INCO` | enum | `Abs` | `Abs`. |
| `NORMCOMBI_DISPPRES` | enum | `And` | `And`, `Or`. |
| `NORMCOMBI_RESFINCO` | enum | `And` | `And`, `Or`. |
| `NORMCOMBI_RESFDISP` | enum | `And` | `And`, `Or`. |
| `STC_SCALING` | enum | `Inactive` | `Inactive`, `Symmetric`, `Right`. Thin-shell scaled director conditioning. |
| `STC_LAYER` | int | `1` | Number of STC layers. |
| `PTCDT` | double | `0.1` | Pseudo time step for PTC-stabilized Newton. |
| `TOLCONSTR` | double | `1.0E-08` | Constraint error norm tolerance. |
| `TOLCONSTRINCR` | double | `1.0E-08` | Constraint Lagrange-multiplier increment tolerance. |
| `MAXITER` | int | `50` | Maximum Newton iterations. |
| `MINITER` | int | `0` | Minimum Newton iterations. |
| `ITERNORM` | enum | `L2` | `L1`, `L2`, `Rms`, `Inf`. |
| `DIVERCONT` | enum | `stop` | `stop`, `continue`, `repeat_step`, `halve_step`, `adapt_step`, `rand_adapt_step`, `rand_adapt_step_ele_err`, `repeat_simulation`, `adapt_penaltycontact`. |
| `MAXDIVCONREFINEMENTLEVEL` | int | `10` | Maximum time-step halvings after divergence. |
| `NLNSOL` | enum | `fullnewton` | `vague`, `fullnewton`, `modnewton`, `lsnewton`, `ptc`, `newtonlinuzawa`, `augmentedlagrange`, `noxnln`, `singlestep`. |
| `LSMAXITER` | int | `30` | Maximum line-search steps. |
| `ALPHA_LS` | double | `0.5` | Line-search step reduction factor. |
| `SIGMA_LS` | double | `1.e-4` | Sufficient descent factor. |
| `MATERIALTANGENT` | enum string | `analytical` | `analytical`, `finitedifferences`. |
| `LOADLIN` | bool | `false` | Linearize external follower loads. Required for structural `orthopressure` Neumann loads according to condition docs. |
| `MASSLIN` | enum | `none` | `none`, `rotations`. |
| `NEGLECTINERTIA` | bool | `false` | Neglect inertia. |
| `PREDICT` | enum | `ConstDis` | `Vague`, `ConstDis`, `ConstVel`, `ConstAcc`, `ConstDisVelAcc`, `TangDis`, `TangDisConstFext`, `ConstDisPres`, `ConstDisVelAccPres`. |
| `UZAWAPARAM` | double | `1.0` | Uzawa parameter for Lagrange multipliers. |
| `UZAWATOL` | double | `1.0E-8` | Uzawa tolerance. |
| `UZAWAMAXITER` | int | `50` | Maximum Uzawa iterations. |
| `UZAWAALGO` | enum | `direct` | `uzawa`, `simple`, `direct`. |
| `ADAPTCONV` | bool | `false` | Adaptive linear-solver tolerance for nonlinear solve. |
| `ADAPTCONV_BETTER` | double | `0.1` | Linear solver must be this much better than current nonlinear residual. |
| `LUMPMASS` | bool | `false` | Lump mass matrix for explicit integration. |
| `MODIFIEDEXPLEULER` | bool | `true` | Use modified explicit Euler. |
| `LINEAR_SOLVER` | int | `-1` | Solver block id; set to `1` and define `SOLVER 1` for ordinary implicit/static runs. |
| `MIDTIME_ENERGY_TYPE` | enum | `vague` | `vague`, `imrLike`, `trLike`. |
| `INITIALDISP` | enum | `zero_displacement` | `zero_displacement`, `displacement_by_function`. |
| `STARTFUNCNO` | int | `-1` | Function id for initial displacement. |

## Time-adaptivity subsections

`STRUCTURAL DYNAMIC/TIMEADAPTIVITY` is optional.

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `KIND` | enum | `None` | `None`, `ZienkiewiczXie`, `JointExplicit`, `AdamsBashforth2`, `ExplicitEuler`, `CentralDifference`. |
| `OUTSYSPERIOD` | double | `0.0` | Period for writing system vectors. |
| `OUTSTRPERIOD` | double | `0.0` | Period for stress/strain output. |
| `OUTENEPERIOD` | double | `0.0` | Period for energy output. |
| `OUTRESTPERIOD` | double | `0.0` | Period for restart output. |
| `OUTSIZEEVERY` | int | `0` | Write step size every given time step. |
| `STEPSIZEMAX` | double | `0.0` | Maximum permitted time-step size. |
| `STEPSIZEMIN` | double | `0.0` | Minimum permitted time-step size. |
| `SIZERATIOMAX` | double | `0.0` | Maximum ratio to previous step size. |
| `SIZERATIOMIN` | double | `0.0` | Minimum ratio to previous step size. |
| `SIZERATIOSCALE` | double | `0.9` | Safety scale for theoretical optimal step. |
| `LOCERRNORM` | enum | `Vague` | `Vague`, `L1`, `L2`, `Rms`, `Inf`. |
| `LOCERRTOL` | double | `0.0` | Target local error tolerance. |
| `ADAPTSTEPMAX` | int | `0` | Maximum step-size reduction attempts. |

`STRUCTURAL DYNAMIC/TIMEADAPTIVITY/JOINT EXPLICIT` is used when `KIND: JointExplicit`; its `DYNAMICTYPE` is limited to `ExplicitEuler`, `CentrDiff`, `AdamsBashforth2`, `AdamsBashforth4`, with default `CentrDiff`. It also accepts `LINEAR_SOLVER`, `INT_STRATEGY: Standard`, `LUMPMASS`, `DAMPING`, `M_DAMP`, and `K_DAMP`.

## Integrator-specific subsections

| Section | Key | Type | Default | Notes |
|---|---|---:|---:|---|
| `STRUCTURAL DYNAMIC/GENALPHA` | `GENAVG` | enum | `TrLike` | `Vague`, `ImrLike`, `TrLike`. |
| `STRUCTURAL DYNAMIC/GENALPHA` | `BETA` | double | `-1.0` | Generalized-alpha factor in `(0, 1/2]`; negative means internal choice. |
| `STRUCTURAL DYNAMIC/GENALPHA` | `GAMMA` | double | `-1.0` | Generalized-alpha factor in `(0, 1]`. |
| `STRUCTURAL DYNAMIC/GENALPHA` | `ALPHA_M` | double | `-1.0` | Generalized-alpha factor in `[0,1)`. |
| `STRUCTURAL DYNAMIC/GENALPHA` | `ALPHA_F` | double | `-1.0` | Generalized-alpha factor in `[0,1)`. |
| `STRUCTURAL DYNAMIC/GENALPHA` | `RHO_INF` | double | `1.0` | Spectral radius in `[0,1]`. |
| `STRUCTURAL DYNAMIC/ONESTEPTHETA` | `THETA` | double | `0.5` | One-step-theta factor in `(0,1]`. |
| `STRUCTURAL DYNAMIC/ERROR EVALUATION` | `EVALUATE_ERROR_ANALYTICAL_REFERENCE` | bool | `false` | Compare with analytical displacement function. |
| `STRUCTURAL DYNAMIC/ERROR EVALUATION` | `ANALYTICAL_DISPLACEMENT_FUNCTION` | int | `-1` | Function id for analytical displacement. |

## Runtime strategy mapping

The structural time-integration factory consumes `INT_STRATEGY` and `DYNAMICTYPE` after parsing:

| Input choice | Runtime family |
|---|---|
| `INT_STRATEGY: Standard` with `DYNAMICTYPE: Statics`, `GenAlpha`, `GenAlphaLieGroup`, `OneStepTheta`, or prestress-related setup | Builds an implicit strategy object; each step assembles a nonlinear residual/Jacobian and uses `LINEAR_SOLVER`. |
| `DYNAMICTYPE: ExplicitEuler`, `CentrDiff`, `AdamsBashforth2`, `AdamsBashforth4` | Builds an explicit strategy object; mass lumping and stable `TIMESTEP` become critical. Explicit FSI combinations are rejected by the factory. |
| `DYNAMICTYPE: GenAlpha` or `OneStepTheta` | Builds dynamic data objects that read `GENALPHA` or `ONESTEPTHETA` subsections. |
| `INT_STRATEGY: Old` | Routes to legacy structural integration paths used by older examples; prefer `Standard` for new decks unless matching an old accepted test. |

## Minimal static structure pattern

Derived from `tests/input_files/beam3r_line2_static_test1.4C.yaml`:

```yaml
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
DISCRETISATION:
  NUMFLUIDDIS: 0
  NUMALEDIS: 0
  NUMTHERMDIS: 0
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 0.1
  NUMSTEP: 10
  MAXTIME: 1
  TOLRES: 1e-06
  MAXITER: 15
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
```

Then add [geometry and elements](geometry-and-elements.md), [materials](materials.md), and [conditions](conditions.md). For a solid or beam element, every `MAT <id>` in `STRUCTURE ELEMENTS` must have a matching material entry.
