---
type: input parameter reference
title: SCALAR TRANSPORT DYNAMIC
description: Scalar transport dynamic controls, nonlinear/stabilization subsections, materials, conditions, and first-attempt deck settings.
tags: [input-format, scalar-transport]
---

# `SCALAR TRANSPORT DYNAMIC`

`SCALAR TRANSPORT DYNAMIC` is registered in `src/inpar/4C_inpar_scatra.cpp`. A `PROBLEMTYPE: "Scalar_Transport"` deck normally needs this section, a transport discretization, scalar material(s), boundary conditions, and a solver block. Theory links: [Scalar transport theory](../theory/scalar-transport.md).

## Main section parameters

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `SOLVERTYPE` | enum | `linear_full` | `linear_full`, `linear_incremental`, `nonlinear`, `nonlinear_multiscale_macrotomicro`, `nonlinear_multiscale_macrotomicro_aitken`, `nonlinear_multiscale_macrotomicro_aitken_dofsplit`, `nonlinear_multiscale_microtomacro`. |
| `TIMEINTEGR` | enum | `One_Step_Theta` | `Stationary`, `One_Step_Theta`, `BDF2`, `Gen_Alpha`. |
| `MAXTIME` | double | `1000.0` | Total simulation time. |
| `NUMSTEP` | int | `20` | Total number of time steps. |
| `TIMESTEP` | double | `0.1` | Time increment. |
| `THETA` | double | `0.5` | One-step-theta factor. |
| `ALPHA_M` | double | `0.5` | Generalized-alpha factor. |
| `ALPHA_F` | double | `0.5` | Generalized-alpha factor. |
| `GAMMA` | double | `0.5` | Generalized-alpha factor. |
| `RESULTSEVERY` | int | `1` | Solution output cadence. |
| `RESTARTEVERY` | int | `1` | Restart output cadence. |
| `MATID` | int | `-1` | Material id for automatic mesh generation. |
| `VELOCITYFIELD` | enum | `zero` | `zero`, `function`, `Navier_Stokes`. |
| `VELFUNCNO` | int | `-1` | Function id for prescribed velocity field. |
| `INITIALFIELD` | enum | `zero_field` | `zero_field`, `field_by_function`, `field_by_condition`, `disturbed_field_by_function`, `1D_DISCONTPV`, `FLAME_VORTEX_INTERACTION`, `RAYTAYMIXFRAC`, `L_shaped_domain`, `facing_flame_fronts`, `oracles_flame`, `high_forced_hit`, `low_forced_hit`, `algebraic_field_dependence`. |
| `INITFUNCNO` | int | `-1` | Function id for initial field. |
| `SPHERICALCOORDS` | bool | `false` | Use spherical coordinates. |
| `CALCERROR` | enum | `No` | `No`, `Kwok_Wu`, `ConcentricCylinders`, `Electroneutrality`, `error_by_function`, `error_by_condition`, `SphereDiffusion`, `AnalyticSeries`. |
| `CALCERRORNO` | int | `-1` | Error function id. |
| `CALCFLUX_DOMAIN` | enum | `No` | `No`, `total`, `diffusive`. |
| `CALCFLUX_DOMAIN_LUMPED` | bool | `true` | Lumped domain flux calculation. |
| `CALCFLUX_BOUNDARY` | enum | `No` | `No`, `total`, `diffusive`, `convective`. |
| `CALCFLUX_BOUNDARY_LUMPED` | bool | `true` | Lumped boundary flux calculation. |
| `WRITEFLUX_IDS` | string | `-1` | Space-separated scalar ids for flux output. |
| `OUTPUTSCALARS` | enum | `none` | `none`, `entire_domain`, `by_condition`, `entire_domain_and_by_condition`. |
| `OUTPUTSCALARSMEANGRAD` | bool | `false` | Output mean gradient. |
| `OUTINTEGRREAC` | bool | `false` | Output integral reaction values. |
| `OUTPUT_GMSH` | bool | `false` | Write Gmsh postprocessing files. |
| `MATLAB_STATE_OUTPUT` | bool | `false` | Write state to Matlab file. |
| `CONVFORM` | enum | `convective` | `convective`, `conservative`. On deforming domains, use conservative unless `IS_INTENSIVE_SCALAR: true`. |
| `NEUMANNINFLOW` | bool | `false` | Activate potential Neumann inflow terms. |
| `CONV_HEAT_TRANS` | bool | `false` | Activate convective heat-transfer boundary conditions. |
| `SKIPINITDER` | bool | `false` | Skip initial time derivative computation. |
| `IS_INTENSIVE_SCALAR` | bool | `false` | Treat scalar as intensive/material quantity; allows non-conservative form on deforming domains. |
| `FSSUGRDIFF` | enum | `No` | `No`, `artificial`, `Smagorinsky_all`, `Smagorinsky_small`. |
| `MESHTYING` | enum | `no` | `no`, `Condensed_Smat`, `Condensed_Bmat`, `Condensed_Bmat_merged`. |
| `FIELDCOUPLING` | enum | `matching` | `matching`, `volmortar`. |
| `LINEAR_SOLVER` | int | `-1` | Solver id for scalar transport and electrochemistry. |
| `L2_PROJ_LINEAR_SOLVER` | int | `-1` | Solver id for L2 projection subproblems. |
| `EQUILIBRATION` | enum | `none` | Global-system equilibration method. |
| `MATRIXTYPE` | enum | `sparse` | Matrix representation. |
| `NATURAL_CONVECTION` | bool | `false` | Include natural convection effects. |
| `FDCHECK` | enum | `none` | `none`, `global`, `global_extended`, `local`. |
| `FDCHECKEPS` | double | `1.e-6` | Perturbation for finite-difference check. |
| `FDCHECKTOL` | double | `1.e-6` | Relative finite-difference tolerance. |
| `COMPUTEINTEGRALS` | enum | `none` | `none`, `initial`, `repeated`. |
| `PADAPTIVITY` | bool | `false` | Activate p-adaptivity. |
| `PADAPTERRORTOL` | double | `1e-6` | Error tolerance for elemental degree variation. |
| `PADAPTERRORBASE` | double | `1.66` | Error tolerance base for degree variation. |
| `PADAPTDEGREEMAX` | int | `4` | Maximum shape-function degree. |
| `SEMIIMPLICIT` | bool | `false` | Semi-implicit reaction calculation. |
| `OUTPUTLINSOLVERSTATS` | bool | `false` | Output linear solver stats to CSV. |
| `OUTPUTNONLINSOLVERSTATS` | bool | `false` | Output nonlinear solver stats to CSV. |
| `NULLSPACE_POINTBASED` | bool | `false` | Point-based null-space calculation. |

## Nonlinear subsection

Section: `SCALAR TRANSPORT DYNAMIC/NONLINEAR`.

| Key | Type | Default | Meaning |
|---|---:|---:|---|
| `ITEMAX_OUTER` | int | `10` | Maximum outer nonlinear iterations. |
| `CONVTOL_OUTER` | double | `1e-6` | Outer nonlinear convergence tolerance. |
| `ITEMAX_INNER` | int | source default | Inner iteration maximum when present in source. |
| `CONVTOL_INNER` | double | source default | Inner convergence tolerance. |
| `EXPLPREDICT` | bool | `false` | Explicit predictor. |
| `ADAPTCONV` | bool | `false` | Adaptive linear-solver tolerance. |
| `ADAPTCONV_BETTER` | double | `0.1` | Linear solve relative improvement. |

## Coupling and force subsections

- `SCALAR TRANSPORT DYNAMIC/STABILIZATION` is the canonical place for scalar stabilization controls; source also composes shared scatra stabilization specs.
- `SCALAR TRANSPORT DYNAMIC/ARTERY COUPLING` controls homogenized coupling between arterial and continuum transport fields.
- `SCALAR TRANSPORT DYNAMIC/EXTERNAL FORCE` controls external force terms. Its high-level keys include `EXTERNAL_FORCE` bool default `false`, `FORCE_FUNCTION_ID` int default `-1`, and `INTRINSIC_MOBILITY_FUNCTION_ID` int default `-1`.

## Minimal scalar diffusion pattern

From `tests/input_files/scatra_1D_line2_diffnumdof.4C.yaml`:

```yaml
PROBLEM SIZE:
  DIM: 1
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  SOLVERTYPE: "nonlinear"
  NUMSTEP: 200
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 0.01
```

Then add `TRANSPORT ELEMENTS` or `TRANSPORT DOMAIN`, and transport Dirichlet/Neumann conditions. `MAT_scatra/DIFFUSIVITY` is required unless a defaulted value is accepted in the compiled spec; set it explicitly for checkable decks.
