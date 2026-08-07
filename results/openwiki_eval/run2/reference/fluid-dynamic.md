---
type: input parameter reference
title: FLUID DYNAMIC
description: Fluid control sections, accepted keys, defaults, enum values, stabilization settings, and minimal Stokes/Navier-Stokes deck patterns.
tags: [input-format, fluid]
---

# `FLUID DYNAMIC`

`FLUID DYNAMIC` is registered in `src/inpar/4C_inpar_fluid.cpp`. A `PROBLEMTYPE: "Fluid"` deck normally needs this section, a fluid mesh, a fluid material, boundary conditions, and a solver block. Theory links: [Fluid theory](../theory/fluid.md).

## Main section parameters

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `PHYSICAL_TYPE` | enum | `Incompressible` | `Incompressible`, `Weakly_compressible`, `Weakly_compressible_stokes`, `Weakly_compressible_dens_mom`, `Weakly_compressible_stokes_dens_mom`, `Artificial_compressibility`, `Varying_density`, `Loma`, `Temp_dep_water`, `Boussinesq`, `Stokes`, `Oseen`. |
| `LINEAR_SOLVER` | int | `-1` | Solver block id for fluid dynamics. |
| `SIMPLER_SOLVER` | int | `-1` | Secondary solver id for selected block/SIMPLER cases. |
| `WSS_TYPE` | enum | `Standard` | `Standard`, `Aggregation`, `Mean`. Wall shear stress calculation. |
| `WSS_ML_AGR_SOLVER` | int | `-1` | ML solver for residual-based WSS smoothing. |
| `TIMEINTEGR` | enum | `One_Step_Theta` | `Stationary`, `Np_Gen_Alpha`, `Af_Gen_Alpha`, `One_Step_Theta`, `BDF2`. |
| `OST_CONT_PRESS` | enum | source default | One-step-theta option for continuity and pressure. |
| `NONLINITER` | enum | source default | Nonlinear iteration scheme. |
| `PREDICTOR` | enum string | `steady_state` | `steady_state`, `zero_acceleration`, `constant_acceleration`, `constant_increment`, `explicit_second_order_midpoint`, `TangVel`. |
| `CONVCHECK` | enum | `L_2_norm` | Convergence norm. |
| `INCONSISTENT_RESIDUAL` | bool | `false` | Skip residual evaluation after convergence for speed. |
| `INITIALFIELD` | enum | `zero_field` | `zero_field`, `field_by_function`, `disturbed_field_from_function`, `FLAME_VORTEX_INTERACTION`, `BELTRAMI-FLOW`, `KIM-MOIN-FLOW`, `hit_comte_bellot_corrsin_initial_field`, `forced_hit_simple_algebraic_spectrum`, `forced_hit_numeric_spectrum`, `forced_hit_passive`, `channel_weakly_compressible`. |
| `OSEENFIELDFUNCNO` | int | `-1` | Oseen advective-field function id. |
| `LIFTDRAG` | bool | `false` | Calculate lift and drag along specified boundary. |
| `CONVFORM` | enum string | `convective` | `convective`, `conservative`. |
| `NONLINEARBC` | bool | `false` | Check for nonlinear boundary conditions. |
| `MESHTYING` | enum | `no` | `no`, `Condensed_Smat`, `Condensed_Bmat`, `Condensed_Bmat_merged`. |
| `GRIDVEL` | enum | source default `BE` | Grid-velocity scheme from displacements. |
| `ALLDOFCOUPLED` | bool | `true` | Couple pressure and velocity dofs. |
| `CALCERROR` | enum | `no` | Error-calculation mode. |
| `CALCERRORFUNCNO` | int | `-1` | Error function id. |
| `CORRTERMFUNCNO` | int | `-1` | Weakly-compressible correction-term function id. |
| `BODYFORCEFUNCNO` | int | `-1` | Weakly-compressible body-force function id. |
| `STAB_DEN_REF` | double | `0.0` | HDG density stabilization reference. |
| `STAB_MOM_REF` | double | `0.0` | HDG momentum stabilization reference. |
| `VARVISCFUNCNO` | int | `-1` | Variable-viscosity function id. |
| `PRESSAVGBC` | bool | `false` | Impose element-average pressure boundary condition. |
| `REFMACH` | double | `1.0` | Reference Mach number. |
| `BLOCKMATRIX` | bool | `false` | Assemble into sparse block matrix. |
| `ADAPTCONV` | bool | `false` | Adaptive linear-solver tolerance. |
| `ADAPTCONV_BETTER` | double | `0.1` | Required improvement of linear residual vs nonlinear residual. |
| `INFNORMSCALING` | bool | `false` | Scale matrix blocks by row infinity norm. |
| `GMSH_OUTPUT` | bool | `false` | Write Gmsh output. |
| `COMPUTE_DIVU` | bool | `false` | Compute velocity divergence at element center. |
| `COMPUTE_EKIN` | bool | `false` | Compute kinetic energy each time step. |
| `NEW_OST` | bool | `false` | Use newer one-step-theta implementation. |
| `RESULTSEVERY` | int | `1` | Solution output cadence. |
| `RESTARTEVERY` | int | `20` | Restart output cadence. |
| `NUMSTEP` | int | `1` | Total time steps. |
| `STEADYSTEP` | int | `-1` | Steady-state check interval. |
| `NUMSTASTEPS` | int | `0` | Starting-scheme steps. |
| `STARTFUNCNO` | int | `-1` | Initial starting field function id. |
| `ITEMAX` | int | `10` | Maximum nonlinear iterations. |
| `INITSTATITEMAX` | int | `5` | Maximum iterations for initial stationary solution. |
| `TIMESTEP` | double | `0.01` | Time increment. |
| `MAXTIME` | double | `1000.0` | Total simulation time. |
| `ALPHA_M` | double | `1.0` | Generalized-alpha factor. |
| `ALPHA_F` | double | `1.0` | Generalized-alpha factor. |
| `GAMMA` | double | `1.0` | Generalized-alpha factor. |
| `THETA` | double | `0.66` | One-step-theta factor. |
| `START_THETA` | double | `1.0` | Starting-scheme theta. |
| `STRONG_REDD_3D_COUPLING_TYPE` | bool | `false` | Activate strong 3D reduced-dimensional coupling. |
| `VELGRAD_PROJ_SOLVER` | int | `-1` | Linear solver for L2 velocity-gradient projection. |
| `VELGRAD_PROJ_METHOD` | enum | `none` | `none`, `superconvergent_patch_recovery`, `L2_projection`. |
| `OFF_PROC_ASSEMBLY` | bool | `false` | Communicate ghosted element contributions instead of evaluating off-process elements. |

## Nonlinear tolerances

Section: `FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES`.

| Key | Type | Default |
|---|---:|---:|
| `TOL_VEL_RES` | double | `1e-6` |
| `TOL_VEL_INC` | double | `1e-6` |
| `TOL_PRES_RES` | double | `1e-6` |
| `TOL_PRES_INC` | double | `1e-6` |

## Residual-based stabilization

Section: `FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION`.

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `STABTYPE` | enum | `residual_based` | `no_stabilization`, `residual_based`, `edge_based`, `pressure_projection`. |
| `INCONSISTENT` | bool | `false` | Use residual without second derivatives. |
| `Reconstruct_Sec_Der` | bool | `false` | Reconstruct second derivatives. |
| `TDS` | enum | `quasistatic` | `quasistatic`, `time_dependent`. |
| `TRANSIENT` | enum | `no_transient` | `no_transient`, `yes_transient`, `transient_complete`. |
| `PSPG` | bool | `true` | Pressure-stabilizing Petrov-Galerkin. |
| `SUPG` | bool | `true` | Streamline-upwind/Petrov-Galerkin. |
| `GRAD_DIV` | bool | `true` | Grad-div stabilization. |
| `VSTAB` | enum | `no_vstab` | `no_vstab`, `vstab_gls`, `vstab_gls_rhs`, `vstab_usfem`, `vstab_usfem_rhs`. |
| `RSTAB` | enum | `no_rstab` | `no_rstab`, `rstab_gls`, `rstab_usfem`. |
| `CROSS-STRESS` | enum | `no_cross` | `no_cross`, `yes_cross`, `cross_rhs`. |
| `REYNOLDS-STRESS` | enum | `no_reynolds` | `no_reynolds`, `yes_reynolds`, `reynolds_rhs`. |
| `DEFINITION_TAU` | enum | source default | Includes `Taylor_Hughes_Zarins`, `Shakib_Hughes_Codina`, `Codina`, `Hughes_Franca_Balestra_wo_dt`, and other source-listed tau definitions. |
| `CHARELELENGTH_U` | enum | `streamlength` | `streamlength`, `volume_equivalent_diameter`, `root_of_volume`. |
| `CHARELELENGTH_PC` | enum | `volume_equivalent_diameter` | `streamlength`, `volume_equivalent_diameter`, `root_of_volume`. |
| `EVALUATION_TAU` | string | `element_center` | Evaluation point for tau. |
| `EVALUATION_MAT` | string | `element_center` | Evaluation point for material data. |

Edge stabilization and porous-flow stabilization are separate subsections with similar names; use them only when `STABTYPE` and element choices require them.

## Turbulence and time adaptivity

High-use optional subsections:

- `FLUID DYNAMIC/TURBULENCE MODEL`: `TURBULENCE_APPROACH` default `DNS_OR_RESVMM_LES`, `PHYSICAL_MODEL` default `no_model`, `FSSUGRVISC` default `No`, sampling/dumping controls, canonical-flow controls.
- `FLUID DYNAMIC/SUBGRID VISCOSITY`: constants for Smagorinsky/Yoshizawa/Vreman-like modeling.
- `FLUID DYNAMIC/WALL MODEL`: `X_WALL`, `Tauw_Type`, wall-normal quadrature, projection solver.
- `FLUID DYNAMIC/TIMEADAPTIVITY`: `ADAPTIVE_TIME_STEP_ESTIMATOR` default `none` or `const_dt` in source enum, `CFL_NUMBER`, `FREEZE_ADAPTIVE_DT_AT`, output cadence and thermo-pressure controls.

## Runtime transfer notes

The fluid adapter reads this section once during algorithm construction. `TIMEINTEGR` selects the time-integration class, `THETA` configures one-step-theta weighting, `NUMSTASTEPS` configures starting steps, `GRIDVEL` determines how mesh displacement becomes grid velocity in ALE/coupled runs, `OST_CONT_PRESS` changes one-step-theta treatment of continuity and pressure, and `NEW_OST` toggles the newer one-step-theta path subject to runtime validity checks.

## Minimal stationary Stokes pattern

From `tests/input_files/f3_stokes_residualbased_rotboxgeom.4C.yaml`:

```yaml
PROBLEM TYPE:
  PROBLEMTYPE: "Fluid"
DISCRETISATION:
  NUMSTRUCDIS: 0
  NUMALEDIS: 0
  NUMARTNETDIS: 0
  NUMTHERMDIS: 0
  NUMAIRWAYSDIS: 0
FLUID DYNAMIC:
  PHYSICAL_TYPE: "Stokes"
  LINEAR_SOLVER: 1
  TIMEINTEGR: "Stationary"
  ITEMAX: 2
FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
  TOL_VEL_RES: 1e-09
  TOL_VEL_INC: 1e-09
  TOL_PRES_RES: 1e-09
  TOL_PRES_INC: 1e-09
FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
  SUPG: false
  GRAD_DIV: false
  DEFINITION_TAU: "Hughes_Franca_Balestra_wo_dt"
SOLVER 1:
  SOLVER: "UMFPACK"
```

Add `MAT_fluid`, fluid boundary conditions, and `FLUID DOMAIN` or `FLUID ELEMENTS`; see [Geometry and elements](geometry-and-elements.md) and [Materials](materials.md).
