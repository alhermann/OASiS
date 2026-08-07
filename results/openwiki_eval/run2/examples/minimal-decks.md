---
type: examples
title: Minimal deck patterns
description: Worked, source-grounded 4C input snippets for first-attempt structure, fluid, scalar transport, and particle decks.
tags: [input-format, examples]
---

# Minimal deck patterns

These examples are compressed from accepted test decks. They show section shape and parse-sensitive keys, not validated physical benchmark models. For real changes, copy a nearby full deck under `tests/input_files` and edit it.

## Stationary Stokes fluid with generated domain

Source pattern: `tests/input_files/f3_stokes_residualbased_rotboxgeom.4C.yaml`.

```yaml
TITLE: ["minimal Stokes fluid"]
PROBLEM SIZE:
  MATERIALS: 1
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
  NAME: "Fluid_Solver"
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: 1
      DENSITY: 1
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 4
    ONOFF: [1, 1, 1, 0]
    VAL: [1, 0, 0, 0]
    FUNCT: [0, 0, 0, 0]
  - E: 2
    NUMDOF: 4
    ONOFF: [1, 1, 1, 0]
    VAL: [0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0]
DSURF-NODE TOPOLOGY:
  - "SIDE fluid x- DSURFACE 1"
  - "SIDE fluid x+ DSURFACE 2"
FLUID DOMAIN:
  bottom_corner_point: [-1, -1, -1]
  top_corner_point: [1, 1, 1]
  subdivisions: [5, 5, 5]
  elements:
    FLUID:
      HEX8:
        MAT: 1
        NA: Euler
  auto_partition: false
```

## Scalar diffusion skeleton

Source pattern: `tests/input_files/scatra_1D_line2_diffnumdof.4C.yaml`.

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

To make it runnable, add `TRANSPORT DOMAIN` or `NODE COORDS` plus `TRANSPORT ELEMENTS`, and transport Dirichlet/Neumann conditions.

## Static beam/structure skeleton

Source pattern: `tests/input_files/beam3r_line2_static_test1.4C.yaml`.

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
MATERIALS:
  - MAT: 1
    MAT_BeamReissnerElastHyper:
      YOUNG: 1e+07
      SHEARMOD: 5e+06
      DENS: 1.3e+09
      CROSSAREA: 1
      SHEARCORR: 1
      MOMINPOL: 0.1406
      MOMIN2: 0.0833333
      MOMIN3: 0.0833333
      FAD: true
```

Add `NODE COORDS`, `STRUCTURE ELEMENTS`, `DNODE-NODE TOPOLOGY`, and Dirichlet/Neumann conditions as in the source test.

## Particle DEM skeleton

Source family: `tests/input_files/particle_dem_*`.

```yaml
PROBLEM TYPE:
  PROBLEMTYPE: "Particle"
PARTICLE DYNAMIC:
  DYNAMICTYPE: "VelocityVerlet"
  INTERACTION: "DEM"
  TIMESTEP: 0.01
  NUMSTEP: 100
PARTICLE DYNAMIC/DEM:
  NORMALCONTACTLAW: "NormalLinearSpring"
MATERIALS:
  - MAT: 1
    MAT_ParticleDEM:
      INITRADIUS: 0.5
      INITDENSITY: 1.0
```

Add the `PARTICLES` legacy section or particle generator data and any particle wall material/settings.

## Parse-error avoidance checklist

- Do not nest slash-named sections under their prefix section.
- Use exact enum spelling from the reference pages.
- Add `SOLVER n` for every `LINEAR_SOLVER: n` where `n > 0`.
- Add `FUNCTn` for every nonzero function id.
- Ensure vector lengths match `NUMDOF`, `NUMMAT`, `NUMSCAL`, or other sizing keys.
- Ensure every element or particle `MAT` id has a matching `MATERIALS` entry.
