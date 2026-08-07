---
type: input parameter reference
title: PARTICLE DYNAMIC
description: Particle dynamic, SPH, DEM, and peridynamics input controls with defaults, material links, and accepted deck patterns.
tags: [input-format, particles]
---

# `PARTICLE DYNAMIC`

Particle sections are registered in `src/particle/src/4C_particle_input.cpp`. A `PROBLEMTYPE: "Particle"` deck uses the main `PARTICLE DYNAMIC` section plus optional subsections for initial/boundary fields, SPH, DEM, and peridynamics. Theory links: [Particle theory](../theory/particles.md).

## Runtime selection

`ParticleAlgorithm::init_particle_time_integration` selects semi-implicit Euler or velocity Verlet directly from `DYNAMICTYPE`. `ParticleAlgorithm::init_particle_interaction` selects no interaction, SPH, or DEM directly from `INTERACTION`; peridynamic body interaction is enabled by `PD_BODY_INTERACTION` and `/PD` settings. If `INTERACTION: "SPH"`, provide `PARTICLE DYNAMIC/SPH` and SPH materials; if `INTERACTION: "DEM"`, provide `PARTICLE DYNAMIC/DEM` and DEM materials.

## Main `PARTICLE DYNAMIC` parameters

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `DYNAMICTYPE` | enum | `VelocityVerlet` | `SemiImplicitEuler`, `VelocityVerlet`. |
| `INTERACTION` | enum | `None` | `None`, `SPH`, `DEM`. Peridynamic body interaction is additionally controlled by `PD_BODY_INTERACTION` and `/PD`. |
| `RESULTSEVERY` | int | `1` | Particle runtime output cadence. |
| `RESTARTEVERY` | int | `1` | Restart output cadence. |
| `WRITE_GHOSTED_PARTICLES` | bool | `false` | Debug output of ghosted particles. |
| `TIMESTEP` | double | `0.01` | Time step. |
| `NUMSTEP` | int | `100` | Maximum number of steps. |
| `MAXTIME` | double | `1.0` | Maximum time. |
| `GRAVITY_ACCELERATION` | string | `0.0 0.0 0.0` | Acceleration vector. |
| `GRAVITY_RAMP_FUNCT` | int | `-1` | Function id for gravity ramp. |
| `VISCOUS_DAMPING` | double | `-1.0` | Viscous damping factor for static equilibrium. |
| `TRANSFER_EVERY` | bool | `false` | Transfer particles to new bins every step. |
| `PHASE_TO_DYNLOADBALFAC` | string | `none` | Phase to dynamic-load-balance factor mapping. |
| `PHASE_TO_MATERIAL_ID` | string | `none` | Phase to material id mapping. |
| `INITIAL_POSITION_AMPLITUDE` | string | `0.0 0.0 0.0` | Random-noise amplitude per spatial direction. |
| `PARTICLE_WALL_SOURCE` | enum | source default `NoParticleWall` | Particle wall source. |
| `PARTICLE_WALL_MAT` | int | `-1` | Material id for bounding-box particle wall. |
| `PARTICLE_WALL_MOVING` | bool | `false` | Moving particle wall. |
| `PARTICLE_WALL_LOADED` | bool | `false` | Loaded particle wall. |
| `RIGID_BODY_MOTION` | bool | `false` | Consider rigid body motion. |
| `RIGID_BODY_PHASECHANGE_RADIUS` | double | `-1.0` | Neighbor search radius for rigid bodies during phase change. |
| `PD_BODY_INTERACTION` | bool | `false` | Consider peridynamic body interaction. |

## Initial and boundary conditions subsection

Section: `PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS`.

| Key | Type | Default | Meaning |
|---|---:|---:|---|
| `INITIAL_TEMP_FIELD` | string | `none` | Function id string for initial particle temperature. |
| `INITIAL_VELOCITY_FIELD` | string | `none` | Function id string for initial velocity. |
| `INITIAL_ANGULAR_VELOCITY_FIELD` | string | `none` | Function id for rigid-body or DEM angular velocity. |
| `INITIAL_ACCELERATION_FIELD` | string | `none` | Function id for initial acceleration. |
| `INITIAL_ANGULAR_ACCELERATION_FIELD` | string | `none` | Function id for angular acceleration. |
| `DIRICHLET_BOUNDARY_CONDITION` | string | `none` | Function id for particle Dirichlet condition. |
| `TEMPERATURE_BOUNDARY_CONDITION` | string | `none` | Function id for temperature condition. |
| `CONSTRAINT` | enum | source default `NoConstraint` | Kinematic constraint type. |

## SPH subsection

Section: `PARTICLE DYNAMIC/SPH`. Use it when `INTERACTION: "SPH"`.

| Key | Type | Default | Meaning |
|---|---:|---:|---|
| `WRITE_PARTICLE_WALL_INTERACTION` | bool | `false` | Write particle-wall interaction output. |
| `KERNEL` | enum | `CubicSpline` | SPH kernel. Source enum also includes other kernels used by tests, e.g. quintic-spline variants. |
| `KERNEL_SPACE_DIM` | enum | `Kernel3D` | Kernel dimension. |
| `INITIALPARTICLESPACING` | double | `0.0` | Initial particle spacing. |
| `EQUATIONOFSTATE` | enum | `GenTait` | SPH equation of state. |
| `MOMENTUMFORMULATION` | enum | `AdamiMomentumFormulation` | Momentum formulation. |
| `DENSITYEVALUATION` | enum | `DensitySummation` | Density evaluation scheme. |
| `DENSITYCORRECTION` | enum | `NoCorrection` | Density correction scheme. |
| `BOUNDARYPARTICLEFORMULATION` | enum | `NoBoundaryFormulation` | Boundary particle formulation. |
| `BOUNDARYPARTICLEINTERACTION` | enum | `NoSlipBoundaryParticle` | Boundary particle interaction. |
| `WALLFORMULATION` | enum | `NoWallFormulation` | Wall formulation. |
| `TRANSPORTVELOCITYFORMULATION` | enum | `NoTransportVelocity` | Transport-velocity formulation. |
| `REDUCED_DIMENSION_SCALE_FACTOR` | double | `1.0` | Force scaling for reduced-dimensional particle fields. |
| `TEMPERATUREEVALUATION` | enum | `NoTemperatureEvaluation` | Temperature evaluation scheme. |
| `TEMPERATUREGRADIENT` | bool | `false` | Evaluate temperature gradient. |
| `HEATSOURCETYPE` | enum | `NoHeatSource` | Heat source type. |
| `HEATSOURCE_FUNCT` | int | `-1` | Heat-source function id. |
| `HEATSOURCE_DIRECTION` | string | `0.0 0.0 0.0` | Surface heat-source direction. |

The source continues with vapor heat-loss, surface-tension, wetting, interface-viscosity, barrier-force, phase-change, and boundary-force keys. Use an accepted `tests/input_files/particle_sph_*` deck when enabling those features.

## DEM subsection

Section: `PARTICLE DYNAMIC/DEM`. Use it when `INTERACTION: "DEM"`.

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `WRITE_PARTICLE_ENERGY` | bool | `false` | Output DEM particle energy. |
| `WRITE_PARTICLE_WALL_INTERACTION` | bool | `false` | Output particle-wall interaction. |
| `NORMALCONTACTLAW` | enum | `NormalLinearSpring` | `NormalLinearSpring`, `NormalLinearSpringDamp`, `NormalHertz`, `NormalLeeHerrmann`, `NormalKuwabaraKono`, `NormalTsuji`. |
| `MIN_RADIUS` | double | `0.0` | Minimum radius for distributions/checks. |
| `MAX_RADIUS` | double | `0.0` | Maximum radius. |
| `MAX_VELOCITY` | double | `-1.0` | Velocity cap/check. |
| `REL_PENETRATION` | double | `-1.0` | Relative penetration control. |
| `NORMAL_DAMP` | double | `-1.0` | Normal damping. |
| `COEFF_RESTITUTION` | double | `-1.0` | Coefficient of restitution. |
| `DAMP_REG_FAC` | double | `-1.0` | Damping regularization factor. |
| `FRICT_COEFF_ROLL` | double | `-1.0` | Rolling friction coefficient. |
| `ADHESION_SURFACE_ENERGY` | double | `-1.0` | Adhesion surface energy. |
| `ADHESION_SURFACE_ENERGY_FACTOR` | double | `1.0` | Adhesion scaling factor. |

## Peridynamics subsection

Section: `PARTICLE DYNAMIC/PD`.

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `PERIDYNAMIC_GRID_SPACING` | double | `0.0` | Particle grid spacing for peridynamic body. |
| `NORMALCONTACTLAW` | enum | `NormalLinearSpring` | `NormalLinearSpring`, `NormalLinearSpringDamp`. |
| `NORMAL_DAMP` | double | `0.0` | Normal damping. |
| `PRE_CRACKS` | string | empty string | Pre-crack definition. |
| `IMPACTOR_VELOCITY` | string | `0.0 0.0 0.0` | Impact velocity vector. |
| `MAT` | int | required in subsection item/source | Peridynamic material id. |

## Materials and particles

Use [Materials](materials.md#particle-materials): `MAT_ParticleSPHFluid`/`MAT_ParticleSPHBoundary` for SPH, `MAT_ParticleDEM`/`MAT_ParticleWallDEM` for DEM, and `MAT_ParticlePD` for peridynamics. Particle coordinates and phases are usually provided in the legacy `PARTICLES` section or generated by particle input utilities.

Representative tests: `tests/input_files/particle_dem_1d_normalcontact_gravity.4C.yaml`, `tests/input_files/particle_sph_2d_dambreak_freesurface_densitynormalizedreinit.4C.yaml`, and `tests/input_files/particle_sph_2d_pdbody_gravity.4C.yaml`.
