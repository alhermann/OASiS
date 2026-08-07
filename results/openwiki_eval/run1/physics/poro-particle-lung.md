---
type: subsystem page
title: Poro, Particle, Lung, and Network Models
description: Covers poroelastic and pressure-based porofluid workflows, particle engine and interaction modules, arterial and airway networks, reduced lung models, cardiovascular0d, and related entrypoints.
tags: [physics, poro, particles, networks]
---

# Poro, Particle, Lung, and Network Models

This page groups physics families that are not a single continuum field but are major runtime problem types in `entrypoint_switch`: poroelasticity, pressure-based porofluid variants, particle simulation, arterial and airway networks, reduced lung models, cardiovascular0d, and Brownian/particle-structure variants.

## Poroelastic runtime

`poroelast_drt()` in `src/poroelast/4C_poroelast_dyn.cpp` shows the common poro lifecycle:

1. get `Global::Problem` and the `structure` discretization communicator;
2. print the poro logo on rank 0;
3. call `PoroElast::Utils::setup_poro<PoroelastCloneStrategy>()` to set up discretizations and cloning;
4. read `poroelast_dynamic_params()`;
5. create the concrete poro algorithm through `PoroElast::Utils::create_poro_algorithm(...)`;
6. read restart or call `post_setup()`;
7. call `setup_system()` to create the coupled system and combined DOF map;
8. run `time_loop()`;
9. run result tests.

`poroelast_scatra` extends this pattern with scalar transport. Pressure-based families split into `porofluid_pressure_based`, `porofluid_pressure_based_ele`, `porofluid_pressure_based_elast`, and `porofluid_pressure_based_elast_scatra`, including artery coupling strategies and phase/variable managers. Problem-type-specific behavior continues beyond the entrypoint in `PoroElast::Utils::create_poro_algorithm`, the pressure-based dynamic functions (`porofluid_pressure_based_dyn`, `porofluid_elast_dyn`, `porofluid_pressure_based_elast_scatra_dyn`), and adapter wrappers for porofluid/poro structure fields. The element-side phase and variable managers in `porofluid_pressure_based_ele` define which DOFs and material phases are available to the coupled system.

## Particle runtime and modules

`particle_drt()` in `src/particle/src/algorithm/4C_particle_algorithm_sim.cpp` obtains local communicator, particle parameters, and initial particle objects from `Global::Problem`, constructs `Particle::ParticleAlgorithm`, calls `init(initialparticles)`, reads restart if present, calls `setup()`, runs `timeloop()`, creates particle result tests, and calls `test_all`.

Particle subpackages:

- `algorithm`: constraints, Dirichlet/temperature boundary conditions, gravity, initial fields, input generation, result tests, time integration, damping;
- `engine`: particle containers, object model, communication utilities, particle reader, unique global ids, runtime VTP writer;
- `interaction`: DEM, SPH, peridynamic, neighbor pairs, contact, surface tension, heat, pressure, material handler;
- `rigidbody` and `wall`: rigid body and particle wall state, output, and result tests.

Focused particle tests live under `src/particle/tests` and exercise containers, DEM contacts, SPH kernels/equations, interaction utilities, and rigidbody utilities.

## Reduced lung and airway/network models

`ReducedLung::reduced_lung_main()` creates a synthetic `Core::FE::Discretization` named `reduced_lung`, builds topology from input, fill-completes it, creates a linear solver, creates visualization writer, instantiates local airway and terminal-unit models, creates global DOF maps, creates evaluators, classifies nodes into boundary conditions and junctions, creates row/domain/column maps, assembles vectors and a sparse Jacobian, and enters a time loop.

Related modules:

- `reduced_lung` and `reduced_lung/1d_pipe_flow`: reduced-dimensional lung tree models and one-dimensional pipe flow;
- `red_airways`: reduced airway elements, acinus/inter-acinar dependencies, implicit integration, result tests;
- `art_net`: arterial network elements, terminal boundary conditions, junctions, explicit and stationary time integration;
- `cardiovascular0d`: 0D cardiovascular circulation, Windkessel, arterial proximal/distal models, respiratory/systemic/pulmonary circulation, NOX linearsystem, structure_new model evaluator;
- `browniandyn`: Brownian dynamics structure model evaluator;
- `pasi`: particle-structure interaction, documented further in [Coupled Multiphysics](coupled-multiphysics.md).

## Invariants

- Poro workflows rely on clone strategies and combined-system setup after restart/post-setup.
- Particle workflows initialize from `Global::Problem::particles()` before restart and setup.
- Reduced lung constructs its own discretization and maps; it does not use a pre-read FE mesh in the same way as continuum modules.
- Network models still depend on global input specs, materials, linalg solvers, and result tests.

## Validation

Use `src/particle/tests`, `src/reduced_lung/tests`, porofluid/poro regression inputs, and network-specific input regressions. For artery-coupled porofluid or scatra changes, include [Scalar Transport](scalar-transport-electrochemistry.md) and [Coupled Multiphysics](coupled-multiphysics.md) checks.
