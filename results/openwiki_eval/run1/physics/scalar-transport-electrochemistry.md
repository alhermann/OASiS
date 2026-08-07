---
type: subsystem page
title: Scalar Transport, Electrochemistry, Thermo, and STI Workflows
description: Documents scalar-transport dispatch, scatra element families, electrochemistry, cardiac monodomain, STI/TSI/SSTI/thermo coupling, mesh coupling modes, and validation points.
tags: [physics, scalar-transport, electrochemistry]
---

# Scalar Transport, Electrochemistry, Thermo, and STI Workflows

Scalar transport is implemented by `src/scatra` and `src/scatra_ele`, with related modules `elch`, `sti`, `tsi`, `ssti`, `thermo`, `loma`, and cardiac monodomain variants. These modules share field-coupling and element-calculation infrastructure with fluid and structure pages.

## `scatra_dyn` flow

`scatra_dyn(int restart)` reads the `fluid` and `scatra` discretizations, fills fluid before scatra, and inspects two key parameters:

- `FIELDCOUPLING` from scalar transport dynamic parameters;
- `VELOCITYFIELD` from the scatra dynamic parameter list.

It enforces mesh/coupling consistency:

- if the scatra discretization is empty, the run must use matching fluid/scatra meshes and Navier-Stokes velocity as appropriate;
- if scatra is non-empty and the velocity comes from Navier-Stokes, nonmatching meshes require volumetric mortar coupling.

For zero or function velocity, scatra uses its own transport elements, creates an auxiliary predefined velocity DOF set, copies transport boundary conditions to generic conditions, fill-completes, initializes and sets up `Adapter::ScaTraBaseAlgorithm`, handles heterogeneous reaction rebalancing, reads restart, sets velocity/external forces, runs `time_loop()`, and tests results. Fresh scalar-transport setup is explicit: `init()` constructs the time integrator, redistribution may occur between `init()` and `setup()`, and `setup()` finalizes the algorithm before restart/state initialization. This entrypoint does not directly implement initial/final output flags; output is scheduled by the scatra time integrator/result-test machinery.

For Navier-Stokes velocity, matching coupling clones scatra elements from the fluid discretization with `ScaTraFluidCloneStrategy`, sets transport implementation types, and then proceeds through coupled fluid/scatra setup.

## Scatra element families

`src/scatra_ele` is a large element-calculation module. It contains:

- standard, HDG, no-physics, anisotropic, LOMA, poro, porofluid-pressure-based, artery, growth/remodel, chemo/reaction, LS/reinit, turbulence, and cardiac monodomain calculators;
- electrochemistry-specific calculators for Nernst-Planck, diffusive conditions, electrodes, STI-electrode and STI-thermo variants;
- boundary calculators and factories;
- parameter classes for standard, boundary, ElCh, diffusive conditions, time integration, LS reinit, and turbulence;
- service files for stabilization, turbulence, ELCH, LS, and cardiac monodomain behavior.

`src/scatra` owns time integrators, meshtying strategies, result tests, algorithm wrappers, and utility/cloning strategies.

## Electrochemistry and coupled thermal/scalar modules

| Module | Runtime responsibility |
| --- | --- |
| `elch` | Electrochemistry algorithms and moving-boundary behavior built on scatra/fluid. |
| `sti` | Scalar-thermo interaction with monolithic and partitioned algorithms. |
| `tsi` | Thermo-structure interaction using structure and thermo fields. |
| `ssti` | Structure-scalar-thermo interaction, including monolithic assembly/evaluate off-diagonal paths. |
| `thermo` | Thermal dynamics, adapter, elements, implicit time integrators, result tests. |
| `cardiac_monodomain` via scatra | Cardiac monodomain dispatch and element schemes. |

`thermo_dyn_drt()` is simple: it gets `thermo` discretization and thermal parameters, creates `Thermo::BaseAlgorithm`, reads restart or writes initial output, calls `integrate()`, creates a field test, and calls `test_all`.

## Coupling dependencies

Scalar transport is a major consumer of [Coupled Multiphysics](coupled-multiphysics.md): it uses matching mesh cloning, volumetric mortar for nonmatching Navier-Stokes coupling, meshtying strategies for artery/fluid/S2I cases, and STI/TSI/SSTI combined systems. Its element behavior also relies on [Materials](materials.md) for scatra, electrochemical, porous, and thermal material laws.

## Validation

Focused tests include `/unittests/scatra`, scatra/scatra_ele source tests when present, ELCH/STI/TSI/SSTI regression inputs, and full regression cases with `RESULT DESCRIPTION`. For input/schema changes, also run metadata validation from [Testing and Validation](../architecture/testing-validation.md).
