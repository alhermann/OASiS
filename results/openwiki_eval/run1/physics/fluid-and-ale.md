---
type: subsystem page
title: Fluid, ALE, Turbulence, and Fluid Elements
description: Covers fluid runtime dispatch, fluid element families, ALE mesh motion, turbulent inflow, low-Mach/lubrication/reduced variants, XFluid/XWall, and DOF-ordering invariants.
tags: [physics, fluid, ale]
---

# Fluid, ALE, Turbulence, and Fluid Elements

Fluid runtime begins with `dyn_fluid_drt(restart)` in `src/fluid/4C_fluid_dyn_nln_drt.cpp`. ALE and FSI-related fluid setup is implemented in `src/fsi/src/4C_fsi_dyn.cpp`. Element assembly lives primarily in `src/fluid_ele` and specialized modules such as `fluid_turbulence`, `fluid_xfluid`, `ale`, `loma`, and `lubrication`.

## Standard fluid runtime

`dyn_fluid_drt` obtains the `fluid` discretization communicator and `fluid_dynamic_params()`. It then branches:

- if `TURBULENT INFLOW/TURBULENTINFLOW` is true and the restart step is less than `NUMINFLOWSTEP`, it constructs `FLD::TurbulentFlowAlgorithm`, optionally reads restart, runs `time_loop()`, adds its result check, and calls `test_all`;
- otherwise it constructs `Adapter::FluidBaseAlgorithm(fdyn, fdyn, "fluid", false)`, optionally reads restart through `fluid_field()`, calls `integrate()`, creates a field test, and calls `test_all`.

Fresh fluid runs do not have an explicit `post_setup()` branch in this entrypoint; setup is hidden inside `Adapter::FluidBaseAlgorithm` and its `fluid_field()` construction/setup. Unlike structural and thermo entrypoints, this function does not inspect `WRITE_INITIAL_STATE` or `WRITE_FINAL_STATE`; fluid output scheduling is handled by the fluid adapter/time integrator and output-control settings during integration.

This keeps turbulent inflow generation as a pre-main-problem path inside the same entrypoint.

## ALE and XFluid variants

Fluid-ALE setup (`fluid_ale_drt`) enforces `fluid dof < ale dof` by filling the fluid discretization before the ALE discretization. If the ALE discretization is empty, it clones from fluid using `ALE::Utils::AleCloneStrategy`, fills ALE, and evaluates ALE elements with `action = setup_material`. If ALE is explicitly provided, fluid and ALE node ids must be disjoint; otherwise Dirichlet boundary conditions become ambiguous.

Fluid-XFEM setup (`fluid_xfem_drt`) fills the structure discretization, calls `FLD::XFluid::setup_fluid_discretization()`, optionally prepares ALE, then chooses either `FSI::FluidXFEMAlgorithm` or a standard `Adapter::FluidBaseAlgorithm`. In fluid-ALE and FSI-ALE paths, `XFLUIDFLUID` replaces normal `fluiddis->fill_complete()` with `FLD::XFluid::setup_fluid_discretization()`, because the XFluid discretization must be built with extended interface state. In `fluid_xfem_drt`, `ALE_XFluid` decides whether an ALE discretization is also filled/cloned and whether the runtime algorithm is `FSI::FluidXFEMAlgorithm` for moving interfaces or the standard fluid base algorithm for non-ALE XFluid.

## Element and algorithm modules

| Module | Role |
| --- | --- |
| `fluid` | Time integrators, implicit integration, BDF2/generalized-alpha/OST/statics/HDG variants, impedance, meshtying, xwall, result tests. |
| `fluid_ele` | Fluid elements, boundary elements, HDG/weak-compressible HDG, poro, XFEM, xwall, parameter objects, factories, action enums. |
| `ale` | ALE elements and algorithms, mesh sliding/tying, cloning strategies, result tests. |
| `fluid_turbulence` | Box filters, dynamic Smagorinsky/Vreman, HIT forcing/initial fields, statistics managers, turbulent inflow transfer. |
| `fluid_xfluid` | Extended fluid state, output service, functions, state creation, result tests. |
| `loma` | Low-Mach algorithm and entrypoint. |
| `lubrication` | Lubrication elements/dynamics and EHL dependencies. |
| `levelset` | Level-set advection/reinitialization used by interface and XFEM workflows. |
| `red_airways`, `art_net` | Fluid-like network models documented in [Poro, Particle, Lung, and Network Models](poro-particle-lung.md). |

## Invariants

- Fluid discretization must not be empty for standard fluid and Navier-Stokes-coupled scalar transport.
- ALE cloned meshes require material setup evaluation after cloning.
- Explicit ALE meshes must use node ids disjoint from fluid node ids.
- Restart is read before integration but after algorithm construction.
- Turbulent inflow uses restart count to decide whether to run inflow-generation steps or the main problem.

## Validation

Run fluid-specific regression inputs for integrator changes, `fluid_ele` unit/regression cases for element changes, and XFluid/Cut tests for interface changes. Changes touching fluid-ALE or FSI ordering should also validate [Coupled Multiphysics](coupled-multiphysics.md).
