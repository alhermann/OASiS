---
type: source map
title: Module Catalog
description: Maps every manifest-backed `src` module and app target to its role, canonical wiki owner, main source entrypoints, dependencies, and validation hints.
tags: [source-map, modules]
---

# Module Catalog

This catalog is the coverage backstop for all CMake-backed modules. Use it to route from a directory name to the canonical page that explains behavior. Dependencies are summarized from module `CMakeLists.txt` files; detailed behavior belongs on the linked page.

## Core and infrastructure modules

| Module | Role | Key dependencies | Canonical page |
| --- | --- | --- | --- |
| `config` | Generated/configuration headers and compile definitions. | none | [Build and Runtime Topology](../architecture/build-and-runtime.md) |
| `core` | Communication, IO, FEM, linalg, search, rebalance, utilities, material base. | none | [Core Platform Services](../core/core-platform.md) |
| `global_data` | Input setup, global problem initialization, discretization reading, metadata emission. | `config`, `core`, `global_legacy_module`, `inpar`, `mat`, `particle` | [Input Schema and Global Problem](../architecture/input-schema-global-problem.md) |
| `global_legacy_module` | Legacy registration hub for elements, materials, conditions, functions, and ParObjects. | most physics/element modules | [Module Registration](../extension/module-registration.md) |
| `module_registry` | `ModuleCallbacks` callback interface. | `config`, `core` | [Module Registration](../extension/module-registration.md) |
| `inpar` | Input parameter definitions for global and physics sections. | `core`, selected physics modules | [Input Schema and Global Problem](../architecture/input-schema-global-problem.md) |
| `timestepping` | Shared multi-step time-stepping helper. | `config`, `core` | [Core Platform Services](../core/core-platform.md) |
| `deal_ii` | Optional deal.II wrappers and conversion utilities gated by `FOUR_C_WITH_DEAL_II`. | optional dependency | [FEM Discretization](../core/fem-discretization.md) |
| `post` | Post-processing writers and filters. | `global_legacy_module`, `inpar` | [Apps and Tooling](../tools/apps-postprocessing.md) |
| `solver_nonlin_nox` | NOX nonlinear solver wrappers, groups, factories, line searches, status tests. | `core`, `inpar`, `structure_new`, contact/constraint/cardiovascular integrations | [Linear Algebra and Solvers](../core/linalg-solvers.md) |

## Materials and element kernels

| Module | Role | Canonical page |
| --- | --- | --- |
| `mat` | Material factory, material parameters, material laws for fluids, solids, poro, scatra, particles, bio/cardiac, thermo, mixtures. | [Materials](../physics/materials.md) |
| `mixture` | Mixture constituents, rules, growth and prestress strategies. | [Materials](../physics/materials.md) |
| `contact_constitutivelaw` | Contact constitutive law input and runtime laws. | [Materials](../physics/materials.md) |
| `solid_3D_ele` | 3D solid elements and formulations. | [Structure and Solid Elements](../physics/structure-and-solid-elements.md) |
| `solid_poro_3D_ele` | Poro solid element formulations. | [Structure and Solid Elements](../physics/structure-and-solid-elements.md) |
| `solid_scatra_3D_ele` | Solid-scalar coupled elements. | [Structure and Solid Elements](../physics/structure-and-solid-elements.md) |
| `fluid_ele` | Fluid, HDG, poro, XFEM, xwall, and boundary elements. | [Fluid and ALE](../physics/fluid-and-ale.md) |
| `scatra_ele` | Scalar transport, electrochemistry, HDG, reaction, poro, LS, STI, and boundary elements. | [Scalar Transport and Electrochemistry](../physics/scalar-transport-electrochemistry.md) |
| `beam3`, `truss3`, `torsion3`, `rigidsphere`, `shell7p`, `shell_kl_nurbs`, `membrane`, `w1`, `bele` | Structural element families and specialized wall/shell/membrane/boundary elements. | [Structure and Solid Elements](../physics/structure-and-solid-elements.md) |
| `porofluid_pressure_based_ele` | Pressure-based porofluid element, phase, and variable managers. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `thermo` | Thermal elements, adapters, implicit integrators, result tests. | [Scalar Transport and Electrochemistry](../physics/scalar-transport-electrochemistry.md) |

## Single-field physics modules

| Module | Entry symbols or files | Canonical page |
| --- | --- | --- |
| `structure` | `caldyn_drt`, `dyn_nlnstructural_drt`, structural time integrators. | [Structure and Solid Elements](../physics/structure-and-solid-elements.md) |
| `structure_new` | New structure model evaluators, functions, result tests, adapter integration. | [Structure and Solid Elements](../physics/structure-and-solid-elements.md) |
| `fluid` | `dyn_fluid_drt`, fluid time integrators and utilities. | [Fluid and ALE](../physics/fluid-and-ale.md) |
| `ale` | `dyn_ale_drt`, ALE elements, mesh sliding/tying and clone strategy. | [Fluid and ALE](../physics/fluid-and-ale.md) |
| `fluid_turbulence` | `TurbulentFlowAlgorithm`, HIT/statistics/inflow helpers. | [Fluid and ALE](../physics/fluid-and-ale.md) |
| `fluid_xfluid` | XFluid setup, state, output, functions, result tests. | [Fluid and ALE](../physics/fluid-and-ale.md) |
| `scatra` | `scatra_dyn`, scalar time integrators, meshtying, result tests. | [Scalar Transport and Electrochemistry](../physics/scalar-transport-electrochemistry.md) |
| `elch`, `sti`, `tsi`, `ssti`, `loma`, `levelset`, `lubrication` | Electrochemistry, scalar/thermo/structure interactions, low-Mach, level set, lubrication dynamics. | [Scalar Transport and Electrochemistry](../physics/scalar-transport-electrochemistry.md), [Fluid and ALE](../physics/fluid-and-ale.md), [Cut and XFEM Geometry](../physics/cut-xfem-geometry.md) |

## Coupling, contact, and geometry modules

| Module | Role | Canonical page |
| --- | --- | --- |
| `adapter` | Field and coupled adapters for structure, fluid, ALE, scatra, poro, FSI, FPSI, SSI, PASI. | [Coupled Multiphysics](../physics/coupled-multiphysics.md) |
| `coupling` | Matching coupling, mortar coupling, volumetric mortar, map/export ownership. | [Coupled Multiphysics](../physics/coupled-multiphysics.md) |
| `fsi`, `fsi_xfem`, `fpsi`, `fs3i`, `ehl`, `ssi`, `pasi`, `fbi` | Major multiphysics orchestration modules. | [Coupled Multiphysics](../physics/coupled-multiphysics.md) |
| `poroelast`, `poroelast_scatra`, `porofluid_pressure_based`, `porofluid_pressure_based_elast`, `porofluid_pressure_based_elast_scatra` | Poroelastic, scalar, porofluid, artery-coupled, and pressure-based coupled systems. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `contact`, `mortar`, `constraint`, `constraint_framework`, `beaminteraction`, `geometry_pair` | Contact/meshtying, mortar, constraints, embedded mesh, beam interactions, geometry pairs. | [Contact, Constraints, and Geometry Pairing](../physics/contact-constraints-geometry.md) |
| `cut`, `xfem` | Cut-cell geometry and extended-FEM interface infrastructure. | [Cut and XFEM Geometry](../physics/cut-xfem-geometry.md) |

## Network, reduced, and particle modules

| Module | Role | Canonical page |
| --- | --- | --- |
| `particle` | Particle algorithm, engine, interactions, rigid bodies, walls, result tests. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `art_net` | Arterial network elements, boundary conditions, time integration. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `red_airways` | Reduced airway/acinus/inter-acinar elements and implicit integration. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `reduced_lung` | Reduced lung and 1D pipe flow models. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `cardiovascular0d` | 0D cardiovascular circuits and structure_new coupling. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `browniandyn` | Brownian dynamics structure model evaluator. | [Poro, Particle, Lung, and Network Models](../physics/poro-particle-lung.md) |
| `stru_multi` | Multiscale structural microstatic support. | [Structure and Solid Elements](../physics/structure-and-solid-elements.md) |

## App targets

| App | Role | Canonical page |
| --- | --- | --- |
| `apps/global_full` | Main `4C` executable and problem dispatch. | [Application Lifecycle](../architecture/application-lifecycle.md) |
| `apps/create_rtdfiles` | Runtime metadata/documentation helper. | [Apps and Tooling](../tools/apps-postprocessing.md) |
| `apps/post_processor` | Post-processing conversion/writer app. | [Apps and Tooling](../tools/apps-postprocessing.md) |
| `apps/post_monitor` | Post-processing monitor app. | [Apps and Tooling](../tools/apps-postprocessing.md) |

## Validation routing

Start with [Testing and Validation](../architecture/testing-validation.md), then follow the owning page above. If a change crosses pages, validate the lowest-level owner first and then one representative runtime problem that consumes it.
