---
type: entrypoint guide
title: 4C OpenWiki Quickstart
description: Entry point for navigating the 4C repository wiki, with a system map, major concepts, source-routing table, focused tests, and minimal validation guidance.
tags: [quickstart, navigation]
---

# 4C OpenWiki Quickstart

4C is a C++20, MPI-based multiphysics simulation framework. The repository builds one broad `lib4C` library plus the main `4C` executable and auxiliary tooling. Most runtime behavior flows through global input/schema setup, `Global::Problem`, named FEM discretizations, field adapters, and problem-type dispatch.

## Start here by intent

| Intent | Read first | Source entrypoints and symbols | Focused tests or validation |
| --- | --- | --- | --- |
| Understand build targets, dependencies, and CMake modules | [Build and Runtime Topology](architecture/build-and-runtime.md) | `/CMakeLists.txt`, `src/CMakeLists.txt`, `cmake/functions/four_c_auto_define_module.cmake` | Configure and build `4C`; build `unittests`. |
| Trace a simulation run | [Application Lifecycle](architecture/application-lifecycle.md) | `apps/global_full/4C_global_full_main.cpp`, `entrypoint_switch()` | Run a tiny input or `4C --parameters`; inspect timing and result-test output. |
| Change input schema or metadata | [Input Schema and Global Problem](architecture/input-schema-global-problem.md), [IO, Input, and Output](core/io-input-output.md) | `Core::IO::InputFile`, `InputSpec`, `Global::set_up_input_file`, `Global::read_parameter` | IO tests, `4C --parameters`, `create-schema-files`, `validate-test-files`. |
| Add a material, element, condition, parameter, or problem type | [Module Registration and Extension Surfaces](extension/module-registration.md), [Materials](physics/materials.md) | `ModuleCallbacks`, `global_legacy_module_callbacks`, `Mat::make_parameter`, `entrypoint_switch` | Unit test for the new surface plus one metadata/regression check. |
| Work on FEM maps, DOFs, cloning, or deal.II wrappers | [FEM Discretization](core/fem-discretization.md) | `Core::FE::Discretization`, `fill_complete`, `evaluate`, `DealiiWrappers::Context` | FEM tests, linalg assembly tests, deal.II tests when enabled. |
| Work on sparse matrices, solvers, or NOX | [Linear Algebra and Solvers](core/linalg-solvers.md) | `Core::LinAlg::SparseMatrix`, sparse algebra utilities, `NOX::Nln::Problem` | Linalg tests, solver-focused regression input. |
| Change structural elements or structural dynamics | [Structure and Solid Elements](physics/structure-and-solid-elements.md) | `caldyn_drt`, `dyn_nlnstructural_drt`, `Adapter::Structure*`, solid/beam/shell element modules | Solid/structure unit tests and representative structural regression. |
| Change fluids, ALE, turbulence, XFluid, or fluid elements | [Fluid and ALE](physics/fluid-and-ale.md) | `dyn_fluid_drt`, `fluid_ale_drt`, `fluid_xfem_drt`, `Adapter::FluidBaseAlgorithm`, `FLD::XFluid::setup_fluid_discretization` | Fluid/fluid_ele regressions, XFluid or ALE cases when relevant. |
| Change scalar transport, electrochemistry, thermo, STI/TSI/SSTI | [Scalar Transport and Electrochemistry](physics/scalar-transport-electrochemistry.md) | `scatra_dyn`, `Thermo::BaseAlgorithm`, `sti_dyn`, `tsi_dyn_drt`, `ssti_drt`, scatra_ele calculators | Scatra tests and selected ELCH/STI/TSI/SSTI regression inputs. |
| Change FSI or generic coupling/mortar/volmortar | [Coupled Multiphysics](physics/coupled-multiphysics.md) | `fsi_ale_drt`, `Coupling::Adapter::Coupling`, `CouplingMortar`, `VolMortarCoupl` | Minimal coupled input; cut/mortar tests for nonmatching geometry. |
| Change contact, constraints, beam interactions, or geometry pairs | [Contact, Constraints, and Geometry Pairing](physics/contact-constraints-geometry.md) | `CONTACT::Manager`, `CONTACT::STRATEGY::Factory`, `constraint`, `constraint_framework`, `geometry_pair` | Geometry-pair tests, contact reference/config tests, contact regression. |
| Change cut-cell, XFEM, level-set geometry | [Cut and XFEM Geometry](physics/cut-xfem-geometry.md) | `Cut::Mesh`, `Cut::Options`, `xfem` managers, levelset algorithms | Build/run `cut_test`; run XFEM or FSI-XFEM regression. |
| Change poro, particles, lung/network, cardiovascular0d | [Poro, Particle, Lung, and Network Models](physics/poro-particle-lung.md) | `poroelast_drt`, `particle_drt`, `ReducedLung::reduced_lung_main`, arterial/airway entrypoints | Particle/reduced-lung tests and representative network/poro regressions. |
| Work on apps, post-processing, docs, Python utilities, or bindings | [Applications and Tooling](tools/apps-postprocessing.md) | `apps/post_processor`, `src/post`, `utilities/four_c_python`, `utilities/py4C` | Script-level checks, schema validation, py4C import in an enabled build. |
| Find the canonical owner for a directory | [Module Catalog](modules/catalog.md) | All CMake-backed `src` modules and app targets | Follow the owning page and run its validation. |

## Core concepts

- **One global runtime path:** `4C` initializes MPI, Kokkos, singleton ownership, communicators, input, and `Global::Problem`, then dispatches by `Core::ProblemType`.
- **Schema before simulation:** input sections are registered as `InputSpec`s or legacy sections before reading; `4C --parameters` emits the same schema for tooling.
- **Named discretizations:** physics modules request named discretizations such as `structure`, `fluid`, `scatra`, `ale`, and `thermo` from `Global::Problem`.
- **Fill-complete matters:** maps, DOFs, element pointers, node adjacency, element initialization, and condition geometry are established before element evaluation. Coupled workflows rely on documented DOF ordering.
- **Registration is explicit:** new materials, elements, functions, conditions, parameters, and problem types need implementation, input schema, factory/callback registration, metadata visibility, and tests.
- **Result tests are part of runtime:** most problem entrypoints create field tests after integration and call `Global::Problem::test_all(comm)`.

## Minimal validation ladder

1. Build the affected target or module tests.
2. Run the focused unit tests named on the owning page.
3. If input schema changed, run `4C --parameters`, schema generation, and YAML validation.
4. If field behavior changed, run one minimal input regression for that field.
5. If coupling, maps, restart, or output changed, add a coupled or restart regression before broad CTest.

## Backlog

No source areas were intentionally deferred. The module catalog assigns every CMake-backed `src` module and app target to a canonical page; optional features such as deal.II and py4C are documented with their build gates.
