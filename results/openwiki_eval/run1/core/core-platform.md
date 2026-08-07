---
type: subsystem page
title: Core Platform Services
description: Summarizes the cross-cutting services under `src/core`, including communication, utility infrastructure, geometric search, rebalance, binning, material base types, and their ownership boundaries.
tags: [core, platform, infrastructure]
---

# Core Platform Services

`src/core` is the shared substrate beneath every physics module. Its `CMakeLists.txt` uses `four_c_auto_define_module(NO_CYCLES)`, making it the dependency root for higher-level modules. Detailed core domains have their own pages: [IO and Output](io-input-output.md), [FEM Discretization](fem-discretization.md), and [Linear Algebra and Solvers](linalg-solvers.md).

## Major core subpackages

| Subpackage | Responsibilities | Primary consumers |
| --- | --- | --- |
| `comm` | MPI utilities, communicator creation, export/import helpers, pack/unpack buffers, `ParObject` and `ParObjectFactory` registration. | Main executable, element/material serialization, parallel mesh/data transfer. |
| `io` | Input files, schema specs, mesh readers, output control, HDF/VTK/VTU/Exodus/Gmsh, runtime visualization. | Global input, apps, post-processing, every problem entrypoint. |
| `fem` | Nodes, elements, conditions, discretizations, DOF sets, geometry, NURBS, assembly utilities, shape functions. | All FE physics, coupling, contact, cut/XFEM, material evaluation. |
| `linalg` | Epetra-backed maps, vectors, sparse/dense matrices, tensor utilities, sparse assembly, projections. | Solvers, element assembly, coupling matrices. |
| `linear_solver` | Trilinos solver and preconditioner configuration, direct/iterative methods, MueLu/Ifpack/Teko integration, Thyra utilities. | NOX, physics algorithms, reduced models. |
| `geometric_search` | Bounding volumes, distributed trees, BVH, matching octree, visualization. | Coupling, contact, XFEM, rebalance, mortar. |
| `rebalance` and `binstrategy` | Partitioning, binning-based redistribution, graph-based rebalance, meshfree bins. | FSI point coupling, scatra heterogeneous reactions, mesh partitioning. |
| `utils` | Exceptions, assertions, function libraries/managers, symbolic expressions, local numeric methods, random helpers, STL extensions, result tests. | Input functions, tests, material laws, algorithms. |
| `material` | Abstract material base and parameter base classes used by `src/mat`. | Material factory and element material access. |
| `legacy_enum_definitions` | Legacy material, condition, and action enumerations. | Global legacy module, input parsing, factories. |

## Communication and registration

The executable calls `global_legacy_module_callbacks().RegisterParObjectTypes()` before reading the problem. The callback forces references to every registered `ParObjectType` and material/element type so the core `ParObject` registry can create objects received over MPI or read from restart data. The callback type itself lives in [Module Registration and Extension Surfaces](../extension/module-registration.md).

A typical cross-rank data path is:

1. a module derives an object from `Core::Communication::ParObject` and provides a type singleton;
2. the legacy module callback forces `.instance().name()` side effects;
3. pack/unpack utilities serialize objects through core communication buffers;
4. exporter/importer wrappers exchange map-owned vectors or objects.

Do not introduce runtime paths that depend on a registered type before the registration callback has run.

## Exceptions and assertions

Core utilities define `FOUR_C_THROW`, `FOUR_C_ASSERT`, and `Core::Exception`. Unit-test executables replace `FOUR_C_THROW` with a throwing `Core::Exception`, so tests can use `EXPECT_THROW(..., Core::Exception)`. Production `main()` catches `Core::Exception`, prints a stack trace, coordinates barriers for multi-group execution, and aborts MPI.

## Function management

Input-defined functions are collected through `Core::Utils::FunctionManager`. `Global::read` attaches module-specific function definitions through `ModuleCallbacks::AttachFunctionDefinitions`, and the schema exports `FUNCT<n>` metadata. Physics modules then query functions through `Global::Problem::instance()->function_manager()` or module-specific parameter containers. Adding a function requires both the implementation and a callback/input-spec registration so metadata and validation remain consistent.

## Geometric search and rebalance

The geometric search package provides search trees and distributed matching utilities. It is a dependency of contact, coupling, XFEM, FEM geometry, rebalance, and several physics modules. Binning and rebalance are used when discretizations must be redistributed after cloning or when point/volume coupling needs nearby entities. Examples in source:

- FSI point coupling rebalances the structure discretization by binning before coupling setup.
- Scalar transport rebalances scatra discretizations when `ScatraHeteroReactionSlave` conditions exist.
- Reduced lung builds a synthetic discretization and constructs row/domain/column maps for parallel equations.

The invariant is that owned, ghosted, row, column, and DOF maps must be rebuilt after redistribution and before element evaluation or sparse assembly.

## Scope boundaries

Core should not encode physics policy. It provides generic maps, discretizations, conditions, IO, search, and utilities. Physics modules own problem-specific parameter groups, element actions, material selection, coupling order, and result tests. When a change requires both a generic primitive and a physics use case, add the generic primitive under `src/core` with focused core tests, then wire the physics behavior in its module with a regression or field test.
