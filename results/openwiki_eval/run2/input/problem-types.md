---
type: input reference
title: Problem types and required deck families
description: Accepted `PROBLEMTYPE` values and the section families a deck normally needs for common 4C analyses.
tags: [input-format, problem-types]
---

# Problem types and required deck families

`PROBLEM TYPE` is the only top-level group marked required in the common parameter registry. It contains the required `PROBLEMTYPE` selector and optional shape/restart/random-seed controls. The accepted spellings are defined by `Core::string_to_problem_type_map()` in `src/global_legacy_module/4C_global_legacy_module_problem_type_string.hpp`.

## `PROBLEM TYPE` parameters

| Key | Type | Required | Default | Accepted values or notes |
|---|---:|---:|---:|---|
| `PROBLEMTYPE` | enum selector | yes | none | See values below. |
| `SHAPEFCT` | enum selector | no | `polynomial` | Shape-function family for spatial approximation. Accepted values come from `Core::FE::string_to_shape_function_type_map()`. Most decks omit it. |
| `RESTART` | int | no | `0` | Legacy restart flag. Command-line restart options are preferred by current docs. |
| `RANDSEED` | int | no | `-1` | If negative, use current time as random seed. |

## Accepted `PROBLEMTYPE` spellings

Common single-field values:

- `Structure`
- `Fluid`
- `Scalar_Transport`
- `Particle`
- `Ale`
- `Lubrication`
- `Level_Set`
- `Electrochemistry`
- `Cardiac_Monodomain`
- `Thermo`
- `ArterialNetwork`
- `ReducedDimensionalAirWays`
- `Reduced_Lung`
- `Reduced_Lung_1D_Pipe_Flow`
- `Poroelasticity`
- `porofluid_pressure_based`
- `Polymer_Network`
- `Low_Mach_Number_Flow`
- `Fluid_XFEM`

Common coupled values:

- `Fluid_Ale`
- `Fluid_Structure_Interaction`
- `Fluid_Structure_Interaction_XFEM`
- `Fluid_Structure_Interaction_RedModels`
- `Fluid_Beam_Interaction`
- `Fluid_Porous_Structure_Interaction`
- `Fluid_Poro_Structure_Interaction_XFEM`
- `Fluid_Porous_Structure_Scalar_Scalar_Interaction`
- `Poroelastic_scalar_transport`
- `porofluid_pressure_based_elast`
- `porofluid_pressure_based_elast_scatra`
- `Structure_Scalar_Interaction`
- `Scalar_Thermo_Interaction`
- `Thermo_Structure_Interaction`
- `Structure_Scalar_Thermo_Interaction`
- `Thermo_Fluid_Structure_Interaction`
- `Particle_Structure_Interaction`
- `Elastohydrodynamic_Lubrication`
- `Biofilm_Fluid_Structure_Interaction`
- `Gas_Fluid_Structure_Interaction`
- `Fluid_RedModels`
- `NP_Supporting_Procs`

Spelling is case-sensitive because the selector map uses literal strings.

## Section routing by common problem type

The generic parser allows many top-level sections to be omitted, but the selected solver needs a coherent set of sections. Use this as the first-attempt routing table.

| `PROBLEMTYPE` | Must include for a useful deck | Usually include | Main references |
|---|---|---|---|
| `Structure` | `STRUCTURAL DYNAMIC`, structure mesh (`STRUCTURE DOMAIN` or `NODE COORDS` plus `STRUCTURE ELEMENTS`), `MATERIALS`, boundary conditions | `SOLVER n` when `STRUCTURAL DYNAMIC/LINEAR_SOLVER >= 1`, `IO`, functions | [Structural dynamic](../reference/structural-dynamic.md), [Structural theory](../theory/structural.md) |
| `Fluid` | `FLUID DYNAMIC`, fluid mesh, `MAT_fluid` or compatible fluid material, fluid boundary conditions | `FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES`, stabilization subsection, `SOLVER n` | [Fluid dynamic](../reference/fluid-dynamic.md), [Fluid theory](../theory/fluid.md) |
| `Scalar_Transport` | `SCALAR TRANSPORT DYNAMIC`, transport mesh, scalar material such as `MAT_scatra`, Dirichlet/Neumann transport conditions as needed | `SOLVER n`, stabilization, functions for velocity or initial field | [Scalar transport dynamic](../reference/scalar-transport-dynamic.md), [Scalar theory](../theory/scalar-transport.md) |
| `Particle` | `PARTICLE DYNAMIC`, `PARTICLES` legacy section or generated particle source, particle materials | `/SPH`, `/DEM`, or `/PD` subsection matching `INTERACTION` or peridynamic flags | [Particle dynamic](../reference/particle-dynamic.md), [Particle theory](../theory/particles.md) |
| Contact within structure | `CONTACT DYNAMIC`, contact/meshtying condition sections, structural sections | `CONTACT CONSTITUTIVE LAWS`, `BINNING STRATEGY`, `SOLVER n` | [Contact dynamic](../reference/contact-dynamic.md), [Contact theory](../theory/contact.md) |
| FSI and coupled problems | Dynamic sections for every participating field plus coupling-specific section, meshes/materials for every field | Field-specific solvers and coupling controls | [Module map](../architecture/module-map.md) |

## A safe first strategy

1. Pick the exact `PROBLEMTYPE` spelling first.
2. Add only the dynamic section for that physics, then set `LINEAR_SOLVER: 1` if the examples for that physics do so.
3. Add `SOLVER 1` with `SOLVER: "UMFPACK"` for the simplest direct solve, or configure an iterative solver in [Solvers](../reference/solvers.md).
4. Use a generated `<FIELD> DOMAIN` while prototyping; it avoids malformed legacy node/element strings.
5. Add `MATERIALS` after choosing element types because elements reference material ids.
6. Add boundary conditions and topology sets last; conditions are list-shaped and parse-sensitive.
