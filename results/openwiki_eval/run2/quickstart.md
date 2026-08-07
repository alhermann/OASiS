---
type: wiki entrypoint
title: 4C input deck quickstart
description: High-level map for writing accepted 4C `.4C.yaml` decks, with routing to format, parameter references, materials, conditions, geometry, solver theory, examples, and source validation.
tags: [quickstart, input-format]
---

# 4C input deck quickstart

This wiki is optimized for an autonomous agent that must write a `.4C.yaml` deck accepted by the 4C binary on the first attempt. Code architecture is secondary; start with the input format and only then follow links into theory or source architecture.

## Fast path to an accepted deck

1. Read [4C input file format](input/file-format.md). Do not nest slash-named sections; `FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES` is a top-level key.
2. Pick the exact `PROBLEMTYPE` spelling from [Problem types and required deck families](input/problem-types.md).
3. Add the physics dynamic section:
   - [STRUCTURAL DYNAMIC](reference/structural-dynamic.md)
   - [FLUID DYNAMIC](reference/fluid-dynamic.md)
   - [SCALAR TRANSPORT DYNAMIC](reference/scalar-transport-dynamic.md)
   - [CONTACT DYNAMIC](reference/contact-dynamic.md)
   - [PARTICLE DYNAMIC](reference/particle-dynamic.md)
4. Add [Linear solver sections](reference/solvers.md) for every positive `LINEAR_SOLVER` id.
5. Add [Materials and contact constitutive laws](reference/materials.md) for every `MAT` id used by elements or particles.
6. Add [Geometry, domains, elements, and topology](reference/geometry-and-elements.md).
7. Add [Design condition sections](reference/conditions.md) and matching `FUNCTn` sections for nonzero function ids.
8. Compare against [Minimal deck patterns](examples/minimal-decks.md) and a nearby full deck in `tests/input_files`.

## Main concept map

| Need | Wiki page |
|---|---|
| Root YAML shape, `.yaml`/`.yml`/`.json`, `INCLUDES`, `input_version`, duplicates, unknown keys, strict scalar typing | [4C input file format](input/file-format.md) |
| Accepted `PROBLEMTYPE` values and section families by problem type | [Problem types](input/problem-types.md) |
| `PROBLEM SIZE`, `DISCRETISATION`, `IO`, `fields` | [Common sections](reference/common-sections.md) |
| `SOLVER 1` through `SOLVER 9` | [Solvers](reference/solvers.md) |
| Structural time integration, prestress, damping, Newton controls | [STRUCTURAL DYNAMIC](reference/structural-dynamic.md) and [Structural theory](theory/structural.md) |
| Fluid physical types, time integration, stabilization, turbulence | [FLUID DYNAMIC](reference/fluid-dynamic.md) and [Fluid theory](theory/fluid.md) |
| Scalar transport solver/time/velocity/reaction controls | [SCALAR TRANSPORT DYNAMIC](reference/scalar-transport-dynamic.md) and [Scalar theory](theory/scalar-transport.md) |
| Contact strategies, friction, penalty, Nitsche, validation failures | [CONTACT DYNAMIC](reference/contact-dynamic.md) and [Contact theory](theory/contact.md) |
| SPH, DEM, peridynamics, particle materials | [PARTICLE DYNAMIC](reference/particle-dynamic.md) and [Particle theory](theory/particles.md) |
| Material selectors and required/defaulted keys | [Materials](reference/materials.md) |
| Dirichlet, Neumann, topology ids, named node sets, functions | [Conditions](reference/conditions.md) |
| Generated domains, external meshes, legacy element strings | [Geometry and elements](reference/geometry-and-elements.md) |
| Where to change input specs in source | [Input registry architecture](architecture/input-registry.md) |
| Registered module/source map beyond the main brief | [Registered module map](architecture/module-map.md) |

## Task-routing table

| Change or authoring intent | Start here | Source entrypoints | Focused tests/examples | Minimal validation |
|---|---|---|---|---|
| Write a new fluid Stokes/Navier-Stokes deck | [FLUID DYNAMIC](reference/fluid-dynamic.md) | `src/inpar/4C_inpar_fluid.cpp`, `src/fluid_ele` | `tests/input_files/f3_stokes_residualbased_rotboxgeom.4C.yaml`, `f2_*`, `f3_*` | `./4C deck.4C.yaml out` or `./4C --parameters` to inspect keys |
| Write a structure/beam deck | [STRUCTURAL DYNAMIC](reference/structural-dynamic.md) | `src/inpar/4C_inpar_structure.cpp`, `src/solid_3D_ele`, `src/beam3` | `tests/input_files/beam3r_line2_static_test1.4C.yaml`, `mat_*` | Direct `SOLVER 1: UMFPACK` first, then run deck |
| Write scalar diffusion/advection/reaction | [SCALAR TRANSPORT DYNAMIC](reference/scalar-transport-dynamic.md) | `src/inpar/4C_inpar_scatra.cpp`, `src/scatra_ele` | `tests/input_files/scatra_1D_line2_diffnumdof.4C.yaml`, `scatra_*` | Ensure `MAT_scatra` and transport BC vector lengths match |
| Add contact to structure | [CONTACT DYNAMIC](reference/contact-dynamic.md) | `src/contact/src/4C_contact_input.cpp`, `src/mortar`, `src/contact_constitutivelaw` | `contact2D_*`, `contact3D_*`, `meshtying*` | Check positive penalty data for penalty/Nitsche/Uzawa and valid system/strategy pair |
| Write SPH/DEM/peridynamic particle deck | [PARTICLE DYNAMIC](reference/particle-dynamic.md) | `src/particle/src/4C_particle_input.cpp` | `particle_sph_*`, `particle_dem_*`, `particle_sph_*pdbody*` | Match `INTERACTION` to subsection and material selector |
| Add or edit material | [Materials](reference/materials.md) | `src/global_legacy_module/4C_global_legacy_module_validmaterials.cpp` | `mat_*`, physics-specific deck using selector | Add a deck with `MATERIALS` entry and referenced `MAT` id |
| Add a boundary/load condition | [Conditions](reference/conditions.md) | `src/global_legacy_module/4C_global_legacy_module_validconditions.cpp`, module condition registrars | Decks with `DESIGN ... CONDITIONS` | Verify `E`/`ENTITY_TYPE`/`NODE_SET_NAME` identification and vector sizes |
| Change parser rules | [Input registry architecture](architecture/input-registry.md) | `src/core/io/src/4C_io_input_file.cpp`, `4C_io_input_spec*` | `src/core/io/tests` | Run parser unit tests and `./4C --parameters` |

## First-attempt parse checklist

- `PROBLEM TYPE` is present and `PROBLEMTYPE` spelling matches [Problem types](input/problem-types.md).
- Slash-named sections are top-level keys.
- Every required key in a present group/list item is set; defaulted keys may be omitted.
- Quoted numbers are not used where a key expects `int`, `double`, or `bool`.
- Every positive `LINEAR_SOLVER` id has a `SOLVER n` section with required `SOLVER`.
- Every element/particle `MAT` id has exactly one non-negative, unique `MATERIALS` entry.
- Every condition uses either `E` plus optional `ENTITY_TYPE`, or `NODE_SET_NAME`, not both.
- `E` is positive for legacy ids and non-negative for `element_block_id`/`node_set_id`.
- Arrays sized by `NUMDOF`, `NUMMAT`, `NUMSCAL`, etc. have exactly that length.
- Every nonzero function id has a matching `FUNCTn` section.
- Included files do not define duplicate top-level sections.

## Validation commands

When a 4C build is available:

```bash
./4C --parameters > input-metadata.yaml
./4C tests/input_files/f3_stokes_residualbased_rotboxgeom.4C.yaml tmp_stokes
./4C your_deck.4C.yaml your_output
```

`--parameters` is the source-generated schema-like metadata. It is the best way to confirm current defaults and enum values after changing source.

## Backlog and valid deferrals

- This initialization deeply covers the brief-priority sections and modules. Many registered coupled physics modules are routed in [Registered module map](architecture/module-map.md) but do not yet have full parameter-table pages because the brief prioritizes agents writing accepted common decks over exhaustive coverage of every coupled module. Source anchor: `Global::valid_parameters()` in `src/global_legacy_module/4C_global_legacy_module_validparameters.cpp`.
- For specialized materials not tabled in [Materials](reference/materials.md), use `src/global_legacy_module/4C_global_legacy_module_validmaterials.cpp` plus the nearest `tests/input_files` deck with the same `MAT_*` selector.
