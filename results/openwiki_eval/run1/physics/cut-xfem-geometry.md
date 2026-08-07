---
type: subsystem page
title: Cut, XFEM, Level Set, and Geometric Interfaces
description: Documents cut-cell geometry, XFEM interface state, level-set integration, mesh/side/facet/volume-cell structures, and the cut-test validation surface.
tags: [physics, cut, xfem, geometry]
---

# Cut, XFEM, Level Set, and Geometric Interfaces

`src/cut` and `src/xfem` support nonconforming interfaces, cut-cell integration, level-set intersections, XFluid, and volumetric mortar. They are infrastructure for FSI-XFEM, fluid-XFEM, contact/interface utilities, and coupling.

## Cut module responsibilities

`src/cut` owns geometric decomposition primitives and algorithms:

- mesh, side, element, node, point, line, edge, facet, volume-cell, boundary-cell, integration-cell, and mesh-handle classes;
- intersection kernels, parent intersections, level-set intersections, self-cut, and tet-mesh intersection;
- cycle and graph utilities for point/facet graphs;
- direct divergence and tessellation-related integration;
- cut options, input, output, parallel support, volume integration, and triangulation utilities.

The module depends only on config, core, global_data, and inpar, making it a reusable geometry engine for higher-level physics.

## XFEM responsibilities

`src/xfem` owns extended-FEM state and coupling support:

- condition manager and interface utilities;
- coupling base, mesh, level-set, coupled-level-set, and FPI mesh coupling;
- XFEM discretization and DOF sets;
- edge stabilization, Neumann handling, mesh projection;
- multi-field map extractor;
- xfield state and field coupling;
- xfluid contact communicator and time integration variants.

Fluid-XFEM and FSI-XFEM entrypoints call into `FLD::XFluid::setup_fluid_discretization()` and XFEM managers after filling or preparing structure/fluid/ALE discretizations.

## Level set

`src/levelset` provides level-set algorithms, reinitialization, intersection utilities, and OST time integration. It is selected directly by `entrypoint_switch` for `level_set` and indirectly by XFluid/XFEM workflows.

## Geometry dependencies

Cut/XFEM code is closely related to:

- [FEM Discretization](../core/fem-discretization.md) for nodes/elements/DOFs and specialized discretizations;
- [Coupled Multiphysics](coupled-multiphysics.md) for volumetric mortar and XFEM FSI;
- [Contact, Constraints, and Geometry Pairing](contact-constraints-geometry.md) for geometric interaction and contact interfaces;
- [Fluid and ALE](fluid-and-ale.md) for XFluid runtime.

## Validation

`tests/cut_test` compiles a standalone `cut_test` executable from dozens of generated and hand-authored geometry cases, including intersection, level set, self-cut, triangulation, volume, utility, and fluid-fluid cases. `cut_test_volume.cpp` shows the canonical volume-cut flow: create `Cut::Options`, call `init_for_cuttests()`, disable position finding for matching-boundary cases, construct two `Cut::Mesh` instances sharing the point pool, build hex meshes, create cut-test side ids, allocate a `Cut::plain_element_set`, call `mesh2.cut(mesh1, elements_done)`, call `cutmesh(mesh1)`, and assign other volume cells with `mesh2.assign_other_volume_cells_cut_test(mesh1)`. Simpler wrapper tests create element coordinate matrices and call wrapper helpers such as `create_hex8`, `create_pyramid5_sides`, and `cut_test_cut()`. Source-level FEM geometry tests and XFEM/FSI-XFEM regression inputs should be used when the change affects runtime discretization or coupled physics, not only low-level geometry.
