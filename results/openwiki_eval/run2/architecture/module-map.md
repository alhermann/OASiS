---
type: architecture map
title: Registered module map
description: Inventory of substantial 4C input modules, owning source files, section names, and representative test deck patterns.
tags: [architecture, modules, input-format]
---

# Registered module map

`Global::valid_parameters()` in `src/global_legacy_module/4C_global_legacy_module_validparameters.cpp` composes all user input parameter groups. Use this map to route a deck-writing or code-change task to the owning input file.

| Domain | Main sections | Owning input source | Representative test patterns |
|---|---|---|---|
| Common parser and global setup | `PROBLEM TYPE`, `PROBLEM SIZE`, `DISCRETISATION`, `* DOMAIN`, `* GEOMETRY`, `fields` | `src/global_legacy_module/4C_global_legacy_module_validparameters.cpp`, `src/global_data/4C_global_data_read.cpp` | almost every `tests/input_files/*.4C.yaml` |
| IO | `IO`, runtime VTK/VTP output sections | `src/inpar/4C_inpar_io.cpp`, `src/inpar/4C_inpar_IO_runtime_output*.cpp` | output-heavy structure/fluid tests |
| Linear solvers | `SOLVER 1` ... `SOLVER 9` | `src/core/linear_solver/src/method/4C_linear_solver_method_input.cpp` | XML and solver examples under `tests/input_files/*.xml` |
| Structure/solid | `STRUCTURAL DYNAMIC` and subsections | `src/inpar/4C_inpar_structure.cpp`, `src/structure`, `src/structure_new`, `src/solid_3D_ele` | `beam3*`, `mat_*`, `contact*`, `meshtying*` |
| Fluid | `FLUID DYNAMIC` and stabilization/turbulence subsections | `src/inpar/4C_inpar_fluid.cpp`, `src/fluid`, `src/fluid_ele`, `src/fluid_turbulence` | `f2_*`, `f3_*`, `hdg_*` |
| Scalar transport | `SCALAR TRANSPORT DYNAMIC` and subsections | `src/inpar/4C_inpar_scatra.cpp`, `src/scatra`, `src/scatra_ele` | `scatra_*`, `elch_*` |
| Contact and mortar | `CONTACT DYNAMIC`, mortar/meshtying sections, contact law list | `src/contact/src/4C_contact_input.cpp`, `src/mortar/src/4C_mortar_input.cpp`, `src/contact_constitutivelaw` | `contact2D_*`, `contact3D_*`, `meshtying*` |
| Particles | `PARTICLE DYNAMIC`, `/SPH`, `/DEM`, `/PD`, `PARTICLES` | `src/particle/src/4C_particle_input.cpp`, `src/particle/src/engine` | `particle_sph_*`, `particle_dem_*`, `particle_nointer_*` |
| ALE | `ALE DYNAMIC` | `src/ale/4C_ale_input.cpp` | `ale*` |
| FSI | `FSI DYNAMIC` and coupling sections | `src/inpar/4C_inpar_fsi.cpp`, `src/fsi`, `src/fsi_xfem` | `fsi_*` |
| Porous and poroelastic | `POROELASTICITY DYNAMIC`, porofluid pressure-based sections | `src/poroelast`, `src/porofluid_pressure_based*` | `poro*`, `fpsi*`, `fps3i*` |
| Electrochemistry and S2I | `ELCH CONTROL`, `S2I` sections | `src/elch/4C_elch_input.cpp`, `src/inpar/4C_inpar_s2i.cpp` | `elch_*` |
| Thermo and coupled thermo | `THERMO DYNAMIC`, `TSI`, `STI`, `SSTI` | `src/thermo/src/utils/4C_thermo_input.cpp`, `src/tsi`, `src/sti`, `src/ssti` | `tsi_*`, `thermo*`, `consinter3D_tsi*` |
| Reduced airways/lung/cardiovascular | airway, reduced lung, cardiovascular 0D sections | `src/red_airways`, `src/reduced_lung`, `src/cardiovascular0d` | `cardiovascular0d_*`, airway/lung tests |
| Beam interaction | beam contact/meshtying/potential/crosslinking sections | `src/beaminteraction/src/*_input.cpp` | `beam3*contact*`, `beam*crosslinking*` |
| Geometry helpers | `SEARCH TREE`, `BOUNDINGVOLUME STRATEGY`, `BINNING STRATEGY` | `src/core/fem/src/geometry`, `src/core/geometric_search`, `src/core/binstrategy` | contact, particle, XFEM tests |
| NOX nonlinear solver | NOX solver sections | `src/solver_nonlin_nox/4C_solver_nonlin_nox_input.cpp` | tests using `noxnln` or NOX XML |

## Coverage rule for this wiki

This wiki gives deep, parameter-table coverage to the brief-priority domains: file format, common sections, solvers, structure, fluid, scalar transport, materials, conditions, geometry, contact, and particles. Other registered modules are routed here with owning source anchors so an agent can find their `valid_parameters()` and nearby tests without scanning the repository.
