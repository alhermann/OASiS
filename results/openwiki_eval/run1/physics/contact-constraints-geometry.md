---
type: subsystem page
title: Contact, Constraints, Mortar, and Geometry Pairing
description: Documents contact managers, mortar/meshtying strategies, constraint and constraint-framework modules, beam interaction, geometry_pair, and their input and map invariants.
tags: [physics, contact, constraints, geometry]
---

# Contact, Constraints, Mortar, and Geometry Pairing

Contact and constraint behavior spans `src/contact`, `src/mortar`, `src/constraint`, `src/constraint_framework`, `src/beaminteraction`, and `src/geometry_pair`. These modules are consumed by structure, FSI, poro, SSI, TSI, beam, and embedded-mesh workflows.

## Contact manager and strategy lifecycle

`CONTACT::STRATEGY::Factory::read_and_check_input` reads the global mortar coupling, contact dynamic, wear, and TSI-contact parameter lists from `Global::Problem`, plus problem type, dimension, and spatial approximation. It rejects unsupported combinations before strategy construction: invalid parallel redistribution settings, TSI redistribution except for Nitsche, MueLu-preconditioned saddle-point contact with redistribution, adhesion combined with wear or friction, nonpositive normal/tangential penalty parameters, invalid Uzawa settings, 3D Tresca outside Nitsche, 3D friction without semi-smooth Newton, 3D crosspoint modification, incompatible Petrov-Galerkin or condensed Lagrange multiplier options, and other mortar/contact combinations. It also emits warnings for zero search inflation and experimental both-sided wear.

`CONTACT::Manager` in `src/contact/src/4C_contact_manager.cpp` is constructed with a filled discretization and `alphaf`. It:

1. stores the discretization communicator;
2. reads and checks contact input parameters;
3. asserts the discretization is fill-complete;
4. retrieves `Contact` conditions;
5. filters beam-to-solid contacts for the beam3contact framework;
6. groups matching contact conditions by `InterfaceID`;
7. validates self-contact and master/slave side data;
8. reads strategy, wear, friction, adhesion, NURBS, constraint-direction, and algorithm settings;
9. creates local interface parameter lists;
10. creates interfaces and selects Lagrange, penalty, Nitsche, poro, TSI, wear, SSI, or meshtying strategies;
11. manages new Lagrange multiplier DOFs above existing displacement DOFs.

Input invariants are strict: contact problems must be 2D or 3D, the discretization must be filled, every non-self group needs matching conditions, and `InterfaceID`/`Side` parameters drive grouping.

## Mortar and meshtying

`src/mortar` provides interface and manager infrastructure used by contact, meshtying, and nonmatching coupling. `src/coupling` uses mortar for generic nonmatching transfers; see [Coupled Multiphysics](coupled-multiphysics.md). Contact-specific meshtying strategies include Lagrange, penalty, poro, and bridge classes.

## Constraint modules

`src/constraint` contains classical constraints: constraint elements, DOF sets, managers, multipoint constraints, penalty constraints, spring-dashpot constraints, Lagrange/penalty NOX linearsystems, and solvers. It is used by structure and coupled problems.

`src/constraint_framework` is a newer framework for embedded mesh and submodel evaluators. It owns equations, input, model evaluators, embedded mesh solid-to-solid mortar utilities, MPC and nullspace submodel evaluators, and parameters. It depends on cut, geometry_pair, inpar, and structure_new.

## Geometry pair and beam interaction

`src/geometry_pair` provides reusable geometric interaction kernels:

- line-to-line, line-to-surface, and line-to-volume pairs;
- Gauss-point projection and segmentation variants;
- scalar type and evaluation data helpers;
- element face and utility functions.

`src/beaminteraction` builds beam-specific interaction evaluators on top of `beam3`, `geometry_pair`, `contact`, `coupling`, and structural modules. It owns beam interaction data, conditions, submodel evaluator factory/generic evaluator, crosslinking links/nodes, and structure model evaluators.

## Extension and validation

Adding a contact or constraint mode requires input-spec changes, condition definitions, strategy/factory registration, map/DOF allocation decisions, and tests that cover both setup validation and solve behavior. Focused tests include `/unittests/geometry_pair`, module-local geometry-pair tests, `src/contact/tests/4C_contact_element_reference_configuration_test.cpp`, contact constitutive-law input/parameter tests when laws change, beaminteraction regressions, contact/constraint input regressions, and coupled tests when mortar or map ownership changes.
