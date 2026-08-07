---
type: extension guide
title: Module Registration and Extension Surfaces
description: Documents the callback, factory, and registration surfaces used to add materials, elements, functions, conditions, parameters, and problem entrypoints safely.
tags: [extension, registration, modules]
---

# Module Registration and Extension Surfaces

4C is modular in CMake but most runtime extension surfaces converge through global registries and callbacks. The central type is `ModuleCallbacks` in `src/module_registry/4C_module_registry_callbacks.hpp`; the central legacy implementation is `src/global_legacy_module/4C_global_legacy_module.cpp`.

## `ModuleCallbacks`

`ModuleCallbacks` contains optional callbacks:

| Callback | Responsibility |
| --- | --- |
| `RegisterParObjectTypes` | Force registration of all `Core::Communication::ParObject`-derived types that can be communicated over MPI or restored from restart data. |
| `AttachFunctionDefinitions` | Add module-specific functions to `Core::Utils::FunctionManager`. |
| `valid_result_description_lines` | Return `RESULT DESCRIPTION` input spec lines for result tests. |
| `materials` | Return material `InputSpec`s keyed by `Core::Materials::MaterialType`. |
| `conditions` | Return condition definitions known to the module. |
| `parameters` | Return valid top-level input sections for module parameters. |

The main executable uses the global callback object before input reading and during input-schema construction. If a feature is not registered here or through the current legacy module, it may compile but be invisible to input, restart, metadata, or result testing.

## Legacy registration hub

`global_legacy_module_callbacks()` currently includes many headers and forces type singletons for elements, nodes, materials, particle objects, conditions, functions, and result tests. This is not a thin file: it is the compatibility hub that connects old-style input, element type names, material type names, and ParObject factories.

Because registration is side-effect based, adding a new element or material usually requires touching more than the implementation file.

## Change recipes

### Add a new material

1. Implement parameter and material classes under `src/mat` or the owning material module.
2. Add the material enum to legacy material definitions if needed.
3. Add an `InputSpec` through the material callback or valid material table.
4. Add a `Mat::make_parameter` case in `4C_mat_material_factory.cpp` so validated input creates the correct parameter object.
5. Ensure `create_material()` returns the runtime material object.
6. Include/register the material type in the legacy module if it participates in restart/communication or metadata.
7. Add focused material tests under `/unittests/mat` or module-local tests and a representative regression if behavior affects a physics solver.

### Add a new element type

1. Implement element, element type singleton, input spec, and evaluation action behavior in the owning element module.
2. Add the module dependency in its `CMakeLists.txt` if it uses new core/physics services.
3. Register the element type in `global_legacy_module` so `ElementDefinition` and ParObject creation can find it.
4. Add legacy element metadata support if the input element line has a new shape or section.
5. Add or update fill-complete, cloning, and material setup paths if the element is created indirectly.
6. Validate with element unit tests plus a minimal input-file regression.

### Add a new input parameter or condition

1. Define the parameter in `src/inpar` or module-local `*_input.cpp` using `InputSpecBuilders`.
2. Return it through `ModuleCallbacks::parameters` or the relevant `valid_parameters()` aggregation.
3. For conditions, return `Core::Conditions::ConditionDefinition` through the condition callback.
4. Update metadata/schema tests and run `4C --parameters` plus Python schema validation.

### Add a new problem entrypoint

1. Add a `Core::ProblemType` value and string mapping in the legacy problem-type definitions.
2. Add input parameters that select the new problem type.
3. Implement a runtime entrypoint that reads `Global::Problem`, fills discretizations in a documented order, handles restart/post-setup, runs the algorithm, creates result tests, and calls `test_all`.
4. Add the `entrypoint_switch()` case in `apps/global_full/4C_global_full_entrypoint_switch.cpp`.
5. Add regression input and focused unit tests for any new core abstractions.

## Public extension surface summary

The complete change surface for user-visible extensions is: implementation class, CMake module, enum/string identifiers, input spec, metadata emission, factory or callback registration, runtime dispatch/consumer, restart/ParObject registration when relevant, focused tests, and at least one validation command.
