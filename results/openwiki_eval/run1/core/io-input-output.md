---
type: subsystem page
title: IO, Input, and Output
description: Covers 4C input-file parsing, input specs, mesh readers, output control, visualization writers, restart/output data flow, and post-processing handoff.
tags: [core, io, input, output]
---

# IO, Input, and Output

`src/core/io` turns user input into validated sections and turns runtime state into control, restart, visualization, and post-processing output. Its highest-level entry is `Core::IO::InputFile`, described in [Input Schema and Global Problem](../architecture/input-schema-global-problem.md). This page focuses on the IO subsystem and its extension seams.

## Input-file API

`Core::IO::InputFile` owns raw section storage and schema-aware matching. It exposes:

- `read(path)` to load YAML/JSON with `INCLUDES`;
- `file_for_section(name)` for diagnostics and provenance;
- `in_section_rank_0_only(name)` for legacy string sections such as element and topology sections;
- `match_section(name, InputParameterContainer&)` for validated sections;
- `has_section(name)` for conditional parsing;
- `get_comm()` for the collective communicator;
- `emit_metadata(YamlNodeRef)` for schema generation;
- `write_as_yaml(...)` for normalized output.

`InputSpec`, `InputSpecBuilders`, and validators are the preferred way to describe valid input. Module parameter files under `src/inpar` and module-local `*_input.cpp` files build these specs. Python schema tooling consumes the metadata emitted by this API.

## Mesh and discretization input

Mesh-related IO is split between legacy accepted section names and structured readers:

- legacy sections include `NODE COORDS`, `DNODE-NODE TOPOLOGY`, `DLINE-NODE TOPOLOGY`, `DSURF-NODE TOPOLOGY`, `DVOL-NODE TOPOLOGY`, `STRUCTURE ELEMENTS`, `FLUID ELEMENTS`, `TRANSPORT ELEMENTS`, `ALE ELEMENTS`, `THERMO ELEMENTS`, `ARTERY ELEMENTS`, `REDUCED D AIRWAYS ELEMENTS`, and `PARTICLES`;
- `Core::IO::MeshReader` is created by `Global::read_discretization` with rebalance and geometric-search parameters;
- `Core::IO::MeshInput` types support mesh construction for FEM and specialized reduced models;
- Exodus, Gmsh, VTU, and HDF readers/writers live in the same core IO package.

A reader should populate nodes/elements/conditions but not assume final DOF numbering. The owning physics entrypoint controls `fill_complete()` order.

## Output control and visualization

IO output surfaces include:

- `Core::IO::OutputControl` for time/restart/output naming and scheduling;
- HDF output through `4C_io_hdf`;
- VTU/VTP/VTK runtime writers through visualization manager/writer factory classes;
- `DiscretizationVisualizationWriterMesh` used by reduced lung runtime visualization;
- runtime CSV writers for monitor-like outputs;
- walltime-based restart helpers.

Output naming is adjusted by `update_io_identifiers` before input reading so grouped communicator runs can produce distinct identifiers.

## Post-processing handoff

The standalone post-processing applications use `src/post` and core IO writer classes. Solver modules produce output files and result descriptions; post-processing tools filter or convert those files. See [Apps and Tooling](../tools/apps-postprocessing.md) for the `post_processor` and `post_monitor` apps.

## Tests and validation

Focused IO tests live under `src/core/io/tests` and `/unittests/io`. They cover command-line helpers, input fields, input files, input parameter containers, input specs and validators, mesh IO, value parsing, VTU, Gmsh, Exodus, output control, and pstream behavior. The `/unittests/io/CMakeLists.txt` currently attaches these tests to the `mat` module because they use concrete material definitions; keep that dependency in mind when refactoring IO tests.

Minimal checks for input changes:

```bash
cmake --build build --target unittests
ctest --test-dir build -R io --output-on-failure
build/4C --parameters > /tmp/4c-parameters.yaml
validate-test-files --metadata /tmp/4c-parameters.yaml tests/input_files
```

Use the exact built executable path for the local build tree.
