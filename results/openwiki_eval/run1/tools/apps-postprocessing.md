---
type: tooling page
title: Applications, Post-Processing, Python Utilities, and Documentation Tooling
description: Covers auxiliary app targets, post-processing modules, documentation/dependency scripts, the `four_c_python` utility package, and optional `py4C` Python bindings.
tags: [tools, apps, python, post-processing]
---

# Applications, Post-Processing, Python Utilities, and Documentation Tooling

The main solver app is documented in [Application Lifecycle](../architecture/application-lifecycle.md). This page covers the other app targets and tooling surfaces under `/apps`, `/src/post`, `/doc`, `/dependencies`, `/docker`, and `/utilities`.

## App targets

| Target directory | Role |
| --- | --- |
| `apps/create_rtdfiles` | Generates runtime documentation/metadata files from 4C input and parameter definitions. |
| `apps/post_processor` | Reads solver output and writes converted post-processing data, using `src/post` writers and core IO. |
| `apps/post_monitor` | Monitors post-processing or runtime result data. |
| `apps/global_full` | Main `4C` executable, documented separately. |

These targets are included by `apps/CMakeLists.txt`; the top-level `full` target depends on the auxiliary app targets in addition to `4C` and `cut_test`.

## `src/post`

`src/post` contains post-processing infrastructure:

- `Post::WriterBase` and derived VTK/Ensight writers;
- VTU/VTI writer variants including node-based writer;
- `Post::FilterBase` for filtering data;
- common post-processing helpers.

Post-processing depends on `global_legacy_module` and `inpar`, because it needs to understand legacy element/material metadata and input/output parameters.

## Documentation and dependencies

`/doc/README.md` documents two build targets:

```bash
cmake --build . --target documentation
cmake --build . --target doxygen
```

The Doxygen target requires configuring with `FOUR_C_ENABLE_DOXYGEN=ON`. Dependency scripts under `/dependencies/current/<dependency>/install.sh` install external dependencies, and `/docker` contains containerized build/runtime support referenced by the top-level README.

## `utilities/four_c_python`

`utilities/four_c_python/pyproject.toml` defines the `four_c_python` package for development, CI, metadata, testing, and post-processing helpers. Script families include:

| Script family | Commands | Purpose |
| --- | --- | --- |
| Development checks | `check-file-header`, `check-filenames`, `check-includes`, `check-includes-for-cycles`, `check-header-guards`, `check-test-files`, `check-companion-test-files`, `check-preprocessor`, `check-python-files`, `check-workflow-dependencies-hash` | Enforce repository conventions, include hygiene, file naming, header guards, workflow dependency hashes, and test placement. |
| CI helpers | `chunk-test-suite`, `clang-tidy-filter-config`, `clang-tidy-filter-output` | Split test suites and process clang-tidy configuration/output. |
| Metadata/schema | `create-schema-files`, `validate-test-files` | Convert metadata emitted by `4C --parameters` into JSON Schema and validate YAML test inputs. |
| Build/test comparison | `diff-with-tolerance`, `vtk-compare`, `post-processing-comparison`, `check-restart-step` | Compare numerical output, VTK data, post-processing output, and restart steps. |

The metadata flow is: build `4C`, run `4C --parameters` to emit YAML metadata, run `create-schema-files` to create schema files, then run `validate-test-files --schema <schema.json> <yaml files...>` against input files. The validator loads the JSON schema, validates each YAML file with `jsonschema_rs`, prints `passed`, `failed`, or `error`, lists all files with validation problems, and exits with status 1 when any file fails. This flow validates input-surface changes without running every solver.

## Optional `py4C` bindings

`cmake/setup_py4C.cmake` enables Python bindings only when `FOUR_C_ENABLE_PYTHON_BINDINGS` is true. It enforces two build constraints:

- `FOUR_C_BUILD_SHARED_LIBS` must be enabled;
- `FOUR_C_WITH_PYBIND11` must be enabled.

It configures an installable package directory in the build tree, writes a generated `pyproject.toml` and `__init__.py`, and adds `utilities/py4C` as a subdirectory.

`utilities/py4C/src/4C_python_bindings_main.cpp` creates the `py4C` module with `PYBIND11_MODULE(py4C, module)`, sets the module docstring, iterates `FourC::Py4C::get_submodule_registry()`, creates each submodule, and calls its initialization function. The generated `4C_config_submodule_registry.hpp` is therefore the binding registry surface: adding bindings means registering a submodule entry so the main module initializes it.

## Validation

For tooling changes, run the affected script directly from an installed/development Python environment. For schema changes, validate the full metadata pipeline. For py4C changes, configure a shared-library build with pybind11 and import `py4C` after building.
