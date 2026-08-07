---
type: testing guide
title: Testing and Validation
description: Maps the GoogleTest, benchmark, cut-test, install-test, and regression-input validation surfaces and explains the narrowest checks for common changes.
tags: [testing, validation, cmake]
---

# Testing and Validation

4C combines unit tests, benchmark tests, standalone geometry regression tests, install tests, and full input-file regression tests. The test build hooks are owned by `/cmake/setup_tests.cmake`, `/unittests`, `/tests/benchmark_tests`, `/tests/cut_test`, and `/tests/list_of_tests.cmake`.

## Test families

| Family | Source | Purpose | Typical validation |
| --- | --- | --- | --- |
| GoogleTest unit tests | `/unittests` and module-local `src/*/tests` directories auto-discovered by CMake helpers | Fast checks for IO, linalg, FEM, materials, particles, geometry, reduced lung, deal.II, etc. | `cmake --build build --target unittests && ctest --test-dir build -R <name>` |
| Google Benchmark | `src/core/*/benchmark_tests`, `tests/benchmark_tests`, selected utility benchmarks | Microbenchmarks for tensors, discretization, symbolic expression, solid elements. | Enable `FOUR_C_WITH_GOOGLE_BENCHMARK`; build `benchmarktests`. |
| Cut regression executable | `/tests/cut_test` | Large collection of cut-cell and intersection cases compiled into `cut_test`. | `cmake --build build --target cut_test` then run the executable or CTest entry. |
| Full input regression tests | `/tests/input_files` plus `/tests/list_of_tests.cmake` | Runs real 4C input files and compares result descriptions/output. | CTest labels or names from generated list. |
| Install test | `/tests/install_test/*.in` configured by `setup_tests.cmake` | Verifies installed CMake package usage. | Install-target dependent CTest. |
| Python utility checks | `/utilities/four_c_python` | Development, CI, schema, comparison, restart and VTK helpers. | See [Apps and Tooling](../tools/apps-postprocessing.md). |

## CMake test setup

`cmake/setup_tests.cmake` scales timeouts through `FOUR_C_TEST_TIMEOUT_SCALE`: debug builds default to a larger scale than release builds. It configures GoogleTest when `FOUR_C_WITH_GOOGLETEST` is enabled, fetching v1.15.2 through `FetchContent`. It similarly configures Google Benchmark only when `FOUR_C_WITH_GOOGLE_BENCHMARK` is enabled.

MPI launch settings are centralized in `FOUR_C_MPIEXEC_ARGS_FOR_TESTING`; address-sanitizer builds append `LSAN_OPTIONS=detect_leaks=0` because the project focuses on undefined-behavior-inducing memory errors rather than leak reports.

## Unit test layout rules

`/unittests/README.md` defines the contributor contract:

1. place a new test under `/unittests` in a subdirectory mirroring the tested `/src` location, or use a module-local `src/<module>/tests` directory when the module already follows that pattern;
2. add the file to that directory's `CMakeLists.txt`;
3. include GoogleTest;
4. keep tests inside an anonymous namespace;
5. for `FOUR_C_THROW`, expect `Core::Exception`;
6. for MPI tests, use `four_c_add_google_test_executable(<name> NP <number> SOURCE ...)` so CTest launches it with the right MPI flags.

Several source modules already own tests under their module directories even though `/unittests/CMakeLists.txt` only lists selected top-level unit-test directories. Examples include `src/core/io/tests`, `src/core/linalg/tests`, `src/core/fem/tests`, `src/particle/tests`, `src/reduced_lung/tests`, and `src/deal_ii/tests`.

`four_c_auto_define_tests()` recursively discovers `*.cpp` files below a test directory. Filename suffixes control launch resources: `.npX.` runs the test with `X` MPI ranks, and `.threadsX.` runs it with `X` OpenMP threads. `_set_up_unit_test_target()` creates a test executable under `build/tests`, links GoogleTest/GMock, the common test target, and `<module>_module`, then registers a CTest command through `MPIEXEC_EXECUTABLE`. Rank 0 writes `--gtest_output=xml:unittest_reports/<target>_report.xml`; additional MPI ranks run the same executable without a report. CTest properties include `TIMEOUT`, label `minimal`, `PROCESSORS = NP * THREADS`, and `OMP_NUM_THREADS` in the environment.

## Integration, restart, and regression contracts

`tests/list_of_tests.cmake` declares full input-file tests with helper calls such as `four_c_test(...)`, including MPI rank counts (`NP`) and result behavior. Restart tests use `four_c_test_restart(...)`, tying a restart run to a prior test case, restart step, and optionally an alternate input file. These tests exercise real solver entrypoints and the `RESULT DESCRIPTION` section.

Problem entrypoints usually create field tests and call `Global::Problem::instance()->test_all(comm)` after time integration. The `RESULT DESCRIPTION` input section is part of the global input schema and is gathered through module callbacks. This is why changing output quantities, field-test names, restart behavior, or final-state writing can affect full input-file regressions even when unit tests pass.

## Input validation and CI chunking

`validate-test-files` loads a generated JSON Schema, validates each YAML file, prints `passed`, `failed`, or `error`, collects files with errors, and exits with status 1 when any file failed validation. `create-schema-files` is fed by metadata from `4C --parameters`.

`chunk-test-suite` reads CTest JSON produced by `ctest --show-only=json-v1`; optional `--junit-report` supplies measured runtimes. For each test it reads `PROCESSORS` and `TIMEOUT` properties, estimates processor-time from JUnit runtime or timeout, and emits chunk ranges. `--max-proctests` caps the sum of processors assigned to a chunk.

## Focused validation by change area

| Change area | Focused check | Broader check when needed |
| --- | --- | --- |
| Input schema, metadata, YAML parsing | IO unit tests under `src/core/io/tests` or `/unittests/io`; `4C --parameters`; `validate-test-files` | Full input regression subset. |
| FEM/DOF/discretization | `src/core/fem/tests`, geometry/search tests, affected element tests | Representative input files for affected physics. |
| Linear algebra wrappers | `src/core/linalg/tests` with MPI variants when maps or imports change | Solver-heavy regression tests. |
| NOX or linear solver integration | solver-specific unit/regression tests and a small nonlinear input | Multiple problem families using the solver. |
| Materials | `/unittests/mat`, material factory tests, representative material input | A regression using the material in its target physics. |
| Cut/XFEM/volmortar | `cut_test`, `src/core/fem/geometry` tests, coupling regression | FSI/XFEM or mesh-tying regressions. |
| Python tooling | package script invocation, `create-schema-files`, `validate-test-files` | CI workflow subset. |

Keep validation narrow first, then expand along dependencies documented in the related system page.
