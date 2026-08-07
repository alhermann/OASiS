# Files

- [Application Lifecycle](application-lifecycle.md) - Traces the `4C` executable from MPI and Kokkos startup through input reading, global problem setup, problem-type dispatch, result testing, and shutdown.
- [Build and Runtime Topology](build-and-runtime.md) - Explains how CMake assembles 4C into one core library plus executables, which dependencies are configured, and how tests, docs, install rules, and Python bindings enter the build graph.
- [Input Schema and Global Problem](input-schema-global-problem.md) - Documents how 4C builds the validated input schema, reads YAML or JSON files, accepts legacy mesh sections, emits metadata, and initializes `Global::Problem`.
- [Testing and Validation](testing-validation.md) - Maps the GoogleTest, benchmark, cut-test, install-test, and regression-input validation surfaces and explains the narrowest checks for common changes.
