# Scope and priorities for this wiki

The reader of this wiki is an autonomous agent that has **never run 4C** and
must write a 4C input deck that the binary accepts on the first attempt. Code
architecture is secondary. Prioritise, in this order:

1. **The input file format.** How a `.4C.yaml` deck is structured: which
   sections exist, which are mandatory for a given PROBLEMTYPE, and how
   sections nest.
2. **Sections and their parameters.** For the most-used sections
   (PROBLEM TYPE, STRUCTURAL DYNAMIC, FLUID DYNAMIC, SCALAR TRANSPORT DYNAMIC,
   SOLVER n, IO, MATERIALS, the DESIGN ... CONDITIONS families, the GEOMETRY
   families): name the accepted parameter keys, give each key's type, its
   default when it has one, and its accepted values when it is an enum.
3. **Materials.** The material names accepted under `MATERIALS`, and for the
   common ones the exact parameter keys each requires, with units.
4. **Required-vs-optional.** State explicitly which keys must be present and
   which may be omitted. An agent that omits a required key gets a parse error.
5. **The theory each solver implements.** For the structural, fluid, scalar
   transport, contact and particle (SPH / DEM / peridynamics) modules, state
   the governing equations, the discretisation, and the time integration
   schemes, so that a reader who knows continuum mechanics and FEM from a
   textbook can connect the theory to the input keys that control it.

Write concrete, checkable statements. Prefer "`TIMESTEP` (double, default 0.1)"
over "the time step is configurable". Worked example decks are welcome.
