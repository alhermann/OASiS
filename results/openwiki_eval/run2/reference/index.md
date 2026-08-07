# Files

- [Common sections](common-sections.md) - Shared 4C input sections for problem size, discretisation, IO, fields, and global metadata.
- [Design condition sections](conditions.md) - Condition list grammar, common Dirichlet and Neumann keys, topology-set coupling, and time-function references for 4C input decks.
- [CONTACT DYNAMIC](contact-dynamic.md) - Contact dynamic parameter keys, defaults, enum values, contact-law dependencies, and accepted contact deck structure.
- [FLUID DYNAMIC](fluid-dynamic.md) - Fluid control sections, accepted keys, defaults, enum values, stabilization settings, and minimal Stokes/Navier-Stokes deck patterns.
- [Geometry, domains, elements, and topology](geometry-and-elements.md) - How 4C decks define meshes using generated DOMAIN sections, external GEOMETRY sections, or legacy node/element/topology string sections.
- [Materials and contact constitutive laws](materials.md) - MATERIALS list grammar, accepted material selector families, required/defaulted keys for common materials, units/dimensions, and the separate CONTACT CONSTITUTIVE LAWS section.
- [PARTICLE DYNAMIC](particle-dynamic.md) - Particle dynamic, SPH, DEM, and peridynamics input controls with defaults, material links, and accepted deck patterns.
- [SCALAR TRANSPORT DYNAMIC](scalar-transport-dynamic.md) - Scalar transport dynamic controls, nonlinear/stabilization subsections, materials, conditions, and first-attempt deck settings.
- [Linear solver sections](solvers.md) - Grammar for `SOLVER 1` through `SOLVER 9` and how physics sections reference solver blocks.
- [STRUCTURAL DYNAMIC](structural-dynamic.md) - Required and optional structural mechanics control keys, defaults, enum values, and solver/material/condition relationships for accepted structure decks.
