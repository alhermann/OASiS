---
type: input format guide
title: Geometry, domains, elements, and topology
description: How 4C decks define meshes using generated DOMAIN sections, external GEOMETRY sections, or legacy node/element/topology string sections.
tags: [input-format, geometry, elements]
---

# Geometry, domains, elements, and topology

A deck must provide a discretization for every active field. The common registry in `src/global_legacy_module/4C_global_legacy_module_validparameters.cpp` creates `GEOMETRY`, `DOMAIN`, and `KNOTVECTORS` sections for known fields: `STRUCTURE`, `FLUID`, `LUBRICATION`, `TRANSPORT`, `TRANSPORT2`, `ALE`, `ARTERY`, `REDUCED D AIRWAYS`, `THERMO`, and `PERIODIC BOUNDINGBOX`. Legacy element sections are also accepted by `InputFile`.

## Generated rectangular domains

`<FIELD> DOMAIN` uses `Core::IO::GridGenerator::RectangularCuboidInputs::spec()` to create a rectangular mesh. A fluid example from `tests/input_files/f3_stokes_residualbased_rotboxgeom.4C.yaml`:

```yaml
FLUID DOMAIN:
  bottom_corner_point: [-1, -1, -1]
  top_corner_point: [1, 1, 1]
  subdivisions: [5, 5, 5]
  rotation_angle: [10, 20, 30]
  elements:
    FLUID:
      HEX8:
        MAT: 1
        NA: Euler
  auto_partition: false
```

The `elements` map selects an element family and cell type; nested keys are validated against registered element definitions. `MAT: 1` must exist in [Materials](materials.md). `NA: Euler` is an element-specific kinematic/analysis flag for fluid element data.

## External mesh geometry

`<FIELD> GEOMETRY` reads external geometry and element block mappings. The common shape is:

```yaml
STRUCTURE GEOMETRY:
  FILE: mesh.exo
  SHOW_INFO: none
  ELEMENT_BLOCKS:
    - ID: 1
      SOLID:
        HEX8:
          MAT: 1
          KINEM: nonlinear
```

| Key | Type | Required | Default | Meaning |
|---|---:|---:|---:|---|
| `FILE` | path | yes | none | External geometry file, absolute or relative to the input file. |
| `SHOW_INFO` | enum | no | `none` | Verbosity of reporting element, node, and set info. |
| `ELEMENT_BLOCKS` | list | yes for external geometry | none | Maps mesh block `ID` values to 4C element definitions. |
| `ID` | int | yes within each block | none | Element block id in the external file. |

## Knot vectors

`<FIELD> KNOTVECTORS` sections define NURBS knot vectors through `Core::FE::Nurbs::Knotvector::spec()`. Include them only for NURBS discretizations; ordinary generated and legacy finite-element meshes omit them.

## Legacy topology and element sections

Legacy sections are accepted as string fragments and parsed by downstream mesh readers. They are not ordinary YAML maps.

```yaml
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
STRUCTURE ELEMENTS:
  - "1 BEAM3R LINE2 1 2 MAT 1 TRIADS 0 0 0 0 0 0 USE_FAD true"
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
```

Registered legacy element sections include:

- `STRUCTURE ELEMENTS`
- `FLUID ELEMENTS`
- `LUBRICATION ELEMENTS`
- `TRANSPORT ELEMENTS`
- `TRANSPORT2 ELEMENTS`
- `ALE ELEMENTS`
- `THERMO ELEMENTS`
- `ARTERY ELEMENTS`
- `REDUCED D AIRWAYS ELEMENTS`
- `PARTICLES`
- `PERIODIC BOUNDINGBOX ELEMENTS`

Registered legacy topology sections include `NODE COORDS`, `DNODE-NODE TOPOLOGY`, `DLINE-NODE TOPOLOGY`, `DSURF-NODE TOPOLOGY`, and `DVOL-NODE TOPOLOGY`.

## Element material ids and field names

Element lines and generated-domain element specs usually contain `MAT: <id>` or `MAT <id>`. That id must match exactly one `MATERIALS` entry. Mismatched ids parse as input but fail when elements/materials are constructed.

Field names are conventional but significant: fluid result descriptions use `DIS: "fluid"`, scalar transport uses `DIS: "scatra"`, and structure uses `DIS: "structure"` in test decks. Keep those names unless a multi-discretization deck deliberately creates alternatives.

## Geometry checklist

- Use `<FIELD> DOMAIN` for the first version of a deck when possible; it is typed YAML and easier to validate.
- Use `<FIELD> GEOMETRY` when importing external mesh blocks; every block needs a valid element definition.
- Use legacy sections only when the example family already uses them or when hand-defining a tiny beam/truss/particle set.
- Define topology sets before using their ids in [conditions](conditions.md).
- Define every referenced material id in [Materials](materials.md).
