---
type: input parameter reference
title: Design condition sections
description: Condition list grammar, common Dirichlet and Neumann keys, topology-set coupling, and time-function references for 4C input decks.
tags: [input-format, conditions, boundary-conditions]
---

# Design condition sections

Condition sections are list-shaped top-level sections registered by `Global::valid_conditions()` in `src/global_legacy_module/4C_global_legacy_module_validconditions.cpp` and by module-specific registrars. Conditions attach physics data to geometric sets such as points, lines, surfaces, or volumes.

## List item shape

Every condition definition adds common geometry-identification fields before condition-specific fields. The common fields are optional at the YAML type level, but a usable condition must identify its geometry by either a numeric design entity or a named node set:

| Key | Type | Rule |
|---|---:|---|
| `E` | optional int | Numeric design entity id. If `ENTITY_TYPE` is omitted or `legacy_id`, `E` must be strictly positive (`> 0`). If `ENTITY_TYPE` is `element_block_id` or `node_set_id`, `E` must be non-negative (`>= 0`). |
| `ENTITY_TYPE` | optional enum | Geometry/entity kind used with `E` when needed by the condition reader. |
| `NODE_SET_NAME` | optional string | Alternative name-based identification from mesh node sets. It is exclusive with `E`/`ENTITY_TYPE`: do not provide `NODE_SET_NAME` together with numeric entity identification. |

Most legacy-style `DESIGN ... CONDITIONS` entries begin with `E`, the id of a design entity set defined by a topology section:

```yaml
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 6
    ONOFF: [1, 1, 1, 1, 1, 1]
    VAL: [0, 0, 0, 0, 0, 0]
    FUNCT: [0, 0, 0, 0, 0, 0]
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
```

`E: 1` refers to `DNODE 1`, `DLINE 1`, `DSURFACE 1`, or `DVOL 1` depending on the condition geometry. The parser validates the keys and array lengths; `Global::read_conditions` then reads condition definitions, builds condition geometry/nodal clouds through the mesh reader, and attaches the resulting condition objects to the owning discretizations.

## Dirichlet condition families

The common Dirichlet maker is used for point, line, surface, and volume conditions, including structural, ALE, transport, thermo, and poro variants.

Common section names:

- `DESIGN POINT DIRICH CONDITIONS`
- `DESIGN LINE DIRICH CONDITIONS`
- `DESIGN SURF DIRICH CONDITIONS`
- `DESIGN VOL DIRICH CONDITIONS`
- `DESIGN POINT TRANSPORT DIRICH CONDITIONS`
- `DESIGN LINE TRANSPORT DIRICH CONDITIONS`
- `DESIGN SURF TRANSPORT DIRICH CONDITIONS`
- `DESIGN VOL TRANSPORT DIRICH CONDITIONS`
- analogous `ALE`, `THERMO`, `PORO`, and NURBS least-squares variants

Common keys:

| Key | Type | Required | Meaning |
|---|---:|---:|---|
| `E` | int | yes | Design entity id. |
| `NUMDOF` | int | yes | Number of dof entries. |
| `ONOFF` | vector int length `NUMDOF` | yes | `1` constrains a component, `0` leaves it free. |
| `VAL` | vector double length `NUMDOF` | yes | Prescribed values. |
| `FUNCT` | vector optional int length `NUMDOF` | yes in common examples | Function ids for time dependence; use `0` for constant values in legacy examples. |

For structural 3D solid displacement, `NUMDOF` is often `3`; for many fluid examples it is `4` for velocity components plus pressure. Beam examples use `NUMDOF: 6`.

## Neumann condition families

Common Neumann sections include point/line/surface/volume variants for structural, transport, thermo, and poro problems.

| Key | Type | Required | Default | Accepted values or notes |
|---|---:|---:|---:|---|
| `E` | int | yes | none | Design entity id. |
| `NUMDOF` | int | yes | none | Number of components. |
| `ONOFF` | vector int length `NUMDOF` | yes | none | Active components. |
| `VAL` | vector double length `NUMDOF` | yes | none | Load/flux values. |
| `FUNCT` | vector optional int length `NUMDOF` | yes in examples | none | Function ids for time dependence. |
| `TYPE` | enum string | no | `Live` | `Live`, `Dead`, `pseudo_orthopressure`, `orthopressure`, `PressureGrad`. `Dead` is not available for solids; structural `orthopressure` requires `STRUCTURAL DYNAMIC/LOADLIN: true`. |

Example structural point load:

```yaml
DESIGN POINT NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 6
    ONOFF: [0, 0, 1, 0, 0, 0]
    VAL: [0, 0, 600, 0, 0, 0]
    FUNCT: [0, 0, 1, 0, 0, 0]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_TIME: "t"
```

## Robin spring-dashpot structural conditions

`src/inpar/4C_inpar_structure.cpp` registers `DESIGN SURF ROBIN SPRING DASHPOT CONDITIONS`, `DESIGN LINE ROBIN SPRING DASHPOT CONDITIONS`, and `DESIGN POINT ROBIN SPRING DASHPOT CONDITIONS`.

Required keys: `E`, `NUMDOF`, and vectors of length `NUMDOF` for `ONOFF`, `STIFF`, `TIMEFUNCTSTIFF`, `VISCO`, `TIMEFUNCTVISCO`, `DISPLOFFSET`, `TIMEFUNCTDISPLOFFSET`, and `FUNCTNONLINSTIFF`. `DIRECTION` is required and selects the spring-dashpot direction. `COUPLING` is optional.

## Functions and time dependence

Function sections are named `FUNCT1`, `FUNCT2`, etc. The input registry treats them specially through a `FUNCT<n>` spec. Any nonzero function id in `FUNCT`, `STARTFUNCNO`, `VELFUNCNO`, `INITFUNCNO`, and similar keys must have a matching `FUNCT<id>` section with a supported function definition.

Common constant and symbolic forms appear in test decks:

```yaml
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_TIME: "t"
```

## Topology sections used by conditions

Legacy topology sections are string lists:

| Section | Defines |
|---|---|
| `DNODE-NODE TOPOLOGY` | Design point ids from nodes. |
| `DLINE-NODE TOPOLOGY` | Design line ids from node sets. |
| `DSURF-NODE TOPOLOGY` | Design surface ids from side/node sets. |
| `DVOL-NODE TOPOLOGY` | Design volume ids. |

See [Geometry and elements](geometry-and-elements.md#legacy-topology-and-element-sections) for examples.
