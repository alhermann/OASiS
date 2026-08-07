---
type: input parameter reference
title: Common sections
description: Shared 4C input sections for problem size, discretisation, IO, fields, and global metadata.
tags: [input-format, common-sections]
---

# Common sections

Common sections are registered in `src/global_legacy_module/4C_global_legacy_module_validparameters.cpp` and apply across problem types. Only `PROBLEM TYPE` is required at parser level; the other sections may be omitted and then their default values are used.

## `PROBLEM SIZE`

| Key | Type | Required | Default | Meaning |
|---|---:|---:|---:|---|
| `DIM` | int | no | `3` | Spatial dimension, usually `1`, `2`, or `3`. |
| `ELEMENTS` | int | no | `0` | Informational total number of elements. |
| `NODES` | int | no | `0` | Informational total number of nodes. |
| `NPATCHES` | int | no | `0` | Number of NURBS patches. |
| `MATERIALS` | int | no | `0` | Informational number of material entries. The actual material list is the top-level `MATERIALS` section. |
| `NUMDF` | int | no | `3` | Maximum number of degrees of freedom. |

## `DISCRETISATION`

| Key | Type | Required | Default | Meaning |
|---|---:|---:|---:|---|
| `NUMFLUIDDIS` | int | no | `1` | Number of fluid discretizations. |
| `NUMSTRUCDIS` | int | no | `1` | Number of structural discretizations. |
| `NUMALEDIS` | int | no | `1` | Number of ALE discretizations. |
| `NUMARTNETDIS` | int | no | `1` | Number of arterial-network discretizations. |
| `NUMTHERMDIS` | int | no | `1` | Number of thermal discretizations. |
| `NUMAIRWAYSDIS` | int | no | `1` | Number of reduced-dimensional airways discretizations. |

When a deck is single-field, examples often set unused discretization counts to `0`, e.g. the Stokes fluid example sets structural, ALE, arterial, thermo, and airways counts to zero.

## `IO`

`IO` controls legacy output quantities and diagnostic formats. All keys are optional.

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `OUTPUT_GMSH` | bool | `false` | Write Gmsh postprocessing files. |
| `OUTPUT_ROT` | bool | `false` | Output rotations. |
| `OUTPUT_SPRING` | bool | `false` | Output spring data. |
| `OUTPUT_BIN` | bool | `true` | Write binary output. |
| `ELEMENT_MAT_ID` | bool | `false` | Output element material ids. |
| `STRUCT_STRESS` | enum | `No` | `No`, `no`, `NO`, `Yes`, `yes`, `YES`, `Cauchy`, `cauchy`, `2PK`, `2pk`. |
| `STRUCT_STRAIN` | enum | `No` | `No`, `no`, `NO`, `Yes`, `yes`, `YES`, `EA`, `ea`, `GL`, `gl`, `LOG`, `log`. |
| `STRUCT_PLASTIC_STRAIN` | enum | `No` | `No`, `Yes`, `EA`, `GL` with case variants defined in source. |
| `STRUCT_SURFACTANT` | bool | `false` | Output structural surfactant data. |
| `STRUCT_JACOBIAN_MATLAB` | bool | `false` | Output structural Jacobian for Matlab. |
| `STRUCT_CONDITION_NUMBER` | enum | `none` | `gmres_estimate`, `max_min_ev_ratio`, `one-norm`, `inf-norm`, `none`. |
| `FLUID_STRESS` | bool | `false` | Output fluid stress. |
| `FLUID_WALL_SHEAR_STRESS` | bool | `false` | Output wall shear stress. |
| `FLUID_ELEDATA_EVERY_STEP` | bool | `false` | Output fluid element data every step. |
| `FLUID_NODEDATA_FIRST_STEP` | bool | `false` | Output fluid node data at first step. |
| `THERM_HEATFLUX` | enum | `None` | `None`, `No`, `NO`, `no`, `Current`, `Initial`. |
| `THERM_TEMPGRAD` | enum | `None` | `None`, `No`, `NO`, `no`, `Current`, `Initial`. |

More modern runtime VTK/VTP output sections are registered separately under `IO/RUNTIME VTK OUTPUT`, `IO/RUNTIME VTK OUTPUT/FLUID`, `IO/RUNTIME VTK OUTPUT/STRUCTURE`, `IO/RUNTIME VTK OUTPUT/BEAMS`, and `IO/RUNTIME VTP OUTPUT STRUCTURE`.

## `fields`

`fields` is an optional list used to define input field data that other parts of a simulation can refer to by name.

| Key | Type | Required | Meaning |
|---|---:|---:|---|
| `name` | string | yes | Unique field name. |
| `discretization` | string | yes | Name of the discretization to which the field belongs. |
| `source` | selection | yes | Either `separate_file` or `from_mesh`. |

`source: separate_file` requires `file` and may include optional `key`; if `key` is omitted, the field name is used. `source: from_mesh` requires `basis` (`cells` or `points`) and may include optional `key`.

```yaml
fields:
  - name: fiber_orientation
    discretization: structure
    source:
      separate_file:
        file: mat_muscle_combo_hex_fiber_orientation.json
```

## Validation behavior

Common sections are ordinary `InputSpec` groups: unknown keys are rejected, required keys without defaults must be present, defaulted keys may be omitted, and enum selectors must use one of the registered strings. For root and include behavior, see [4C input file format](../input/file-format.md).
