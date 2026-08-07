---
type: input parameter reference
title: CONTACT DYNAMIC
description: Contact dynamic parameter keys, defaults, enum values, contact-law dependencies, and accepted contact deck structure.
tags: [input-format, contact]
---

# `CONTACT DYNAMIC`

`CONTACT DYNAMIC` is registered in `src/contact/src/4C_contact_input.cpp`. It is optional to the generic parser but required when a structural deck uses contact, meshtying, or related contact constraints. Theory links: [Contact theory](../theory/contact.md).

## Parameters

| Key | Type | Default | Accepted values or notes |
|---|---:|---:|---|
| `LINEAR_SOLVER` | int | `-1` | Solver block for contact systems. |
| `RESTART_WITH_CONTACT` | bool | `false` | Restart with contact state. |
| `ADHESION` | enum | `none` | Adhesion law type. Avoid combining adhesion with friction or wear unless an accepted example demonstrates it. |
| `FRICTION` | enum | `None` | `None`, `Stick`, `Tresca`, `Coulomb`. |
| `FRLESS_FIRST` | bool | `false` | Perform frictionless first step/iteration. |
| `GP_SLIP_INCR` | bool | `false` | Use Gauss-point slip increment handling. |
| `STRATEGY` | enum | `LagrangianMultipliers` | `LagrangianMultipliers`, `lagrange`, `Lagrange`, `penalty`, `Penalty`, `Uzawa`, `Nitsche`, `Ehl`, `MultiScale`. |
| `SYSTEM` | enum | source default | `Condensed`, `condensed`, `cond`, `Condensedlagmult`, `condensedlagmult`, `condlm`, `SaddlePoint`, `Saddlepoint`, `saddlepoint`, `sp`, `none`. |
| `PENALTYPARAM` | double | `0.0` | Penalty stiffness/scale for penalty-style contact. Penalty, Nitsche, and Uzawa-style strategies require a positive value at runtime. |
| `PENALTYPARAMTAN` | double | `0.0` | Tangential penalty parameter for penalty or Uzawa augmented strategy. |
| `UZAWAMAXSTEPS` | int | `10` | Maximum Uzawa steps; runtime checks reject invalid limits. |
| `UZAWACONSTRTOL` | double | `1.0e-8` | Uzawa constraint tolerance. |
| `SEMI_SMOOTH_NEWTON` | bool | `true` | Use semi-smooth Newton for contact nonlinearities. |
| `SEMI_SMOOTH_CN` | double | `1.0` | Normal semi-smooth parameter. |
| `SEMI_SMOOTH_CT` | double | `1.0` | Tangential semi-smooth parameter. |
| `CONTACTFORCE_ENDTIME` | bool | `false` | Evaluate contact forces at end time. |
| `VELOCITY_UPDATE` | bool | `false` | Apply velocity update method. |
| `INITCONTACTBYGAP` | bool | `false` | Initialize contact by weighted gap vector. |
| `INITCONTACTGAPVALUE` | double | `0.0` | Initial contact gap value. |
| `NORMCOMBI_RESFCONTCONSTR` | enum | `And` | `And`, `Or`; combines residual-force and contact-constraint convergence. |
| `NORMCOMBI_DISPLAGR` | enum | `And` | `And`, `Or`; combines displacement and Lagrange multiplier increments. |
| `TOLCONTCONSTR` | double | `1.0E-6` | Contact constraint norm tolerance for saddle-point formulation. |
| `TOLLAGR` | double | `1.0E-6` | Lagrange multiplier norm tolerance for saddle-point formulation. |
| `CONSTRAINT_DIRECTIONS` | enum | `ntt` | Formulate constraints in normal/tangential or xyz directions. |
| `NONSMOOTH_GEOMETRIES` | bool | `false` | Combine mortar and node-to-segment formulations for mixed-dimension contact. |
| `NONSMOOTH_CONTACT_SURFACE` | bool | `false` | Alter qualified-vector criterion for nonsmooth self-contact surfaces. |
| `HYBRID_ANGLE_MIN` | double | `-1.0` | Non-smooth contact transition start angle. |
| `HYBRID_ANGLE_MAX` | double | `-1.0` | Non-smooth contact transition end angle. |
| `CPP_NORMALS` | bool | `false` | Create averaged CPP nodal normals. |
| `TIMING_DETAILS` | bool | `false` | Print detailed contact timings. |
| `NITSCHE_THETA` | double | `0.0` | `+1` symmetric, `0` non-symmetric, `-1` skew-symmetric. |
| `NITSCHE_THETA_2` | double | `1.0` | `+1` Chouly-type, `0` Burman penalty-free when `NITSCHE_THETA=-1`. |
| `NITSCHE_WEIGHTING` | enum | `harmonic` | Weighting of consistency terms. |
| `NITSCHE_PENALTY_ADAPTIVE` | bool | `true` | Adapt penalty after each converged time step. |
| `REGULARIZED_NORMAL_CONTACT` | bool | `false` | Add regularized normal contact formulation. |
| `REGULARIZATION_THICKNESS` | double | `-1.0` | Maximum contact penetration. |
| `REGULARIZATION_STIFFNESS` | double | `-1.0` | Initial regularization stiffness. |

## Related sections

A contact-capable deck normally combines:

1. `PROBLEM TYPE: Structure` or a coupled problem containing structure.
2. [Structural dynamic](structural-dynamic.md) with `LINEAR_SOLVER` and, for certain follower pressure/contact loads, `LOADLIN: true`.
3. `CONTACT DYNAMIC`.
4. Contact or meshtying condition sections, for example contact surface conditions from `Global::valid_conditions()` and module-specific contact registrations.
5. Optional `CONTACT CONSTITUTIVE LAWS` for multiscale or law-driven contact; see [Materials](materials.md#contact-constitutive-laws).
6. `SOLVER n` blocks referenced by structural/contact sections.

## Runtime validation failures to avoid

`CONTACT::STRATEGY::Factory::read_and_check_input` validates combinations after parsing. Common failures:

- `SYSTEM: Condensedlagmult` is only valid for the strategy combinations implemented for condensed Lagrange multiplier systems; use an accepted contact deck before selecting it.
- Penalty, Nitsche, and Uzawa-style contact require positive normal penalty data (`PENALTYPARAM > 0`); tangential penalty data must be positive when a tangential penalty/frictional strategy requires it.
- Adhesion is not freely combinable with friction and wear; source validation rejects unsupported adhesion-friction-wear combinations.
- Uzawa settings must have meaningful positive limits and tolerances; invalid `UZAWAMAXSTEPS` or missing penalty data fails during setup.
- Geometry flags such as `NONSMOOTH_GEOMETRIES`, `NONSMOOTH_CONTACT_SURFACE`, `HYBRID_ANGLE_MIN`, `HYBRID_ANGLE_MAX`, and `CPP_NORMALS` should match the contact surface topology; wrong choices may parse but fail geometrically.

## Strategy selection consequences

| Strategy family | Input consequence |
|---|---|
| Lagrange multiplier (`LagrangianMultipliers`, `lagrange`, `Lagrange`) | Use a system type compatible with multiplier dofs, often `SaddlePoint` or condensed variants depending on examples. |
| Penalty (`penalty`, `Penalty`) | Set `PENALTYPARAM` to a physically scaled positive stiffness. |
| `Uzawa` | Set `UZAWAMAXSTEPS` and `UZAWACONSTRTOL`; structural `UZAWA*` controls may also matter. |
| `Nitsche` | Use Nitsche-specific contact examples for required conditions and stabilization settings. |
| `MultiScale` | Usually needs `CONTACT CONSTITUTIVE LAWS` entries such as `CoConstLaw_*`. |

## Minimal pattern

```yaml
CONTACT DYNAMIC:
  LINEAR_SOLVER: 1
  FRICTION: "None"
  STRATEGY: "Penalty"
  SYSTEM: "none"
  PENALTYPARAM: 1000.0
  SEMI_SMOOTH_NEWTON: true
SOLVER 1:
  SOLVER: "UMFPACK"
```

Do not copy this blindly into a production deck: contact is geometry-sensitive. Start from a nearby accepted deck under `tests/input_files/contact2D_*`, `tests/input_files/contact3D_*`, `tests/input_files/meshtying2D_*`, or `tests/input_files/meshtying3D_*`, then adjust surfaces and material ids.
