---
type: input parameter reference
title: Materials and contact constitutive laws
description: MATERIALS list grammar, accepted material selector families, required/defaulted keys for common materials, units/dimensions, and the separate CONTACT CONSTITUTIVE LAWS section.
tags: [input-format, materials, contact]
---

# Materials and contact constitutive laws

`MATERIALS` is registered by `Global::gather_all_section_specs()` in `src/global_data/4C_global_data_read.cpp`. Each list item requires a material id `MAT` and exactly one `MAT_*` selector group from `src/global_legacy_module/4C_global_legacy_module_validmaterials.cpp`.

```yaml
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: 1.0
      DENSITY: 1.0
```

4C does not encode a unit system in the input parser. Units are therefore whatever consistent system the deck uses. The “units” below are dimensions: density is mass per volume, viscosity is dynamic viscosity, Young's modulus is stress, diffusivity is length squared per time, etc.

## Required/default rules

- `MAT` is required for every material item.
- The selector group name is required and case-sensitive.
- A key with no default in the source must be present when that material is selected.
- A key with a source default may be omitted, but for first-attempt decks set important physical parameters explicitly.
- Internally, the material selector is a `one_of` alternative and stores `_material_type`; `_material_type` is not a user-facing YAML key.
- During `Global::read_materials`, `MAT` ids must be non-negative and unique. A negative id or a duplicate id is a parse/setup error.
- Material objects are inserted as lazy constructors after the material list has matched; the actual material implementation is constructed when the problem later requests that material id.

## Common fluid materials

| Selector | Required keys | Optional/defaulted keys | Dimensions |
|---|---|---|---|
| `MAT_fluid` | `DYNVISCOSITY` double, `DENSITY` double | `GAMMA` double default `0.0` | dynamic viscosity, density, surface tension coefficient |
| `MAT_fluid_murnaghantait` | `DYNVISCOSITY`, `REFDENSITY`, `REFPRESSURE`, `REFBULKMODULUS`, `MATPARAMETER` doubles | `GAMMA` default `0.0` | viscosity, density, pressure, bulk modulus |
| `MAT_fluid_linear_density_viscosity` | `REFDENSITY`, `REFVISCOSITY`, `REFPRESSURE`, `COEFFDENSITY`, `COEFFVISCOSITY` doubles | `GAMMA` default `0.0` | density/viscosity pressure law |
| `MAT_fluid_weakly_compressible` | `VISCOSITY`, `REFDENSITY`, `REFPRESSURE`, `COMPRCOEFF` doubles | none | viscosity, density, pressure, compressibility |
| `MAT_carreauyasuda` | `NU_0`, `NU_INF`, `LAMBDA`, `APARAM`, `BPARAM`, `DENSITY` doubles | none | non-Newtonian viscosity law plus density |
| `MAT_modpowerlaw` | `MCONS`, `DELTA`, `AEXP`, `DENSITY` doubles | none | modified power-law viscosity plus density |
| `MAT_herschelbulkley` | `TAU_0`, `KFAC`, `NEXP`, `MEXP`, `LOLIMSHEARRATE`, `UPLIMSHEARRATE`, `DENSITY` doubles | none | yield stress, rheology factors, shear-rate limits, density |

Minimal fluid material:

```yaml
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: 1.0
      DENSITY: 1.0
```

## Common scalar transport materials

| Selector | Required keys | Optional/defaulted keys | Dimensions |
|---|---|---|---|
| `MAT_scatra` | `DIFFUSIVITY` double | `REACOEFF` default `0.0`, `SCNUM` default `0.0`, `DENSIFICATION` default `0.0`, `REACTS_TO_EXTERNAL_FORCE` default `false` | diffusivity, reaction rate coefficient, Schmidt number |
| `MAT_scatra_reaction` | `NUMSCAL` int, `STOICH` vector int sized by `NUMSCAL`, `REACCOEFF` double, `ROLE` vector double sized by `NUMSCAL` | `DISTRFUNCT` default `0`, `COUPLING` default `no_coupling`, optional `REACSTART` vector | stoichiometry and reaction coefficient |
| `MAT_scatra_reaction_poro` | `NUMSCAL`, `STOICH`, `REACCOEFF`, `REACSCALE`, `ROLE` | `DISTRFUNCT` default `0`, `COUPLING` default `no_coupling`, optional `REACSTART` | porous ECM reaction material |
| `MAT_matlist` | `LOCAL` bool, `NUMMAT` int, `MATIDS` vector int sized by `NUMMAT` | none | material-composition list |

Minimal scalar material:

```yaml
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: 0.01
```

## Common structural and beam materials

| Selector | Required keys | Optional/defaulted keys | Dimensions |
|---|---|---|---|
| `MAT_Struct_StVenantKirchhoff` | `YOUNG` double, `NUE` double, `DENS` double | none | Young's modulus, Poisson ratio, density |
| `MAT_Struct_DruckerPrager` | `YOUNG`, `NUE`, `DENS`, `ISOHARD`, `TOL`, `C`, `ETA`, `XI`, `ETABAR` doubles | `TANG` default `consistent` | elastoplastic Drucker-Prager parameters |
| `MAT_Struct_PlasticLinElast` | `YOUNG`, `NUE`, `DENS`, `YIELD`, `ISOHARD`, `KINHARD`, `TOL` doubles | none | linear elastic-plastic parameters |
| `MAT_ElastHyper` | `NUMMAT` int and material ids for sublaws, `DENS` double | `POLYCONVEX` default `0` | hyperelastic material composition and density |
| `MAT_ViscoElastHyper` | `NUMMAT` int and material ids for sublaws, `DENS` double | `POLYCONVEX` default `0` | visco-hyperelastic material composition |
| `MAT_BeamReissnerElastHyper` | `YOUNG`, `SHEARMOD`, `DENS`, `CROSSAREA`, `SHEARCORR`, `MOMINPOL`, `MOMIN2`, `MOMIN3` doubles | `FAD` bool appears in examples | beam modulus, area, moments of inertia, density |
| `MAT_LinElast1D` | `YOUNG` double, `DENS` double | none | one-dimensional stiffness and density |
| `MAT_Struct_Spring` | `STIFFNESS` double, `DENS` double | none | spring stiffness and density |
| `MAT_Kirchhoff_Love_shell` | `YOUNG_MODULUS`, `POISSON_RATIO`, `THICKNESS` doubles | none | shell elastic constants and thickness |

Example from a beam test:

```yaml
MATERIALS:
  - MAT: 1
    MAT_BeamReissnerElastHyper:
      YOUNG: 1e+07
      SHEARMOD: 5e+06
      DENS: 1.3e+09
      CROSSAREA: 1
      SHEARCORR: 1
      MOMINPOL: 0.1406
      MOMIN2: 0.0833333
      MOMIN3: 0.0833333
      FAD: true
```

## Particle materials

| Selector | Required keys | Optional/defaulted keys | Dimensions |
|---|---|---|---|
| `MAT_ParticleSPHFluid` | `INITRADIUS`, `INITDENSITY`, `EXPONENT`, `BACKGROUNDPRESSURE` doubles | `BULK_MODULUS`, `DYNAMIC_VISCOSITY`, `BULK_VISCOSITY`, `ARTIFICIAL_VISCOSITY`, `THERMALCONDUCTIVITY`, `THERMALABSORPTIVITY` default `0.0` | SPH particle radius, density, EOS and transport constants |
| `MAT_ParticleSPHBoundary` | none required in source excerpt | `INITRADIUS`, `INITDENSITY`, `THERMALCONDUCTIVITY`, `THERMALABSORPTIVITY` default `0.0` | boundary particle properties |
| `MAT_ParticleDEM` | `INITRADIUS`, `INITDENSITY` doubles | none | DEM sphere radius and density |
| `MAT_ParticleWallDEM` | none required | `FRICT_COEFF_ROLL` default `-1.0`, `ADHESION_SURFACE_ENERGY` default `-1.0` | wall contact properties |
| `MAT_ParticlePD` | `INITRADIUS`, `INITDENSITY`, `YOUNG`, `CRITICAL_STRETCH` doubles | none | peridynamic particle radius/density, bond stiffness scale, failure stretch |

## Poro, mixture, thermal, and other families

The material registry also includes poroelastic (`MAT_StructPoro`, `MAT_PoroLaw*`, `MAT_FluidPoro*`), mixture/growth (`MAT_Mixture`, `MAT_ConstraintMixture`), thermal (`MAT_Fourier`, `MAT_soret`), arterial (`MAT_CNST_ART`), membrane, muscle, myocardium, electrochemistry (`MAT_ion`, `MAT_newman`, `MAT_electrode`), and reduced lung families. For these, use `src/global_legacy_module/4C_global_legacy_module_validmaterials.cpp` as the canonical source and prefer an existing test deck with the same selector.

## Accepted selector inventory

The source currently registers these selector families, among others: `MAT_fluid`, `MAT_fluid_murnaghantait`, `MAT_fluid_linear_density_viscosity`, `MAT_fluid_weakly_compressible`, `MAT_carreauyasuda`, `MAT_modpowerlaw`, `MAT_herschelbulkley`, `MAT_lubrication`, `MAT_lubrication_law_constant`, `MAT_lubrication_law_barus`, `MAT_lubrication_law_roeland`, `MAT_scatra`, `MAT_scatra_reaction`, `MAT_scatra_reaction_poro`, `MAT_scatra_multiporo_fluid`, `MAT_scatra_multiporo_solid`, `MAT_scatra_chemotaxis`, `MAT_scatra_gr`, `MAT_scatra_multiscale`, `MAT_ion`, `MAT_newman`, `MAT_electrode`, `MAT_matlist`, `MAT_elchmat`, `MAT_elchphase`, `MAT_Struct_StVenantKirchhoff`, `MAT_Struct_DruckerPrager`, `MAT_Struct_PlasticLinElast`, `MAT_ElastHyper`, `MAT_ViscoElastHyper`, `MAT_PlasticElastHyper`, `MAT_BeamReissnerElastHyper`, `MAT_BeamKirchhoffElastHyper`, `MAT_BeamKirchhoffTorsionFreeElastHyper`, `MAT_CNST_ART`, `MAT_Fourier`, `MAT_Membrane_ElastHyper`, `MAT_StructPoro`, `MAT_FluidPoro`, `MAT_FluidPoroMultiPhase`, `MAT_Struct_Spring`, `MAT_Kirchhoff_Love_shell`, `MAT_Crosslinker`, `MAT_ParticleSPHFluid`, `MAT_ParticleSPHBoundary`, `MAT_ParticleDEM`, `MAT_ParticleWallDEM`, `MAT_ParticlePD`, `MAT_Mixture`, `MAT_IterativePrestress`, `MAT_crystal_plasticity`, `MAT_LinElast1D`, and `MAT_LinElast1DGrowth`.

## Contact constitutive laws

`CONTACT CONSTITUTIVE LAWS` is not `MATERIALS`. It is a separate optional list. Each item requires `LAW` and exactly one `CoConstLaw_*` selector from `src/contact_constitutivelaw/src/4C_contact_constitutivelaw_valid_laws.cpp`.

```yaml
CONTACT CONSTITUTIVE LAWS:
  - LAW: 1
    CoConstLaw_linear:
      A: 1.0
      B: 0.0
```

| Selector | Required keys | Optional/defaulted keys |
|---|---|---|
| `CoConstLaw_brokenrational` | `A`, `B`, `C` doubles | `Offset` default `0.0` |
| `CoConstLaw_python_surrogate` | `Python_Filename` path | `Offset` default `0.0` |
| `CoConstLaw_power` | `A`, `B` doubles | `Offset` default `0.0` |
| `CoConstLaw_cubic` | `A`, `B`, `C`, `D` doubles | `Offset` default `0.0` |
| `CoConstLaw_linear` | `A`, `B` doubles | `Offset` default `0.0` |
| `CoConstLaw_mirco` | `FirstMatID`, `SecondMatID`, `LateralLength`, `Resolution` plus roughness/model controls | many defaults including `PressureGreenFunFlag: true`, `Tolerance: 0.01`, `MaxIteration: 1000`, `Offset: 0.0` |

Use contact law ids only with contact features that explicitly consume them; otherwise the section may be omitted.
