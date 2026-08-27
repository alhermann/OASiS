# Coupled Simulation Setup

## Problem Description
- **Global domain**: Unit square (0,1)×(0,1) with square (3/4,1)×(3/4,1) removed (L-shaped non-convex domain)
- **Subdomain A**: Everything except (1/2,1)×(0,1/2) - L-shaped region solved with scikit-fem
- **Subdomain B**: Rectangle (1/2,1)×(0,1/2) solved with FEniCSx
- **Interface**: Bent polyline with two legs:
  - Leg 1: x=1/2, 0<y<1/2 (vertical)
  - Leg 2: y=1/2, 1/2<x<1 (horizontal)
- **Equation**: -div(k grad u) = f with piecewise constant k
- **Coupling scheme**: Dirichlet-Neumann iteration with A as DIRICHLET side, B as NEUMANN side

## Material Coefficients
- Region 1 (0,1/2)×(0,1/2): k = 1
- Region 2 (1/2,1)×(0,1/2): k = 5/2 (subdomain B)
- Region 3 (1/2,1)×(1/2,1): k = 5
- Region 4 (0,1/2)×(1/2,1): k = 2

## Conductance Ratios
- Vertical leg: rho = 0.4 (k_A/k_B = 1/(5/2))
- Horizontal leg: rho = 0.8 (k_A/k_B = 2/(5/2))
- Recommended theta: ~0.6-0.7 for optimal convergence

## Mesh Levels
- Level 1: h = 1/8 (nx = 8)
- Level 2: h = 1/16 (nx = 16)
- Level 3: h = 1/32 (nx = 32)

## Files Structure
```
/tmp/coupled_sim/
├── participant_A_skfem.py    # Subdomain A solver (scikit-fem)
├── participant_B_fenics.py   # Subdomain B solver (FEniCSx)
├── run_coupled_simulation.py # Main driver script
├── level1_A/                 # Work directory for level 1, subdomain A
├── level1_B/                 # Work directory for level 1, subdomain B
├── level2_A/
├── level2_B/
├── level3_A/
└── level3_B/
```

## Execution
Run the main driver script which will:
1. Set up work directories for each mesh level
2. Copy participant scripts to work directories
3. Run the coupled simulation using OASiS's `couple` tool
4. Generate all required output files
