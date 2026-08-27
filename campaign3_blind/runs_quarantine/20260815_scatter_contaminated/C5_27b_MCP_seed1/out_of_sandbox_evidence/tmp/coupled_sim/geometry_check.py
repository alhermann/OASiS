"""Check the domain decomposition and interface definition."""
import numpy as np

# Global domain: unit square minus (3/4,1)x(3/4,1)
# Subdomain A: everything except (1/2,1)x(0,1/2)
# Subdomain B: (1/2,1)x(0,1/2)

# Interface has two legs:
# Leg 1: x = 1/2, 0 < y < 1/2 (vertical segment)
# Leg 2: y = 1/2, 1/2 < x < 1 (horizontal segment)

# For subdomain A (Dirichlet side):
# - Receives u from B on interface
# - Applies u as Dirichlet BC on interface
# - Exports normal flux qn = -k * grad(u) . n_out
# Outward normal from A:
#   On leg 1 (x=1/2): n_A = (+1, 0) pointing right INTO B
#   On leg 2 (y=1/2): n_A = (0, +1) pointing up INTO B

# For subdomain B (Neumann side):
# - Receives flux from A on interface  
# - Applies flux as Neumann BC on interface
# - Exports u on interface
# Outward normal from B:
#   On leg 1 (x=1/2): n_B = (-1, 0) pointing left OUT of B
#   On leg 2 (y=1/2): n_B = (0, -1) pointing down OUT of B

# Conductance ratios:
# Vertical leg (x=1/2): k_A=1, k_B=5/2, ratio = k_A/k_B = 1/(5/2) = 2/5 = 0.4
# Horizontal leg (y=1/2): k_A=2, k_B=5/2, ratio = k_A/k_B = 2/(5/2) = 4/5 = 0.8

# Wait, need to reconsider. The conductance ratio rho is defined as:
# rho = (interface conductance of DIRICHLET side) / (interface conductance of NEUMANN side)
# Interface conductance ~ k / distance_to_outer_boundary

# For vertical leg at x=1/2:
#   Subdomain A (left of interface): k=1, distance to outer boundary x=0 is 1/2
#   Subdomain B (right of interface): k=5/2, distance to outer boundary x=1 is 1/2
#   So rho_vertical = (1/(1/2)) / ((5/2)/(1/2)) = 2 / 5 = 0.4

# For horizontal leg at y=1/2:
#   Subdomain A (above interface): k=2, distance to outer boundary y=1 is 1/2
#   Subdomain B (below interface): k=5/2, distance to outer boundary y=0 is 1/2
#   So rho_horizontal = (2/(1/2)) / ((5/2)/(1/2)) = 4 / 5 = 0.8

# The effective rho is somewhere between these values depending on how we weight them.
# For theta choice: theta_optimal = 1/(1+rho)
# If rho ~ 0.6 (average), then theta ~ 1/1.6 = 0.625

print("Geometry analysis:")
print("Subdomain A: L-shaped region (unit square minus (1/2,1)x(0,1/2) minus (3/4,1)x(3/4,1))")
print("Subdomain B: Rectangle (1/2,1)x(0,1/2)")
print()
print("Interface legs:")
print("  Leg 1 (vertical): x=1/2, 0<y<1/2")
print("    - n_A = (+1, 0), n_B = (-1, 0)")
print("    - k_A = 1, k_B = 5/2")
print("    - rho_vertical = 0.4")
print("  Leg 2 (horizontal): y=1/2, 1/2<x<1")
print("    - n_A = (0, +1), n_B = (0, -1)")
print("    - k_A = 2, k_B = 5/2")
print("    - rho_horizontal = 0.8")
print()
print("Recommended theta: ~0.6-0.7 (since rho < 1, Dirichlet side is softer)")
print("This should converge well with Aitken acceleration.")
