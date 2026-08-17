#!/usr/bin/env python3
"""Test DUNE-fem basic usage."""

from dune.grid import structuredGrid, cartesianDomain
from dune.fem import space, discretefunction, operator, scheme, model
import numpy as np

# Create a simple grid directly
grid = structuredGrid([0.0, 0.0], [1.0, 1.0], [10, 10])

print(f"Grid created: {grid}")
print(f"Number of leaves: {grid.size(0)}")

# Create Lagrange space (P1) with vector range
vector_space = space.lagrange(grid, 1, dimRange=2)
print(f"Vector space: {vector_space}")

# Create grid function using DiscreteFunction - first argument is space
u = vector_space.DiscreteFunction(vector_space, "u")
print(f"Grid function u: {u}")
print(f"NDOF: {len(u.dofVector)}")

# Test interpolation
def test_func(x):
    return np.array([x[0] * x[1], x[0] + x[1]])

u.interpolate(test_func)
print("Interpolation done")

# Print some values
print(f"u at center: {u.evaluate([0.5, 0.5])}")

print("SUCCESS!")
