#!/usr/bin/env python
import sys
sys.path.insert(0, '/home/alexander/miniconda3/envs/dune-fem-env/lib/python3.12/site-packages')

from dune.grid import yaspGrid, reader
from dune.grid.grid_generator import levelView
from dune.fem import space, operator, discretefunction, scheme
from dune.fem import assemble
from dune.ufl import GridFunction, DirichletBC
import ufl
from ufl import dx, inner, grad, TrialFunction, TestFunction

# Create a DGF string for a 4x4 grid
dgf_string = """DGF
INTERVAL
0 0
1 1
4 4
#
BOUNDARYDOMAIN
1 -0.002525 -2.5e-05 2.5e-05 1.000025
2 0.999975 -2.5e-05 1.0025249999999999 1.000025
3 -2.5e-05 -0.002525 1.000025 2.5e-05
4 -2.5e-05 0.999975 1.000025 1.0025249999999999
#
GRIDPARAMETER
REFINEMENTEDGE ARBITRARY
BISECTIONCOMPATIBILITY 1
#
"""

# Create a grid from the DGF string
grid = yaspGrid((reader.dgfString, dgf_string), dimgrid=2)
print("Grid created:", grid)
print("Grid type:", type(grid))

# Get a grid view
gv = levelView(grid, 0)
print("Grid view created:", gv)
print("Grid view type:", type(gv))

# Create a finite element space
X = space.lagrange(gv, 1)
print("Space created:", X)
print("Space dim:", X.dim)

# Create trial and test functions
u = TrialFunction(X)
v = TestFunction(X)

# Define the bilinear form
K = ufl.as_tensor([[2.5, 0], [0, 1]])
a = inner(K * grad(u), grad(v)) * dx

# Define the source term
f = 9*ufl.x[0]**3*ufl.x[1]/2 - 17*ufl.x[0]**3/6 + 15*ufl.x[0]**2*ufl.x[1]/2 + 41*ufl.x[0]**2/6 + 45*ufl.x[0]*ufl.x[1]**3/4 - 85*ufl.x[0]*ufl.x[1]**2/4 - 2*ufl.x[0]*ufl.x[1] - 4*ufl.x[0] + 25*ufl.x[1]**3/4 + 205*ufl.x[1]**2/12 - 70*ufl.x[1]/3
L = f * v * dx

# Define Dirichlet BC
bc = DirichletBC(X, 0, "on_boundary")

# Assemble the system
A, b = assemble([a - L, bc])
print("Matrix assembled:", A)
print("RHS assembled:", b)

# Solve the system
x = X.zero.copy()
print("Initial solution:", x)

# Use a solver
from dune.fem import discretefunction as df
solver = df.suitesparsesolver(X)
solver.solve(A, b, x)
print("Solution:", x)
