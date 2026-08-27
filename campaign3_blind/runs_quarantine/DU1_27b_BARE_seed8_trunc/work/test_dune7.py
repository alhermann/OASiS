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

# Try to load a grid from a file
mshfile = "/home/alexander/miniconda3/envs/dune-fem-env/lib/python3.12/site-packages/dune/grid/tutorial/test2d_offset.dgf"
print("Loading grid from:", mshfile)

try:
    grid = yaspGrid((reader.dgf, mshfile), dimgrid=2)
    print("Grid created:", grid)
    print("Grid type:", type(grid))
except Exception as e:
    print("Error creating grid:", e)
