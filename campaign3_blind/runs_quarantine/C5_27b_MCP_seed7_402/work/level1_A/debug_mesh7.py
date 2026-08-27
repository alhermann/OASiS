import numpy as np
from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, LinearForm, condense, solve
from skfem.helpers import dot, grad

# Use init_tensor which works correctly
mesh = MeshTri.init_tensor(np.linspace(0, 1, 9), np.linspace(0, 1, 9))
print(f"Full mesh: {mesh.nelements} elements")
print(f"mesh.nnodes = {mesh.nnodes}, mesh.p.shape = {mesh.p.shape}")

elem = ElementTriP1()
basis = Basis(mesh, elem)
print(f"Basis NDOF = {basis.N}")

# Simple test problem
@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm  
def source(v, w):
    return 1.0 * v

A = stiffness.assemble(basis)
b = source.assemble(basis)

# Dirichlet on all boundary
D = basis.get_dofs()
print(f"Boundary dofs: {len(D)}")

sol = solve(*condense(A, b, D=D))
print(f"Solution: min={sol.min():.6g}, max={sol.max():.6g}")
