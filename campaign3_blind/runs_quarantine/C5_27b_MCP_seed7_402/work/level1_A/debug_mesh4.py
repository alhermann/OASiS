import numpy as np
from skfem import MeshTri

# Test the exact format expected
# According to docs: doflocs should be (ndim, nvertices)
# t should be (n_dofs_per_element, n_elements)

# Simple test - unit square divided into 2 triangles
p = np.array([[0.0, 1.0, 0.0, 1.0],   # x coords of 4 nodes
              [0.0, 0.0, 1.0, 1.0]])  # y coords of 4 nodes
t = np.array([[0, 2, 1],    # triangle 1: nodes 0, 2, 1
              [1, 2, 3]])   # triangle 2: nodes 1, 2, 3

print(f"p shape: {p.shape}, t shape: {t.shape}")
mesh = MeshTri(p, t)
print(f"Mesh: {mesh.nelements} elements, {mesh.nnodes} nodes")
print(f"mesh.p:\n{mesh.p}")
print(f"mesh.t:\n{mesh.t}")
