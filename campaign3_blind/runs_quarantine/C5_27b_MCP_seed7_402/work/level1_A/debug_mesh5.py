import numpy as np
from skfem import MeshTri

# Correct format: t should be (3, nelements) - 3 nodes per triangle
p = np.array([[0.0, 1.0, 0.0, 1.0],   # x coords of 4 nodes
              [0.0, 0.0, 1.0, 1.0]])  # y coords of 4 nodes
t = np.array([[0, 1],    # node 0 for both triangles
              [2, 2],    # node 2 for both triangles  
              [1, 3]])   # nodes 1 and 3 for triangles 1 and 2

print(f"p shape: {p.shape}, t shape: {t.shape}")
mesh = MeshTri(p, t)
print(f"Mesh: {mesh.nelements} elements, {mesh.nnodes} nodes")
print(f"mesh.p:\n{mesh.p}")
print(f"mesh.t:\n{mesh.t}")
