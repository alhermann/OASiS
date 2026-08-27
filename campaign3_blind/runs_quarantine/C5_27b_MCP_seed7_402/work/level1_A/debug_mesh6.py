import numpy as np
from skfem import MeshTri

# Try with validate=False
p = np.array([[0.0, 1.0, 0.0, 1.0],
              [0.0, 0.0, 1.0, 1.0]])
t = np.array([[0, 1],
              [2, 2],  
              [1, 3]])

print("With validate=True (default):")
mesh1 = MeshTri(p, t, validate=True)
print(f"  {mesh1.nelements} elements, {mesh1.nnodes} nodes")

print("\nWith validate=False:")
mesh2 = MeshTri(p, t, validate=False)
print(f"  {mesh2.nelements} elements, {mesh2.nnodes} nodes")
print(f"  mesh2.p shape: {mesh2.p.shape}")
print(f"  mesh2.t shape: {mesh2.t.shape}")

# Check what init_tensor does
print("\nUsing init_tensor:")
mesh3 = MeshTri.init_tensor(np.linspace(0, 1, 3), np.linspace(0, 1, 3))
print(f"  {mesh3.nelements} elements, {mesh3.nnodes} nodes")
print(f"  mesh3.p:\n{mesh3.p}")
print(f"  mesh3.t:\n{mesh3.t}")
