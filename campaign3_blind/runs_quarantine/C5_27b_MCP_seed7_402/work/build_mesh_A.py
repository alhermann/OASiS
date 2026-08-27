"""Build mesh for subdomain A using meshio and skfem."""
import numpy as np
import meshio
from skfem import MeshTri, ElementTriP1, Basis

NX = 8
h = 1.0 / NX

# Collect all nodes in subdomain A
nodes_x, nodes_y = [], []
node_set = set()

for i in range(NX + 1):
    for j in range(NX + 1):
        x, y = i * h, j * h
        
        # Check if node is in subdomain A
        # Exclude interior of subdomain B: x > 0.5 and y < 0.5
        # Exclude interior of notch: x > 0.75 and y > 0.75
        if x > 0.5 and y < 0.5:
            continue
        if x > 0.75 and y > 0.75:
            continue
        
        node_set.add((i, j))
        nodes_x.append(x)
        nodes_y.append(y)

nodes_x = np.array(nodes_x)
nodes_y = np.array(nodes_y)
node_map = {(i, j): idx for idx, (i, j) in enumerate(node_set)}

print(f"Number of nodes: {len(node_set)}")

# Build elements
elems = []
for i in range(NX):
    for j in range(NX):
        corners = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        
        # Skip if cell is in subdomain B or notch
        if xc > 0.5 and yc < 0.5:
            continue
        if xc > 0.75 and yc > 0.75:
            continue
        
        cell_nodes = [node_map[c] for c in corners if c in node_map]
        
        if len(cell_nodes) == 4:
            n0, n1, n2, n3 = cell_nodes
            # Two triangles per quad
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
        elif len(cell_nodes) >= 3:
            # Use available nodes
            elems.append(cell_nodes[:3])

elems = np.array(elems)
print(f"Number of elements: {len(elems)}")

# Create meshio mesh
points = np.column_stack([nodes_x, nodes_y])
cells = [("triangle", elems)]
mesh_io = meshio.Mesh(points, cells)

# Convert to skfem MeshTri
# meshio uses (n_cells, ndim) format, skfem uses (ndim, n_cells)
p_skfem = points.T  # (2, nnodes)
t_skfem = elems.T   # (3, nelements)

print(f"p_skfem shape: {p_skfem.shape}, t_skfem shape: {t_skfem.shape}")

mesh = MeshTri(p_skfem, t_skfem, validate=False)
print(f"skfem Mesh: {mesh.nelements} elements")
print(f"mesh.p.shape: {mesh.p.shape}")

elem = ElementTriP1()
basis = Basis(mesh, elem)
print(f"Basis NDOF: {basis.N}")
