import numpy as np
from skfem import MeshTri

# Test with a simple known mesh first
p_simple = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])  # 4 nodes
t_simple = np.array([[0, 1, 2], [1, 3, 2]])  # 2 triangles
mesh_simple = MeshTri(p_simple, t_simple)
print(f"Simple mesh: {mesh_simple.nelements} elements, {mesh_simple.nnodes} nodes")

# Now test our mesh
NX = 8
h = 1.0 / NX

nodes_x, nodes_y = [], []
node_set = set()
for i in range(NX + 1):
    for j in range(NX + 1):
        x, y = i * h, j * h
        if x > 0.5 and y < 0.5: continue
        if x > 0.75 and y > 0.75: continue
        node_set.add((i, j))
        nodes_x.append(x)
        nodes_y.append(y)

nodes_x, nodes_y = np.array(nodes_x), np.array(nodes_y)
node_map = {(i, j): idx for idx, (i, j) in enumerate(node_set)}

elems = []
for i in range(NX):
    for j in range(NX):
        corners = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        if xc > 0.5 and yc < 0.5: continue
        if xc > 0.75 and yc > 0.75: continue
        
        cell_nodes = [node_map[c] for c in corners if c in node_map]
        if len(cell_nodes) == 4:
            n0, n1, n2, n3 = cell_nodes
            p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                         np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            if 0.5 * abs(np.cross(p1-p0, p2-p0)) > 1e-12:
                elems.append([n0, n1, n2])
            p1, p2, p3 = np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n3], nodes_y[n3]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            if 0.5 * abs(np.cross(p2-p1, p3-p1)) > 1e-12:
                elems.append([n1, n3, n2])
        elif len(cell_nodes) == 3:
            n0, n1, n2 = cell_nodes
            p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                         np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            if 0.5 * abs(np.cross(p1-p0, p2-p0)) > 1e-12:
                elems.append(cell_nodes)

p = np.array([nodes_x, nodes_y])
t = np.array(elems).T

print(f"Input p shape: {p.shape}, t shape: {t.shape}")
print(f"Max index in t: {t.max()}, expected max: {len(nodes_x)-1}")

# Check if indices are valid
if t.max() >= len(nodes_x):
    print("ERROR: Element indices exceed number of nodes!")
else:
    print("Indices are valid")

# Try creating mesh
try:
    mesh = MeshTri(p, t)
    print(f"Mesh created: {mesh.nelements} elements, {mesh.nnodes} nodes")
except Exception as e:
    print(f"Error creating mesh: {e}")

# Alternative: use init_tensor and filter
from skfem import MeshTri
mesh_full = MeshTri.init_tensor(np.linspace(0, 1, NX+1), np.linspace(0, 1, NX+1))
print(f"\nFull tensor mesh: {mesh_full.nelements} elements, {mesh_full.nnodes} nodes")

# Filter elements
keep = []
for e in range(mesh_full.nelements):
    xc = mesh_full.p[0, mesh_full.t[:, e]].mean()
    yc = mesh_full.p[1, mesh_full.t[:, e]].mean()
    if xc > 0.5 and yc < 0.5: continue
    if xc > 0.75 and yc > 0.75: continue
    keep.append(e)

print(f"Elements to keep: {len(keep)}")
# This approach won't work directly - we need to remap nodes
