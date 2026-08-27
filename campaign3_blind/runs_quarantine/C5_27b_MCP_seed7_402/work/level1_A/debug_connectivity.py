import numpy as np
from skfem import MeshTri, ElementTriP1, Basis

NX = 8
h = 1.0 / NX

# Build mesh
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

mesh = MeshTri(np.array([nodes_x, nodes_y]), np.array(elems).T)
print(f"Mesh: {mesh.nelements} elements, {mesh.nnodes} nodes")

# Check connectivity using graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Build adjacency from elements
nnodes = mesh.nnodes
adj_data = []
adj_rows = []
adj_cols = []
for e in range(mesh.nelements):
    nodes = mesh.t[:, e]
    for i in range(3):
        for j in range(i+1, 3):
            adj_rows.extend([nodes[i], nodes[j]])
            adj_cols.extend([nodes[j], nodes[i]])
            adj_data.extend([1, 1])

adj = csr_matrix((adj_data, (adj_rows, adj_cols)), shape=(nnodes, nnodes))
n_components, labels = connected_components(adj, directed=False)
print(f"Number of connected components: {n_components}")
print(f"Component sizes: {[np.sum(labels == i) for i in range(n_components)]}")

# Print which nodes are in each component
for comp in range(n_components):
    comp_nodes = np.where(labels == comp)[0]
    print(f"Component {comp}: nodes {comp_nodes[:10]}... (total {len(comp_nodes)})")
    if len(comp_nodes) <= 20:
        print(f"  Coordinates: {list(zip(nodes_x[comp_nodes], nodes_y[comp_nodes]))}")
