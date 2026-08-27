import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

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

nodes_x = np.array(nodes_x)
nodes_y = np.array(nodes_y)
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
            if 0.5 * abs(np.cross(p1-p0, p2-p0)) > 1e-10:
                elems.append([n0, n1, n2])
            p1, p2, p3 = np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n3], nodes_y[n3]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            if 0.5 * abs(np.cross(p2-p1, p3-p1)) > 1e-10:
                elems.append([n1, n3, n2])
        elif len(cell_nodes) == 3:
            n0, n1, n2 = cell_nodes
            p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                         np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            if 0.5 * abs(np.cross(p1-p0, p2-p0)) > 1e-10:
                elems.append(cell_nodes)

elems = np.array(elems)
nnodes = len(nodes_x)

# Build adjacency graph
adj_rows, adj_cols = [], []
for e in elems:
    for i in range(3):
        for j in range(i+1, 3):
            adj_rows.extend([e[i], e[j]])
            adj_cols.extend([e[j], e[i]])

adj = csr_matrix(([1]*len(adj_rows), (adj_rows, adj_cols)), shape=(nnodes, nnodes))
n_comp, labels = connected_components(adj, directed=False)

print(f"Number of connected components: {n_comp}")
for c in range(n_comp):
    comp_nodes = np.where(labels == c)[0]
    print(f"Component {c}: {len(comp_nodes)} nodes")
    if len(comp_nodes) <= 10:
        coords = list(zip(nodes_x[comp_nodes], nodes_y[comp_nodes]))
        print(f"  Coordinates: {coords}")
