import numpy as np
from skfem import MeshTri, ElementTriP1, Basis

NX = 8
TOL = 1e-9 * max(1.0, 1.0/NX)
h = 1.0 / NX

# Collect all nodes in subdomain A
nodes_x = []
nodes_y = []
node_set = set()

for i in range(NX + 1):
    for j in range(NX + 1):
        x = i * h
        y = j * h
        
        # Check if node is in subdomain A
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

print(f"Total nodes: {len(node_set)}")

# Build elements
elems = []
for i in range(NX):
    for j in range(NX):
        corners = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
        xc = (i + 0.5) * h
        yc = (j + 0.5) * h
        
        if xc > 0.5 and yc < 0.5:
            continue
        if xc > 0.75 and yc > 0.75:
            continue
        
        cell_nodes = []
        for corner in corners:
            if corner in node_map:
                cell_nodes.append(node_map[corner])
        
        if len(cell_nodes) == 4:
            n0, n1, n2, n3 = cell_nodes
            p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                         np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            area1 = 0.5 * abs(np.cross(p1-p0, p2-p0))
            if area1 > 1e-12:
                elems.append([n0, n1, n2])
            
            p1, p2, p3 = np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n3], nodes_y[n3]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            area2 = 0.5 * abs(np.cross(p2-p1, p3-p1))
            if area2 > 1e-12:
                elems.append([n1, n3, n2])
        elif len(cell_nodes) == 3:
            n0, n1, n2 = cell_nodes
            p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                         np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            area = 0.5 * abs(np.cross(p1-p0, p2-p0))
            if area > 1e-12:
                elems.append(cell_nodes)

elems = np.array(elems).T
p = np.array([nodes_x, nodes_y])

mesh = MeshTri(p, elems)
elem = ElementTriP1()
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs[0]

px, py = mesh.p[0], mesh.p[1]

print(f"Mesh: {mesh.nelements} elements, {mesh.nnodes} nodes")
print(f"Basis NDOF: {basis.N}")

# Interface nodes
iface_leg1 = np.where((np.abs(px - 0.5) < TOL) & (py > TOL) & (py < 0.5 - TOL))[0]
iface_leg2 = np.where((np.abs(py - 0.5) < TOL) & (px > 0.5 + TOL) & (px < 1.0 - TOL))[0]

print(f"Interface leg 1 nodes: {len(iface_leg1)} at y={py[iface_leg1]}")
print(f"Interface leg 2 nodes: {len(iface_leg2)} at x={px[iface_leg2]}")

iface_nodes = np.concatenate([iface_leg1, iface_leg2])
iface_dofs = n2d[iface_nodes]

# Outer boundary
outer_mask = (
    (np.abs(px - 0.0) < TOL) |
    (np.abs(py - 0.0) < TOL) |
    (np.abs(py - 1.0) < TOL) |
    (np.abs(px - 1.0) < TOL) |
    ((np.abs(px - 0.75) < TOL) & (py > 0.75)) |
    ((np.abs(py - 0.75) < TOL) & (px > 0.75))
)
outer_nodes = np.where(outer_mask)[0]
outer_dofs = n2d[outer_nodes]

print(f"Outer boundary nodes: {len(outer_nodes)}")
print(f"Outer dofs: {outer_dofs}")

# Check for overlap between interface and outer
overlap = np.intersect1d(iface_dofs, outer_dofs)
print(f"Overlap between interface and outer: {overlap}")

# Total constrained
D = np.concatenate([outer_dofs, iface_dofs])
D_unique = np.unique(D)
print(f"Total unique constrained dofs: {len(D_unique)} out of {basis.N}")
print(f"Free dofs: {basis.N - len(D_unique)}")
