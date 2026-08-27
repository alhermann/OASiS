"""Build mesh for subdomain A - fixed node indexing."""
import numpy as np
from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, LinearForm, condense, solve
from skfem.helpers import dot, grad
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

NX = 8
h = 1.0 / NX

# First pass: identify all valid nodes
valid_nodes = []
for i in range(NX + 1):
    for j in range(NX + 1):
        x, y = i * h, j * h
        
        # Node is valid if it's NOT strictly inside subdomain B or notch
        # Subdomain B: (0.5, 1) x (0, 0.5) - exclude if x > 0.5 AND y < 0.5
        # Notch: (0.75, 1) x (0.75, 1) - exclude if x > 0.75 AND y > 0.75
        in_B = (x > 0.5) and (y < 0.5)
        in_notch = (x > 0.75) and (y > 0.75)
        
        if not in_B and not in_notch:
            valid_nodes.append((i, j, x, y))

print(f"Valid nodes: {len(valid_nodes)}")

# Create node map from (i,j) to index
node_map = {(i, j): idx for idx, (i, j, x, y) in enumerate(valid_nodes)}
nodes_x = np.array([x for i, j, x, y in valid_nodes])
nodes_y = np.array([y for i, j, x, y in valid_nodes])

# Build elements
elems = []
for i in range(NX):
    for j in range(NX):
        # Cell center
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        
        # Skip cell if center is in subdomain B or notch
        in_B = (xc > 0.5) and (yc < 0.5)
        in_notch = (xc > 0.75) and (yc > 0.75)
        
        if in_B or in_notch:
            continue
        
        # Get corner nodes
        corners = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
        cell_nodes = [node_map[c] for c in corners if c in node_map]
        
        if len(cell_nodes) == 4:
            n0, n1, n2, n3 = cell_nodes
            # Two triangles
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])
        elif len(cell_nodes) >= 3:
            elems.append(cell_nodes[:3])

elems = np.array(elems)
print(f"Elements: {len(elems)}")

# Check connectivity
nnodes = len(nodes_x)
adj_rows, adj_cols = [], []
for e in elems:
    for ii in range(3):
        for jj in range(ii+1, 3):
            adj_rows.extend([e[ii], e[jj]])
            adj_cols.extend([e[jj], e[ii]])

adj = csr_matrix(([1]*len(adj_rows), (adj_rows, adj_cols)), shape=(nnodes, nnodes))
n_comp, labels = connected_components(adj, directed=False)
print(f"Connected components: {n_comp}")

if n_comp > 1:
    for c in range(n_comp):
        comp_nodes = np.where(labels == c)[0]
        print(f"  Component {c}: {len(comp_nodes)} nodes")

# Create mesh
p_skfem = np.array([nodes_x, nodes_y])
t_skfem = elems.T

mesh = MeshTri(p_skfem, t_skfem, validate=False)
elem = ElementTriP1()
basis = Basis(mesh, elem)

print(f"\nMesh: {mesh.nelements} elements, {basis.N} DOFs")

# Test solve
@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm  
def source(v, w):
    return 1.0 * v

A = stiffness.assemble(basis)
b = source.assemble(basis)

# Outer boundary
TOL = 1e-9
px, py = mesh.p[0], mesh.p[1]
outer_mask = (
    (np.abs(px - 0.0) < TOL) |
    (np.abs(py - 0.0) < TOL) |
    (np.abs(py - 1.0) < TOL) |
    (np.abs(px - 1.0) < TOL) |
    ((np.abs(px - 0.75) < TOL) & (py > 0.75)) |
    ((np.abs(py - 0.75) < TOL) & (px > 0.75))
)
D = basis.nodal_dofs[0][np.where(outer_mask)[0]]
print(f"Outer boundary dofs: {len(D)}")

sol = solve(*condense(A, b, D=D))
print(f"Solution: min={sol.min():.6g}, max={sol.max():.6g}")
print("SUCCESS!" if np.isfinite(sol).all() else "FAILED!")
