"""Build mesh for subdomain A - filtering degenerate elements."""
import numpy as np
from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, LinearForm, condense, solve
from skfem.helpers import dot, grad

NX = 8
h = 1.0 / NX

# Collect all nodes in subdomain A
nodes_x, nodes_y = [], []
node_set = set()

for i in range(NX + 1):
    for j in range(NX + 1):
        x, y = i * h, j * h
        
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

# Build elements, filtering degenerate ones
elems = []
for i in range(NX):
    for j in range(NX):
        corners = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
        xc, yc = (i + 0.5) * h, (j + 0.5) * h
        
        if xc > 0.5 and yc < 0.5:
            continue
        if xc > 0.75 and yc > 0.75:
            continue
        
        cell_nodes = [node_map[c] for c in corners if c in node_map]
        
        if len(cell_nodes) == 4:
            n0, n1, n2, n3 = cell_nodes
            # Check areas before adding
            p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                         np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            area1 = 0.5 * abs(np.cross(p1-p0, p2-p0))
            if area1 > 1e-10:
                elems.append([n0, n1, n2])
            
            p1, p2, p3 = np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n3], nodes_y[n3]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            area2 = 0.5 * abs(np.cross(p2-p1, p3-p1))
            if area2 > 1e-10:
                elems.append([n1, n3, n2])
        elif len(cell_nodes) == 3:
            n0, n1, n2 = cell_nodes
            p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                         np.array([nodes_x[n1], nodes_y[n1]]), \
                         np.array([nodes_x[n2], nodes_y[n2]])
            area = 0.5 * abs(np.cross(p1-p0, p2-p0))
            if area > 1e-10:
                elems.append(cell_nodes)

elems = np.array(elems)
print(f"Number of valid elements: {len(elems)}")

p_skfem = np.array([nodes_x, nodes_y])
t_skfem = elems.T

mesh = MeshTri(p_skfem, t_skfem, validate=False)
print(f"skfem Mesh: {mesh.nelements} elements, p.shape={mesh.p.shape}")

elem = ElementTriP1()
basis = Basis(mesh, elem)
print(f"Basis NDOF: {basis.N}")

# Test solve
@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm  
def source(v, w):
    return 1.0 * v

A = stiffness.assemble(basis)
b = source.assemble(basis)

# Outer boundary DOFs
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
print(f"Success!" if np.isfinite(sol).all() else "FAILED!")
