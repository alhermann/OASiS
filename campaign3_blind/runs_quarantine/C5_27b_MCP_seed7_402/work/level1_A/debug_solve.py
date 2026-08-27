import numpy as np
from skfem import MeshTri, ElementTriP1, Basis, BilinearForm, LinearForm, condense, solve
from skfem.helpers import dot, grad

NX = 8
TOL = 1e-9 * max(1.0, 1.0/NX)
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
elem = ElementTriP1()
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs[0]
px, py = mesh.p[0], mesh.p[1]

# Simple constant k and f for testing
@BilinearForm
def stiffness(u, v, w):
    return 1.0 * dot(grad(u), grad(v))

@LinearForm
def source(v, w):
    return 1.0 * v

A = stiffness.assemble(basis)
b = source.assemble(basis)

# Outer boundary only (no interface constraint for now)
outer_mask = (
    (np.abs(px - 0.0) < TOL) |
    (np.abs(py - 0.0) < TOL) |
    (np.abs(py - 1.0) < TOL) |
    (np.abs(px - 1.0) < TOL) |
    ((np.abs(px - 0.75) < TOL) & (py > 0.75)) |
    ((np.abs(py - 0.75) < TOL) & (px > 0.75))
)
outer_dofs = n2d[np.where(outer_mask)[0]]

print(f"A shape: {A.shape}, nnz: {A.nnz}")
print(f"Outer dofs: {len(outer_dofs)}")

# Try solving with just outer BC
sol = basis.zeros()
sol[outer_dofs] = 0.0
try:
    sol = solve(*condense(A, b, x=sol, D=outer_dofs))
    print(f"Solution with outer BC only: min={sol.min():.6g}, max={sol.max():.6g}")
except Exception as e:
    print(f"Error with outer BC only: {e}")

# Now add interface
iface_leg1 = np.where((np.abs(px - 0.5) < TOL) & (py > TOL) & (py < 0.5 - TOL))[0]
iface_leg2 = np.where((np.abs(py - 0.5) < TOL) & (px > 0.5 + TOL) & (px < 1.0 - TOL))[0]
iface_nodes = np.concatenate([iface_leg1, iface_leg2])
iface_dofs = n2d[iface_nodes]

D = np.concatenate([outer_dofs, iface_dofs])
print(f"Total constrained: {len(np.unique(D))}")

sol = basis.zeros()
sol[outer_dofs] = 0.0
sol[iface_dofs] = 0.1  # Non-zero interface value
try:
    sol = solve(*condense(A, b, x=sol, D=D))
    print(f"Solution with interface BC: min={sol.min():.6g}, max={sol.max():.6g}")
except Exception as e:
    print(f"Error with interface BC: {e}")
