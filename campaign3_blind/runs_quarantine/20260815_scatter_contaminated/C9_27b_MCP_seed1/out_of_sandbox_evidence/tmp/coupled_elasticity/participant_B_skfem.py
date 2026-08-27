"""scikit-fem participant for coupled elasticity - Subdomain B (Neumann side).

Subdomain B: (0.625, 1.5) x (0, 1), lambda=450, mu=900 (E=2100, nu=1/6)
Interface at x = 0.625 (left edge of B)
Outer BC: u=0 on right, top, bottom edges (including interface corners)
Role: Neumann side - imports traction, exports displacement
"""
import json
from pathlib import Path
import sys
import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters

# Problem parameters for Subdomain B
X0, X1 = 0.625, 1.5  # Subdomain B extent
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625      # Interface is at left edge of B
LAMBDA_B = 450.0
MU_B = 900.0
E_B = 2100.0         # Young's modulus
NU_B = 1.0/6.0       # Poisson's ratio

PARTNER = "A"  # Name of partner in couple()

# Fallback values for iteration 1
T_INIT = [0.0, 0.0]


def body_force_x_B(x, y):
    """Body force f_x in subdomain B."""
    return (-3*x**2*y**3/6125 - 9*x**2*y**2/17500 + 117*x**2*y/122500 
            - 9*x**2/49000 + 1907*x*y**3/140000 + 5721*x*y**2/400000 
            - 74373*x*y/2800000 + 5721*x/1120000 - 3*y**5/24500 
            - 3*y**4/14000 - 168333*y**3/15680000 
            - 3941073*y**2/313600000 + 7051707*y/313600000 
            - 542439/125440000)


def body_force_y_B(x, y):
    """Body force f_y in subdomain B."""
    return (8829*x**2*y**2/392000 + 8829*x**2*y/560000 - 114777*x**2/7840000
            - 9*x*y**4/24500 - 9*x*y**3/17500 - 425601*x*y**2/15680000
            - 644571*x*y/31360000 + 1164969*x/62720000
            + 6333*y**4/1960000 + 6333*y**3/1400000
            - 3201021*y**2/156800000 - 39051*y/62720000
            + 127413/25088000)


def read_imports():
    """Read imports.json and return partner's interface data."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER)


def sample_vector(imp, y_coords):
    """Interpolate partner's vector field onto our y-coordinates."""
    if imp is None or "coordinates" not in imp:
        n = len(y_coords)
        return np.array([[T_INIT[0], T_INIT[1]]] * n)
    
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vals = np.asarray(imp.get("values", []), float)
    
    if vals.ndim == 1:
        vals = vals.reshape(-1, 2)
    
    if vals.shape[0] != len(ys):
        n = len(y_coords)
        return np.array([[T_INIT[0], T_INIT[1]]] * n)
    
    order = np.argsort(ys)
    tx_interp = np.interp(y_coords, ys[order], vals[order, 0])
    ty_interp = np.interp(y_coords, ys[order], vals[order, 1])
    
    return np.column_stack([tx_interp, ty_interp])


def main():
    global NX, NY
    
    # Get mesh resolution from command line
    if len(sys.argv) > 1:
        NX = int(sys.argv[1])
        NY = int(sys.argv[2])
    else:
        NX = 8
        NY = 8
    
    # Build mesh using quadrilaterals
    xs = np.linspace(X0, X1, NX + 1)
    ys = np.linspace(Y0, Y1, NY + 1)
    mesh = MeshQuad.init_tensor(xs, ys)
    
    # Vector element for elasticity
    elem = ElementVector(ElementQuad1())
    basis = Basis(mesh, elem)
    
    # Lamé parameters
    lam, mu = lame_parameters(E_B, NU_B)
    
    # Bilinear form for linear elasticity
    A = linear_elasticity(lam, mu).assemble(basis)
    
    # Body force linear form - use proper skfem syntax
    @LinearForm
    def load(v, w):
        fx = body_force_x_B(w.x[0], w.x[1])
        fy = body_force_y_B(w.x[0], w.x[1])
        return fx * v[0] + fy * v[1]
    
    b = load.assemble(basis)
    
    # Find interface and outer boundary DOFs
    tol = 1e-9 * max(X1 - X0, Y1 - Y0)
    
    # mesh.p is (2, n_nodes), so px = mesh.p[0] gives x-coords
    px = mesh.p[0]
    py = mesh.p[1]
    nodal_dofs = basis.nodal_dofs  # shape (2, n_nodes)
    
    # Interface nodes at x = IFACE_X (left edge), EXCLUDING corners
    iface_node_mask = np.abs(px - IFACE_X) < tol
    iface_nodes_all = np.flatnonzero(iface_node_mask)
    
    # Exclude corners (y=0 and y=1)
    iface_nodes_interior = iface_nodes_all[(py[iface_nodes_all] > tol) & (py[iface_nodes_all] < (Y1 - tol))]
    iface_nodes_sorted = iface_nodes_interior[np.argsort(py[iface_nodes_interior])]
    iface_y = py[iface_nodes_sorted]
    
    print(f"Found {len(iface_nodes_sorted)} interface nodes (excluding corners)")
    
    # Interface DOFs (both components)
    iface_dofs_x = nodal_dofs[0, iface_nodes_sorted]
    iface_dofs_y = nodal_dofs[1, iface_nodes_sorted]
    
    # Outer boundary nodes: right edge, top edge, bottom edge, AND interface corners
    outer_node_mask = (
        (np.abs(px - X1) < tol) |  # Right edge
        (np.abs(py - Y1) < tol) |  # Top edge  
        (np.abs(py - Y0) < tol)    # Bottom edge
    )
    outer_nodes = np.flatnonzero(outer_node_mask)
    outer_dofs_x = nodal_dofs[0, outer_nodes]
    outer_dofs_y = nodal_dofs[1, outer_nodes]
    outer_dofs = np.concatenate([outer_dofs_x, outer_dofs_y])
    
    # Read imports (traction from partner)
    imp = read_imports()
    
    # Sample traction at interface
    traction = sample_vector(imp, iface_y)
    
    # Apply traction as Neumann BC on interface
    dy = (Y1 - Y0) / NY
    for i, (dof_x, dof_y) in enumerate(zip(iface_dofs_x, iface_dofs_y)):
        # Tributary length for trapezoidal rule
        weight = dy / 2.0 if i == 0 or i == len(iface_nodes_sorted) - 1 else dy
        b[dof_x] += traction[i, 0] * weight
        b[dof_y] += traction[i, 1] * weight
    
    # Dirichlet BC: u = 0 on outer boundary
    x = solve(*condense(A, b, D=outer_dofs))
    
    # Extract displacement at interface
    u_iface_x = x[iface_dofs_x]
    u_iface_y = x[iface_dofs_y]
    u_iface = np.column_stack([u_iface_x, u_iface_y])
    
    # Write exports.json
    exports = {
        "field_name": "displacement",
        "n_points": len(iface_nodes_sorted),
        "coordinates": [[float(IFACE_X), float(y)] for y in iface_y],
        "values": [[float(u_iface[i, 0]), float(u_iface[i, 1])] for i in range(len(iface_nodes_sorted))],
        "normal_fluxes": [[0.0, 0.0] for _ in range(len(iface_nodes_sorted))]
    }
    
    Path("exports.json").write_text(json.dumps(exports, indent=2))
    
    # Write run log
    ndof = A.shape[0]
    with open("run.log", "w") as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"NX = {NX}, NY = {NY}\n")
        f.write(f"Interface nodes = {len(iface_nodes_sorted)}\n")
    
    print(f"Subdomain B (skfem): NDof={ndof}, iface_u_mean={u_iface.mean():.6e}")


if __name__ == "__main__":
    main()
