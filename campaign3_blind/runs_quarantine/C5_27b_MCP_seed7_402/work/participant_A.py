"""scikit-fem participant for subdomain A (Dirichlet side) - OASiS couple driver.

Subdomain A: L-shaped domain = unit square minus (0.75,1)x(0.75,1), 
             further minus subdomain B = (0.5,1)x(0,0.5)
             
So A consists of:
- Region 1: (0, 0.5) x (0, 1)  [k=1 for y<0.5, k=2 for y>=0.5]
- Region 2: (0.5, 1) x (0.5, 1) minus (0.75,1)x(0.75,1) [k=5]

Interface (two legs):
- Leg 1: x=0.5, 0<y<0.5 (vertical) - normal from A is (+1, 0)
- Leg 2: y=0.5, 0.5<x<1 (horizontal) - normal from A is (0, +1)

Outer BC: u=0 on all outer boundaries including notch faces at x=0.75,y>0.75 and y=0.75,x>0.75
"""
import json
from pathlib import Path
import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm,
                   MeshTri, condense, solve, asm)
from skfem.helpers import dot, grad

# ── CONFIGURATION ───────────────────────────────────────────────────────────
SIDE      = "dirichlet"   # A is Dirichlet side
PARTNER   = "side_B"      # partner name in couple() call

# Read mesh level from file if present
config_path = Path("level_config.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    NX = config.get("nx", 8)
else:
    NX = 8  # default

# Tolerance for boundary detection
TOL = 1e-9 * max(1.0, 1.0/NX)

# Conductivity function - piecewise constant
def K(x, y):
    """Conductivity in subdomain A."""
    k_vals = np.zeros_like(x)
    # Region (0, 0.5) x (0, 0.5): k = 1
    mask = (x < 0.5) & (y < 0.5)
    k_vals[mask] = 1.0
    # Region (0, 0.5) x (0.5, 1): k = 2
    mask = (x < 0.5) & (y >= 0.5)
    k_vals[mask] = 2.0
    # Region (0.5, 1) x (0.5, 1) minus notch: k = 5
    mask = (x >= 0.5) & (y >= 0.5)
    k_vals[mask] = 5.0
    return k_vals

# Source term f(x,y) - piecewise polynomial
def F_SRC(x, y):
    """Source term in subdomain A."""
    f_vals = np.zeros_like(x)
    
    # Region (0, 0.5) x (0, 0.5): k=1
    mask = (x < 0.5) & (y < 0.5)
    xm, ym = x[mask], y[mask]
    f_vals[mask] = (3*xm**4*ym - 11*xm**4/8 + 12*xm**3*ym**2 - 123*xm**3*ym/20 
                    - xm**3/40 + 6*xm**2*ym**3 - 477*xm**2*ym**2/20 
                    + 2799*xm**2*ym/400 + 1423*xm**2/800 + 6*xm*ym**4 
                    - 123*xm*ym**3/20 + 993*xm*ym**2/200 + 63*xm*ym/1600 
                    - 609*xm/800 - 13*ym**4/5 + 279*ym**3/200 
                    + 1423*ym**2/800 - 327*ym/320)
    
    # Region (0, 0.5) x (0.5, 1): k=2
    mask = (x < 0.5) & (y >= 0.5)
    xm, ym = x[mask], y[mask]
    f_vals[mask] = (3*xm**4*ym/4 - 5*xm**4/16 + 3*xm**3*ym**2/2 - 3*xm**3*ym/80 
                    - 13*xm**3/32 + 3*xm**2*ym**3/2 - 153*xm**2*ym**2/40 
                    - 873*xm**2*ym/800 + 119*xm**2/80 + 3*xm*ym**4/4 
                    - 3*xm*ym**3/80 - 471*xm*ym**2/800 + 9*xm*ym/16 
                    - 3*xm/800 - 13*ym**4/40 - 241*ym**3/800 + 37*ym**2/40 
                    - 107*ym/3200 - 849/3200)
    
    # Region (0.5, 1) x (0.5, 1) minus notch: k=5
    mask = (x >= 0.5) & (y >= 0.5) & ((x < 0.75) | (y < 0.75))
    xm, ym = x[mask], y[mask]
    f_vals[mask] = (6*xm**4*ym/125 - xm**4/50 + 6*xm**3*ym**2/25 + 69*xm**3*ym/500 
                    - xm**3/8 + 12*xm**2*ym**3/125 - 9*xm**2*ym**2/25 
                    - 9*xm**2*ym/40 + 769*xm**2/4000 + 3*xm*ym**4/25 
                    + 69*xm*ym**3/500 - 51*xm*ym**2/100 - 549*xm*ym/8000 
                    + 2853*xm/16000 - ym**4/25 - 71*ym**3/1000 
                    + 233*ym**2/800 + 419*ym/4000 - 2031/16000)
    
    return f_vals

# Initial guesses for iteration 1
T_INIT = 0.0
Q_INIT = 0.0


def read_imports():
    """Read imports.json; return None or partner data."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, coords):
    """Interpolate partner's data onto our interface points."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(coords[0]), float(fallback))
    
    ys_partner = np.array([c[1] for c in imp["coordinates"]], float)
    xs_partner = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    
    if vs.size != len(ys_partner):
        return np.full(len(coords[0]), float(fallback))
    
    # Sort by coordinate for interpolation
    o = np.argsort(xs_partner)
    xs_sorted = xs_partner[o]
    ys_sorted = ys_partner[o]
    vs_sorted = vs[o]
    
    result = np.zeros(len(coords[0]))
    for i, (xx, yy) in enumerate(zip(coords[0], coords[1])):
        # Determine which leg this point is on
        if abs(xx - 0.5) < 1e-6:  # Vertical leg (x=0.5)
            result[i] = np.interp(yy, ys_sorted, vs_sorted)
        else:  # Horizontal leg (y=0.5)
            result[i] = np.interp(xx, xs_sorted, vs_sorted)
    
    return result


def build_subdomain_A_mesh(NX):
    """Build mesh for subdomain A with NX divisions per unit length.
    
    Subdomain A = unit square minus (0.75,1)x(0.75,1) minus (0.5,1)x(0,0.5)
    """
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
    
    # Map (i,j) to node index
    node_map = {(i, j): idx for idx, (i, j) in enumerate(node_set)}
    
    # Build elements - only include valid triangles
    elems = []
    for i in range(NX):
        for j in range(NX):
            # Cell corners
            corners = [(i, j), (i+1, j), (i, j+1), (i+1, j+1)]
            
            # Check cell center
            xc = (i + 0.5) * h
            yc = (j + 0.5) * h
            
            # Skip if cell is in subdomain B or notch
            if xc > 0.5 and yc < 0.5:
                continue
            if xc > 0.75 and yc > 0.75:
                continue
            
            # Get node indices for this cell
            cell_nodes = []
            for corner in corners:
                if corner in node_map:
                    cell_nodes.append(node_map[corner])
            
            if len(cell_nodes) == 4:
                # Split into two triangles
                n0, n1, n2, n3 = cell_nodes
                # Check that triangles are non-degenerate
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
                # Triangle at boundary - check it's non-degenerate
                n0, n1, n2 = cell_nodes
                p0, p1, p2 = np.array([nodes_x[n0], nodes_y[n0]]), \
                             np.array([nodes_x[n1], nodes_y[n1]]), \
                             np.array([nodes_x[n2], nodes_y[n2]])
                area = 0.5 * abs(np.cross(p1-p0, p2-p0))
                if area > 1e-12:
                    elems.append(cell_nodes)
    
    if len(elems) == 0:
        raise RuntimeError("No valid elements created!")
    
    elems = np.array(elems).T
    p = np.array([nodes_x, nodes_y])
    
    return MeshTri(p, elems)


mesh = build_subdomain_A_mesh(NX)
elem = ElementTriP1()
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs[0]

px, py = mesh.p[0], mesh.p[1]

# Identify interface nodes (two legs)
# Leg 1: x = 0.5, 0 < y < 0.5 (vertical)
# Leg 2: y = 0.5, 0.5 < x < 1 (horizontal)
# EXCLUDE corners: (0.5, 0), (0.5, 0.5), (1, 0.5) - these are outer BC or ambiguous

iface_leg1 = np.where((np.abs(px - 0.5) < TOL) & (py > TOL) & (py < 0.5 - TOL))[0]
iface_leg2 = np.where((np.abs(py - 0.5) < TOL) & (px > 0.5 + TOL) & (px < 1.0 - TOL))[0]

# Sort each leg
iface_leg1 = iface_leg1[np.argsort(py[iface_leg1])]
iface_leg2 = iface_leg2[np.argsort(px[iface_leg2])]

# Combine: leg 1 first, then leg 2
iface_nodes = np.concatenate([iface_leg1, iface_leg2])
iface_dofs = n2d[iface_nodes]

# Interface coordinates for export
iface_x = px[iface_nodes]
iface_y = py[iface_nodes]

# Identify outer boundary nodes (u=0)
# Outer boundary: x=0, y=0, y=1, x=1, plus notch faces x=0.75 (y>0.75) and y=0.75 (x>0.75)
outer_mask = (
    (np.abs(px - 0.0) < TOL) |  # left
    (np.abs(py - 0.0) < TOL) |  # bottom
    (np.abs(py - 1.0) < TOL) |  # top
    (np.abs(px - 1.0) < TOL) |  # right
    ((np.abs(px - 0.75) < TOL) & (py > 0.75)) |  # notch vertical face
    ((np.abs(py - 0.75) < TOL) & (px > 0.75))    # notch horizontal face
)
outer_nodes = np.where(outer_mask)[0]
outer_dofs = n2d[outer_nodes]

print(f"[skfem A] NDOF = {basis.N}, nelements = {mesh.nelements}, interface nodes = {len(iface_nodes)}, outer dofs = {len(outer_dofs)}")

# Write run log
with open("run.log", "w") as f:
    f.write(f"NDOF = {basis.N}\n")
    f.write(f"interface_nodes = {len(iface_nodes)}\n")
    f.write(f"outer_dofs = {len(outer_dofs)}\n")

# Stiffness form with variable k - evaluate K at quadrature points
@BilinearForm
def stiffness(u, v, w):
    return K(w.x[0], w.x[1]) * dot(grad(u), grad(v))

# Source form
@LinearForm
def source(v, w):
    return F_SRC(w.x[0], w.x[1]) * v

# Interface facet basis for flux recovery
facets_if = mesh.facets_satisfying(lambda p: 
    ((np.abs(p[0] - 0.5) < TOL) & (p[1] > TOL) & (p[1] < 0.5 - TOL)) |  # leg 1
    ((np.abs(p[1] - 0.5) < TOL) & (p[0] > 0.5 + TOL) & (p[0] < 1.0 - TOL)))  # leg 2

fbasis = FacetBasis(mesh, elem, facets=facets_if)

# Unit load for flux normalization
@LinearForm
def unit_load(v, w):
    return 1.0 * v

# Assemble
A = stiffness.assemble(basis)
b = source.assemble(basis)

# Read imports and apply interface BC
imp = read_imports()

sol = basis.zeros()
sol[outer_dofs] = 0.0  # Outer Dirichlet BC

if SIDE == "dirichlet":
    # Import temperature from partner
    T_if = sample(imp, "values", T_INIT, (iface_x, iface_y))
    sol[iface_dofs] = T_if
    D = np.concatenate([outer_dofs, iface_dofs])
else:
    # Neumann side would import flux here
    D = outer_dofs

# Solve
sol = solve(*condense(A, b, x=sol, D=D))

# Export interface data
# Values: temperature at interface
T_export = sol[iface_dofs]

# Flux: consistent (reaction) flux for Dirichlet side
# q_out = -(k grad u) . n_out where n_out is outward normal from subdomain A
# On leg 1 (x=0.5): n_out = (+1, 0)
# On leg 2 (y=0.5): n_out = (0, +1)

r = A @ sol - b  # Unconstrained residual

# Compute nodal weights on interface
wgt = unit_load.assemble(fbasis)

# Extract flux at interface dofs
Q = np.zeros(len(iface_dofs))
ok = np.abs(wgt[iface_dofs]) > 1e-14
Q[ok] = -r[iface_dofs][ok] / wgt[iface_dofs][ok]

# Fix suspect nodes (those also on outer boundary)
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Write exports.json
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(iface_x[i]), float(iface_y[i])] for i in range(len(iface_dofs))],
    "values": [float(t) for t in T_export],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

print(f"[skfem A] Exported {len(iface_dofs)} interface points")
print(f"[skfem A] T_range = [{T_export.min():.6g}, {T_export.max():.6g}]")
print(f"[skfem A] q_range = [{Q.min():.6g}, {Q.max():.6g}]")
