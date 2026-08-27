"""NGSolve participant for subdomain A (DIRICHLET side) - coupled heat conduction.

Subdomain A: (0, 0.625) x (0, 1), k = 1
Interface at x = 0.625 (right edge of this subdomain)
Outer BCs: u = 0 on left, top, bottom; interface gets Dirichlet from partner
"""
import json
from pathlib import Path
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, GridFunction, H1, LinearForm, 
                     Mesh, NodeId, TaskManager, ds, dx, grad, x, y)

# Problem parameters for subdomain A
SIDE      = "dirichlet"   # This is the DIRICHLET side
PARTNER   = "B"           # Partner name in couple()
X0, X1    = 0.0, 0.625    # Subdomain A x-range
Y0, Y1    = 0.0, 1.0      # y-range
IFACE_X   = 0.625         # Interface location (x = 5/8)
K         = 1.0           # Thermal conductivity in A
T_OUTER   = 0.0           # Dirichlet value on outer boundaries
T_INIT    = 0.0           # Iteration-1 fallback interface temperature
Q_INIT    = 0.0           # Iteration-1 fallback flux

# Read resolution from environment or use default
import os
RESOLUTION = int(os.environ.get('RESOLUTION', 8))  # h = 1/RESOLUTION
NX = max(1, int(0.625 * RESOLUTION))  # Elements in x for subdomain A
NY = max(1, int(1.0 * RESOLUTION))    # Elements in y

MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)
ORDER = 1

# Interface is on the right (x = X1)
ON_RIGHT = True
OUTER_X = X0
S = 1.0  # Outward normal at interface points in +x direction
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, y_coords):
    """Interpolate partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y_coords), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y_coords), float(fallback))
    o = np.argsort(ys)
    return np.interp(y_coords, ys[o], vs[o])


# Read imports
imp = read_imports()

# Create mesh with named boundaries
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=("bottom", "interface", "top", "outer"))  # bottom, right, top, left
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

# Dirichlet on outer boundary and interface (for Dirichlet side)
fes = H1(mesh, order=ORDER, dirichlet="outer|interface")
uh, vh = fes.TnT()

# Build vertex -> dof map
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[0] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

# Find interface vertices (on x = IFACE_X)
iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]  # Sort by y
y_if = vxy[iface_v, 1]
iface_dofs = vdof[iface_v]

# Find outer boundary vertices (on x = OUTER_X = 0)
outer_x_dofs = vdof[np.where(np.abs(vxy[:, 0] - OUTER_X) < TOL)[0]]

# Assemble bilinear form
a = BilinearForm(fes)
a += K * grad(uh) * grad(vh) * dx

# Source term f(x,y) in subdomain A using NGSolve's symbolic expression
f = LinearForm(fes)
f += (-18*x**3*y/5 + 2*x**3/5 + 5063*x**2*y/20000 - 95021*x**2/60000 
      - 18*x*y**3/5 + 6*x*y**2/5 + 14607*x*y/4000 + 1669*x/2000 
      + 5063*y**3/60000 - 95021*y**2/60000 + 14993*y/10000) * vh * dx

gfu = GridFunction(fes)
gfu.vec[:] = 0.0

# Apply outer Dirichlet BCs (u = 0 everywhere on outer boundary)
for d in outer_x_dofs:
    gfu.vec[int(d)] = T_OUTER

if SIDE == "dirichlet":
    # Import temperature from partner and apply as Dirichlet on interface
    T_if = sample(imp, "values", T_INIT, y_if)
    for d, t in zip(iface_dofs, T_if):
        gfu.vec[int(d)] = float(t)

with TaskManager():
    a.Assemble()
    f.Assemble()
    
    # Solve: modify RHS for Dirichlet and solve on free dofs
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * res

    # Compute outward normal flux using consistent (reaction) flux method
    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f.vec  # Reaction = Au - b
    
    # Weight function w_i = integral of phi_i over interface
    fw = LinearForm(fes)
    fw += vh * ds("interface")
    fw.Assemble()
    
    r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
    w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w_if) > 1e-14
    Q[ok] = -r_if[ok] / w_if[ok]
    
    # Handle corner nodes
    suspect = ~ok
    good = np.where(~suspect)[0]
    if len(good) > 0:
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Get interface temperatures
T_if_out = np.array([gfu.vec[int(d)] for d in iface_dofs], float)

# Write solution data as simple text file for post-processing
# Format: x y temperature
with open("solution.dat", 'w') as f:
    for i in range(mesh.nv):
        vx, vy = vxy[i]
        val = gfu.vec[int(vdof[i])]
        f.write(f"{vx} {vy} {val}\n")

# Write exports
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(iface_v),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(t) for t in T_if_out],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

# Write run log with NDOF
ndof = fes.ndof
with open("run.log", "w") as logfile:
    logfile.write(f"NDOF = {ndof}\n")
    logfile.write(f"Elements: {mesh.ne}\n")
    logfile.write(f"Interface points: {len(iface_v)}\n")

print(f"NGSolve A (Dirichlet): ndof={ndof}, T_interface_mean={T_if_out.mean():.6f}, q_out_mean={Q.mean():.6f}")
