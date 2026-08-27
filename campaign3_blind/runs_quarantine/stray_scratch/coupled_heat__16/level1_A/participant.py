"""NGSolve participant for subdomain A (DIRICHLET side) in coupled heat conduction.

Physics: -div(k grad u) = f on [0, 0.625] x [0, 1] with k=1
Interface at x = 0.625 (right edge of this subdomain)
Outer BC: u = 0 on left, top, bottom edges
Role: DIRICHLET - imports temperature from partner, exports flux

IMPORTANT: Interface corners (y=0 and y=1) are EXCLUDED from exchange.
They belong to the outer boundary and are constrained by u=0.
"""
import json
from pathlib import Path

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, GridFunction, H1, LinearForm, Mesh,
                     NodeId, TaskManager, ds, dx, grad)

# ── PROBLEM PARAMETERS ───────────────────────────────────────────────────────
SIDE      = "dirichlet"   # This is the DIRICHLET side
PARTNER   = "B"           # Name of the partner participant
X0, X1    = 0.0, 0.625    # Subdomain A: x in [0, 0.625]
Y0, Y1    = 0.0, 1.0      # y in [0, 1]
IFACE_X   = 0.625         # Interface at x = 5/8
K         = 1.0           # Thermal conductivity in subdomain A

# Source term f(x,y) in subdomain A (exactly as given in problem)
def F_SRC(x, y):
    """Source term for subdomain A."""
    return (-18*x**3*y/5 + 2*x**3/5 + 5063*x**2*y/20000 - 95021*x**2/60000 
            - 18*x*y**3/5 + 6*x*y**2/5 + 14607*x*y/4000 + 1669*x/2000 
            + 5063*y**3/60000 - 95021*y**2/60000 + 14993*y/10000)

T_OUTER   = 0.0           # Dirichlet value on outer boundary (u=0 everywhere)
NX, NY    = 8, 8          # Will be set per mesh level via environment
T_INIT    = 0.0           # Iteration-1 fallback interface temperature
Q_INIT    = 0.0           # Iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────────

# Parse NX, NY from environment (set by driver script)
import os
NX = int(os.environ.get('NX', '8'))
NY = int(os.environ.get('NY', '8'))

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)
ORDER = 1  # P1 elements

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)  # True: interface is right edge
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0  # Outward normal at interface = +e_x
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


def sample(imp, key, fallback, y):
    """Interpolate partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


imp = read_imports()

# Build mesh with named boundaries
geo = SplineGeometry()
# Edge order: bottom, right, top, left
# Right edge is interface, others are outer (Dirichlet u=0)
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=("outer", "interface", "outer", "outer"))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = H1(mesh, order=ORDER, dirichlet="outer|interface")
u, v = fes.TnT()

# Vertex -> dof map (order 1: one dof per vertex)
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[0] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

# Find interface vertices (sorted by y)
iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]
y_if = vxy[iface_v, 1]
iface_dofs = vdof[iface_v]

# EXCLUDE CORNER NODES from interface exchange (y=0 and y=1)
# These belong to outer boundary and are constrained by u=0
corner_mask = (np.abs(y_if - Y0) < TOL) | (np.abs(y_if - Y1) < TOL)
interior_mask = ~corner_mask
iface_v_interior = iface_v[interior_mask]
y_if_interior = y_if[interior_mask]
iface_dofs_interior = iface_dofs[interior_mask]

# Outer boundary dofs (left edge at x=0, plus top/bottom)
outer_dofs = vdof[np.where(np.abs(vxy[:, 0] - OUTER_X) < TOL)[0]]
top_bottom = np.where((np.abs(vxy[:, 1] - Y0) < TOL) | (np.abs(vxy[:, 1] - Y1) < TOL))[0]
outer_dofs = np.unique(np.concatenate([outer_dofs, vdof[top_bottom]]))

# Bilinear form: k * grad(u) . grad(v)
a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx

# Linear form with source term
f = LinearForm(fes)
gff = GridFunction(fes)
gff.vec[:] = 0.0
gff.vec.FV().NumPy()[vdof] = np.broadcast_to(
    np.asarray(F_SRC(vxy[:, 0], vxy[:, 1]), float), (mesh.nv,))
f += gff * v * dx

# Solution grid function
gfu = GridFunction(fes)
gfu.vec[:] = 0.0

# Set outer Dirichlet values (u=0) - includes corners
for d in outer_dofs:
    gfu.vec[int(d)] = T_OUTER

# Apply interface condition based on role (only interior points!)
if SIDE == "dirichlet":
    # Import temperature from partner (only interior points)
    T_if = sample(imp, "values", T_INIT, y_if_interior)
    for d, t in zip(iface_dofs_interior, T_if):
        gfu.vec[int(d)] = float(t)
else:
    # Neumann case (not used here but kept for completeness)
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if_interior)
    gfun = GridFunction(fes)
    gfun.vec[:] = 0.0
    for d, q in zip(iface_dofs_interior, q_if):
        gfun.vec[int(d)] = float(q)
    f += gfun * v * ds("interface")

# Solve
with TaskManager():
    a.Assemble()
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * res

    # Compute outward normal flux using consistent (reaction) flux method
    # q = -(k grad u) . n_out where n_out = +e_x at right interface
    # Only compute for INTERIOR interface points (exclude corners)
    if SIDE == "dirichlet":
        rvec = f.vec.CreateVector()
        rvec.data = a.mat * gfu.vec - f.vec  # r = A*u - b (unconstrained residual)
        
        fw = LinearForm(fes)
        fw += v * ds("interface")  # w_i = int_Gamma phi_i ds
        fw.Assemble()

        r_if = np.array([rvec[int(d)] for d in iface_dofs_interior], float)
        w_if = np.array([fw.vec[int(d)] for d in iface_dofs_interior], float)
        Q = np.zeros(len(iface_dofs_interior))
        ok = np.abs(w_if) > 1e-14
        Q[ok] = -r_if[ok] / w_if[ok]
        
        # Fill any remaining bad values with neighbors
        good = np.where(ok)[0]
        if len(good) < len(Q):
            for i in np.where(~ok)[0]:
                if len(good) > 0:
                    Q[i] = Q[good[np.argmin(np.abs(good - i))]]
    else:
        # Neumann side flux computation (not used here)
        fesq = H1(mesh, order=ORDER)
        p, w = fesq.TnT()
        m = BilinearForm(fesq)
        m += p * w * dx
        m.Assemble()
        fq = LinearForm(fesq)
        fq += (-K * S) * grad(gfu)[0] * w * dx
        fq.Assemble()
        qh = GridFunction(fesq)
        qh.vec.data = m.mat.Inverse(fesq.FreeDofs(), inverse="sparsecholesky") * fq.vec
        qdofs = np.array([fesq.GetDofNrs(NodeId(VERTEX, int(i)))[0] for i in iface_v_interior], int)
        Q = np.array([qh.vec[int(d)] for d in qdofs], float)

# Write exports.json LAST - only interior interface points
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_v_interior)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if_interior],
    "values": [float(gfu.vec[int(d)]) for d in iface_dofs_interior],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

# Write run log with NDOF
ndof = fes.ndof
with open("run.log", "w") as logfile:
    logfile.write(f"NDOF = {ndof}\n")
    logfile.write(f"Elements = {mesh.ne}\n")
    logfile.write(f"Vertices = {mesh.nv}\n")
    logfile.write(f"Interface points (interior only) = {len(iface_v_interior)}\n")

print(f"[NGSolve A] NDOF={ndof}, interface T=[{min(T_if):.6g},{max(T_if):.6g}], q_out=[{min(Q):.6g},{max(Q):.6g}]")
