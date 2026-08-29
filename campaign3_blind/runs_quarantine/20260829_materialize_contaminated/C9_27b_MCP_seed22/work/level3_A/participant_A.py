"""NGSolve participant for subdomain A (NEUMANN side) - coupled elasticity.

Also saves the full solution field for post-processing.
"""
import json
from pathlib import Path
import sys

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, CF, GridFunction, InnerProduct,
                     LinearForm, Mesh, NodeId, TaskManager, VectorH1, ds, dx,
                     grad, SetMeshSize)

# Get level from directory name
work_dir = Path.cwd().name
level = int(work_dir.replace("level", "").replace("_A", "")) if "_A" in work_dir else 1

# ── PROBLEM SPECIFIC PARAMETERS ─
SIDE      = "neumann"   # A is the NEUMANN side
PARTNER   = "B"
X0, X1    = 0.0, 1.0
Y0, Y1    = 0.0, 0.625
IFACE_Y   = 0.625
LAMBDA    = 480.0
MU        = 1200.0

def B_SRC(x, y):
    fx = (-63*x**4*y/3125 + 99*x**4/5000 + 168*x**3*y/15625 - 33*x**3/3125 
          + 216*x**2*y**2/625 - 1557*x**2*y/3125 - 99*x**2/5000 
          - 288*x*y**2/3125 + 1992*x*y/15625 + 33*x/3125 
          - 36*y**2/625 + 54*y/625)
    fy = (-108*x**5/15625 + 72*x**4/15625 - 18*x**3*y**2/625 + 153*x**3*y/1250 
          - 279*x**3/3125 + 36*x**2*y**2/3125 - 153*x**2*y/3125 + 486*x**2/15625 
          + 9*x*y**2/625 - 153*x*y/2500 + 63*x/1250 
          - 12*y**2/3125 + 51*y/3125 - 42/3125)
    return fx, fy

h = 1.0 / (8 * level)
NX = int(1.0 / h)
NY = int(0.625 / h)
UI_X, UI_Y = 0.0, 0.0
TI_X, TI_Y = 0.0, 0.0

MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)
ORDER = 1
ON_TOP = abs(IFACE_Y - Y1) < abs(IFACE_Y - Y0)
S = 1.0 if ON_TOP else -1.0
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

def sample(imp, key, fallback, x):
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(x), 1))
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != xs.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(x), 1))
    o = np.argsort(xs)
    return np.column_stack([np.interp(x, xs[o], vs[o, c])
                            for c in range(vs.shape[1])])

imp = read_imports()

geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("outer", "outer", "interface", "outer") if ON_TOP else
                      ("outer", "outer", "outer", "interface")))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = VectorH1(mesh, order=ORDER,
               dirichlet=("outer|interface" if SIDE == "dirichlet" else "outer"))
u, v = fes.TnT()

vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[:2] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, 1] - IFACE_Y) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 0])]
x_if = vxy[iface_v, 0]

outer_v = np.where((np.abs(vxy[:, 0] - X0) < TOL) |
                   (np.abs(vxy[:, 0] - X1) < TOL) |
                   (np.abs(vxy[:, 1] - Y0) < TOL))[0]

corner = (np.abs(x_if - X0) < TOL) | (np.abs(x_if - X1) < TOL)

def eps_of(g):
    exx, eyy = g[0, 0], g[1, 1]
    return exx, eyy, 0.5 * (g[0, 1] + g[1, 0])

a = BilinearForm(fes)
gu, gv = grad(u), grad(v)
eu = eps_of(gu)
ev = eps_of(gv)
a += (2.0 * MU * (eu[0] * ev[0] + eu[1] * ev[1] + 2.0 * eu[2] * ev[2])
      + LAMBDA * (eu[0] + eu[1]) * (ev[0] + ev[1])) * dx

f = LinearForm(fes)
gfb = GridFunction(fes)
gfb.vec[:] = 0.0
bx, by = B_SRC(vxy[:, 0], vxy[:, 1])
bvals = gfb.vec.FV().NumPy()
bvals[vdof[:, 0]] = np.broadcast_to(np.asarray(bx, float), (mesh.nv,))
bvals[vdof[:, 1]] = np.broadcast_to(np.asarray(by, float), (mesh.nv,))
f += InnerProduct(gfb, v) * dx

f_vol = LinearForm(fes)
f_vol += InnerProduct(gfb, v) * dx

gfu = GridFunction(fes)
gfu.vec[:] = 0.0

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), x_if)
    for k, vtx in enumerate(iface_v):
        if corner[k]:
            continue
        gfu.vec[int(vdof[vtx, 0])] = float(u_if[k, 0])
        gfu.vec[int(vdof[vtx, 1])] = float(u_if[k, 1])
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), x_if)
    gfun = GridFunction(fes)
    gfun.vec[:] = 0.0
    for k, vtx in enumerate(iface_v):
        gfun.vec[int(vdof[vtx, 0])] = float(t_if[k, 0])
        gfun.vec[int(vdof[vtx, 1])] = float(t_if[k, 1])
    f += InnerProduct(gfun, v) * ds("interface")

for k, vtx in enumerate(outer_v):
    gfu.vec[int(vdof[vtx, 0])] = 0.0
    gfu.vec[int(vdof[vtx, 1])] = 0.0

with TaskManager():
    a.Assemble()
    f.Assemble()
    f_vol.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * res

    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec
    fw = LinearForm(fes)
    fw += InnerProduct(CF((1.0, 1.0)), v) * ds("interface")
    fw.Assemble()

    Q = np.zeros((len(iface_v), 2))
    ok = np.ones((len(iface_v), 2), bool)
    for k, vtx in enumerate(iface_v):
        for c in (0, 1):
            d = int(vdof[vtx, c])
            wi = float(fw.vec[d])
            if abs(wi) > 1e-14:
                Q[k, c] = -float(rvec[d]) / wi
            else:
                ok[k, c] = False

    suspect = np.isin(iface_v, outer_v) | ~ok.all(axis=1)
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Save full solution field as JSON for post-processing
solution_data = {
    "vertices": [[float(vxy[i, 0]), float(vxy[i, 1])] for i in range(mesh.nv)],
    "displacement": [[float(gfu.vec[int(vdof[i, 0])]), float(gfu.vec[int(vdof[i, 1])])] 
                     for i in range(mesh.nv)]
}
Path("solution_field.json").write_text(json.dumps(solution_data, indent=2))

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(xx), float(IFACE_Y)] for xx in x_if],
    "values": [[float(gfu.vec[int(vdof[i, 0])]), float(gfu.vec[int(vdof[i, 1])])]
               for i in iface_v],
    "normal_fluxes": [[float(q0), float(q1)] for q0, q1 in Q],
}, indent=2))

ndof = len(gfu.vec)
log_file = f"run_level{level}_A.log"
with open(log_file, "w") as f:
    f.write(f"NDOF = {ndof}\n")
    f.write(f"Interface nodes: {len(iface_v)}\n")
    f.write(f"Side: {SIDE}\n")
    f.write(f"Level: {level}\n")
    f.write(f"NX, NY: {NX}, {NY}\n")
