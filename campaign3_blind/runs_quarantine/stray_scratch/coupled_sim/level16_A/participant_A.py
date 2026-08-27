"""DUNE-fem participant for subdomain A (Dirichlet side).

Problem: -div(k grad u) + c*u = f on (0,1)x(0,0.625)
k=1, c=10, Dirichlet BCs on outer boundary and interface.
"""
import json
from pathlib import Path
import numpy as np
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.fem import assemble
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 conditional, dot, ds, dx, grad, lt)

# Problem parameters
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.0, 0.625
IFACE_Y = 0.625
K = 1.0
C = 10.0
PARTNER = "B"

import os
resolution_str = os.environ.get('RESOLUTION', '8')
NX = int(resolution_str)
NY = max(1, int(NX * Y1 / X1))

T_OUTER = 0.0
T_INIT = 0.0
Q_INIT = 0.0
EPS = 1e-8

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None

def sample(imp, key, fallback, xs):
    if not imp or not imp.get("coordinates"):
        return np.full(len(xs), float(fallback))
    xx = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != xx.size:
        return np.full(len(xs), float(fallback))
    o = np.argsort(xx)
    return np.interp(xs, xx[o], vs[o])

imp = read_imports()

gridView = structuredGrid([X0, Y0], [X1, Y1], [NX, NY])
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)

xd = np.array(space.interpolate(x[0], name="xcoord").as_numpy)
yd = np.array(space.interpolate(x[1], name="ycoord").as_numpy)

iface_dofs = np.where(np.abs(yd - IFACE_Y) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xd[iface_dofs])]
x_if = xd[iface_dofs]

# Source term
f_src_expr = (-10*x[0]**3*x[1]**3/3 + 325*x[0]**3*x[1]**2/24 - 69*x[0]**3*x[1]/8 - 65*x[0]**3/24 
              + 16*x[0]**2*x[1]**3/3 - 77*x[0]**2*x[1]**2/3 + 361*x[0]**2*x[1]/20 + 77*x[0]**2/15 
              + 4*x[0]*x[1]**2 - 61*x[0]*x[1]/20 - 97*x[0]/40 - 16*x[1]**3/15 + 77*x[1]**2/15 - 17*x[1]/4)

u, v = TrialFunction(space), TestFunction(space)
a = K * dot(grad(u), grad(v)) * dx + C * u * v * dx
b = f_src_expr * v * dx

bc_outer = (conditional(lt(x[0] + EPS, X0 + EPS), 1, 0) +
            conditional(lt(X1 - EPS, x[0] + EPS), 1, 0) +
            conditional(lt(x[1] + EPS, Y0 + EPS), 1, 0))
bcs = [DirichletBC(space, T_OUTER, bc_outer)]

gfun = space.interpolate(0, name="iface_data")
gdofs = gfun.as_numpy
gdofs[:] = 0.0
gdofs[iface_dofs] = sample(imp, "values", T_INIT, x_if)
bc_iface = conditional(lt(IFACE_Y - EPS, x[1] + EPS), 
                       conditional(lt(x[1] + EPS, IFACE_Y + EPS), 1, 0), 0)
bcs.append(DirichletBC(space, gfun, bc_iface))

scheme = galerkin([a == b] + bcs, solver="cg")
uh = space.interpolate(0, name="solution")
scheme.solve(target=uh)

# Flux projection
p, w = TrialFunction(space), TestFunction(space)
proj_form = p * w * dx == -K * grad(uh)[1] * w * dx
proj_scheme = galerkin([proj_form], solver="cg")
qh = space.interpolate(0, name="normal_flux")
proj_scheme.solve(target=qh)

Q = np.array(qh.as_numpy)[iface_dofs]
T_dofs = np.array(uh.as_numpy)

Path("exports.json").write_text(json.dumps({
    "field_name": "u",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(xx), float(IFACE_Y)] for xx in x_if],
    "values": [float(t) for t in T_dofs[iface_dofs]],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

ndof = len(T_dofs)
with open("run_log.txt", "w") as f:
    f.write(f"NDOF = {ndof}\n")
    f.write(f"NX = {NX}, NY = {NY}\n")
    f.write(f"Interface points = {len(iface_dofs)}\n")

print(f"[DUNE-fem A] DOFs={ndof}, interface_points={len(iface_dofs)}")
print(f"[DUNE-fem A] u_range=[{T_dofs.min():.6g},{T_dofs.max():.6g}]")
print(f"[DUNE-fem A] flux_range=[{Q.min():.6g},{Q.max():.6g}]")

# Write VTU for post-processing
gridView.writeVTK("result", pointdata={"u": uh})
