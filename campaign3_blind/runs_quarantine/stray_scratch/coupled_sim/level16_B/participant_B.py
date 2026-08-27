"""4C participant for subdomain B (Neumann side).

Problem: -div(k grad u) = f on (0,1)x(0.625,1.5)
k=2, no reaction term.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
import numpy as np

# Problem parameters
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.625, 1.5
IFACE_Y = 0.625
K = 2.0
PARTNER = "A"
FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"
FIT_DEG = 3

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None

def sample(imp, key, fallback, xs):
    if not imp or not imp.get("coordinates"):
        return np.full(len(xs), float(fallback))
    xx = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != xx.size:
        return np.full(len(xs), float(fallback))
    o = np.argsort(xx)
    return np.interp(xs, xx[o], vs[o])

def funct_expr(xs, vals, deg):
    if float(np.ptp(vals)) < 1e-14:
        return f"{float(vals[0]):.12e}"
    c = np.polyfit(xs, vals, int(min(deg, len(xs) - 1)))[::-1]
    return " + ".join(f"({v:.12e})*x^{i}" if i else f"({v:.12e})"
                      for i, v in enumerate(c))

resolution_str = os.environ.get('RESOLUTION', '8')
NX = int(resolution_str)
NY = max(1, int(NX * (Y1 - Y0) / (X1 - X0)))

T_INIT = 0.0
Q_INIT = 0.0

imp = read_imports()
xs = np.linspace(X0, X1, NX + 1)
iface_fluxes = sample(imp, "normal_fluxes", Q_INIT, xs)
flux_expr = funct_expr(xs, iface_fluxes, FIT_DEG)

nid, nodes, grid = 1, [], {}
for j in range(NY + 1):
    for i in range(NX + 1):
        nx = X0 + i * (X1 - X0) / NX
        ny = Y0 + j * (Y1 - Y0) / NY
        nodes.append(f"NODE {nid} COORD {nx:.12f} {ny:.12f} 0.0")
        grid[(i, j)] = nid
        nid += 1

elems = []
for j in range(NY):
    for i in range(NX):
        elems.append(f"{len(elems) + 1} TRANSP QUAD4 {grid[(i, j)]} "
                     f"{grid[(i + 1, j)]} {grid[(i + 1, j + 1)]} "
                     f"{grid[(i, j + 1)]} MAT 1 TYPE Std")

deck = f"""TITLE:
  - "OASiS coupling participant B"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  CALCFLUX_DOMAIN: "diffusive"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: "ascii"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {K}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{flux_expr}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x*x*x*y/2 - 25*x*x*x/24 - 4*x*x*y/5 + 31*x*x/15 + x*y*y*y/2 - 25*x*y*y/8 + 1397*x*y/640 + 1913*x/1280 - 4*y*y*y/15 + 31*y*y/15 - 65*y/48 - 55/32"
NODE COORDS:
"""
deck += "".join(f'  - "{n}"\n' for n in nodes)
deck += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e}"\n' for e in elems)
deck += "DLINE-NODE TOPOLOGY:\n"
deck += "".join(f'  - "NODE {grid[(0, j)]} DLINE 1"\n' for j in range(NY + 1))
deck += "".join(f'  - "NODE {grid[(NX, j)]} DLINE 1"\n' for j in range(NY + 1))
deck += "".join(f'  - "NODE {grid[(i, NY)]} DLINE 1"\n' for i in range(NX + 1))
deck += "".join(f'  - "NODE {grid[(i, 0)]} DLINE 2"\n' for i in range(NX + 1))
deck += "DSURF-NODE TOPOLOGY:\n"
deck += "".join(f'  - "NODE {n} DSURFACE 3"\n' for n in range(1, len(nodes) + 1))

Path("input.4C.yaml").write_text(deck)

env = dict(os.environ)
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
r = subprocess.run([FOURC_BIN, "input.4C.yaml", "out"],
                   capture_output=True, text=True, env=env)

if r.returncode != 0:
    print(f"4C failed with rc={r.returncode}")
    print(f"stdout: {r.stdout[-2000:]}")
    print(f"stderr: {r.stderr[-2000:]}")
    sys.exit(1)

vtus = sorted(Path("out-vtk-files").glob("scatra-*-0.vtu"))
if not vtus:
    sys.exit("4C produced no VTU files")

def step(p):
    m = re.match(r"scatra-(\d+)-\d+\.vtu$", p.name)
    return int(m.group(1)) if m else -1

import meshio
m = meshio.read(str(max(vtus, key=step)))
pts = np.asarray(m.points)[:, :2]
phi = np.asarray(m.point_data["phi_1"]).ravel()
flux = m.point_data.get("flux_domain_phi_1")
if flux is None:
    sys.exit("no flux field in the 4C VTU")
flux = np.asarray(flux)

mask = np.abs(pts[:, 1] - IFACE_Y) < 1e-9
if not mask.any():
    sys.exit(f"no 4C nodes at y={IFACE_Y}")

ux, inv = np.unique(np.round(pts[mask, 0], 10), return_inverse=True)
T = np.zeros(len(ux)); Q = np.zeros(len(ux)); n = np.zeros(len(ux))
np.add.at(T, inv, phi[mask])
np.add.at(Q, inv, flux[mask][:, 1])
np.add.at(n, inv, 1.0)
T /= n
Q = -Q  # Outward normal (down for subdomain B)

print(f"[4C B] interface n={len(ux)} T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

Path("exports.json").write_text(json.dumps({
    "field_name": "u",
    "n_points": int(len(ux)),
    "coordinates": [[float(x), float(IFACE_Y)] for x in ux],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

ndof = len(nodes)
with open("run_log.txt", "w") as f:
    f.write(f"NDOF = {ndof}\n")
    f.write(f"NX = {NX}, NY = {NY}\n")
    f.write(f"Interface points = {len(ux)}\n")
