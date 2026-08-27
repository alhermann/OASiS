"""4C participant for OASiS couple driver - Subdomain A (DIRICHLET side).

Subdomain A: (0, 0.625) x (0, 1), k=1
Interface at x=0.625 (right boundary of A)
This is the DIRICHLET side: imports temperature from partner, exports flux.

Flux convention: Export OUTWARD normal flux.
For subdomain A, outward at interface is +e_x, so qn_A = -K_A * du/dx
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# Problem parameters for Subdomain A
PARTNER = "kratos_B"
X0, X1 = 0.0, 0.625  # Subdomain A x-extent
Y0, Y1 = 0.0, 1.0    # y-extent
IFACE_X = 0.625      # Interface at right boundary
K = 1.0              # Thermal conductivity in A
T_OUTER = 0.0        # Dirichlet value on outer boundary (x=0)
T_INIT = 0.0         # Iteration-1 fallback interface temperature
FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"
FIT_DEG = 5          # Polynomial degree for interface profile fitting

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None

def sample(imp, key, fallback, ys):
    """Interpolate imported data at given y coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(ys), float(fallback))
    yy = np.array([c[1] for c in imp["coordinates"]], float)
    vv = np.asarray(imp.get(key, []), float).ravel()
    if vv.size != yy.size:
        return np.full(len(ys), float(fallback))
    o = np.argsort(yy)
    return np.interp(ys, yy[o], vv[o])

def funct_expr(ys, vals, deg):
    """Fit a polynomial to the interface values for 4C FUNCT expression."""
    if float(np.ptp(vals)) < 1e-14:
        return f"{float(vals[0]):.12e}"
    c = np.polyfit(ys, vals, int(min(deg, len(ys) - 1)))[::-1]
    terms = []
    for i, v in enumerate(c):
        if abs(v) < 1e-15:
            continue
        if i == 0:
            terms.append(f"({v:.12e})")
        elif i == 1:
            terms.append(f"({v:.12e})*y")
        else:
            terms.append(f"({v:.12e})*y^{i}")
    return " + ".join(terms) if terms else "0.0"

def get_mesh_params(level):
    """Get mesh divisions for each level."""
    base_divisions = [8, 16, 32][level - 1]
    nx = int(round(0.625 * base_divisions))
    ny = int(round(1.0 * base_divisions))
    return nx, ny

if len(sys.argv) > 1:
    level = int(sys.argv[1])
else:
    level = int(os.environ.get("LEVEL", 1))

nx, ny = get_mesh_params(level)

# Outward normal at interface for A is +e_x
S = 1.0

imp = read_imports()
ys = np.linspace(Y0, Y1, ny + 1)
iface_vals = sample(imp, "values", T_INIT, ys)
expr = funct_expr(ys, iface_vals, FIT_DEG)

print(f"[4C A] Importing T: min={iface_vals.min():.6f}, max={iface_vals.max():.6f}", file=sys.stderr)

# Build inline QUAD4 TRANSPORT mesh
nid, nodes, grid = 1, [], {}
for j in range(ny + 1):
    for i in range(nx + 1):
        x = X0 + i * (X1 - X0) / nx
        y = Y0 + j * (Y1 - Y0) / ny
        nodes.append(f"NODE {nid} COORD {x:.12f} {y:.12f} 0.0")
        grid[(i, j)] = nid
        nid += 1

elems = []
for j in range(ny):
    for i in range(nx):
        elems.append(f"{len(elems) + 1} TRANSP QUAD4 {grid[(i, j)]} "
                     f"{grid[(i + 1, j)]} {grid[(i + 1, j + 1)]} "
                     f"{grid[(i, j + 1)]} MAT 1 TYPE Std")

i_out = 0   # Outer boundary at x=X0
i_if = nx   # Interface at x=X1

# Source term in subdomain A (simplified for 4C parser)
F_SRC_EXPR = "-6*x^3*y + 3.2*x^3 - 4.22375*x^2*y + 12.6526666667*x^2 - 6*x*y^3 + 9.6*x*y^2 + 1.435*x*y - 9.2308333333*x - 1.4079166667*y^3 + 12.6526666667*y^2 - 11.24475*y"

deck = f"""TITLE:
  - "OASiS coupling participant 4C - Subdomain A (Dirichlet side)"
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
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{expr}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_OUTER}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{F_SRC_EXPR}"
NODE COORDS:
""" + "".join(f'  - "{n}"\n' for n in nodes)

deck += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e}"\n' for e in elems)
deck += "DLINE-NODE TOPOLOGY:\n"
deck += "".join(f'  - "NODE {grid[(i_out, j)]} DLINE 1"\n' for j in range(ny + 1))
deck += "".join(f'  - "NODE {grid[(i_if, j)]} DLINE 2"\n' for j in range(ny + 1))
deck += "DSURF-NODE TOPOLOGY:\n" + "".join(
    f'  - "NODE {n} DSURFACE 1"\n' for n in range(1, len(nodes) + 1))

Path("input.4C.yaml").write_text(deck)

env = dict(os.environ)
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")

r = subprocess.run([FOURC_BIN, "input.4C.yaml", "out"],
                   capture_output=True, text=True, env=env)

if r.returncode != 0:
    print(f"4C failed with rc={r.returncode}", file=sys.stderr)
    print(f"stdout: {r.stdout[-2000:]}", file=sys.stderr)
    print(f"stderr: {r.stderr[-2000:]}", file=sys.stderr)
    sys.exit(r.returncode)

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

mask = np.abs(pts[:, 0] - IFACE_X) < 1e-9
if not mask.any():
    sys.exit(f"no 4C nodes at x={IFACE_X}")

uy, inv = np.unique(np.round(pts[mask, 1], 10), return_inverse=True)
T = np.zeros(len(uy)); Q_raw = np.zeros(len(uy)); n = np.zeros(len(uy))
np.add.at(T, inv, phi[mask])
np.add.at(Q_raw, inv, flux[mask][:, 0])
np.add.at(n, inv, 1.0)
T /= n
Q_raw = Q_raw / n

# 4C flux_domain is -D grad(phi), which is the diffusive flux vector
# For outward normal +e_x at interface, qn = flux . n = flux_x
# So we just take the x-component (which is what we already have)
Q = S * Q_raw

print(f"[4C A DIRICHLET] interface n={len(uy)} T=[{T.min():.6g},{T.max():.6g}] q_out=[{Q.min():.6g},{Q.max():.6g}]")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(uy)),
    "coordinates": [[float(IFACE_X), float(y)] for y in uy],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

ndof = len(nodes)
with open("run.log", "w") as f:
    f.write(f"NDOF = {ndof}\n")
    f.write(f"Level = {level}\n")
    f.write(f"Elements = {len(elems)}\n")
