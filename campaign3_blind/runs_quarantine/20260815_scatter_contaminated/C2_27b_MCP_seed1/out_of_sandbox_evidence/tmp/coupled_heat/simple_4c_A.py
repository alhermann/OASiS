#!/usr/bin/env python3
"""Simple 4C participant for testing - Subdomain A (Dirichlet side)."""
import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np

PARTNER = "kratos_B"
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K = 1.0
T_OUTER = 0.0
T_INIT = 0.0
FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"

# Get level from environment or default to 1
level = int(os.environ.get("LEVEL", 1))
base_divisions = [8, 16, 32][level - 1]
nx, ny = int(round(0.625 * base_divisions)), int(round(1.0 * base_divisions))

print(f"[4C A Level {level}] Mesh: {nx}x{ny}", file=sys.stderr)

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get(PARTNER)
    except:
        return None

def sample(imp, key, fallback, ys):
    if not imp or not imp.get("coordinates"):
        return np.full(len(ys), float(fallback))
    yy = np.array([c[1] for c in imp["coordinates"]], float)
    vv = np.asarray(imp.get(key, []), float).ravel()
    if vv.size != yy.size:
        return np.full(len(ys), float(fallback))
    o = np.argsort(yy)
    return np.interp(ys, yy[o], vv[o])

def funct_expr(ys, vals, deg=3):
    """Fit polynomial for 4C boundary condition."""
    if len(vals) < 2 or float(np.ptp(vals)) < 1e-14:
        return f"{float(vals[0] if len(vals) > 0 else 0):.12e}"
    deg = min(deg, len(ys) - 1)
    c = np.polyfit(ys, vals, deg)[::-1]
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

# Read imports
imp = read_imports()
ys = np.linspace(Y0, Y1, ny + 1)
iface_vals = sample(imp, "values", T_INIT, ys)
expr = funct_expr(ys, iface_vals)

print(f"[4C A] Imported T: min={iface_vals.min():.6f}, max={iface_vals.max():.6f}", file=sys.stderr)

# Build mesh
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
        elems.append(f"{len(elems) + 1} TRANSP QUAD4 {grid[(i, j)]} {grid[(i + 1, j)]} {grid[(i + 1, j + 1)]} {grid[(i, j + 1)]} MAT 1 TYPE Std")

# Source term (simplified)
F_SRC = "-6*x^3*y + 3.2*x^3 - 4.22375*x^2*y + 12.6526666667*x^2 - 6*x*y^3 + 9.6*x*y^2 + 1.435*x*y - 9.2308333333*x - 1.4079166667*y^3 + 12.6526666667*y^2 - 11.24475*y"

deck = f"""TITLE:
  - "4C Subdomain A Level {level}"
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
SOLVER 1:
  SOLVER: "UMFPACK"
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
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{F_SRC}"
NODE COORDS:
""" + "".join(f'  - "{n}"\n' for n in nodes)

deck += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e}"\n' for e in elems)
deck += "DLINE-NODE TOPOLOGY:\n"
deck += "".join(f'  - "NODE {grid[(0, j)]} DLINE 1"\n' for j in range(ny + 1))
deck += "".join(f'  - "NODE {grid[(nx, j)]} DLINE 2"\n' for j in range(ny + 1))
deck += "DSURF-NODE TOPOLOGY:\n" + "".join(f'  - "NODE {n} DSURFACE 1"\n' for n in range(1, len(nodes) + 1))

Path("input.4C.yaml").write_text(deck)

env = dict(os.environ)
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")

r = subprocess.run([FOURC_BIN, "input.4C.yaml", "out"], capture_output=True, text=True, env=env)
if r.returncode != 0:
    print(f"4C failed: {r.stderr[:500]}", file=sys.stderr)
    sys.exit(r.returncode)

vtus = sorted(Path("out-vtk-files").glob("scatra-*-0.vtu"))
if not vtus:
    sys.exit("No VTU files")

def step(p):
    m = re.match(r"scatra-(\d+)-\d+\.vtu$", p.name)
    return int(m.group(1)) if m else -1

import meshio
m = meshio.read(str(max(vtus, key=step)))
pts = np.asarray(m.points)[:, :2]
phi = np.asarray(m.point_data["phi_1"]).ravel()
flux = np.asarray(m.point_data.get("flux_domain_phi_1", np.zeros((len(phi), 2))))

mask = np.abs(pts[:, 0] - IFACE_X) < 1e-9
uy, inv = np.unique(np.round(pts[mask, 1], 10), return_inverse=True)
T = np.zeros(len(uy)); Q_raw = np.zeros(len(uy)); n = np.zeros(len(uy))
np.add.at(T, inv, phi[mask])
np.add.at(Q_raw, inv, flux[mask][:, 0])
np.add.at(n, inv, 1.0)
T /= n
Q = Q_raw / n  # Outward normal is +e_x for A

print(f"[4C A] Exporting: T=[{T.min():.6g},{T.max():.6g}], q_out=[{Q.min():.6g},{Q.max():.6g}]", file=sys.stderr)

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(uy)),
    "coordinates": [[float(IFACE_X), float(y)] for y in uy],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

with open("run.log", "w") as f:
    f.write(f"NDOF = {len(nodes)}\n")
    f.write(f"Level = {level}\n")
