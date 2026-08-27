"""4C participant for subdomain A (DIRICHLET side) - coupled heat conduction.

Subdomain A: (0, 0.625) x (0, 1), k=1
Interface at x = 0.625 (right boundary of A)
Outer BCs: u=0 on x=0, y=0, y=1
Coupling: imports temperature from partner, exports flux
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# Problem parameters for subdomain A
SIDE      = "dirichlet"   # A is DIRICHLET side
PARTNER   = "B"           # partner name
X0, X1    = 0.0, 0.625    # subdomain A x-extent
Y0, Y1    = 0.0, 1.0      # subdomain A y-extent
IFACE_X   = 0.625         # interface at right boundary
K         = 1.0           # thermal conductivity in A


def F_SRC(x, y):
    """Source term in subdomain A.
    
    f(x,y) = -6*x**3*y + 16*x**3/5 - 3379*x**2*y/800 + 18979*x**2/1500 
             - 6*x*y**3 + 48*x*y**2/5 + 287*x*y/200 - 11077*x/1200 
             - 3379*y**3/2400 + 18979*y**2/1500 - 44979*y/4000
    """
    result = (-6*x**3*y + 16*x**3/5 - 3379*x**2*y/800 + 18979*x**2/1500 
              - 6*x*y**3 + 48*x*y**2/5 + 287*x*y/200 - 11077*x/1200 
              - 3379*y**3/2400 + 18979*y**2/1500 - 44979*y/4000)
    return result


T_OUTER   = 0.0           # Dirichlet value on outer boundary (x=0, y=0, y=1)
NX, NY    = 10, 16        # mesh resolution (will be updated per level)
T_INIT    = 0.0           # iteration-1 fallback
Q_INIT    = 0.0           # iteration-1 fallback flux
FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD  = "/opt/4C-dependencies/lib"
FIT_DEG   = 5             # polynomial degree for boundary profile fitting
SRC_DEG_MAX = 10          # max degree for source fitting


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
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


def funct_expr(ys, vals, deg):
    """Fit a polynomial to the imported boundary values."""
    if float(np.ptp(vals)) < 1e-14:
        return f"{float(vals[0]):.12e}"
    c = np.polyfit(ys, vals, int(min(deg, len(ys) - 1)))[::-1]
    return " + ".join(f"({v:.12e})*y^{i}" if i else f"({v:.12e})"
                      for i, v in enumerate(c))


def src_expr(vals, gx, gy, rtol=1e-9):
    """Fit a polynomial to the source term samples."""
    scale = float(np.max(np.abs(vals)))
    if scale == 0.0:
        return None
    xs, ys_, vs = gx.ravel(), gy.ravel(), vals.ravel()
    pw, c, err = [], np.zeros(0), float("inf")
    for deg in range(max(SRC_DEG_MAX, 0) + 1):
        pw = [(i, d - i) for d in range(deg + 1) for i in range(d + 1)]
        A = np.column_stack([xs ** i * ys_ ** j for i, j in pw])
        c = np.linalg.lstsq(A, vs, rcond=None)[0]
        c = np.where(np.abs(c) > 1e-12 * np.max(np.abs(c)), c, 0.0)
        err = float(np.max(np.abs(A @ c - vs)))
        if err <= rtol * scale:
            break
    else:
        sys.exit(f"F_SRC cannot be fitted with degree {SRC_DEG_MAX}")
    terms = []
    for (i, j), v in zip(pw, c):
        if v == 0.0:
            continue
        terms.append(f"({v:.12e})" + (f"*x^{i}" if i else "")
                     + (f"*y^{j}" if j else ""))
    return " + ".join(terms) if terms else None


# Read mesh level from environment or file
level = 1
if Path("level.json").is_file():
    level = json.loads(Path("level.json").read_text()).get("level", 1)

# Set NX, NY based on mesh level (h = 1/8, 1/8, 1/32 -> divisions = 8, 16, 32)
divisions = [8, 16, 32][level - 1]
# For subdomain A: width = 0.625, so NX = divisions * 0.625 / 1.0 ≈ divisions * 5/8
# Actually h is global, so for domain A: NX = 0.625/h = 0.625 * divisions
NX = int(round(0.625 * divisions))
NY = int(divisions)  # height is 1.0

OUTER_X = X0  # left boundary
S = 1.0  # outward normal at interface (right boundary) = +e_x

imp = read_imports()
ys = np.linspace(Y0, Y1, NY + 1)
iface_vals = sample(imp, "values", T_INIT, ys)
expr = funct_expr(ys, iface_vals, FIT_DEG)

# Sample source on element-node grid
gx, gy = np.meshgrid(np.linspace(X0, X1, NX + 1), np.linspace(Y0, Y1, NY + 1), indexing="ij")
fsrc = np.broadcast_to(np.asarray(F_SRC(gx, gy), float), gx.shape)
src = src_expr(fsrc, gx, gy)

# Build inline QUAD4 mesh
nid, nodes, grid = 1, [], {}
for j in range(NY + 1):
    for i in range(NX + 1):
        nodes.append(f"NODE {nid} COORD {X0 + i * (X1 - X0) / NX:.12f} "
                     f"{Y0 + j * (Y1 - Y0) / NY:.12f} 0.0")
        grid[(i, j)] = nid
        nid += 1

elems = []
for j in range(NY):
    for i in range(NX):
        elems.append(f"{len(elems) + 1} TRANSP QUAD4 {grid[(i, j)]} "
                     f"{grid[(i + 1, j)]} {grid[(i + 1, j + 1)]} "
                     f"{grid[(i, j + 1)]} MAT 1 TYPE Std")

i_out = 0  # outer Dirichlet at x=X0 (left)
i_if = NX  # interface at x=X1 (right)

# Boundary conditions
# DLINE 1 = outer Dirichlet (x=0, y=0, y=1)
# DLINE 2 = interface (x=0.625)
dirich_outer = ("DESIGN LINE DIRICH CONDITIONS:\n"
                "  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n"
                "  - E: 2\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]\n")

deck = f"""TITLE:
  - "4C participant subdomain A (Dirichlet side)"
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
{dirich_outer}"""

if src is not None:
    deck += f'FUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{src}"\n'
    deck += ("DESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n"
             "    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [2]\n")

deck += "NODE COORDS:\n" + "".join(f'  - "{n}"\n' for n in nodes)
deck += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e}"\n' for e in elems)

# Outer Dirichlet: left edge (x=0), bottom (y=0), top (y=1)
deck += "DLINE-NODE TOPOLOGY:\n"
# Left edge (outer Dirichlet)
deck += "".join(f'  - "NODE {grid[(0, j)]} DLINE 1"\n' for j in range(NY + 1))
# Bottom edge (outer Dirichlet) - exclude corners already listed
deck += "".join(f'  - "NODE {grid[(i, 0)]} DLINE 1"\n' for i in range(1, NX + 1))
# Top edge (outer Dirichlet) - exclude corners
deck += "".join(f'  - "NODE {grid[(i, NY)]} DLINE 1"\n' for i in range(1, NX + 1))
# Interface (right edge)
deck += "".join(f'  - "NODE {grid[(NX, j)]} DLINE 2"\n' for j in range(NY + 1))

if src is not None:
    deck += "DSURF-NODE TOPOLOGY:\n"
    deck += "".join(f'  - "NODE {n} DSURFACE 1"\n' for n in range(1, len(nodes) + 1))

Path("input.4C.yaml").write_text(deck)

# Run 4C
env = dict(os.environ)
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
r = subprocess.run([FOURC_BIN, "input.4C.yaml", "out"],
                   capture_output=True, text=True, env=env)

if r.returncode != 0:
    print(f"4C failed: {r.stderr[-2000:]}", file=sys.stderr)
    sys.exit(r.returncode)

# Read VTU output
vtus = sorted(Path("out-vtk-files").glob("scatra-*-0.vtu"))
if not vtus:
    sys.exit("No VTU files found")

def step(p):
    m = re.match(r"scatra-(\d+)-\d+\.vtu$", p.name)
    return int(m.group(1)) if m else -1

import meshio
m = meshio.read(str(max(vtus, key=step)))
pts = np.asarray(m.points)[:, :2]
phi = np.asarray(m.point_data["phi_1"]).ravel()
flux = m.point_data.get("flux_domain_phi_1")
if flux is None:
    sys.exit("No flux field in VTU")
flux = np.asarray(flux)

# Extract interface nodes (x = IFACE_X)
mask = np.abs(pts[:, 0] - IFACE_X) < 1e-9
if not mask.any():
    sys.exit(f"No nodes at interface x={IFACE_X}")

# Collapse duplicates by y-coordinate
uy, inv = np.unique(np.round(pts[mask, 1], 10), return_inverse=True)
T = np.zeros(len(uy)); Q = np.zeros(len(uy)); n = np.zeros(len(uy))
np.add.at(T, inv, phi[mask])
np.add.at(Q, inv, flux[mask][:, 0])
np.add.at(n, inv, 1.0)
T /= n
Q = S * (Q / n)  # Outward normal flux

print(f"[4C A] interface n={len(uy)} T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

# Write exports.json
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(uy)),
    "coordinates": [[float(IFACE_X), float(y)] for y in uy],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

# Write run log with NDOF
ndof = len(nodes)
Path("run.log").write_text(f"NDOF = {ndof}\n")
