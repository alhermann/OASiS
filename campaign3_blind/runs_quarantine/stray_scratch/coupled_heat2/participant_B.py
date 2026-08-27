#!/usr/bin/env python3
"""deal.II participant for subdomain B (Neumann side) - transient heat with coupling"""
import json
import subprocess
import sys
from pathlib import Path
import os

# Parameters from command line or defaults
NX = int(sys.argv[1]) if len(sys.argv) > 1 else 8
DT = float(sys.argv[2]) if len(sys.argv) > 2 else 1/32

# Problem setup
X0, X1 = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K = 4.0
F_SRC = 0.0
T_OUTER = 0.0
T_END = 0.25
PARTNER = "A"

NY = int(round(NX * (Y1-Y0)/(X1-X0)))

DEALII_EXE = "/tmp/coupling_build/build/heat_iface_dealii_transient"
DEGREE = 1

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get(PARTNER)
    except:
        return None

def sample_interface(imp, key, fallback):
    """Get partner's samples as sorted (y, value) pairs"""
    if imp and imp.get("coordinates"):
        ys = [float(c[1]) for c in imp["coordinates"]]
        vs = [float(v) for v in (imp.get(key) or [])]
        if len(vs) == len(ys) and len(ys) > 0:
            return sorted(zip(ys, vs))
    return [(float(Y0), float(fallback)), (float(Y1), float(fallback))]

# Read imported flux data
imp = read_imports()
side_flag = 1  # Neumann side
pairs = sample_interface(imp, "normal_fluxes", 0.0)

# Write input file for deal.II solver
header = f"{side_flag} {K!r} {X0!r} {X1!r} {Y0!r} {Y1!r} {IFACE_X!r} {T_OUTER!r} {F_SRC!r} {NX} {NY} {DEGREE} {DT!r} {T_END!r} {len(pairs)}"
lines = [header, str(len(pairs))]
lines += [f"{y:.16g} {v:.16g}" for y, v in pairs]

Path("dealii_input.txt").write_text("\n".join(lines) + "\n")

# Run deal.II solver
out_txt = Path("dealii_output.txt")
if out_txt.exists():
    out_txt.unlink()

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/opt/4C-dependencies/lib"

result = subprocess.run([DEALII_EXE, "dealii_input.txt", "dealii_output.txt"],
                       capture_output=True, text=True, env=env)

if result.returncode != 0 or not out_txt.is_file():
    sys.stderr.write(f"deal.II solver failed (rc={result.returncode})\n")
    sys.stderr.write(f"stdout: {result.stdout[-2000:]}\n")
    sys.stderr.write(f"stderr: {result.stderr[-2000:]}\n")
    sys.exit(1)

# Parse output
coords, temps, fluxes = [], [], []
for line in out_txt.read_text().splitlines():
    tok = line.split()
    if len(tok) == 3:
        coords.append([float(IFACE_X), float(tok[0])])
        temps.append(float(tok[1]))
        fluxes.append(float(tok[2]))

if not coords:
    sys.stderr.write("deal.II solver produced no interface points\n")
    sys.exit(1)

# Write exports.json LAST
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": len(coords),
    "coordinates": coords,
    "values": temps,
    "normal_fluxes": fluxes,
}, indent=2))

# Estimate NDOF for Q1 elements on rectangular mesh
ndof = (NX + 1) * (NY + 1)
Path("run.log").write_text(f"NDOF = {ndof}\n")

print(f"[B] Exported {len(temps)} interface points, T=[{min(temps):.4g},{max(temps):.4g}]", flush=True)
