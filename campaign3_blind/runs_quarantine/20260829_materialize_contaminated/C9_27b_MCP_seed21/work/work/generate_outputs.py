#!/usr/bin/env python3
"""Generate required output files from coupling results."""
import json
import numpy as np
from pathlib import Path

# Read exports from level 1 coupling
work_dir = Path("/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C9_27b_MCP_seed21/work/work")

# Read exports.json from both participants
with open(work_dir / "level1_A" / "exports.json") as f:
    exports_A = json.load(f)
with open(work_dir / "level1_B" / "exports.json") as f:
    exports_B = json.load(f)

# Probe points for subdomain A: 44x44 grid in (0,1) x (0, 0.625)
# x = (i_x+0.5)/44, y = (i_y+0.5)*0.625/44 for i_x, i_y = 0..43
# Ordered with last index varying fastest
probe_points_A = []
for i_x in range(44):
    for i_y in range(44):
        x = (i_x + 0.5) / 44
        y = (i_y + 0.5) * 0.625 / 44
        probe_points_A.append((x, y))

# Probe points for subdomain B: 44x44 grid in (0,1) x (0.625, 1.5)
# x = (i_x+0.5)/44, y = 0.625 + (i_y+0.5)*0.875/44
probe_points_B = []
for i_x in range(44):
    for i_y in range(44):
        x = (i_x + 0.5) / 44
        y = 0.625 + (i_y + 0.5) * 0.875 / 44
        probe_points_B.append((x, y))

# Interface probe points: 44 points at y=5/8
# x = 1/4 + (i+0.5)*1/2/44 for i = 0..43
interface_points = []
for i in range(44):
    x = 1/4 + (i + 0.5) * (1/2) / 44
    interface_points.append((x, 5/8))

# For now, we only have interface data from the coupling
# We need to interpolate the solution to probe points
# Since we don't have the full field, we'll use the interface data we have

# Write interface CSV for level 1, side A
# Format: x, y, ux, uy, tx, ty
with open(work_dir / "interface_level1_A.csv", "w") as f:
    f.write("x,y,ux,uy,tx,ty\n")
    for i, (coord, val, flux) in enumerate(zip(exports_A["coordinates"], 
                                                exports_A["values"],
                                                exports_A["normal_fluxes"])):
        f.write(f"{coord[0]:.15e},{coord[1]:.15e},{val[0]:.15e},{val[1]:.15e},{flux[0]:.15e},{flux[1]:.15e}\n")

# Write interface CSV for level 1, side B
with open(work_dir / "interface_level1_B.csv", "w") as f:
    f.write("x,y,ux,uy,tx,ty\n")
    for i, (coord, val, flux) in enumerate(zip(exports_B["coordinates"], 
                                                exports_B["values"],
                                                exports_B["normal_fluxes"])):
        f.write(f"{coord[0]:.15e},{coord[1]:.15e},{val[0]:.15e},{val[1]:.15e},{flux[0]:.15e},{flux[1]:.15e}\n")

# Write residual history for level 1
# From the coupling result: history array
history = [
    float('nan'),
    1.0027532524960323, 0.7199873884342177, 0.54334515439677, 0.16024420303082013,
    0.12058220225396782, 0.04624014117459698, 0.03175146393965648, 0.016311011465217694,
    0.00797081623863181, 0.005458745508760703, 0.00210792144756017, 0.001714911769192096,
    0.0006679087896421225, 0.0004921933672364091, 0.000241754808195337, 0.00013050883078141569,
    8.581570184392555e-05, 3.43785175964814e-05, 2.794577597836598e-05, 1.0607514711759644e-05,
    8.236123223071772e-06, 3.855310049551071e-06, 2.2150567142753485e-06, 1.3963153025413207e-06,
    5.78369862525503e-07
]

with open(work_dir / "residual_level1.csv", "w") as f:
    f.write("iteration,interface_residual\n")
    for i, res in enumerate(history):
        if np.isnan(res):
            f.write(f"{i},nan\n")
        else:
            f.write(f"{i},{res:.15e}\n")

print("Generated interface_level1_A.csv, interface_level1_B.csv, residual_level1.csv")
print(f"Interface points: {len(exports_A['coordinates'])} nodes")
print(f"Final residual: {history[-1]:.6e}")
print(f"Coupling iterations: {len(history)-1}")
