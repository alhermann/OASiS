#!/usr/bin/env python3
"""Generate all required output files from coupling results."""
import json
import numpy as np
from pathlib import Path

work_dir = Path("/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C9_27b_MCP_seed21/work")

# Coupling results
results = {
    1: {"iterations": 25, "residual": 5.78369862525503e-07, "history": [
        float('nan'), 1.0027532524960323, 0.7199873884342177, 0.54334515439677, 0.16024420303082013,
        0.12058220225396782, 0.04624014117459698, 0.03175146393965648, 0.016311011465217694,
        0.00797081623863181, 0.005458745508760703, 0.00210792144756017, 0.001714911769192096,
        0.0006679087896421225, 0.0004921933672364091, 0.000241754808195337, 0.00013050883078141569,
        8.581570184392555e-05, 3.43785175964814e-05, 2.794577597836598e-05, 1.0607514711759644e-05,
        8.236123223071772e-06, 3.855310049551071e-06, 2.2150567142753485e-06, 1.3963153025413207e-06,
        5.78369862525503e-07
    ]},
    2: {"iterations": 27, "residual": 4.3523105965908707e-07, "history": [
        float('nan'), 0.9257299533410493, 0.5820166607420535, 0.4161493319630871, 0.1541158830877259,
        0.10800440355397727, 0.04800073793345691, 0.03169591363482978, 0.017876891799778534,
        0.009040114254473385, 0.006368913368324831, 0.00273245584551085, 0.0021403725625989425,
        0.0009572618783710408, 0.0006624588364019057, 0.0003658424008171184, 0.00019321913531185613,
        0.0001362732633117819, 5.7865009271748987e-05, 4.6978561548946716e-05, 2.012966766356618e-05,
        1.4836434467793556e-05, 7.824016964276272e-06, 4.360737993342546e-06, 2.9868860958622035e-06,
        1.2849021553298086e-06, 1.0554211284896215e-06, 4.3523105965908707e-07
    ]},
    3: {"iterations": 27, "residual": 8.855053019340767e-07, "history": [
        float('nan'), 0.8932546795510392, 0.5352166698453009, 0.37028139395760806, 0.16273870597381573,
        0.10574530102999664, 0.053341681758187556, 0.03310890863970731, 0.02007185706793786,
        0.010356177279846177, 0.0072573585527570305, 0.003454024300635951, 0.0025041504592973793,
        0.0012845551510795477, 0.000811953775566305, 0.0005016547941360631, 0.00025610411691633035,
        0.00019071306698417454, 8.58172323781153e-05, 6.797574260015335e-05, 3.276900234680191e-05,
        2.2581392594056476e-05, 1.3370650763286156e-05, 7.16242166019724e-06, 5.310177111655311e-06,
        2.35299680521457e-06, 1.971072512381238e-06, 8.855053019340767e-07
    ]}
}

# Level 1 is in work/work, levels 2-3 are in work
for level in [1, 2, 3]:
    if level == 1:
        level_dir = work_dir / "work" / f"level{level}_A"
        level_dir_B = work_dir / "work" / f"level{level}_B"
    else:
        level_dir = work_dir / f"level{level}_A"
        level_dir_B = work_dir / f"level{level}_B"
    
    with open(level_dir / "exports.json") as f:
        exports_A = json.load(f)
    
    with open(level_dir_B / "exports.json") as f:
        exports_B = json.load(f)
    
    # Write interface CSV for side A
    with open(work_dir / f"interface_level{level}_A.csv", "w") as f:
        f.write("x,y,ux,uy,tx,ty\n")
        for coord, val, flux in zip(exports_A["coordinates"], exports_A["values"], exports_A["normal_fluxes"]):
            f.write(f"{coord[0]:.15e},{coord[1]:.15e},{val[0]:.15e},{val[1]:.15e},{flux[0]:.15e},{flux[1]:.15e}\n")
    
    # Write interface CSV for side B
    with open(work_dir / f"interface_level{level}_B.csv", "w") as f:
        f.write("x,y,ux,uy,tx,ty\n")
        for coord, val, flux in zip(exports_B["coordinates"], exports_B["values"], exports_B["normal_fluxes"]):
            f.write(f"{coord[0]:.15e},{coord[1]:.15e},{val[0]:.15e},{val[1]:.15e},{flux[0]:.15e},{flux[1]:.15e}\n")
    
    # Write residual history
    with open(work_dir / f"residual_level{level}.csv", "w") as f:
        f.write("iteration,interface_residual\n")
        for i, res in enumerate(results[level]["history"]):
            if np.isnan(res):
                f.write(f"{i},nan\n")
            else:
                f.write(f"{i},{res:.15e}\n")
    
    print(f"Level {level}: {len(exports_A['coordinates'])} interface nodes, {results[level]['iterations']} iterations, residual={results[level]['residual']:.6e}")

# Compute mesh independence from interface displacements
print("\nComputing mesh independence...")

def parse_interface_csv(path):
    data = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split(',')
            data.append({
                'x': float(parts[0]),
                'y': float(parts[1]),
                'ux': float(parts[2]),
                'uy': float(parts[3]),
                'tx': float(parts[4]),
                'ty': float(parts[5])
            })
    return data

level2_data = parse_interface_csv(work_dir / "interface_level2_A.csv")
level3_data = parse_interface_csv(work_dir / "interface_level3_A.csv")

max_rel_change = 0.0
for d2 in level2_data:
    best_d3 = min(level3_data, key=lambda d3: abs(d3['x'] - d2['x']))
    if abs(best_d3['x'] - d2['x']) < 0.01:
        mag2 = np.sqrt(d2['ux']**2 + d2['uy']**2)
        mag3 = np.sqrt(best_d3['ux']**2 + best_d3['uy']**2)
        if mag2 > 1e-15:
            rel_change = abs(mag3 - mag2) / mag2
            max_rel_change = max(max_rel_change, rel_change)

print(f"Max relative change between levels 2 and 3: {max_rel_change:.6e}")

mesh_independent = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
print(f"Mesh independence: {mesh_independent}")

# Write final RESULT.txt
files = []
for level in [1, 2, 3]:
    files.extend([
        f"interface_level{level}_A.csv",
        f"interface_level{level}_B.csv",
        f"residual_level{level}.csv"
    ])

with open(work_dir / "RESULT.txt", "w") as f:
    f.write("LEVELS = 3\n")
    f.write(f"FILES = {', '.join(files)}\n")
    f.write(f"INTERFACE_RESIDUAL = {results[3]['residual']:.15e}\n")
    f.write(f"COUPLING_ITERATIONS = {results[3]['iterations']}\n")
    f.write(f"MESH_INDEPENDENCE = {mesh_independent}\n")
    f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    f.write("\n")
    f.write("NOTE: Validation checks showed interface flux imbalance (5-11%).\n")
    f.write("Coupling converged but solution may not be physically correct.\n")
    f.write("Solution CSV files at probe points not generated - only interface data available.\n")

print("\nRESULT.txt written to", work_dir / "RESULT.txt")
