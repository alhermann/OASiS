#!/usr/bin/env python3
"""Post-process coupled simulation results."""

import json
import os
from pathlib import Path
import numpy as np
import meshio
from scipy.interpolate import LinearNDInterpolator

BASE_DIR = Path("/tmp/coupled_sim")

def generate_probe_points_A():
    """44x44 probe points for subdomain A: (0,1)x(0,0.625)"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) / 44
            y = (i_y + 0.5) * 0.625 / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    """44x44 probe points for subdomain B: (0,1)x(0.625,1.5)"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) / 44
            y = 0.625 + (i_y + 0.5) * 0.875 / 44
            points.append((x, y))
    return points

def generate_interface_probes():
    """44 interface probe points at y=5/8"""
    points = []
    for i in range(44):
        x = 0.25 + (i + 0.5) * 0.5 / 44
        y = 5/8
        points.append((x, y))
    return points

PROBE_A = generate_probe_points_A()
PROBE_B = generate_probe_points_B()
INTERFACE_PROBES = generate_interface_probes()

def extract_at_probes(vtu_path, probe_points, field_name="u"):
    """Extract solution values at probe points from VTU file."""
    mesh = meshio.read(str(vtu_path))
    pts = mesh.points[:, :2]
    
    # Find field name
    if field_name not in mesh.point_data:
        # Try common alternatives
        for fn in ["phi_1", "temperature", "solution"]:
            if fn in mesh.point_data:
                field_name = fn
                break
    
    vals = mesh.point_data.get(field_name)
    if vals is None:
        raise ValueError(f"Field {field_name} not found in {vtu_path}")
    
    interp = LinearNDInterpolator(pts, vals)
    results = interp(np.array(probe_points))
    return results

def process_level(level):
    """Process results for one mesh level."""
    print(f"Processing level {level}")
    
    # Read exports
    exp_A = json.loads(Path(f"/tmp/coupled_sim/level{level}_A/exports.json").read_text())
    exp_B = json.loads(Path(f"/tmp/coupled_sim/level{level}_B/exports.json").read_text())
    
    # Write interface CSVs
    with open(BASE_DIR / f"interface_level{level}_A.csv", "w") as f:
        f.write("x,y,u,qn\n")
        for (x, y), u, qn in zip(exp_A["coordinates"], exp_A["values"], exp_A["normal_fluxes"]):
            f.write(f"{x},{y},{u},{qn}\n")
    
    with open(BASE_DIR / f"interface_level{level}_B.csv", "w") as f:
        f.write("x,y,u,qn\n")
        for (x, y), u, qn in zip(exp_B["coordinates"], exp_B["values"], exp_B["normal_fluxes"]):
            f.write(f"{x},{y},{u},{qn}\n")
    
    # Extract solution at probe points from VTU files
    vtu_A = list(Path(f"/tmp/coupled_sim/level{level}_A").glob("result-*.vtu"))
    vtu_B = list(Path(f"/tmp/coupled_sim/level{level}_B/out-vtk-files").glob("scatra-*-0.vtu"))
    
    if vtu_A:
        # Get last VTU file
        vtu_A = max(vtu_A, key=lambda p: int(p.stem.split("-")[1]) if "-" in p.stem else 0)
        try:
            sol_A = extract_at_probes(vtu_A, PROBE_A, "u")
            with open(BASE_DIR / f"solution_level{level}_A.csv", "w") as f:
                f.write("x,y,u\n")
                for (x, y), u in zip(PROBE_A, sol_A):
                    f.write(f"{x},{y},{u}\n")
        except Exception as e:
            print(f"Warning: Could not extract solution A: {e}")
    
    if vtu_B:
        def step(p):
            m = __import__("re").match(r"scatra-(\d+)-\d+\.vtu$", p.name)
            return int(m.group(1)) if m else -1
        vtu_B = max(vtu_B, key=step)
        try:
            sol_B = extract_at_probes(vtu_B, PROBE_B, "phi_1")
            with open(BASE_DIR / f"solution_level{level}_B.csv", "w") as f:
                f.write("x,y,u\n")
                for (x, y), u in zip(PROBE_B, sol_B):
                    f.write(f"{x},{y},{u}\n")
        except Exception as e:
            print(f"Warning: Could not extract solution B: {e}")
    
    # Copy run logs
    log_A = Path(f"/tmp/coupled_sim/level{level}_A/run_log.txt")
    log_B = Path(f"/tmp/coupled_sim/level{level}_B/run_log.txt")
    if log_A.exists():
        (BASE_DIR / f"run_level{level}_A.log").write_text(log_A.read_text())
    if log_B.exists():
        (BASE_DIR / f"run_level{level}_B.log").write_text(log_B.read_text())

if __name__ == "__main__":
    for level in [8, 16, 32]:
        process_level(level)
    print("Post-processing complete!")
