#!/usr/bin/env python3
"""Master script to run coupled DUNE-fem + 4C simulation across mesh levels."""

import json
import os
import sys
from pathlib import Path
import numpy as np
import subprocess

# Configuration
BASE_DIR = Path("/tmp/coupled_sim")
LEVELS = [8, 16, 32]  # Resolution values (h = 1/resolution)
MAX_ITER = 100
TOL = 1e-6
THETA = 0.5

# Probe point definitions
def generate_probe_points_A():
    """Generate 44x44 probe points for subdomain A: (0,1)x(0,0.625)"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) / 44
            y = (i_y + 0.5) * 0.625 / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate 44x44 probe points for subdomain B: (0,1)x(0.625,1.5)"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) / 44
            y = 0.625 + (i_y + 0.5) * 0.875 / 44
            points.append((x, y))
    return points

def generate_interface_probes():
    """Generate 44 interface probe points at y=5/8"""
    points = []
    for i in range(44):
        x = 0.25 + (i + 0.5) * 0.5 / 44  # Centered in (0,1)
        y = 5/8
        points.append((x, y))
    return points

PROBE_A = generate_probe_points_A()
PROBE_B = generate_probe_points_B()
INTERFACE_PROBES = generate_interface_probes()

def copy_participants(level, side):
    """Copy participant scripts to work directory."""
    work_dir = BASE_DIR / f"level{level}_A" if side == "A" else BASE_DIR / f"level{level}_B"
    src = BASE_DIR / f"participant_{side}.py"
    dst = work_dir / f"participant_{side}.py"
    dst.write_text(src.read_text())

def run_single_level(level):
    """Run coupled simulation for one mesh level using OASiS couple tool."""
    print(f"\n{'='*60}")
    print(f"Running level {level} (resolution={level})")
    print(f"{'='*60}")
    
    # Copy participant scripts
    copy_participants(level, "A")
    copy_participants(level, "B")
    
    # Set up work directories
    work_dir_A = str(BASE_DIR / f"level{level}_A")
    work_dir_B = str(BASE_DIR / f"level{level}_B")
    
    # Clean previous runs
    for d in [work_dir_A, work_dir_B]:
        for f in ["imports.json", "exports.json"]:
            p = Path(d) / f
            if p.exists():
                p.unlink()
    
    # Use OASiS couple tool
    from oasis_mcp import couple
    
    participants = json.dumps([
        {
            "name": "A",
            "command": ["/home/alexander/miniconda3/envs/dune-fem-env/bin/python", "participant_A.py"],
            "work_dir": work_dir_A,
            "imports_from": ["B"],
            "timeout": 900
        },
        {
            "name": "B", 
            "command": ["python3", "participant_B.py"],
            "work_dir": work_dir_B,
            "imports_from": ["A"],
            "timeout": 900
        }
    ])
    
    result = couple(
        participants=participants,
        max_iter=MAX_ITER,
        tol=TOL,
        accelerator="constant",
        theta=THETA,
        critic_approved=False
    )
    
    return result

def extract_solution_from_vtu(vtu_path, probe_points, field_name="phi_1"):
    """Extract solution values at probe points from VTU file."""
    import meshio
    mesh = meshio.read(str(vtu_path))
    pts = mesh.points[:, :2]
    vals = mesh.point_data.get(field_name)
    if vals is None:
        raise ValueError(f"Field {field_name} not found in {vtu_path}")
    
    # Interpolate at probe points
    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(pts, vals)
    results = interp(np.array(probe_points))
    return results

def process_level_results(level, couple_result):
    """Process results from one level and write output files."""
    print(f"\nProcessing results for level {level}")
    
    # Read exports from both sides
    exp_A = Path(f"/tmp/coupled_sim/level{level}_A/exports.json").read_text()
    exp_B = Path(f"/tmp/coupled_sim/level{level}_B/exports.json").read_text()
    data_A = json.loads(exp_A)
    data_B = json.loads(exp_B)
    
    # Write interface CSVs
    iface_csv_A = BASE_DIR / f"interface_level{level}_A.csv"
    with open(iface_csv_A, "w") as f:
        f.write("x,y,u,qn\n")
        for (x, y), u, qn in zip(data_A["coordinates"], data_A["values"], data_A["normal_fluxes"]):
            f.write(f"{x},{y},{u},{qn}\n")
    
    iface_csv_B = BASE_DIR / f"interface_level{level}_B.csv"
    with open(iface_csv_B, "w") as f:
        f.write("x,y,u,qn\n")
        for (x, y), u, qn in zip(data_B["coordinates"], data_B["values"], data_B["normal_fluxes"]):
            f.write(f"{x},{y},{u},{qn}\n")
    
    # Write residual history
    residual_csv = BASE_DIR / f"residual_level{level}.csv"
    history = couple_result.get("history", [])
    with open(residual_csv, "w") as f:
        f.write("iteration,interface_residual\n")
        for i, res in enumerate(history):
            f.write(f"{i+1},{res}\n")
    
    # Copy run logs
    log_A = Path(f"/tmp/coupled_sim/level{level}_A/run_log.txt")
    log_B = Path(f"/tmp/coupled_sim/level{level}_B/run_log.txt")
    if log_A.exists():
        (BASE_DIR / f"run_level{level}_A.log").write_text(log_A.read_text())
    if log_B.exists():
        (BASE_DIR / f"run_level{level}_B.log").write_text(log_B.read_text())
    
    return data_A, data_B, couple_result

if __name__ == "__main__":
    print("Starting coupled DUNE-fem + 4C simulation")
    print(f"Levels: {LEVELS}")
    print(f"Max iterations: {MAX_ITER}, Tolerance: {TOL}, Theta: {THETA}")
    
    all_results = {}
    
    for level in LEVELS:
        try:
            result = run_single_level(level)
            data_A, data_B, couple_result = process_level_results(level, result)
            all_results[level] = {
                "converged": result.get("converged", False),
                "iterations": result.get("iterations", 0),
                "final_residual": result.get("residual", float('inf'))
            }
            print(f"Level {level}: converged={result.get('converged')}, "
                  f"iterations={result.get('iterations')}, "
                  f"residual={result.get('residual')}")
        except Exception as e:
            print(f"ERROR at level {level}: {e}")
            import traceback
            traceback.print_exc()
            all_results[level] = {"error": str(e)}
    
    # Write RESULT.txt
    print("\nWriting RESULT.txt")
    files_written = []
    for level in LEVELS:
        files_written.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv",
            f"run_level{level}_A.log",
            f"run_level{level}_B.log"
        ])
    
    # Determine mesh independence (placeholder - needs actual comparison)
    finest = LEVELS[-1]
    second_finest = LEVELS[-2] if len(LEVELS) > 1 else None
    
    result_txt = f"""LEVELS = {len(LEVELS)}
FILES = {','.join(files_written)}
INTERFACE_RESIDUAL = {all_results[finest].get('final_residual', 'N/A')}
COUPLING_ITERATIONS = {all_results[finest].get('iterations', 'N/A')}
MESH_INDEPENDENCE = NOT_CONVERGED
MAX_REL_CHANGE = N/A
"""
    
    (BASE_DIR / "RESULT.txt").write_text(result_txt)
    print("Done!")
