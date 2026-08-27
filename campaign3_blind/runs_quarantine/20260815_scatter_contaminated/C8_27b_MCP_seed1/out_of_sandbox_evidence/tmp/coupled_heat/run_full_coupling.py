#!/usr/bin/env python3
"""
Full coupled simulation runner.
This script runs the coupled heat conduction problem for all mesh levels.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np

BASE_DIR = Path("/tmp/coupled_heat")
NGSOLVE_PYTHON = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
KRATOS_PYTHON = "/usr/bin/python3"
NGSOLVE_SCRIPT = BASE_DIR / "ngsolve_A.py"
KRATOS_SCRIPT = BASE_DIR / "kratos_B.py"

# Mesh levels: h = 1/8, 1/16, 1/32
LEVELS = [8, 16, 32]

def generate_probe_points_A():
    """Generate 1936 probe points for subdomain A."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate 1936 probe points for subdomain B."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return points

def generate_interface_points():
    """Generate 44 interface probe points at x = 5/8."""
    points = []
    for i in range(44):
        x = 5/8
        y = 1/4 + (i + 0.5) * 1/2 / 44
        points.append((x, y))
    return points

PROBE_A = generate_probe_points_A()
PROBE_B = generate_probe_points_B()
INTERFACE_POINTS = generate_interface_points()

def prepare_level(level_idx, resolution):
    """Prepare directories and copy scripts for one mesh level."""
    level_dir = BASE_DIR / f"level{level_idx}"
    dir_A = level_dir / "A"
    dir_B = level_dir / "B"
    
    # Create directories
    dir_A.mkdir(parents=True, exist_ok=True)
    dir_B.mkdir(parents=True, exist_ok=True)
    
    # Copy scripts
    import shutil
    shutil.copy(NGSOLVE_SCRIPT, dir_A / "participant.py")
    shutil.copy(KRATOS_SCRIPT, dir_B / "participant.py")
    
    # Remove any stale files
    for d in [dir_A, dir_B]:
        for f in ["imports.json", "exports.json"]:
            fp = d / f
            if fp.exists():
                fp.unlink()
    
    print(f"Prepared level {level_idx} (resolution={resolution})")
    return dir_A, dir_B


def read_solution_dat(dat_path):
    """Read solution data from .dat file."""
    nodes = []
    temps = []
    with open(dat_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                nodes.append((float(parts[0]), float(parts[1])))
                temps.append(float(parts[2]))
    return np.array(nodes), np.array(temps)


def interpolate_nearest_neighbor(nodes, temps, probe_points):
    """Interpolate using nearest neighbor."""
    values = []
    for px, py in probe_points:
        # Find nearest node
        dists = np.sum((nodes - [px, py])**2, axis=1)
        nearest = np.argmin(dists)
        values.append(temps[nearest])
    return np.array(values)


def extract_interface_data(exports_json_path, interface_points):
    """Extract interface temperature and flux at specified points."""
    with open(exports_json_path) as f:
        data = json.load(f)
    
    coords = np.array(data["coordinates"])
    values = np.array(data["values"])
    fluxes = np.array(data.get("normal_fluxes", [0]*len(values)))
    
    results = []
    for ix, iy in interface_points:
        # Find closest point in exports
        dists = np.sum((coords - [ix, iy])**2, axis=1)
        idx = np.argmin(dists)
        results.append({
            'x': ix, 'y': iy, 
            'u': float(values[idx]), 
            'qn': float(fluxes[idx])
        })
    
    return results


def write_solution_csv(filename, probe_points, values):
    """Write solution values at probe points to CSV."""
    with open(filename, 'w') as f:
        f.write("x, y, u\n")
        for (x, y), v in zip(probe_points, values):
            f.write(f"{x}, {y}, {v}\n")


def write_interface_csv(filename, interface_data):
    """Write interface data to CSV."""
    with open(filename, 'w') as f:
        f.write("x, y, u, qn\n")
        for d in interface_data:
            f.write(f"{d['x']}, {d['y']}, {d['u']}, {d['qn']}\n")


def write_residual_csv(filename, history):
    """Write coupling residual history to CSV."""
    with open(filename, 'w') as f:
        f.write("iteration, interface_residual\n")
        for entry in history:
            f.write(f"{entry['iteration']}, {entry['residual']}\n")


def run_single_iteration(dir_A, dir_B, resolution):
    """Run one iteration of both participants."""
    import subprocess
    
    env_A = os.environ.copy()
    env_A['RESOLUTION'] = str(resolution)
    
    env_B = os.environ.copy()
    env_B['RESOLUTION'] = str(resolution)
    
    # Run NGSolve
    result_A = subprocess.run(
        [NGSOLVE_PYTHON, str(NGSOLVE_SCRIPT)],
        cwd=str(dir_A), capture_output=True, text=True, env=env_A
    )
    
    if result_A.returncode != 0:
        print(f"NGSolve failed: {result_A.stderr[:500]}")
        return False
    
    # Read exports from A and create imports for B
    with open(dir_A / "exports.json") as f:
        exports_A = json.load(f)
    
    with open(dir_B / "imports.json", 'w') as f:
        json.dump({"A": exports_A}, f)
    
    # Run Kratos
    result_B = subprocess.run(
        [KRATOS_PYTHON, str(KRATOS_SCRIPT)],
        cwd=str(dir_B), capture_output=True, text=True, env=env_B
    )
    
    if result_B.returncode != 0:
        print(f"Kratos failed: {result_B.stderr[:500]}")
        return False
    
    # Read exports from B and create imports for A
    with open(dir_B / "exports.json") as f:
        exports_B = json.load(f)
    
    with open(dir_A / "imports.json", 'w') as f:
        json.dump({"B": exports_B}, f)
    
    return True


def compute_residual(exports_A, exports_B):
    """Compute relative interface residual."""
    vals_A = np.array(exports_A["values"])
    vals_B = np.array(exports_B["values"])
    
    # Relative difference in temperatures
    diff = np.abs(vals_A - vals_B)
    ref = np.maximum(np.abs(vals_A), np.abs(vals_B))
    ref = np.where(ref < 1e-10, 1.0, ref)
    
    return float(np.max(diff / ref))


def main():
    print("="*60)
    print("Starting coupled heat conduction simulation")
    print("="*60)
    
    all_files = []
    coupling_results = {}
    solution_data = {}
    
    for level_idx, resolution in enumerate(LEVELS, 1):
        print(f"\n{'='*60}")
        print(f"Mesh Level {level_idx} (resolution={resolution})")
        print(f"{'='*60}")
        
        # Prepare directories
        dir_A, dir_B = prepare_level(level_idx, resolution)
        
        # Run coupling iterations
        max_iter = 100
        tol = 1e-6
        history = []
        
        converged = False
        for iteration in range(1, max_iter + 1):
            success = run_single_iteration(dir_A, dir_B, resolution)
            
            if not success:
                print(f"Iteration {iteration}: FAILED")
                break
            
            # Read exports
            with open(dir_A / "exports.json") as f:
                exports_A = json.load(f)
            with open(dir_B / "exports.json") as f:
                exports_B = json.load(f)
            
            # Compute residual
            if iteration > 1:
                residual = compute_residual(exports_A, exports_B)
            else:
                residual = 1.0  # First iteration has no previous to compare
            
            history.append({"iteration": iteration, "residual": residual})
            print(f"Iteration {iteration}: residual = {residual:.6e}")
            
            if residual < tol:
                converged = True
                print(f"Converged at iteration {iteration}")
                break
        
        coupling_results[level_idx] = {
            "converged": converged,
            "iterations": len(history),
            "final_residual": history[-1]["residual"] if history else None,
            "history": history
        }
        
        # Extract final solutions
        with open(dir_A / "exports.json") as f:
            exports_A = json.load(f)
        with open(dir_B / "exports.json") as f:
            exports_B = json.load(f)
        
        # Read solution data from .dat files
        dat_A = dir_A / "solution.dat"
        dat_B = dir_B / "solution.dat"
        
        if dat_A.exists():
            nodes_A, temps_A = read_solution_dat(dat_A)
            sol_A = interpolate_nearest_neighbor(nodes_A, temps_A, PROBE_A)
            solution_data[(level_idx, 'A')] = sol_A
        else:
            print(f"Warning: {dat_A} not found")
            solution_data[(level_idx, 'A')] = None
        
        if dat_B.exists():
            nodes_B, temps_B = read_solution_dat(dat_B)
            sol_B = interpolate_nearest_neighbor(nodes_B, temps_B, PROBE_B)
            solution_data[(level_idx, 'B')] = sol_B
        else:
            print(f"Warning: {dat_B} not found")
            solution_data[(level_idx, 'B')] = None
        
        # Write output files
        # Solution CSVs
        if sol_A is not None:
            sol_file_A = BASE_DIR / f"solution_level{level_idx}_A.csv"
            write_solution_csv(sol_file_A, PROBE_A, sol_A)
            all_files.append(f"solution_level{level_idx}_A.csv")
        
        if sol_B is not None:
            sol_file_B = BASE_DIR / f"solution_level{level_idx}_B.csv"
            write_solution_csv(sol_file_B, PROBE_B, sol_B)
            all_files.append(f"solution_level{level_idx}_B.csv")
        
        # Interface CSVs
        iface_data_A = extract_interface_data(dir_A / "exports.json", INTERFACE_POINTS)
        iface_file_A = BASE_DIR / f"interface_level{level_idx}_A.csv"
        write_interface_csv(iface_file_A, iface_data_A)
        all_files.append(f"interface_level{level_idx}_A.csv")
        
        iface_data_B = extract_interface_data(dir_B / "exports.json", INTERFACE_POINTS)
        iface_file_B = BASE_DIR / f"interface_level{level_idx}_B.csv"
        write_interface_csv(iface_file_B, iface_data_B)
        all_files.append(f"interface_level{level_idx}_B.csv")
        
        # Residual CSV
        res_file = BASE_DIR / f"residual_level{level_idx}.csv"
        write_residual_csv(res_file, history)
        all_files.append(f"residual_level{level_idx}.csv")
        
        # Copy run logs
        import shutil
        if (dir_A / "run.log").exists():
            shutil.copy(dir_A / "run.log", BASE_DIR / f"run_level{level_idx}_A.log")
            all_files.append(f"run_level{level_idx}_A.log")
        if (dir_B / "run.log").exists():
            shutil.copy(dir_B / "run.log", BASE_DIR / f"run_level{level_idx}_B.log")
            all_files.append(f"run_level{level_idx}_B.log")
        
        print(f"Level {level_idx} complete")
    
    # Check mesh independence
    print("\n" + "="*60)
    print("Checking mesh independence...")
    print("="*60)
    
    rel_change_A = None
    rel_change_B = None
    
    # Compare finest two levels
    if solution_data.get((3, 'A')) is not None and solution_data.get((2, 'A')) is not None:
        sol_A_3 = solution_data[(3, 'A')]
        sol_A_2 = solution_data[(2, 'A')]
        # Only consider points where level 2 has non-zero values
        mask = np.abs(sol_A_2) > 1e-10
        if mask.sum() > 0:
            rel_change_A = np.max(np.abs(sol_A_3[mask] - sol_A_2[mask]) / np.abs(sol_A_2[mask]))
        print(f"Subdomain A max relative change (level 2->3): {rel_change_A:.6e}")
    
    if solution_data.get((3, 'B')) is not None and solution_data.get((2, 'B')) is not None:
        sol_B_3 = solution_data[(3, 'B')]
        sol_B_2 = solution_data[(2, 'B')]
        mask = np.abs(sol_B_2) > 1e-10
        if mask.sum() > 0:
            rel_change_B = np.max(np.abs(sol_B_3[mask] - sol_B_2[mask]) / np.abs(sol_B_2[mask]))
        print(f"Subdomain B max relative change (level 2->3): {rel_change_B:.6e}")
    
    changes = [c for c in [rel_change_A, rel_change_B] if c is not None]
    max_rel_change = max(changes) if changes else None
    mesh_independent = max_rel_change < 0.01 if max_rel_change is not None else False
    
    # Write RESULT.txt
    final_residual = coupling_results[3]["final_residual"] if 3 in coupling_results else None
    coupling_iterations = coupling_results[3]["iterations"] if 3 in coupling_results else None
    
    with open(BASE_DIR / "RESULT.txt", 'w') as f:
        f.write(f"LEVELS = {len(LEVELS)}\n")
        f.write(f"FILES = {', '.join(all_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residual}\n")
        f.write(f"COUPLING_ITERATIONS = {coupling_iterations}\n")
        f.write(f"MESH_INDEPENDENCE = {'CONVERGED' if mesh_independent else 'NOT_CONVERGED'}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change}\n")
    
    print(f"\nResults written to {BASE_DIR / 'RESULT.txt'}")
    print("="*60)


if __name__ == "__main__":
    main()
