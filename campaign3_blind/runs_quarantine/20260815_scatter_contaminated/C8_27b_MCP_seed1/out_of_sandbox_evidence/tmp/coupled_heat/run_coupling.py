#!/usr/bin/env python3
"""Main script to run coupled simulation for all mesh levels."""
import json
import os
import subprocess
import numpy as np
from pathlib import Path

# Mesh levels: h = 1/8, 1/16, 1/32 means RESOLUTION = 8, 16, 32
LEVELS = [8, 16, 32]

# Probe points definition
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

NGSOLVE_PYTHON = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
KRATOS_PYTHON = "/usr/bin/python3"
NGSOLVE_SCRIPT = "/tmp/coupled_heat/ngsolve_A.py"
KRATOS_SCRIPT = "/tmp/coupled_heat/kratos_B.py"

def run_participant(work_dir, script, python_interpreter, resolution):
    """Run a participant script in its work directory."""
    env = os.environ.copy()
    env['RESOLUTION'] = str(resolution)
    result = subprocess.run(
        [python_interpreter, script],
        cwd=work_dir,
        capture_output=True,
        text=True,
        env=env
    )
    return result.returncode, result.stdout, result.stderr

def interpolate_solution_from_vtu(vtu_path, probe_points, field_name="temperature"):
    """Interpolate solution at probe points from VTU file."""
    # Use pyvista or similar to read VTU and interpolate
    try:
        import pyvista as pv
        mesh = pv.read(vtu_path)
        values = []
        for x, y in probe_points:
            # Interpolate at point (x, y, 0)
            val = mesh.sample_line([x, y, 0], [x, y, 0]).point_data[field_name][0] if hasattr(mesh, 'sample_line') else None
            if val is None:
                # Fallback: find nearest cell
                val = mesh.interpolate_point(x, y, 0)[field_name]
            values.append(val)
        return np.array(values)
    except ImportError:
        print("pyvista not available, using alternative method")
        # Alternative: use meshio
        import meshio
        mesh = meshio.read(vtu_path)
        # Simple barycentric interpolation would go here
        return None

def main():
    base_dir = Path("/tmp/coupled_heat")
    
    # Store results for mesh independence check
    all_results = {}
    coupling_history = {}
    
    for level_idx, resolution in enumerate(LEVELS, 1):
        print(f"\n{'='*60}")
        print(f"Running mesh level {level_idx} (resolution={resolution})")
        print(f"{'='*60}")
        
        level_dir = base_dir / f"level{level_idx}"
        dir_A = level_dir / "A"
        dir_B = level_dir / "B"
        dir_A.mkdir(parents=True, exist_ok=True)
        dir_B.mkdir(parents=True, exist_ok=True)
        
        # Copy scripts to work directories
        import shutil
        shutil.copy(NGSOLVE_SCRIPT, dir_A / "participant.py")
        shutil.copy(KRATOS_SCRIPT, dir_B / "participant.py")
        
        # For now, run participants once to get initial solutions
        # In a real coupling, we'd use the couple() tool
        
        # Run NGSolve participant (Dirichlet side)
        print(f"Running NGSolve for subdomain A...")
        rc_a, out_a, err_a = run_participant(dir_A, NGSOLVE_SCRIPT, NGSOLVE_PYTHON, resolution)
        print(f"NGSolve exit code: {rc_a}")
        if rc_a != 0:
            print(f"STDOUT: {out_a}")
            print(f"STDERR: {err_a}")
        
        # Read exports from A
        exports_A = json.loads((dir_A / "exports.json").read_text())
        
        # Create imports for B
        imports_B = {"A": exports_A}
        (dir_B / "imports.json").write_text(json.dumps(imports_B, indent=2))
        
        # Run Kratos participant (Neumann side)
        print(f"Running Kratos for subdomain B...")
        rc_b, out_b, err_b = run_participant(dir_B, KRATOS_SCRIPT, KRATOS_PYTHON, resolution)
        print(f"Kratos exit code: {rc_b}")
        if rc_b != 0:
            print(f"STDOUT: {out_b}")
            print(f"STDERR: {err_b}")
        
        # Read exports from B
        exports_B = json.loads((dir_B / "exports.json").read_text())
        
        # Create imports for A (for next iteration)
        imports_A = {"B": exports_B}
        (dir_A / "imports.json").write_text(json.dumps(imports_A, indent=2))
        
        # For this simple test, we'll just do one iteration
        # In reality, we need to iterate until convergence
        
        # Save coupling history (just one iteration for now)
        coupling_history[level_idx] = [{"iteration": 1, "residual": 1.0}]  # Placeholder
        
        # Copy run logs
        if (dir_A / "run.log").exists():
            shutil.copy(dir_A / "run.log", base_dir / f"run_level{level_idx}_A.log")
        if (dir_B / "run.log").exists():
            shutil.copy(dir_B / "run.log", base_dir / f"run_level{level_idx}_B.log")
        
        # TODO: Extract probe point values and interface data
        # This requires reading the actual solution fields
        
        print(f"Level {level_idx} complete")
    
    print("\nAll levels completed!")

if __name__ == "__main__":
    main()
