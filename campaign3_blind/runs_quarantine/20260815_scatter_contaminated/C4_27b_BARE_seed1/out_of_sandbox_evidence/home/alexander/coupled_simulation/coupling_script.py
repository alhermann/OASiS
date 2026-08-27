#!/usr/bin/env python3
"""
Coupling script for Dirichlet-Neumann iteration between FEniCSx (subdomain A) and deal.II (subdomain B).

Dirichlet-Neumann scheme:
- Subdomain A (DIRICHLET side): receives interface field from B, imposes as Dirichlet BC, returns flux
- Subdomain B (NEUMANN side): receives flux from A, applies as Neumann BC, returns interface field

Convergence criterion: relative interface mismatch < 1e-6
"""

import os
import sys
import numpy as np
import subprocess
import shutil

# Paths to executables
FENICS_PYTHON = "/home/alexander/miniconda3/envs/fenics/bin/python"
DEAL_II_BINARY = None  # Will be set after compilation

def find_deal_ii_binary():
    """Find the compiled deal.II binary."""
    possible_paths = [
        "/home/alexander/coupled_simulation/dealii_solver",
        "./dealii_solver",
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return None

def compile_deal_ii():
    """Compile the deal.II solver."""
    print("Compiling deal.II solver...")
    
    build_dir = "/home/alexander/coupled_simulation/build"
    os.makedirs(build_dir, exist_ok=True)
    
    cmake_cmd = f"""
cd {build_dir} &&
cmake -DCMAKE_BUILD_TYPE=Release \
      -DDEAL_II_DIR=/home/alexander/dealii/build \
      /home/alexander/coupled_simulation/dealii_solver.cpp
"""
    
    result = subprocess.run(cmake_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"CMake failed:\n{result.stdout}\n{result.stderr}")
        return False
    
    make_cmd = f"cd {build_dir} && make -j4"
    result = subprocess.run(make_cmd, shell=True, capture_output=True, text=True, 
                          env={**os.environ, "LD_LIBRARY_PATH": "/opt/4C-dependencies/lib"})
    if result.returncode != 0:
        print(f"Make failed:\n{result.stdout}\n{result.stderr}")
        return False
    
    global DEAL_II_BINARY
    DEAL_II_BINARY = os.path.join(build_dir, "dealii_solver.cpp")
    if not os.path.exists(DEAL_II_BINARY):
        # Try alternative name
        DEAL_II_BINARY = os.path.join(build_dir, "dealii_solver")
    
    if os.path.exists(DEAL_II_BINARY):
        print(f"Compiled deal.II solver: {DEAL_II_BINARY}")
        return True
    else:
        print("Could not find compiled binary")
        return False

def run_fenicsx_solver(level, dt, t_end, interface_u_file, flux_output_file, solution_output_file, log_file):
    """Run FEniCSx solver for subdomain A."""
    cmd = f"{FENICS_PYTHON} /home/alexander/coupled_simulation/fenicsx_solver.py " \
          f"{level} {dt} {t_end} {interface_u_file} {flux_output_file} {solution_output_file} {log_file}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FEniCSx solver failed:\n{result.stdout}\n{result.stderr}")
        return False
    return True

def run_dealii_solver(level, dt, t_end, interface_flux_file, flux_output_file, solution_output_file, log_file):
    """Run deal.II solver for subdomain B."""
    if DEAL_II_BINARY is None or not os.path.exists(DEAL_II_BINARY):
        print("deal.II binary not found!")
        return False
    
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/opt/4C-dependencies/lib:/home/alexander/dealii/build/lib:" + env.get("LD_LIBRARY_PATH", "")
    
    cmd = f"{DEAL_II_BINARY} {level} {dt} {t_end} {interface_flux_file} {flux_output_file} {solution_output_file} {log_file}"
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        print(f"deal.II solver failed:\n{result.stdout}\n{result.stderr}")
        return False
    return True

def read_interface_data(filename):
    """Read interface data from CSV file."""
    u_values = []
    y_coords = []
    with open(filename, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                y_coords.append(float(parts[1]))
                u_values.append(float(parts[2]))
    return y_coords, u_values

def write_interface_u(filename, u_values):
    """Write interface u values to CSV file for the other solver to read."""
    # Interface probe points: x = 5/8, y = 1/4 + (i+0.5)*1/2/44
    with open(filename, 'w') as f:
        f.write("x, y, u\n")
        for i, u_val in enumerate(u_values):
            x = 5/8
            y = 0.25 + (i + 0.5) * 0.5 / 44
            f.write(f"{x:.16e}, {y:.16e}, {u_val:.16e}\n")

def write_interface_flux(filename, qn_values):
    """Write interface flux values to CSV file for the other solver to read."""
    with open(filename, 'w') as f:
        f.write("x, y, qn\n")
        for i, qn_val in enumerate(qn_values):
            x = 5/8
            y = 0.25 + (i + 0.5) * 0.5 / 44
            f.write(f"{x:.16e}, {y:.16e}, {qn_val:.16e}\n")

def compute_residual(u_A, u_B):
    """Compute relative interface residual."""
    u_A_arr = np.array(u_A)
    u_B_arr = np.array(u_B)
    
    diff = np.abs(u_A_arr - u_B_arr)
    mean_u = np.mean(np.abs(u_A_arr) + np.abs(u_B_arr))
    
    if mean_u < 1e-12:
        return np.max(diff)
    else:
        return np.max(diff) / mean_u

def solve_level(level, h, dt, t_end, output_dir):
    """Solve for a single mesh level using Dirichlet-Neumann iteration."""
    
    max_iterations = 100
    tolerance = 1e-6
    
    # Files for this level
    level_prefix = f"level{level}"
    
    # Temporary files for interface data exchange
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    
    interface_u_from_B = os.path.join(temp_dir, f"interface_u_from_B.csv")
    interface_flux_from_A = os.path.join(temp_dir, f"interface_flux_from_A.csv")
    
    # Output files
    solution_A_file = os.path.join(output_dir, f"solution_{level_prefix}_A.csv")
    solution_B_file = os.path.join(output_dir, f"solution_{level_prefix}_B.csv")
    interface_A_file = os.path.join(output_dir, f"interface_{level_prefix}_A.csv")
    interface_B_file = os.path.join(output_dir, f"interface_{level_prefix}_B.csv")
    residual_file = os.path.join(output_dir, f"residual_{level_prefix}.csv")
    log_A_file = os.path.join(output_dir, f"run_{level_prefix}_A.log")
    log_B_file = os.path.join(output_dir, f"run_{level_prefix}_B.log")
    
    # Initialize interface guess (zero)
    u_interface = [0.0] * 44
    
    residuals = []
    
    print(f"\n=== Solving level {level}, h={h}, dt={dt} ===")
    
    for iteration in range(max_iterations):
        # Step 1: Write interface u from B to file (for A to use as Dirichlet BC)
        write_interface_u(interface_u_from_B, u_interface)
        
        # Step 2: Run FEniCSx solver for subdomain A (DIRICHLET side)
        # A receives u from B, solves, returns flux
        temp_solution_A = os.path.join(temp_dir, f"solution_A_iter{iteration}.csv")
        temp_interface_A = os.path.join(temp_dir, f"interface_A_iter{iteration}.csv")
        
        success = run_fenicsx_solver(level, dt, t_end, interface_u_from_B, 
                                     temp_interface_A, temp_solution_A, log_A_file)
        if not success:
            print(f"FEniCSx solver failed at iteration {iteration}")
            return None
        
        # Read flux from A's interface output
        _, qn_A = read_interface_data(temp_interface_A)
        
        # Step 3: Write flux from A to file (for B to use as Neumann BC)
        write_interface_flux(interface_flux_from_A, qn_A)
        
        # Step 4: Run deal.II solver for subdomain B (NEUMANN side)
        # B receives flux from A, solves, returns u
        temp_solution_B = os.path.join(temp_dir, f"solution_B_iter{iteration}.csv")
        temp_interface_B = os.path.join(temp_dir, f"interface_B_iter{iteration}.csv")
        
        success = run_dealii_solver(level, dt, t_end, interface_flux_from_A,
                                    temp_interface_B, temp_solution_B, log_B_file)
        if not success:
            print(f"deal.II solver failed at iteration {iteration}")
            return None
        
        # Read new interface u from B
        _, u_new = read_interface_data(temp_interface_B)
        
        # Compute residual
        residual = compute_residual(u_interface, u_new)
        residuals.append((iteration + 1, residual))
        
        print(f"Iteration {iteration + 1}: residual = {residual:.6e}")
        
        # Update interface
        u_interface = u_new
        
        # Check convergence
        if residual < tolerance:
            print(f"Converged at iteration {iteration + 1} with residual {residual:.6e}")
            break
    
    # Final run to get proper output files
    write_interface_u(interface_u_from_B, u_interface)
    success = run_fenicsx_solver(level, dt, t_end, interface_u_from_B, 
                                 interface_A_file, solution_A_file, log_A_file)
    
    _, qn_A_final = read_interface_data(interface_A_file)
    write_interface_flux(interface_flux_from_A, qn_A_final)
    
    success = run_dealii_solver(level, dt, t_end, interface_flux_from_A,
                                interface_B_file, solution_B_file, log_B_file)
    
    # Write residual history
    with open(residual_file, 'w') as f:
        f.write("iteration, interface_residual\n")
        for iter_num, res in residuals:
            f.write(f"{iter_num}, {res:.16e}\n")
    
    final_residual = residuals[-1][1] if residuals else 1.0
    n_iterations = len(residuals)
    
    print(f"Level {level} complete: {n_iterations} iterations, final residual = {final_residual:.6e}")
    
    return {
        'final_residual': final_residual,
        'n_iterations': n_iterations,
        'solution_A': solution_A_file,
        'solution_B': solution_B_file,
        'interface_A': interface_A_file,
        'interface_B': interface_B_file,
        'residual': residual_file,
        'log_A': log_A_file,
        'log_B': log_B_file
    }

def read_solution_csv(filename):
    """Read solution values from CSV file."""
    values = []
    with open(filename, 'r') as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                values.append(float(parts[2]))
    return np.array(values)

def check_mesh_independence(results):
    """Check mesh independence by comparing solutions at different levels."""
    if len(results) < 2:
        return False, 0.0
    
    # Compare finest two levels
    level2_results = results[-2]
    level3_results = results[-1]
    
    # Read solutions
    sol_A_l2 = read_solution_csv(level2_results['solution_A'])
    sol_A_l3 = read_solution_csv(level3_results['solution_A'])
    sol_B_l2 = read_solution_csv(level2_results['solution_B'])
    sol_B_l3 = read_solution_csv(level3_results['solution_B'])
    
    # Compute relative changes
    rel_change_A = np.max(np.abs(sol_A_l3 - sol_A_l2) / (np.abs(sol_A_l2) + 1e-12))
    rel_change_B = np.max(np.abs(sol_B_l3 - sol_B_l2) / (np.abs(sol_B_l2) + 1e-12))
    
    max_rel_change = max(rel_change_A, rel_change_B)
    
    # Consider converged if relative change is small (< 1%)
    converged = max_rel_change < 0.01
    
    return converged, max_rel_change

def main():
    output_dir = "/home/alexander/coupled_simulation/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compile deal.II solver first
    if not compile_deal_ii():
        print("Failed to compile deal.II solver")
        # Write failure result
        with open(os.path.join(output_dir, "RESULT.txt"), 'w') as f:
            f.write("COULD_NOT_COMPLETE\n")
            f.write("Reason: Failed to compile deal.II solver\n")
        return
    
    # Mesh levels
    h_values = [1/8, 1/16, 1/32]
    dt_values = [1/32, 1/64, 1/128]
    t_end = 1/4
    
    all_results = []
    all_files = []
    
    for level_idx in range(3):
        level = level_idx + 1
        h = h_values[level_idx]
        dt = dt_values[level_idx]
        
        result = solve_level(level, h, dt, t_end, output_dir)
        if result is None:
            print(f"Failed to solve level {level}")
            continue
        
        all_results.append(result)
        all_files.extend([
            result['solution_A'],
            result['solution_B'],
            result['interface_A'],
            result['interface_B'],
            result['residual'],
            result['log_A'],
            result['log_B']
        ])
    
    # Check mesh independence
    converged, max_rel_change = check_mesh_independence(all_results)
    
    # Get final level info
    final_result = all_results[-1] if all_results else None
    final_residual = final_result['final_residual'] if final_result else 1.0
    n_iterations = final_result['n_iterations'] if final_result else 0
    
    # Write RESULT.txt
    with open(os.path.join(output_dir, "RESULT.txt"), 'w') as f:
        f.write(f"LEVELS = {len(all_results)}\n")
        f.write(f"FILES = {','.join([os.path.basename(fn) for fn in all_files])}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residual:.16e}\n")
        f.write(f"COUPLING_ITERATIONS = {n_iterations}\n")
        f.write(f"MESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.16e}\n")
    
    print("\n=== Simulation Complete ===")
    print(f"Levels solved: {len(all_results)}")
    print(f"Final interface residual: {final_residual:.6e}")
    print(f"Mesh independence: {'CONVERGED' if converged else 'NOT_CONVERGED'}")
    print(f"Max relative change: {max_rel_change:.6e}")

if __name__ == "__main__":
    main()
