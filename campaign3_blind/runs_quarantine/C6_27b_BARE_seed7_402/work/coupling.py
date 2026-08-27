#!/usr/bin/env python3
"""
Coupling script for Dirichlet-Neumann iteration between deal.II (subdomain A) and NGSolve (subdomain B).

Subdomain A (deal.II): Dirichlet side - receives interface field, returns flux
Subdomain B (NGSolve): Neumann side - receives flux, returns interface field

Interface: x = 5/8 = 0.625
"""

import os
import sys
import subprocess
import numpy as np

def generate_interface_points():
    """Generate the 44 interface probe points."""
    points = []
    for i in range(44):
        y = 0.25 + (i + 0.5) * 0.5 / 44.0
        points.append((5.0/8.0, y))
    return points

def write_interface_file(filename, points, values):
    """Write interface data to CSV file."""
    with open(filename, 'w') as f:
        f.write("x,y,u\n")
        for i, (x, y) in enumerate(points):
            f.write("{:.15e},{:.15e},{:.15e}\n".format(x, y, values[i]))

def read_flux_file(filename):
    """Read flux values from CSV file."""
    fluxes = []
    with open(filename, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                fluxes.append(float(parts[2]))
    return np.array(fluxes)

def read_values_file(filename):
    """Read u values from CSV file."""
    values = []
    with open(filename, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                values.append(float(parts[2]))
    return np.array(values)

def compute_residual(u_A, u_B):
    """Compute relative interface residual."""
    diff = np.abs(u_A - u_B)
    max_val = max(np.max(np.abs(u_A)), np.max(np.abs(u_B)), 1e-10)
    return np.max(diff) / max_val

def run_coupling(level, work_dir):
    """Run the coupled simulation for a given mesh level."""
    
    print(f"\n{'='*60}")
    print(f"Running coupling for level {level}")
    print(f"{'='*60}")
    
    # Generate interface points
    interface_points = generate_interface_points()
    
    # Initial guess for interface values (zero)
    u_interface = np.zeros(len(interface_points))
    
    # File names
    input_interface = os.path.join(work_dir, "input_interface.csv")
    output_sol_A = os.path.join(work_dir, f"solution_level{level}_A.csv")
    output_iface_A = os.path.join(work_dir, f"interface_level{level}_A.csv")
    output_log_A = os.path.join(work_dir, f"run_level{level}_A.log")
    output_flux_A = os.path.join(work_dir, "flux_A.csv")
    
    input_flux = os.path.join(work_dir, "flux_A.csv")
    output_sol_B = os.path.join(work_dir, f"solution_level{level}_B.csv")
    output_iface_B = os.path.join(work_dir, f"interface_level{level}_B.csv")
    output_log_B = os.path.join(work_dir, f"run_level{level}_B.log")
    output_values_B = os.path.join(work_dir, "values_B.csv")
    
    residual_file = os.path.join(work_dir, f"residual_level{level}.csv")
    
    # Open residual file
    with open(residual_file, 'w') as f:
        f.write("iteration,interface_residual\n")
    
    # Coupling parameters
    max_iter = 100
    tol = 1e-6
    
    # Path to executables
    subdomain_A_exe = os.path.join(work_dir, "subdomain_A")
    subdomain_B_py = os.path.join(work_dir, "subdomain_B.py")
    python_exe = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
    
    residuals = []
    
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}...")
        
        # Write interface values for subdomain A
        write_interface_file(input_interface, interface_points, u_interface)
        
        # Run subdomain A (deal.II) - Dirichlet side
        cmd_A = [subdomain_A_exe, str(level), input_interface, output_sol_A, 
                 output_iface_A, output_log_A, output_flux_A]
        result_A = subprocess.run(cmd_A, capture_output=True, text=True)
        if result_A.returncode != 0:
            print(f"Error running subdomain A: {result_A.stderr}")
            break
        
        # Read flux from subdomain A
        flux_A = read_flux_file(output_flux_A)
        
        # Run subdomain B (NGSolve) - Neumann side
        cmd_B = [python_exe, subdomain_B_py, str(level), input_flux, output_sol_B,
                 output_iface_B, output_log_B, output_values_B]
        result_B = subprocess.run(cmd_B, capture_output=True, text=True)
        if result_B.returncode != 0:
            print(f"Error running subdomain B: {result_B.stderr}")
            break
        
        # Read interface values from subdomain B
        u_new = read_values_file(output_values_B)
        
        # Compute residual
        residual = compute_residual(u_interface, u_new)
        residuals.append(residual)
        print(f"  Residual: {residual:.6e}")
        
        # Write residual to file
        with open(residual_file, 'a') as f:
            f.write(f"{iteration + 1},{residual:.15e}\n")
        
        # Update interface values (simple relaxation)
        omega = 0.5  # Relaxation parameter
        u_interface = (1 - omega) * u_interface + omega * u_new
        
        # Check convergence
        if residual < tol:
            print(f"Converged after {iteration + 1} iterations!")
            break
    
    # Final iteration to get consistent results
    if iteration == max_iter - 1:
        print("Warning: Did not converge within maximum iterations")
    
    # Run one more time with final interface values to get consistent outputs
    write_interface_file(input_interface, interface_points, u_interface)
    
    cmd_A = [subdomain_A_exe, str(level), input_interface, output_sol_A, 
             output_iface_A, output_log_A, output_flux_A]
    subprocess.run(cmd_A, capture_output=True)
    
    flux_A = read_flux_file(output_flux_A)
    
    cmd_B = [python_exe, subdomain_B_py, str(level), input_flux, output_sol_B,
             output_iface_B, output_log_B, output_values_B]
    subprocess.run(cmd_B, capture_output=True)
    
    return len(residuals), residuals[-1] if residuals else float('inf')

def main():
    work_dir = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C6_27b_BARE_seed7/work"
    
    # Mesh levels
    levels = [1, 2, 3]  # h = 1/8, 1/16, 1/32
    
    all_files = []
    final_residuals = []
    coupling_iterations = []
    
    for level in levels:
        n_iter, final_res = run_coupling(level, work_dir)
        coupling_iterations.append(n_iter)
        final_residuals.append(final_res)
        
        # Collect files
        all_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv",
            f"run_level{level}_A.log",
            f"run_level{level}_B.log"
        ])
    
    # Compute mesh independence
    # Compare solutions at finest two levels
    print("\nChecking mesh independence...")
    
    # Read solution files for levels 2 and 3
    def read_solution(filename):
        values = []
        with open(os.path.join(work_dir, filename), 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    values.append(float(parts[2]))
        return np.array(values)
    
    sol_A_2 = read_solution("solution_level2_A.csv")
    sol_A_3 = read_solution("solution_level3_A.csv")
    sol_B_2 = read_solution("solution_level2_B.csv")
    sol_B_3 = read_solution("solution_level3_B.csv")
    
    # Compute relative changes
    max_val_A = max(np.max(np.abs(sol_A_2)), np.max(np.abs(sol_A_3)), 1e-10)
    max_val_B = max(np.max(np.abs(sol_B_2)), np.max(np.abs(sol_B_3)), 1e-10)
    
    rel_change_A = np.max(np.abs(sol_A_3 - sol_A_2)) / max_val_A
    rel_change_B = np.max(np.abs(sol_B_3 - sol_B_2)) / max_val_B
    
    max_rel_change = max(rel_change_A, rel_change_B)
    print(f"Max relative change between levels 2 and 3: {max_rel_change:.6e}")
    
    # Determine convergence
    mesh_independence = "CONVERGED" if max_rel_change < 1e-3 else "NOT_CONVERGED"
    
    # Write RESULT.txt
    with open(os.path.join(work_dir, "RESULT.txt"), 'w') as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {','.join(all_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residuals[-1]:.15e}\n")
        f.write(f"COUPLING_ITERATIONS = {coupling_iterations[-1]}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\nResults written to RESULT.txt")
    print(f"Mesh independence: {mesh_independence}")

if __name__ == "__main__":
    main()
