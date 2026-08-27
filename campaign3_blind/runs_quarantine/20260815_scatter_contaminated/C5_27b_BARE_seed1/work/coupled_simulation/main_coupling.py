#!/usr/bin/env python3
"""
Main coupling script for Dirichlet-Neumann iteration between scikit-fem (A) and dolfinx (B).
"""

import numpy as np
from pathlib import Path
import sys

# Import solvers
sys.path.insert(0, '/home/alexander/coupled_simulation')
from solver_A import SubdomainA, get_k, get_f
from solver_B import SubdomainB

def get_probe_points_A():
    """Generate probe points for subdomain A."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            # Exclude points in subdomain B: (0.5, 1) x (0, 0.5)
            if x > 0.5 and y < 0.5:
                continue
            # Exclude points in removed square: (0.75, 1) x (0.75, 1)
            if x > 0.75 and y > 0.75:
                continue
            points.append((x, y))
    return np.array(points)

def get_probe_points_B():
    """Generate probe points for subdomain B."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.5 + (i_x + 0.5) * 0.5 / 44.0
            y = (i_y + 0.5) * 0.5 / 44.0
            points.append((x, y))
    return np.array(points)

def get_interface_probes():
    """Generate interface probe points: leg 1 first, then leg 2."""
    points = []
    # Leg 1: x = 1/2, y = 1/8 + (i+0.5)*(1/4)/44 for i = 0..43
    for i in range(44):
        x = 0.5
        y = 1/8 + (i + 0.5) * (1/4) / 44.0
        points.append((x, y))
    # Leg 2: x = 5/8 + (i+0.5)*(1/4)/44, y = 1/2 for i = 0..43
    for i in range(44):
        x = 5/8 + (i + 0.5) * (1/4) / 44.0
        y = 0.5
        points.append((x, y))
    return np.array(points)


def interpolate_on_mesh_A(solver_A, x, y):
    """Interpolate solution at point (x, y) on subdomain A mesh."""
    try:
        cell_idx = solver_A.mesh.point_locator.locate(np.array([[x], [y]]))[0]
        if cell_idx >= 0:
            cell_nodes = solver_A.mesh.t[:, cell_idx]
            # Get barycentric coordinates for proper interpolation
            from skfem.helpers import barycentric_coordinates
            coords = solver_A.mesh.p[:, cell_nodes]
            bc = barycentric_coordinates(coords, np.array([[x], [y]]))
            if bc is not None and len(bc[0]) == 3:
                return np.sum(bc[0] * solver_A.u[cell_nodes])
            else:
                return np.mean(solver_A.u[cell_nodes])
    except:
        pass
    
    # Nearest neighbor fallback
    min_dist = float('inf')
    closest_val = 0.0
    for i, (xi, yi) in enumerate(solver_A.mesh.p.T):
        dist = (xi - x)**2 + **(yi - y)2
        if dist < min_dist:
            min_dist = dist
            closest_val = solver_A.u[i]
    
    return closest_val


def run_coupling(level, h, output_dir):
    """Run the coupled Dirichlet-Neumann iteration for one mesh level."""
    
    print(f"\n{'='*60}")
    print(f"Running coupling for level {level}, h = {h}")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize solvers
    print("Initializing subdomain A solver (scikit-fem)...")
    solver_A = SubdomainA(h)
    ndof_A = solver_A.setup()
    print(f"Subdomain A: {ndof_A} DOFs, {len(solver_A.interface_nodes)} interface nodes")
    
    print("Initializing subdomain B solver (dolfinx)...")
    solver_B = SubdomainB(h)
    ndof_B = solver_B.setup()
    print(f"Subdomain B: {ndof_B} DOFs")
    
    # Write log files
    with open(output_dir / f"run_level{level}_A.log", "w") as f:
        f.write(f"NDOF = {ndof_A}\n")
    
    with open(output_dir / f"run_level{level}_B.log", "w") as f:
        f.write(f"NDOF = {ndof_B}\n")
    
    # Coupling parameters
    max_iter = 200
    tol = 1e-6
    relaxation = 0.5
    
    # Number of interface points
    n_interface = len(solver_A.interface_nodes)
    print(f"Interface has {n_interface} nodes")
    
    # Initial guess: zero on interface
    u_interface_A = np.zeros(n_interface)
    
    residual_history = []
    
    for iteration in range(max_iter):
        # Step 1: Solve subdomain A with Dirichlet BC from B
        u_A = solver_A.solve_dirichlet(u_interface_A)
        solver_A.u = u_A  # Store for interpolation
        
        # Compute flux from A on interface (outward from A)
        qn_A = solver_A.compute_flux(u_A)
        
        # Step 2: Solve subdomain B with Neumann BC from A
        # Flux from A outward = flux into B (with opposite sign convention)
        u_B_vals = solver_B.solve_neumann(qn_A, solver_A.interface_nodes)
        
        # Extract interface values from B
        interface_values_B = solver_B.get_interface_values()
        
        # Map to A's interface node ordering
        u_interface_B = np.zeros(n_interface)
        tol_match = h * 0.5
        
        # Match interface points
        for i, node_idx in enumerate(solver_A.interface_nodes):
            x_a, y_a = solver_A.mesh.p[:, node_idx]
            
            # Find matching point in B's interface values
            best_match = None
            best_dist = float('inf')
            for j, (xb, yb, ub) in enumerate(interface_values_B):
                dist = (xb - x_a)**2 + **(yb - y_a)2
                if dist < best_dist:
                    best_dist = dist
                    best_match = ub
            
            if best_dist < tol_match**2 and best_match is not None:
                u_interface_B[i] = best_match
        
        # Compute residual
        if len(u_interface_B) > 0:
            diff = np.abs(u_interface_B - u_interface_A)
            norm_A = np.max(np.abs(u_interface_A)) + 1e-15
            residual = np.max(diff) / norm_A
        else:
            residual = 1.0
        
        residual_history.append(residual)
        
        if (iteration + 1) % 10 == 0 or iteration < 5:
            print(f"Iteration {iteration + 1}: residual = {residual:.6e}")
        
        if residual < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Relaxation update
        u_interface_A = (1 - relaxation) * u_interface_A + relaxation * u_interface_B
    
    # Final solutions
    u_A_final = solver_A.solve_dirichlet(u_interface_A)
    solver_A.u = u_A_final
    qn_A_final = solver_A.compute_flux(u_A_final)
    
    u_B_final_vals = solver_B.solve_neumann(qn_A_final, solver_A.interface_nodes)
    
    # Write solution files
    write_solution_files(level, solver_A, solver_B, output_dir)
    
    # Write interface files
    write_interface_files(level, solver_A, solver_B, qn_A_final, output_dir)
    
    # Write residual history
    write_residual_file(level, residual_history, output_dir)
    
    final_residual = residual_history[-1] if residual_history else 1.0
    n_iterations = len(residual_history)
    
    print(f"Final residual: {final_residual:.6e}, iterations: {n_iterations}")
    
    return final_residual, n_iterations, solver_A, solver_B


def write_solution_files(level, solver_A, solver_B, output_dir):
    """Write solution CSV files for probe points."""
    
    # Probe points for A
    probes_A = get_probe_points_A()
    
    results_A = []
    for x, y in probes_A:
        u_val = interpolate_on_mesh_A(solver_A, x, y)
        results_A.append((x, y, u_val))
    
    with open(output_dir / f"solution_level{level}_A.csv", "w") as f:
        f.write("x, y, u\n")
        for x, y, u in results_A:
            f.write(f"{x}, {y}, {u}\n")
    
    # Probe points for B
    probes_B = get_probe_points_B()
    
    results_B = []
    for x, y in probes_B:
        u_val = solver_B.interpolate_at_point(x, y)
        results_B.append((x, y, u_val))
    
    with open(output_dir / f"solution_level{level}_B.csv", "w") as f:
        f.write("x, y, u\n")
        for x, y, u in results_B:
            f.write(f"{x}, {y}, {u}\n")


def write_interface_files(level, solver_A, solver_B, qn_A, output_dir):
    """Write interface CSV files."""
    
    interface_probes = get_interface_probes()
    
    # For subdomain A
    results_A = []
    for i, (x, y) in enumerate(interface_probes):
        u_val = interpolate_on_mesh_A(solver_A, x, y)
        # Map probe to interface node index
        qn_val = qn_A[i] if i < len(qn_A) else 0.0
        results_A.append((x, y, u_val, qn_val))
    
    with open(output_dir / f"interface_level{level}_A.csv", "w") as f:
        f.write("x, y, u, qn\n")
        for x, y, u, qn in results_A:
            f.write(f"{x}, {y}, {u}, {qn}\n")
    
    # For subdomain B
    results_B = []
    for i, (x, y) in enumerate(interface_probes):
        u_val = solver_B.interpolate_at_point(x, y)
        # Outward normal from B is opposite to A
        qn_val = -qn_A[i] if i < len(qn_A) else 0.0
        results_B.append((x, y, u_val, qn_val))
    
    with open(output_dir / f"interface_level{level}_B.csv", "w") as f:
        f.write("x, y, u, qn\n")
        for x, y, u, qn in results_B:
            f.write(f"{x}, {y}, {u}, {qn}\n")


def write_residual_file(level, residual_history, output_dir):
    """Write residual history CSV file."""
    with open(output_dir / f"residual_level{level}.csv", "w") as f:
        f.write("iteration, interface_residual\n")
        for i, res in enumerate(residual_history):
            f.write(f"{i + 1}, {res}\n")


def main():
    """Main driver for coupled simulation."""
    
    output_dir = Path("/home/alexander/coupled_simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mesh levels
    levels = [1, 2, 3]
    hs = [1/8, 1/16, 1/32]
    
    final_residuals = []
    iteration_counts = []
    solution_data = {}
    solvers_A = []
    solvers_B = []
    
    for level, h in zip(levels, hs):
        try:
            residual, n_iter, solver_A, solver_B = run_coupling(level, h, output_dir)
            final_residuals.append(residual)
            iteration_counts.append(n_iter)
            solvers_A.append(solver_A)
            solvers_B.append(solver_B)
            
            # Store solution for convergence check
            probes_A = get_probe_points_A()
            with open(output_dir / f"solution_level{level}_A.csv", "r") as f:
                lines = f.readlines()[1:]
                vals_A = [float(line.split(',')[2]) for line in lines]
            solution_data[f"A_{level}"] = np.array(vals_A)
            
            probes_B = get_probe_points_B()
            with open(output_dir / f"solution_level{level}_B.csv", "r") as f:
                lines = f.readlines()[1:]
                vals_B = [float(line.split(',')[2]) for line in lines]
            solution_data[f"B_{level}"] = np.array(vals_B)
            
        except Exception as e:
            print(f"Error at level {level}: {e}")
            import traceback
            traceback.print_exc()
            final_residuals.append(None)
            iteration_counts.append(0)
    
    # Check mesh independence
    converged = "NOT_CONVERGED"
    max_rel_change = 0.0
    
    if solution_data.get("A_2") is not None and solution_data.get("A_3") is not None:
        rel_change_A = np.max(np.abs(solution_data["A_3"] - solution_data["A_2"]) / 
                              (np.abs(solution_data["A_2"]) + 1e-15))
        rel_change_B = np.max(np.abs(solution_data.get("B_3", np.zeros_like(solution_data["A_3"])) - 
                                     solution_data.get("B_2", np.zeros_like(solution_data["A_2"]))) /
                              (np.abs(solution_data["B_2"]) + 1e-15))
        max_rel_change = max(rel_change_A, rel_change_B)
        
        if max_rel_change < 0.01:
            converged = "CONVERGED"
    
    # Collect all CSV files
    csv_files = []
    for level in levels:
        csv_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv"
        ])
    
    # Write RESULT.txt
    with open(output_dir / "RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {','.join(csv_files)}\n")
        final_res = final_residuals[-1] if final_residuals[-1] is not None else "COULD_NOT_COMPLETE"
        f.write(f"INTERFACE_RESIDUAL = {final_res}\n")
        f.write(f"COUPLING_ITERATIONS = {iteration_counts[-1]}\n")
        f.write(f"MESH_INDEPENDENCE = {converged}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change}\n")
    
    print(f"\nSimulation complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
