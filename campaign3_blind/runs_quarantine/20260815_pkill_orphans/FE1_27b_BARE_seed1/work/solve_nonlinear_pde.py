#!/usr/bin/env python3
"""
Solve nonlinear PDE: -div(a(u) grad u) = f with a(u) = 1 + u^2/2
using FEniCSx (dolfinx) with Newton solver.
"""

import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh
from dolfinx.fem import functionspace, Form, Function, Constant
from dolfinx.fem.petsc import NonlinearProblem
import ufl
from petsc4py import PETSc

# Source term f(x, y) - very long polynomial
def source_term(x):
    # x[0] is x-coordinate, x[1] is y-coordinate
    X = x[0]
    Y = x[1]
    
    f = (320*X**9*Y**4 - 640*X**9*Y**3 + 384*X**9*Y**2 - 64*X**9*Y 
         + 112*X**8*Y**5 - 1008*X**8*Y**4 + 1696*X**8*Y**3 - 4768*X**8*Y**2/5 + 768*X**8*Y/5
         + 7024*X**7*Y**6/9 - 13116*X**7*Y**5/5 + 16852*X**7*Y**4/5 - 91496*X**7*Y**3/45 + 14752*X**7*Y**2/25 - 2112*X**7*Y/25
         + 1348*X**6*Y**7/9 - 51772*X**6*Y**6/27 + 1132663*X**6*Y**5/225 - 3243077*X**6*Y**4/675 + 327088*X**6*Y**3/225 + 14144*X**6*Y**2/125 - 3584*X**6*Y/125
         + 28*X**5*Y**8/3 - 4808*X**5*Y**7/15 + 343336*X**5*Y**6/225 - 203819*X**5*Y**5/75 + 90457*X**5*Y**4/45 - 6304*X**5*Y**3/15 - 12992*X**5*Y**2/125 + 2112*X**5*Y/125
         + 5*X**4*Y**9/27 - 173*X**4*Y**8/9 + 9229*X**4*Y**7/45 - 244697*X**4*Y**6/675 + 309*X**4*Y**5/25 + 69931*X**4*Y**4/225 - 28112*X**4*Y**3/225 - 3488*X**4*Y**2/125 + 768*X**4*Y/125
         - 10*X**3*Y**9/27 + 106*X**3*Y**8/9 - 1318*X**3*Y**7/45 - 25786*X**3*Y**6/675 + 38573*X**3*Y**5/225 - 114751*X**3*Y**4/675 + 12584*X**3*Y**3/225 - 224*X**3*Y**2/125 + 64*X**3*Y/125 + 8*X**3
         + 2*X**2*Y**9/9 - 26*X**2*Y**8/15 - 382*X**2*Y**7/75 + 12566*X**2*Y**6/1125 + 4144*X**2*Y**5/375 - 3488*X**2*Y**4/125 + 1536*X**2*Y**3/125 + 2*X**2*Y - 106*X**2/15
         - X*Y**9/27 - 7*X*Y**8/45 + 11*X*Y**7/225 + 2177*X*Y**6/3375 - 44*X*Y**5/375 - 112*X*Y**4/125 + 64*X*Y**3/125 + 24*X*Y**2 - 26*X*Y - 14*X/15
         + 2*Y**3/3 - 106*Y**2/15 + 32*Y/5)
    return f

def solve_level(N, level_idx):
    """Solve the problem for a given mesh size N."""
    print(f"\n{'='*60}")
    print(f"Solving level {level_idx} with N={N} elements per side")
    print(f"{'='*60}")
    
    # Create mesh
    msh = mesh.create_unit_square(MPI.COMM_WORLD, N, N, cell_type=mesh.CellType.triangle)
    
    # Create function space - Lagrange P1 (degree 1)
    V = functionspace(msh, ("Lagrange", 1))
    
    # Apply Dirichlet BCs on all boundaries
    ndim = msh.topology.dim
    facets = mesh.locate_entities_boundary(msh, ndim - 1, lambda x: True)
    dofs = fem.locate_dofs_topological(V, ndim - 1, facets)
    
    # Create zero function for BC
    bc = fem.dirichletbc(0.0, dofs, V)
    
    # Define trial and test functions
    u = Function(V)
    du = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # Coefficient a(u) = 1 + u^2/2
    a_u = 1.0 + u**2 / 2.0
    
    # Weak form: R(u; v) = int(a(u) grad u . grad v) - int(f v)
    # For Newton: J(u; du, v) = int(a'(u) du grad u . grad v + a(u) grad du . grad v)
    # where a'(u) = u
    
    # Residual form
    F = a_u * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx - source_term(msh.x) * v * ufl.dx
    
    # Linearized form (Jacobian)
    # dF/du[du] = int((u*du)*grad(u).grad(v) + a(u)*grad(du).grad(v)) - 0
    a_prime = u  # derivative of a(u) = 1 + u^2/2 is u
    J = a_prime * du * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx + a_u * ufl.dot(ufl.grad(du), ufl.grad(v)) * ufl.dx
    
    # Compile forms
    F_form = Form(F)
    J_form = Form(J)
    
    # Set up Newton solver using high-level NonlinearProblem
    petsc_options = {
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_linesearch_type": "bt",
        "snes_rtol": 1e-10,
        "snes_atol": 1e-14,
        "snes_max_it": 50,
    }
    
    problem = NonlinearProblem(
        F_form, u, bcs=[bc], J=J_form,
        petsc_options_prefix="nonlinear_",
        petsc_options=petsc_options
    )
    
    # Get SNES solver
    snes = problem.solver
    
    # Initial guess (zero)
    u.vector[:] = 0.0
    
    # Solve
    print("Starting Newton iteration...")
    n_iterations, converged = snes.solve(u.vector)
    
    if not converged:
        print(f"WARNING: Newton solver did not converge! Iterations: {n_iterations}")
    else:
        print(f"Newton solver converged in {n_iterations} iterations")
    
    # Get final residual
    norm = snes.getTolerances()[0]  # This might not be right
    # Actually get the residual norm from SNES
    rnorm = snes.getTolerances()
    print(f"SNES tolerances: {rnorm}")
    
    # Count DOFs
    ndof = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
    print(f"NDOF = {ndof}")
    
    # Write run log
    with open(f"run_level{level_idx}.log", "w") as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"Newton iterations: {n_iterations}\n")
        f.write(f"Converged: {converged}\n")
    
    # Generate probe points
    probe_points = []
    for i_y in range(44):
        for i_x in range(44):
            x_coord = (i_x + 0.5) / 44.0
            y_coord = (i_y + 0.5) / 44.0
            probe_points.append([x_coord, y_coord])
    
    probe_points = np.array(probe_points)
    
    # Interpolate solution at probe points
    print(f"Evaluating solution at {len(probe_points)} probe points...")
    
    # Evaluate at all probe points
    from dolfinx.fem import evaluate_function
    u_values = evaluate_function(u, probe_points.T)
    
    # Write CSV file
    csv_filename = f"solution_level{level_idx}.csv"
    with open(csv_filename, "w") as f:
        f.write("x, y, u\n")
        for i in range(len(probe_points)):
            x_val = probe_points[i, 0]
            y_val = probe_points[i, 1]
            u_val = u_values[i]
            f.write(f"{x_val:.11e}, {y_val:.11e}, {u_val:.11e}\n")
    
    print(f"Wrote {csv_filename}")
    
    return u_values, ndof, converged, n_iterations

if __name__ == "__main__":
    # Mesh levels
    N_values = [8, 16, 32, 64]
    
    results = []
    all_converged = True
    
    for idx, N in enumerate(N_values, start=1):
        try:
            u_vals, ndof, converged, n_iter = solve_level(N, idx)
            results.append((idx, N, u_vals, ndof, converged, n_iter))
            if not converged:
                all_converged = False
        except Exception as e:
            print(f"ERROR at level {idx} (N={N}): {e}")
            import traceback
            traceback.print_exc()
            all_converged = False
    
    # Compute mesh independence
    print("\n" + "="*60)
    print("Computing mesh independence...")
    print("="*60)
    
    max_rel_change = 0.0
    if len(results) >= 2:
        # Compare finest two levels (indices 3 and 2, i.e., N=64 and N=32)
        _, _, u_fine, _, _, _ = results[-1]  # N=64
        _, _, u_coarse, _, _, _ = results[-2]  # N=32
        
        # Both should have same number of probe points
        assert len(u_fine) == len(u_coarse), "Different number of probe points!"
        
        # Compute relative change at each point
        rel_changes = np.abs(u_fine - u_coarse) / (np.abs(u_coarse) + 1e-15)
        max_rel_change = np.max(rel_changes)
        
        print(f"Max relative change between N=32 and N=64: {max_rel_change:.6e}")
        
        # Check convergence criterion (typically < 1% or similar)
        if max_rel_change < 0.01:  # 1% threshold
            mesh_independence = "CONVERGED"
        else:
            mesh_independence = "NOT_CONVERGED"
    else:
        max_rel_change = float('inf')
        mesh_independence = "NOT_CONVERGED"
    
    print(f"MESH_INDEPENDENCE = {mesh_independence}")
    
    # Write RESULT.txt
    csv_files = ", ".join([f"solution_level{k}.csv" for k in range(1, len(results)+1)])
    
    with open("RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(results)}\n")
        f.write(f"FILES = {csv_files}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.11e}\n")
    
    print("\nWrote RESULT.txt")
    print("Done!")
