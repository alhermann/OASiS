#!/usr/bin/env python3
"""
NGSolve solver for -div(K grad u) = f on unit square
K = [[3, -1], [-1, 2]]
u = 0 on boundary
"""

from ngsolve import *
from math import *
import numpy as np

# Define probe points
def generate_probe_points():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            px = (i_x + 0.5) / 44
            py = (i_y + 0.5) / 44
            points.append((px, py))
    return points

# K tensor components
K11 = 3
K12 = -1
K21 = -1
K22 = 2

def solve_level(h, level):
    """Solve for a given mesh size h"""
    print(f"\n{'='*50}")
    print(f"Solving level {level} with h = {h}")
    print(f"{'='*50}")
    
    # Create mesh from unit_square geometry
    geom = unit_square
    mesh = Mesh(geom.GenerateMesh(maxh=h))
    
    # Create finite element space (H1, order=1)
    V = H1(mesh, order=1)
    
    # Get number of DOFs
    ndof = V.ndof
    print(f"NDOF = {ndof}")
    
    # Define trial and test functions
    u, v = V.TnT()
    
    # Define the bilinear form: a(u,v) = int(K grad u . grad v)
    # K grad u = [K11*ux + K12*uy, K21*ux + K22*uy]
    # K grad u . grad v = (K11*ux + K12*uy)*vx + (K21*ux + K22*uy)*vy
    a = BilinearForm(V)
    a += (K11*grad(u)[0] + K12*grad(u)[1])*grad(v)[0] * dx
    a += (K21*grad(u)[0] + K22*grad(u)[1])*grad(v)[1] * dx
    a.Assemble()
    
    # Define the linear form: L(v) = int(f*v)
    # Define f using x and y coordinate functions
    f_expr = 36*x**3*y - 20*x**3/3 - 54*x**2*y**2 - 32*x**2*y/5 + 92*x**2/15 + 54*x*y**3 - 18*x*y**2/5 - 448*x*y/15 - 32*x/15 - 66*y**3/5 + 2*y**2 + 112*y/15 + 8/3
    L = LinearForm(V)
    L += f_expr * v * dx
    L.Assemble()
    
    # Apply Dirichlet boundary conditions (u=0 on all boundaries)
    # We need to modify the matrix and RHS to enforce u=0 on boundaries
    # First, get the boundary DOFs
    bnd_dofs = V.GetDofs(mesh.Boundaries('.*'))
    print(f"Boundary DOFs: {len(bnd_dofs)}")
    
    # Create a BitArray for the boundary DOFs
    from pyngcore import BitArray
    bnd_mask = BitArray(a.mat.height)
    for dof in bnd_dofs:
        bnd_mask.Set(dof)
    
    # Modify the matrix and RHS to enforce u=0 on boundaries
    # Set diagonal entries to 1 and RHS to 0 for boundary DOFs
    for dof in bnd_dofs:
        a.mat.Set(dof, dof, 1.0)
        # Zero out off-diagonal entries
        for j in range(a.mat.width):
            if j != dof:
                a.mat.Set(dof, j, 0.0)
                a.mat.Set(j, dof, 0.0)
        L.vec[dof] = 0.0
    
    # Create preconditioner
    pre = a.mat.CreateSmoother()
    
    # Solve using CGSolver
    gfu = GridFunction(V)
    solver = CGSolver(a.mat, pre)
    solver.Mult(gfu.vec, L.vec)
    
    # Evaluate at probe points
    probe_points = generate_probe_points()
    results = []
    
    for (px, py) in probe_points:
        u_val = gfu(px, py)
        results.append((px, py, u_val))
    
    # Write CSV file
    csv_filename = f"solution_level{level}.csv"
    with open(csv_filename, 'w') as f:
        f.write("x, y, u\n")
        for (px, py, u_val) in results:
            f.write(f"{px:.15e}, {py:.15e}, {u_val:.15e}\n")
    print(f"Written {csv_filename}")
    
    # Write log file
    log_filename = f"run_level{level}.log"
    num_elements = len(list(mesh.Elements()))
    with open(log_filename, 'w') as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"h = {h}\n")
        f.write(f"Mesh elements: {num_elements}\n")
    print(f"Written {log_filename}")
    
    return results, ndof

def main():
    # Mesh levels
    h_values = [1/8, 1/16, 1/32, 1/64]
    levels = [1, 2, 3, 4]
    
    all_results = []
    all_ndofs = []
    
    for i, h in enumerate(h_values):
        level = levels[i]
        results, ndof = solve_level(h, level)
        all_results.append(results)
        all_ndofs.append(ndof)
    
    # Compute convergence metrics
    # Compare finest two levels (level 3 and 4)
    results_3 = all_results[2]  # h = 1/32
    results_4 = all_results[3]  # h = 1/64
    
    max_rel_change = 0
    for i in range(len(results_3)):
        u3 = results_3[i][2]
        u4 = results_4[i][2]
        if abs(u3) > 1e-15:
            rel_change = abs(u4 - u3) / abs(u3)
        else:
            rel_change = abs(u4 - u3)
        max_rel_change = max(max_rel_change, rel_change)
    
    print(f"\n{'='*50}")
    print(f"MAX_REL_CHANGE = {max_rel_change:.15e}")
    
    # Determine convergence (if max relative change is small enough)
    # For linear elements, we expect O(h) convergence
    # If max_rel_change < 0.01 (1%), consider converged
    converged = max_rel_change < 0.01
    convergence_status = "CONVERGED" if converged else "NOT_CONVERGED"
    print(f"MESH_INDEPENDENCE = {convergence_status}")
    
    # Write RESULT.txt
    csv_files = "solution_level1.csv, solution_level2.csv, solution_level3.csv, solution_level4.csv"
    with open("RESULT.txt", 'w') as f:
        f.write(f"LEVELS = 4\n")
        f.write(f"FILES = {csv_files}\n")
        f.write(f"MESH_INDEPENDENCE = {convergence_status}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\nWritten RESULT.txt")
    print("Done!")

if __name__ == "__main__":
    main()
