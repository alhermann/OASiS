#!/usr/bin/env python3
"""
Mixed displacement-pressure formulation for nearly incompressible elasticity.
Taylor-Hood pair: P2 for displacement, P1 for pressure.
"""

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, io, mesh, la
from dolfinx.fem import functionspace, Form, Expression
from ufl import (dx, inner, grad, tr, Identity, div, dot, 
                 TrialFunctions, TestFunctions)
from petsc4py.PETSc import ScalarType

# Material parameters
lambda_ = 74998500.0
mu = 1500.0

# Source term f(x, y)
def f_ux(x):
    # x[0] is x-coordinate, x[1] is y-coordinate
    X = x[0]
    Y = x[1]
    return (36000*X**5*Y - 18000*X**5 
            - 3001740036*X**4*Y**2/49999 + 151797036*X**4*Y/49999 
            + 824683494*X**4/49999 
            + 6011880240*X**3*Y**3/49999 
            - 15045300912*X**3*Y**2/249995 
            - 13508130168*X**3*Y/249995 
            + 5254295088*X**3/249995 
            - 3001740036*X**2*Y**4/49999 
            + 285593712*X**2*Y**3/49999 
            + 9825406512*X**2*Y**2/249995 
            + 7496249928*X**2*Y/249995 
            - 4878802578*X**2/249995 
            + 3600072*X*Y**5/49999 
            + 14976899544*X*Y**4/249995 
            - 28477829568*X*Y**3/249995 
            + 15732884664*X*Y**2/249995 
            - 449390988*X*Y/49999 
            + 200004*X/49999 
            - 1800036*Y**5/49999 
            - 2485849718*Y**4/249995 
            + 6248674976*Y**3/249995 
            - 4878802578*Y**2/249995 
            + 4500*Y)

def f_uy(x):
    X = x[0]
    Y = x[1]
    return (-3600072*X**5*Y/249995 + 1800036*X**5/249995 
            - 4490909820*X**4*Y**2/49999 
            + 22476749544*X**4*Y/249995 
            - 3752025042*X**4/249995 
            + 3997519952*X**3*Y**3/49999 
            - 314394288*X**3*Y**2/49999 
            - 16449268992*X**3*Y/249995 
            + 3748724976*X**3/249995 
            - 4490909820*X**2*Y**4/49999 
            + 14954099088*X**2*Y**3/249995 
            + 20281995648*X**2*Y**2/249995 
            - 15766485336*X**2*Y/249995 
            + 225295506*X**2/49999 
            + 24000*X*Y**5 
            - 158997180*X*Y**4/49999 
            - 6443468872*X*Y**3/249995 
            - 7503450072*X*Y**2/249995 
            + 9742004844*X*Y/249995 
            - 4500*X 
            - 12000*Y**5 
            + 1426471530*Y**4/49999 
            - 1051979040*Y**3/49999 
            + 225295506*Y**2/49999 
            + 200004*Y/49999)

def solve_level(N, level_idx):
    """Solve for a given mesh size N."""
    print(f"\n=== Solving level {level_idx} with N={N} ===")
    
    # Create mesh
    domain = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N, 
                                              dolfinx.mesh.CellType.triangle)
    
    # Define function spaces - Taylor-Hood P2-P1
    V_elem = ("Lagrange", 2)  # P2 for displacement
    Q_elem = ("Lagrange", 1)  # P1 for pressure
    
    # Displacement space (vector)
    V = functionspace(domain, (V_elem, V_elem))
    # Pressure space (scalar)
    Q = functionspace(domain, Q_elem)
    
    # Mixed space: (u, p)
    W = V * Q
    
    # Boundary condition: u = 0 on entire boundary
    def boundary_facet(x):
        return np.ones(len(x[0]), dtype=bool)  # All facets are boundary
    
    # Find boundary dofs for displacement only
    bdy_dofs_V = fem.locate_dofs_topological(V, domain.topology.dim - 1, boundary_facet)
    
    # For mixed space, we need to map these to the correct indices
    # The mixed space has structure [u_x, u_y, p], so displacement dofs come first
    bc_dofs = bdy_dofs_V.copy()
    
    # Create Dirichlet BC for displacement components
    u_bc = fem.dirichletbc(0.0, bc_dofs, V)
    
    # Weak form: Mixed displacement-pressure formulation
    # sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I
    # But we use p = -lambda*div(u), so we reformulate
    # The weak form becomes:
    # a((u,p), (v,q)) = L(v,q)
    # where:
    #   a((u,p), (v,q)) = integral(2*mu*eps(u):eps(v) dx) 
    #                    + integral(p*div(v) dx) 
    #                    + integral(q*div(u) dx)
    #   L(v) = integral(f*v dx)
    
    u, p = TrialFunctions(W)
    v, q = TestFunctions(W)
    
    # Strain tensor
    eps_u = (grad(u) + grad(u).T) / 2
    eps_v = (grad(v) + grad(v).T) / 2
    
    # Bilinear form
    a_form = (2*mu*inner(eps_u, eps_v)*dx 
              + p*div(v)*dx 
              + q*div(u)*dx)
    
    # Linear form - define source term
    class FSource:
        def __call__(self, x):
            result = np.zeros((2, x.shape[1]))
            result[0, :] = f_ux(x)
            result[1, :] = f_uy(x)
            return result
    
    f_vec = fem.Function(V)
    f_vec.interpolate(lambda x: np.column_stack([f_ux(x), f_uy(x)]))
    
    L_form = dot(f_vec, v)*dx
    
    # Compile forms
    a = Form(a_form)
    L = Form(L_form)
    
    # Assemble system
    A = fem.petsc.assemble_matrix(a, bcs=[u_bc])
    A.assemble()
    
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], bcs=[u_bc])
    b.ghostUpdate(addv=dolfinx.la.InsertMode.add)
    
    # Apply BCs
    fem.petsc.set_bc(b, [u_bc])
    
    # Solve
    w = fem.Function(W)
    ksp = la.KrylovSolver("mf_gmres", "ilu")
    ksp.set_operator(A)
    ksp.solve(w.x.vector, b)
    
    # Extract displacement and pressure
    u_sol, p_sol = w.split()
    
    # Count DOFs
    ndof = W.dofmap.index_map.size_local * W.dofmap.index_map_bs
    print(f"NDOF = {ndof}")
    
    # Write log file
    with open(f"run_level{level_idx}.log", "w") as logfile:
        logfile.write(f"NDOF = {ndof}\n")
        logfile.write(f"Mesh: unit square with N={N} elements per side\n")
        logfile.write(f"Element type: Taylor-Hood P2-P1\n")
        logfile.write(f"Displacement space: P2 vector\n")
        logfile.write(f"Pressure space: P1 scalar\n")
    
    # Generate probe points
    probe_points = []
    for i_y in range(44):
        for i_x in range(44):
            x_coord = (i_x + 0.5) / 44.0
            y_coord = (i_y + 0.5) / 44.0
            probe_points.append((x_coord, y_coord))
    
    # Evaluate solution at probe points
    results = []
    for (px, py) in probe_points:
        # Use dolfinx's evaluation utilities
        x_arr = np.array([[px, py]], dtype=np.float64)
        
        # Interpolate at point
        ux_val = u_sol.eval(x_arr)[0][0]
        uy_val = u_sol.eval(x_arr)[0][1]
        
        results.append((px, py, ux_val, uy_val))
    
    # Write CSV file
    with open(f"solution_level{level_idx}.csv", "w") as csvfile:
        csvfile.write("x,y,ux,uy\n")
        for (px, py, ux_val, uy_val) in results:
            csvfile.write(f"{px:.11e},{py:.11e},{ux_val:.11e},{uy_val:.11e}\n")
    
    print(f"Wrote solution_level{level_idx}.csv with {len(results)} probe points")
    
    return u_sol, p_sol, ndof

if __name__ == "__main__":
    # Mesh levels
    N_values = [8, 16, 32]
    
    all_ndofs = []
    all_solutions = []
    
    for idx, N in enumerate(N_values, start=1):
        u_sol, p_sol, ndof = solve_level(N, idx)
        all_ndofs.append(ndof)
        all_solutions.append(u_sol)
    
    print("\n=== Summary ===")
    print(f"LEVELS = {len(N_values)}")
    print(f"FILES = solution_level1.csv,solution_level2.csv,solution_level3.csv")
    
    # Check convergence by comparing solutions at probe points
    # Read back the CSV files and compare
    def read_solution(filename):
        data = []
        with open(filename, 'r') as f:
            header = f.readline()  # Skip header
            for line in f:
                parts = line.strip().split(',')
                data.append([float(p) for p in parts])
        return np.array(data)
    
    sol1 = read_solution("solution_level1.csv")
    sol2 = read_solution("solution_level2.csv")
    sol3 = read_solution("solution_level3.csv")
    
    # Compare level 2 vs level 3 (two finest levels)
    # Relative change = |sol3 - sol2| / |sol2|
    ux2 = sol2[:, 2]
    uy2 = sol2[:, 3]
    ux3 = sol3[:, 2]
    uy3 = sol3[:, 3]
    
    # Avoid division by zero
    mag2 = np.sqrt(ux2**2 + uy2**2)
    mag3 = np.sqrt(ux3**2 + uy3**2)
    
    # Compute relative changes
    rel_change_ux = np.abs(ux3 - ux2) / (np.abs(ux2) + 1e-30)
    rel_change_uy = np.abs(uy3 - uy2) / (np.abs(uy2) + 1e-30)
    
    max_rel_change = np.max(np.maximum(rel_change_ux, rel_change_uy))
    
    print(f"MAX_REL_CHANGE = {max_rel_change:.11e}")
    
    # Determine convergence (heuristic: if max relative change < 1%, consider converged)
    # For a proper FEM with P2 elements, we expect O(h^2) convergence
    # Let's also check level 1 vs level 2
    ux1 = sol1[:, 2]
    uy1 = sol1[:, 3]
    
    rel_change_ux_12 = np.abs(ux2 - ux1) / (np.abs(ux1) + 1e-30)
    rel_change_uy_12 = np.abs(uy2 - uy1) / (np.abs(uy1) + 1e-30)
    max_rel_change_12 = np.max(np.maximum(rel_change_ux_12, rel_change_uy_12))
    
    print(f"Max relative change level 1->2: {max_rel_change_12:.11e}")
    print(f"Max relative change level 2->3: {max_rel_change:.11e}")
    
    # Convergence criterion: relative change should decrease with refinement
    # For quadratic elements, error ~ h^2, so halving h should reduce error by ~4x
    if max_rel_change < max_rel_change_12 and max_rel_change < 0.01:
        mesh_independence = "CONVERGED"
    else:
        mesh_independence = "NOT_CONVERGED"
    
    print(f"MESH_INDEPENDENCE = {mesh_independence}")
    
    # Write RESULT.txt
    with open("RESULT.txt", "w") as result_file:
        result_file.write(f"LEVELS = {len(N_values)}\n")
        result_file.write("FILES = solution_level1.csv,solution_level2.csv,solution_level3.csv\n")
        result_file.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        result_file.write(f"MAX_REL_CHANGE = {max_rel_change:.11e}\n")
    
    print("\nWrote RESULT.txt")
