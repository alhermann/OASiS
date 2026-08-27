"""
DUNE-fem solver for -div(K grad u) = f on unit square
K = [[5/2, 0], [0, 1]] (constant diagonal tensor)
f(x,y) = complex polynomial
u = 0 on entire boundary
Continuous Lagrange order 1 on triangles
Mesh levels: N = 8, 16, 32, 64 cells per side
"""

from dune.alugrid import aluConformGrid, cartesianDomain
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC, SpatialCoordinate
from ufl import TrialFunction, TestFunction, inner, grad, dx, as_tensor
import numpy as np
import json
import os

# Problem parameters
K_11 = 5.0 / 2.0  # 2.5
K_22 = 1.0

def source_term(x, y):
    """f(x, y) = 9*x**3*y/2 - 17*x**3/6 + 15*x**2*y/2 + 41*x**2/6 + 
                45*x*y**3/4 - 85*x*y**2/4 - 2*x*y - 4*x + 
                25*y**3/4 + 205*y**2/12 - 70*y/3"""
    return (9*x**3*y/2 - 17*x**3/6 + 15*x**2*y/2 + 41*x**2/6 + 
            45*x*y**3/4 - 85*x*y**2/4 - 2*x*y - 4*x + 
            25*y**3/4 + 205*y**2/12 - 70*y/3)

def generate_probe_points():
    """Generate 1936 probe points: x=(i_x+0.5)/44, y=(i_y+0.5)/44
    Ordered with last index varying fastest (i_y varies fastest)"""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            points.append([x, y])
    return points

def solve_level(N, level_idx, probe_points):
    """Solve at mesh level with N cells per side"""
    
    print(f"\n{'='*60}")
    print(f"Level {level_idx}: N = {N} cells per side")
    print(f"{'='*60}")
    
    # Create triangular mesh using ALUGrid
    domain = cartesianDomain([0.0, 0.0], [1.0, 1.0])
    gridView = aluConformGrid(domain, [N, N])
    
    # Count elements to verify triangles
    n_elements = sum(1 for _ in gridView.elements)
    print(f"Number of elements: {n_elements} (expected ~{2*N*N} triangles)")
    
    # Create finite element space: continuous Lagrange order 1
    space = lagrange(gridView, order=1)
    ndof = space.size
    print(f"NDOF = {ndof}")
    
    # Write log file
    with open(f"run_level{level_idx}.log", "w") as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"Elements = {n_elements}\n")
        f.write(f"N = {N} cells per side\n")
    
    # Define trial and test functions
    u = TrialFunction(space)
    v = TestFunction(space)
    
    # Define spatial coordinates for source term
    x_coord = SpatialCoordinate(space)
    
    # Define tensor K (diagonal)
    K = as_tensor([[K_11, 0.0], [0.0, K_22]])
    
    # Weak form: a(u,v) = integral(K * grad(u) . grad(v)) dx
    # For -div(K grad u) = f, we have a(u,v) = int K grad u . grad v
    a = inner(K * grad(u), grad(v)) * dx
    
    # Source term: b(v) = integral(f * v) dx
    f_expr = source_term(x_coord[0], x_coord[1])
    b = f_expr * v * dx
    
    # Dirichlet BC: u = 0 on entire boundary (no indicator = whole boundary)
    dbc = DirichletBC(space, 0)
    
    # Create scheme - MUST include dbc in the list!
    scheme = galerkin([a == b, dbc], solver="cg")
    
    # Initial guess
    uh = space.interpolate(0, name="solution")
    
    # Solve
    info = scheme.solve(target=uh)
    print(f"Solver converged: {info.get('converged', 'unknown')}")
    if 'linear_iterations' in info:
        print(f"Linear iterations: {info['linear_iterations']}")
    
    # Check convergence
    if not info.get('converged', False):
        raise RuntimeError(f"Solver did not converge at level {level_idx}")
    
    # Extract solution values
    sol_values = np.array(uh.as_numpy)
    print(f"Solution range: [{sol_values.min():.6e}, {sol_values.max():.6e}]")
    
    # Evaluate at probe points using pointSample from dune.fem.utility
    from dune.fem.utility import pointSample
    
    probe_results = []
    for pt in probe_points:
        # pointSample returns the value at a single point
        val = float(pointSample(uh, pt))
        probe_results.append((pt[0], pt[1], val))
    
    # Write CSV file
    csv_filename = f"solution_level{level_idx}.csv"
    with open(csv_filename, "w") as f:
        f.write("x,y,u\n")
        for (px, py, val) in probe_results:
            f.write(f"{px:.15e},{py:.15e},{val:.15e}\n")
    
    print(f"Wrote {csv_filename} with {len(probe_results)} probe points")
    
    # Also write VTK for visualization
    vtk_name = f"result_level{level_idx}"
    gridView.writeVTK(vtk_name, pointdata={"u": uh})
    print(f"Wrote {vtk_name}.vtu")
    
    return ndof, probe_results, csv_filename

def main():
    """Main driver: solve at all mesh levels"""
    
    # Mesh levels
    N_values = [8, 16, 32, 64]
    
    # Generate probe points once
    probe_points = generate_probe_points()
    print(f"Generated {len(probe_points)} probe points")
    
    # Store results for convergence analysis
    all_ndofs = []
    all_csv_files = []
    all_probe_results = []
    
    # Solve at each level
    for idx, N in enumerate(N_values, start=1):
        ndof, probe_results, csv_file = solve_level(N, idx, probe_points)
        all_ndofs.append(ndof)
        all_csv_files.append(csv_file)
        all_probe_results.append(probe_results)
    
    # Analyze convergence between finest two levels
    # Compare level 3 (N=32) and level 4 (N=64)
    results_coarse = all_probe_results[-2]  # Level 3
    results_fine = all_probe_results[-1]    # Level 4
    
    max_rel_change = 0.0
    for (xc, yc, uc), (xf, yf, uf) in zip(results_coarse, results_fine):
        if abs(uf) > 1e-15:
            rel_change = abs(uc - uf) / abs(uf)
        else:
            rel_change = abs(uc - uf)
        if rel_change > max_rel_change:
            max_rel_change = rel_change
    
    print(f"\n{'='*60}")
    print(f"CONVERGENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Max relative change between levels 3 and 4: {max_rel_change:.6e}")
    
    # Determine mesh independence
    # Using a threshold of 1% (0.01) for convergence
    tol = 0.01
    mesh_independence = "CONVERGED" if max_rel_change < tol else "NOT_CONVERGED"
    print(f"MESH_INDEPENDENCE = {mesh_independence}")
    
    # Write RESULT.txt
    with open("RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(N_values)}\n")
        f.write(f"FILES = {','.join(all_csv_files)}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\nWrote RESULT.txt")
    print("Simulation complete!")

if __name__ == "__main__":
    main()
