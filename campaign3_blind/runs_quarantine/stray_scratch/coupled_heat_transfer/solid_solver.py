#!/usr/bin/env python3
"""
FEniCSx (dolfinx) solver for the solid side (side A) of the conjugate heat transfer problem.

Geometry: rectangle (0, 0.001) x (0, 0.0006) meters
Equation: -div(ks grad T) = 0 (steady heat conduction)
ks = 0.02 W/(m K)

Boundary conditions:
- x = 0: T = 400 K (Dirichlet)
- y = 0, y = 0.0006: insulated (natural)
- x = 0.001: receives gas heat flux as Neumann BC (from coupling)

Discretization: Lagrange P1 elements on uniform mesh with at least 40 elements in x and 24 in y.
"""

import numpy as np
from mpi4py import MPI
import dolfinx
from dolfinx import fem, io, mesh, default_scalar_type
from dolfinx.fem import functionspace, Function, dirichletbc, Constant
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import create_unit_square, locate_entities_boundary
from ufl import dx, ds, dot, grad, inner, TestFunction, TrialFunction, FacetNormal, conditional, gt

def main():
    # Read command line arguments
    import sys
    
    if len(sys.argv) < 5:
        print(f"Usage: {sys.argv[0]} <flux_W_per_m2> <output_temp_file> <log_file> <iteration>")
        sys.exit(1)
    
    flux_value = float(sys.argv[1])  # Heat flux from gas (W/m^2), positive = into solid
    output_temp_file = sys.argv[2]   # File to write interface temperatures
    log_file = sys.argv[3]           # Log file
    iteration = int(sys.argv[4])     # Coupling iteration number
    
    # Physical parameters
    ks = 0.02  # Thermal conductivity W/(m K)
    T_left = 400.0  # Temperature at x=0 (K)
    
    # Geometry
    x_min, x_max = 0.0, 0.001  # Solid domain in x
    y_min, y_max = 0.0, 0.0006  # Solid domain in y
    
    # Mesh parameters (at least 40 in x, 24 in y)
    nx = 40
    ny = 24
    
    # Create unit square mesh and scale it
    L_x = x_max - x_min
    L_y = y_max - y_min
    
    # Create unit square and then transform coordinates
    mesh_obj = create_unit_square(MPI.COMM_WORLD, nx, ny, dolfinx.mesh.CellType.triangle)
    
    # Scale the mesh coordinates
    x_coords = mesh_obj.geometry.x
    x_coords[:, 0] = x_min + x_coords[:, 0] * L_x
    x_coords[:, 1] = y_min + x_coords[:, 1] * L_y
    
    # Number of degrees of freedom
    V = functionspace(mesh_obj, ("Lagrange", 1))
    ndof = V.dofmap.index_map.size_global
    
    # Write log file
    with open(log_file, 'w') as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"Iteration = {iteration}\n")
        f.write(f"Heat flux from gas = {flux_value} W/m^2\n")
        f.write(f"Mesh: {nx} x {ny} elements\n")
        f.write(f"Total DOFs = {ndof}\n")
    
    # Define function space
    u = Function(V)  # Solution (temperature)
    v = TestFunction(V)
    du = TrialFunction(V)
    
    # Define thermal conductivity (scalar constant)
    k = Constant(mesh_obj, ks)
    
    # Get DOF coordinates
    dof_coords = V.tabulate_dof_coordinates()
    
    # Find DOFs on left boundary (x = 0) for Dirichlet BC
    tol = 1e-12
    left_dofs = np.array([i for i in range(dof_coords.shape[0]) 
                          if np.abs(dof_coords[i, 0] - x_min) < tol], dtype=np.int32)
    
    # Apply Dirichlet BC on left boundary
    bc = dirichletbc(T_left, left_dofs, V)
    
    # Flux constant
    flux = Constant(mesh_obj, flux_value)
    
    # Use normal vector to restrict flux to right boundary only
    n = FacetNormal(mesh_obj)
    
    # Define the variational form
    # Weak form: integral(k grad T . grad v dx) = integral(flux * v ds_right)
    # We use conditional to restrict to right boundary where n[0] > 0
    a = inner(k * grad(du), grad(v)) * dx
    L = flux * v * conditional(gt(n[0], 0), 1, 0) * ds
    
    # Solve the linear system
    problem = LinearProblem(a, L, bcs=[bc], petsc_options_prefix="solid_heat")
    u = problem.solve()
    
    # Extract interface temperatures (at x = x_max)
    # Find DOFs on right boundary
    right_dofs = np.array([i for i in range(dof_coords.shape[0]) 
                           if np.abs(dof_coords[i, 0] - x_max) < tol], dtype=np.int32)
    
    # Get the temperature values at interface DOFs
    u_values = u.x.array
    interface_temps = u_values[right_dofs]
    
    # Compute mean interface temperature
    mean_interface_temp = np.mean(interface_temps)
    
    # Write interface temperatures to output file
    with open(output_temp_file, 'w') as f:
        f.write(f"{mean_interface_temp}\n")
        for t in interface_temps:
            f.write(f"{t}\n")
    
    # Compute total heat flux through the solid
    # In steady state 1D conduction with constant k:
    # Flux = -k * dT/dx = -k * (T_interface - T_left) / L
    computed_flux = -ks * (mean_interface_temp - T_left) / L_x
    
    # Update log file with results
    with open(log_file, 'a') as f:
        f.write(f"Mean interface temperature = {mean_interface_temp} K\n")
        f.write(f"Computed heat flux through solid = {computed_flux} W/m^2\n")
    
    # Print summary
    print(f"Iteration {iteration}: Mean interface temp = {mean_interface_temp:.6f} K, Flux = {computed_flux:.6f} W/m^2")

if __name__ == "__main__":
    main()
