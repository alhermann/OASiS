#!/usr/bin/env python3
"""
FEniCSx (dolfinx) solver for subdomain A in the coupled heat equation problem.
Subdomain A: (0, 0.625) x (0, 1), k=1, c=1
This is the DIRICHLET side of the coupling - receives interface field from B, 
imposes it as Dirichlet BC, returns outward normal flux.
"""

import numpy as np
from mpi4py import MPI
import dolfinx
import basix.ufl
from ufl import dx, ds, grad, inner, dot, lhs, rhs, TestFunction
from dolfinx.fem import functionspace, Function, Constant, form, assemble_matrix, assemble_vector
from dolfinx.mesh import create_rectangle, locate_entities_boundary, CellType
from petsc4py import PETSc
import os
import sys

def main():
    # Parse command line arguments
    if len(sys.argv) < 8:
        print(f"Usage: {sys.argv[0]} <level> <dt> <t_end> <interface_u_file> <flux_output_file> <solution_output_file> <log_file>")
        sys.exit(1)
    
    level = int(sys.argv[1])
    dt = float(sys.argv[2])
    t_end = float(sys.argv[3])
    interface_u_file = sys.argv[4]
    flux_output_file = sys.argv[5]
    solution_output_file = sys.argv[6]
    log_file = sys.argv[7]
    
    # Mesh parameters
    h_values = [1/8, 1/16, 1/32]
    h = h_values[level - 1]
    
    # Subdomain A geometry
    x_max_A = 0.625
    y_max = 1.0
    
    # Number of cells
    nx = int(x_max_A / h + 0.5)
    ny = int(y_max / h + 0.5)
    
    # Create mesh for subdomain A using create_rectangle
    domain_A = ((0.0, 0.0), (x_max_A, y_max))
    mesh = create_rectangle(MPI.COMM_WORLD, domain_A, [nx, ny], cell_type=CellType.triangle, dtype=np.float64)
    
    # Create function space (P1 elements)
    V = functionspace(mesh, ("Lagrange", 1))
    
    # Count DOFs
    ndof = V.dofmap.index_map.size_local * V.dofmap.index_map.block_size
    print(f"NDOF = {ndof}")
    
    # Write log file
    with open(log_file, 'w') as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"Level = {level}\n")
        f.write(f"h = {h}\n")
        f.write(f"dt = {dt}\n")
        f.write(f"t_end = {t_end}\n")
    
    # Material properties for subdomain A
    k_A = 1.0
    
    # Define boundaries
    def left_boundary(x):
        return np.isclose(x[0], 0.0)
    
    def right_boundary(x):
        return np.isclose(x[0], x_max_A)
    
    def bottom_boundary(x):
        return np.isclose(x[1], 0.0)
    
    def top_boundary(x):
        return np.isclose(x[1], y_max)
    
    # Find boundary facets
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    
    # Outer boundary (left, bottom, top) - Dirichlet u=0
    outer_facets_left = locate_entities_boundary(mesh, fdim, left_boundary)
    outer_facets_bottom = locate_entities_boundary(mesh, fdim, bottom_boundary)
    outer_facets_top = locate_entities_boundary(mesh, fdim, top_boundary)
    outer_facets = np.concatenate([outer_facets_left, outer_facets_bottom, outer_facets_top])
    
    # Interface (right boundary of A at x=0.625)
    interface_facets = locate_entities_boundary(mesh, fdim, right_boundary)
    
    # Create Dirichlet BCs for outer boundary
    dofmap = V.dofmap.list
    outer_dofs = []
    for facet in outer_facets:
        dofs = dofmap[facet]
        outer_dofs.extend(dofs)
    outer_dofs = np.unique(outer_dofs).astype(np.int32)
    
    # Read interface u values from file
    interface_u_values = []
    if os.path.exists(interface_u_file):
        with open(interface_u_file, 'r') as f:
            header = f.readline()
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    interface_u_values.append(float(parts[2]))
    
    # Create Dirichlet BC for outer boundary only (u=0)
    bcs = []
    bc = dolfinx.dirichletbc.ConstantBC(0.0, outer_dofs, V)
    bcs.append(bc)
    
    # Test and trial functions
    u = Function(V, name="u")
    v = TestFunction(V)
    
    # Source term for subdomain A
    def source_A(x, t_val):
        result = (-384*t_val*x[0]**3*x[1]**3 + 1184*t_val*x[0]**3*x[1]**2 + 3808*t_val*x[0]**3*x[1] - 4736*t_val*x[0]**3 
                 - 316*t_val*x[0]**2*x[1]**3 - 199*t_val*x[0]**2*x[1]**2 + 4307*t_val*x[0]**2*x[1] + 796*t_val*x[0]**2 
                 + 5148*t_val*x[0]*x[1]**3 - 14883*t_val*x[0]*x[1]**2 + 3255*t_val*x[0]*x[1] + 2700*t_val*x[0] 
                 + 1264*t_val*x[1]**3 + 796*t_val*x[1]**2 - 2060*t_val*x[1] 
                 - 768*x[0]**3*x[1]**3 + 2368*x[0]**3*x[1]**2 - 1600*x[0]**3*x[1] 
                 - 632*x[0]**2*x[1]**3 - 398*x[0]**2*x[1]**2 + 1030*x[0]**2*x[1] 
                 + 1080*x[0]*x[1]**3 - 1350*x[0]*x[1]**2 + 270*x[0]*x[1]) * np.exp(t_val/2) / 1280
        return result
    
    # Time stepping
    n_steps = int(t_end / dt + 0.5)
    
    # Initial condition: u = 0
    u.x.array[:] = 0.0
    
    # Pre-assemble LHS matrix
    # Crank-Nicolson: u*v - (dt/2)*k*grad(u)*grad(v)
    a_form = form(inner(u, v) * dx - (dt/2) * k_A * inner(grad(u), grad(v)) * dx)
    A_mat = assemble_matrix(a_form, bcs=bcs)
    A_mat.assemble()
    
    # Store previous solution
    u_prev = Function(V)
    u_prev.x.array[:] = 0.0
    
    # Run time steps
    for step in range(n_steps):
        t_half = (step + 0.5) * dt
        
        # Copy current solution to previous
        u_prev.x.array[:] = u.x.array[:]
        
        # Compute source at all quadrature points
        x_coords = mesh.geometry.x
        source_vals = source_A(x_coords, t_half)
        
        # Create source function
        source_func = Function(V)
        source_func.x.array[:] = source_vals
        
        # RHS form: u_prev*v + (dt/2)*k*grad(u_prev)*grad(v) + dt*f*v
        rhs_form = form(inner(u_prev, v) * dx + (dt/2) * k_A * inner(grad(u_prev), grad(v)) * dx + dt * inner(source_func, v) * dx)
        
        # Assemble RHS
        b = assemble_vector(rhs_form)
        dolfinx.fem.petsc.apply_lifting(b, [rhs_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b, bcs)
        
        # Solve linear system
        ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        ksp.setOperators(A_mat)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")
        ksp.solve(b, u.x.vec)
    
    print(f"FEniCSx solver completed for level {level}, final time = {t_end}")
    
    # Evaluate solution at probe points for subdomain A
    # Probe points: x = 0 + (i_x+0.5)*0.625/44; y = 0 + (i_y+0.5)*1/44
    probe_points_A = []
    for i_x in range(44):
        for i_y in range(44):
            x_probe = (i_x + 0.5) * 0.625 / 44
            y_probe = (i_y + 0.5) * 1 / 44
            probe_points_A.append((x_probe, y_probe))
    
    # Interpolate solution at probe points using cell search
    probe_values = []
    cell_vertices = mesh.topology.connectivity[mesh.topology.dim, 0]
    
    for (px, py) in probe_points_A:
        found = False
        val = 0.0
        
        for cell_idx in range(mesh.topology.index_map[mesh.topology.dim].size_local):
            vertices = cell_vertices[cell_idx]
            verts_coords = mesh.geometry.x[vertices]
            
            # Check bounding box
            if (np.min(verts_coords[:, 0]) <= px <= np.max(verts_coords[:, 0]) and
                np.min(verts_coords[:, 1]) <= py <= np.max(verts_coords[:, 1])):
                
                v0, v1, v2 = verts_coords
                denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
                
                if abs(denom) > 1e-12:
                    a = ((v1[1] - v2[1]) * (px - v2[0]) + (v2[0] - v1[0]) * (py - v2[1])) / denom
                    b = ((v2[1] - v0[1]) * (px - v2[0]) + (v0[0] - v2[0]) * (py - v2[1])) / denom
                    c = 1 - a - b
                    
                    if -1e-6 <= a <= 1+1e-6 and -1e-6 <= b <= 1+1e-6 and -1e-6 <= c <= 1+1e-6:
                        dofs = dofmap[cell_idx]
                        val = a * u.x.array[dofs[0]] + b * u.x.array[dofs[1]] + c * u.x.array[dofs[2]]
                        found = True
                        break
        
        probe_values.append(val)
    
    # Write solution CSV
    with open(solution_output_file, 'w') as f:
        f.write("x, y, u\n")
        for i, ((px, py), val) in enumerate(zip(probe_points_A, probe_values)):
            f.write(f"{px:.16e}, {py:.16e}, {val:.16e}\n")
    
    # Compute and write interface flux
    # Interface probe points: x = 5/8, y = 1/4 + (i+0.5)*1/2/44
    interface_probes = []
    for i in range(44):
        y_probe = 0.25 + (i + 0.5) * 0.5 / 44
        interface_probes.append((0.625, y_probe))
    
    interface_u_values_out = []
    interface_fluxes = []
    
    for (px, py) in interface_probes:
        found = False
        u_val = 0.0
        qn = 0.0
        
        for cell_idx in range(mesh.topology.index_map[mesh.topology.dim].size_local):
            vertices = cell_vertices[cell_idx]
            verts_coords = mesh.geometry.x[vertices]
            dofs = dofmap[cell_idx]
            
            # Check if any vertex is at x = 0.625
            if np.any(np.isclose(verts_coords[:, 0], 0.625)):
                if np.min(verts_coords[:, 1]) <= py <= np.max(verts_coords[:, 1]):
                    v0, v1, v2 = verts_coords
                    u0, u1, u2 = u.x.array[dofs]
                    
                    denom = v0[0]*(v1[1]-v2[1]) + v1[0]*(v2[1]-v0[1]) + v2[0]*(v0[1]-v1[1])
                    
                    if abs(denom) > 1e-12:
                        dudx = (u0*(v1[1]-v2[1]) + u1*(v2[1]-v0[1]) + u2*(v0[1]-v1[1])) / denom
                        
                        # Outward normal from A is (1, 0)
                        # qn = -k * grad(u) . n = -k * dudx
                        qn = -k_A * dudx
                        
                        # Interpolate u at interface point
                        iface_verts = []
                        for j, vc in enumerate(verts_coords):
                            if np.isclose(vc[0], 0.625):
                                iface_verts.append((vc, u.x.array[dofs[j]]))
                        
                        if len(iface_verts) == 2:
                            dy_edge = iface_verts[1][0][1] - iface_verts[0][0][1]
                            if abs(dy_edge) > 1e-12:
                                t_param = (py - iface_verts[0][0][1]) / dy_edge
                                u_val = iface_verts[0][1] + t_param * (iface_verts[1][1] - iface_verts[0][1])
                            else:
                                u_val = iface_verts[0][1]
                        
                        found = True
                        break
        
        interface_u_values_out.append(u_val)
        interface_fluxes.append(qn)
    
    # Write interface CSV
    with open(flux_output_file, 'w') as f:
        f.write("x, y, u, qn\n")
        for i, ((px, py), u_val, qn) in enumerate(zip(interface_probes, interface_u_values_out, interface_fluxes)):
            f.write(f"{px:.16e}, {py:.16e}, {u_val:.16e}, {qn:.16e}\n")
    
    print(f"FEniCSx solver finished. Output written to {solution_output_file} and {flux_output_file}")

if __name__ == "__main__":
    main()
