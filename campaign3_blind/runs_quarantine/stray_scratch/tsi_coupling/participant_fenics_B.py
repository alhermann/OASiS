#!/usr/bin/env python3
"""
FEniCSx/dolfinx participant for subdomain B (Neumann side) - coupled thermo-structural
Subdomain B: x in [0.625, 1.5], y in [0, 1]
Material: k=3, lambda=600, mu=1600, beta=1, E=40000/11, nu=3/22
Role: Neumann - receives fluxes/tractions from partner, exports T,u
"""
import json
import os
import sys
import numpy as np
from pathlib import Path

from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem import functionspace, form, dirichletbc, locate_dofs_topological
from dolfinx.mesh import locate_entities_boundary
import ufl

# Problem parameters
X_INTERFACE = 0.625
X_MAX = 1.5
Y_MIN, Y_MAX = 0.0, 1.0
Lx_B = X_MAX - X_INTERFACE
Ly = Y_MAX - Y_MIN

# Material properties for subdomain B
k_B = 3.0
lambda_B = 600.0
mu_B = 1600.0
beta_B = 1.0
E_B = 40000/11
nu_B = 3/22

def get_n_divisions():
    return int(os.environ.get('N_DIVISIONS', 8))

def generate_interface_points(n_interface=44):
    points = []
    for i in range(n_interface):
        y = Y_MIN + (i + 0.5) * Ly / n_interface
        points.append([X_INTERFACE, y])
    return np.array(points)

def source_T_B(x):
    """Source term f_T for subdomain B"""
    val = (-2*x[0]**3*x[1]/3 + 8*x[0]**3/45 - 8*x[0]**2*x[1]/3 + 4*x[0]**2/9 
           - 2*x[0]*x[1]**3/3 + 8*x[0]*x[1]**2/15 + 251*x[0]*x[1]/120 - 41*x[0]/90 
           - 8*x[1]**3/9 + 4*x[1]**2/9 + 829*x[1]/144 - 11/12)
    return val

def main():
    work_dir = Path.cwd()
    
    # Read imports
    imports_path = work_dir / "imports.json"
    if imports_path.exists():
        with open(imports_path) as f:
            imports = json.load(f)
    else:
        imports = {}
    
    # Get imported values or initial guess
    if imports and 'side_A' in imports:
        partner_data = imports['side_A']
        interface_coords = np.array(partner_data['coordinates'])
        imported_qn = np.array(partner_data['normal_fluxes']['qn'])
        imported_tx = np.array(partner_data['normal_fluxes']['tx'])
        imported_ty = np.array(partner_data['normal_fluxes']['ty'])
    else:
        interface_coords = generate_interface_points(44)
        imported_qn = np.zeros(len(interface_coords))
        imported_tx = np.zeros(len(interface_coords))
        imported_ty = np.zeros(len(interface_coords))
    
    n_divisions = get_n_divisions()
    mesh_nx = max(int(Lx_B * n_divisions), 1)
    mesh_ny = max(int(Ly * n_divisions), 1)
    
    # Create mesh
    domain = mesh.create_rectangle(
        MPI.COMM_WORLD,
        [[X_INTERFACE, Y_MIN], [X_MAX, Y_MAX]],
        [mesh_nx, mesh_ny],
        mesh.CellType.quadrilateral
    )
    
    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, tdim-1)
    
    # Function spaces
    V_T = functionspace(domain, ("Lagrange", 1))
    V_u = functionspace(domain, ("Lagrange", 1, (tdim,)))
    
    ndof_T = V_T.dofmap.index_map.size_global * V_T.dofmap.index_map_bs
    ndof_u = V_u.dofmap.index_map.size_global * V_u.dofmap.index_map_bs
    total_ndof = ndof_T + ndof_u
    
    with open(work_dir / "run_log.txt", 'w') as f:
        f.write(f"NDOF = {total_ndof}\n")
        f.write(f"Thermal DOFs: {ndof_T}\n")
        f.write(f"Structural DOFs: {ndof_u}\n")
    
    # Trial/test functions
    T = ufl.TrialFunction(V_T)
    v_T = ufl.TestFunction(V_T)
    u = ufl.TrialFunction(V_u)
    v_u = ufl.TestFunction(V_u)
    
    def epsilon(u):
        return ufl.sym(ufl.grad(u))
    
    def sigma(u, T_val):
        return (lambda_B * ufl.nabla_div(u) * ufl.Identity(tdim) 
                + 2 * mu_B * epsilon(u) 
                - beta_B * T_val * ufl.Identity(tdim))
    
    # Source terms
    f_T = fem.Function(V_T)
    f_T.interpolate(source_T_B)
    
    f_u = fem.Function(V_u)
    f_u.interpolate(lambda x: np.zeros((2, len(x[0])), dtype=default_scalar_type))
    
    # Weak forms
    a_T = ufl.dot(k_B * ufl.grad(T), ufl.grad(v_T)) * ufl.dx
    L_T = f_T * v_T * ufl.dx
    
    a_u = ufl.inner(sigma(u, 0), epsilon(v_u)) * ufl.dx
    L_u = ufl.dot(f_u, v_u) * ufl.dx
    
    # Boundary conditions - outer boundaries only (NOT interface)
    def right_boundary(x):
        return np.isclose(x[0], X_MAX)
    
    def top_boundary(x):
        return np.isclose(x[1], Y_MAX)
    
    def bottom_boundary(x):
        return np.isclose(x[1], Y_MIN)
    
    fdim = tdim - 1
    right_facets = locate_entities_boundary(domain, fdim, right_boundary)
    top_facets = locate_entities_boundary(domain, fdim, top_boundary)
    bottom_facets = locate_entities_boundary(domain, fdim, bottom_boundary)
    outer_facets = np.unique(np.concatenate([right_facets, top_facets, bottom_facets]))
    
    dofs_T = locate_dofs_topological(V_T, fdim, outer_facets)
    bc_T = dirichletbc(np.array([0.0], dtype=default_scalar_type), dofs_T, V_T)
    
    dofs_u = locate_dofs_topological(V_u, fdim, outer_facets)
    bc_u = dirichletbc(np.zeros(2, dtype=default_scalar_type), dofs_u, V_u)
    
    # Solve thermal
    a_T_form = form(a_T)
    L_T_form = form(L_T)
    
    A_T = fem.petsc.assemble_matrix(a_T_form, bcs=[bc_T])
    A_T.assemble()
    b_T = fem.petsc.assemble_vector(L_T_form)
    fem.petsc.apply_lifting(b_T, [a_T_form], [bc_T])
    b_T.ghostUpdate(addv=fem.petsc.InsertMode.add, mode=fem.petsc.InsertMode.insert)
    fem.petsc.set_bc(b_T, [bc_T])
    
    T_sol = fem.Function(V_T)
    fem.petsc.solve(A_T, T_sol.x.vector, b_T, {"ksp_type": "preonly", "pc_type": "lu"})
    
    # Solve structural with temperature coupling
    a_u_updated = ufl.inner(sigma(u, T_sol), epsilon(v_u)) * ufl.dx
    a_u_form = form(a_u_updated)
    L_u_form = form(L_u)
    
    A_u = fem.petsc.assemble_matrix(a_u_form, bcs=[bc_u])
    A_u.assemble()
    b_u = fem.petsc.assemble_vector(L_u_form)
    fem.petsc.apply_lifting(b_u, [a_u_form], [bc_u])
    b_u.ghostUpdate(addv=fem.petsc.InsertMode.add, mode=fem.petsc.InsertMode.insert)
    fem.petsc.set_bc(b_u, [bc_u])
    
    u_sol = fem.Function(V_u)
    fem.petsc.solve(A_u, u_sol.x.vector, b_u, {"ksp_type": "preonly", "pc_type": "lu"})
    
    # Extract at interface
    coords = domain.geometry.x
    T_out, ux_out, uy_out = [], [], []
    
    T_values = T_sol.x.array.reshape(-1, V_T.value_size)
    u_values = u_sol.x.array.reshape(-1, V_u.value_size)
    
    for pt in interface_coords:
        dists = np.linalg.norm(coords - pt, axis=1)
        idx = np.argmin(dists)
        T_out.append(float(T_values[idx, 0]))
        ux_out.append(float(u_values[idx, 0]))
        uy_out.append(float(u_values[idx, 1]))
    
    # Placeholder fluxes/tractions
    n_pts = len(interface_coords)
    qn_out = [0.0] * n_pts
    tx_out = [0.0] * n_pts
    ty_out = [0.0] * n_pts
    
    exports = {
        'field_name': 'thermo_structural_interface',
        'n_points': len(interface_coords),
        'coordinates': interface_coords.tolist(),
        'values': {'T': T_out, 'ux': ux_out, 'uy': uy_out},
        'normal_fluxes': {'qn': qn_out, 'tx': tx_out, 'ty': ty_out}
    }
    
    with open(work_dir / "exports.json", 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"Participant B completed. NDOF={total_ndof}")

if __name__ == "__main__":
    main()
