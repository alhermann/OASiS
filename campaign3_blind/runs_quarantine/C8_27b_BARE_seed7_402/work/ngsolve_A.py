#!/usr/bin/env python3
"""
NGSolve solver for subdomain A (Dirichlet side in Dirichlet-Neumann coupling).
Subdomain A: (0, 0.625) x (0, 1), k = 1
Interface at x = 0.625 receives field from partner and returns flux.
"""
import sys
import json
from pathlib import Path

# Add venv to path
sys.path.insert(0, '/home/alexander/Schreibtisch/open-fem-agent/.venv/lib/python3.12/site-packages')

import ngsolve as ng
from ngsolve import *
import numpy as np


def main():
    if len(sys.argv) < 3:
        print("Usage: python ngsolve_A.py <level> <work_dir>")
        sys.exit(1)
    
    level = int(sys.argv[1])
    work_dir = sys.argv[2]
    
    # Domain parameters for subdomain A
    x_max_A = 0.625
    y_max = 1.0
    k_A = 1.0
    
    # Mesh size
    h = 1.0 / (8 * level)
    nx = int(x_max_A / h + 0.5)
    ny = int(y_max / h + 0.5)
    
    print(f"NGSolve A: Creating mesh with nx={nx}, ny={ny}")
    
    # Create mesh
    mesh = Mesh()
    mesh.Generate("rect", nx=nx, ny=ny, xmin=0, xmax=x_max_A, ymin=0, ymax=y_max)
    
    # Finite element space - P1 elements
    V = H1(mesh, order=1)
    u = V.TrialFunction()
    v = V.TestFunction()
    
    # Source term f(x,y) for subdomain A
    def source_A(x, y):
        return (-18*x**3*y/5 + 2*x**3/5 + 5063*x**2*y/20000 - 95021*x**2/60000 
                - 18*x*y**3/5 + 6*x*y**2/5 + 14607*x*y/4000 + 1669*x/2000 
                + 5063*y**3/60000 - 95021*y**2/60000 + 14993*y/10000)
    
    # Bilinear form
    a = BilinearForm(V)
    a += k_A * InnerProduct(grad(u), grad(v)) * dx
    
    # Linear form
    f_form = LinearForm(V)
    f_form += source_A(x, y) * v * dx
    
    a.Assemble()
    f_form.Assemble()
    
    gfu = GridFunction(V)
    ndof = V.GetNDof()
    
    print(f"NGSolve A: ndof={ndof}")
    
    # Identify boundary DOFs
    tol = 1e-10
    outer_dofs = []
    interface_dofs = []
    
    for vd in mesh.Vertices():
        x_coord = vd.x
        y_coord = vd.y
        
        try:
            dof = V.DofNumber(vd.nr)
        except:
            continue
            
        if abs(x_coord) < tol or abs(y_coord) < tol or abs(y_coord - y_max) < tol:
            outer_dofs.append(dof)
        elif abs(x_coord - x_max_A) < tol:
            interface_dofs.append(dof)
    
    outer_dofs = sorted(set(outer_dofs))
    
    # Sort interface_dofs by y-coordinate
    interface_nodes = []
    for dof in interface_dofs:
        for vd in mesh.Vertices():
            try:
                if V.DofNumber(vd.nr) == dof:
                    interface_nodes.append((vd.y, dof))
                    break
            except:
                pass
    
    interface_nodes.sort(key=lambda x: x[0])
    interface_dofs_sorted = [x[1] for x in interface_nodes]
    interface_y_coords = [x[0] for x in interface_nodes]
    
    print(f"NGSolve A: {len(interface_dofs_sorted)} interface DOFs")
    
    # Read imports.json for interface data
    imports_path = Path(work_dir) / "imports.json"
    interface_field_data = None
    if imports_path.exists():
        imports = json.loads(imports_path.read_text())
        if "B_to_A" in imports and "field" in imports["B_to_A"]:
            interface_field_data = imports["B_to_A"]["field"]
    
    # Apply outer boundary conditions (u = 0)
    for dof in outer_dofs:
        gfu.vec[dof] = 0
    
    # Apply interface boundary conditions (from partner)
    if interface_field_data is not None and len(interface_field_data) > 0:
        partner_y = [pt[0] for pt in interface_field_data]
        partner_u = [pt[1] for pt in interface_field_data]
        
        for i, dof in enumerate(interface_dofs_sorted):
            y_pos = interface_y_coords[i]
            u_val = np.interp(y_pos, partner_y, partner_u)
            gfu.vec[dof] = u_val
    
    # Solve with Dirichlet BCs
    mat = a.mat
    rhs = f_form.vec
    
    for dof in outer_dofs + interface_dofs_sorted:
        mat.AddMultiple(dof, dof, -mat(dof, dof))
        for j in range(ndof):
            if j != dof:
                mat.AddMultiple(dof, j, -mat(dof, j))
                mat.AddMultiple(j, dof, -mat(j, dof))
        mat.AddSingle(dof, dof, 1)
        rhs[dof] = gfu.vec[dof]
    
    mat.Inverse().Mult(rhs, gfu.vec)
    
    # Compute interface flux
    # Outward normal from A is +x direction
    # qn = -k * du/dx
    h_local = x_max_A / nx
    
    interface_results = []
    for i, dof in enumerate(interface_dofs_sorted):
        y_pos = interface_y_coords[i]
        u_val = gfu.vec[dof]
        
        # Find interior node
        interior_dof = None
        for vd in mesh.Vertices():
            if abs(vd.x - (x_max_A - h_local)) < h_local/2 and abs(vd.y - y_pos) < h_local/2:
                try:
                    interior_dof = V.DofNumber(vd.nr)
                    break
                except:
                    pass
        
        if interior_dof is not None:
            u_interior = gfu.vec[interior_dof]
            du_dx = (u_val - u_interior) / h_local
        else:
            du_dx = 0
        
        qn = -k_A * du_dx
        interface_results.append([y_pos, u_val, qn])
    
    # Write run log
    log_path = Path(work_dir) / f"run_level{level}_A.log"
    with open(log_path, 'w') as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"Mesh level: {level}\n")
        f.write(f"Elements: {mesh.NumElements()}\n")
    
    # Write exports.json
    exports = {
        "A_to_B": {
            "flux": [[r[0], r[2]] for r in interface_results],  # [y, qn]
            "field": [[r[0], r[1]] for r in interface_results]   # [y, u]
        }
    }
    with open(Path(work_dir) / "exports.json", 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"NGSolve A: level={level}, ndof={ndof}, done")


if __name__ == "__main__":
    main()
