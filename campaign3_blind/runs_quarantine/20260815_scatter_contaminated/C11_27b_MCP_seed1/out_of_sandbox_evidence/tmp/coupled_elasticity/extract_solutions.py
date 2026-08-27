#!/usr/bin/env python3
"""Extract solutions at probe points for all levels."""
import json
import numpy as np
from pathlib import Path
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters

WORK_DIR = Path("/tmp/coupled_elasticity")

# Problem parameters
X0_A, X1_A = 0.0, 0.625
X0_B, X1_B = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625

E_A, NU_A = 600.0, 1.0/3.0
E_B, NU_B = 2100.0, 1.0/6.0

MESH_LEVELS = [
    {"h": 1/8, "nx_A": 5, "ny_A": 8, "nx_B": 7, "ny_B": 8},
    {"h": 1/16, "nx_A": 10, "ny_A": 16, "nx_B": 14, "ny_B": 16},
    {"h": 1/32, "nx_A": 20, "ny_A": 32, "nx_B": 28, "ny_B": 32},
]


def generate_probe_points_A():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.0 + (i_x + 0.5) * 0.625 / 44
            y = 0.0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)


def generate_probe_points_B():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0.0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)


PROBE_A = generate_probe_points_A()
PROBE_B = generate_probe_points_B()


def body_force_x_A(x, y):
    return (3*x**2*y**3/500 + 63*x**2*y**2/10000 - 117*x**2*y/10000 
            + 9*x**2/4000 - 97*x*y**3/5000 - 2037*x*y**2/100000 
            + 3783*x*y/100000 - 291*x/40000 + 3*y**5/1250 
            + 21*y**4/5000 - 69*y**3/5000 + 1089*y**2/100000 
            - 351*y/100000 + 27/40000)


def body_force_y_A(x, y):
    return (-18*x**2*y**2/625 - 63*x**2*y/3125 + 117*x**2/6250
            + 9*x*y**4/1000 + 63*x*y**3/5000 - 279*x*y**2/10000
            + 927*x*y/50000 - 117*x/25000 - 99*y**4/20000
            - 693*y**3/100000 + 3861*y**2/200000 - 297*y/40000)


def body_force_x_B(x, y):
    return (-3*x**2*y**3/6125 - 9*x**2*y**2/17500 + 117*x**2*y/122500 
            - 9*x**2/49000 + 1907*x*y**3/140000 + 5721*x*y**2/400000 
            - 74373*x*y/2800000 + 5721*x/1120000 - 3*y**5/24500 
            - 3*y**4/14000 - 168333*y**3/15680000 
            - 3941073*y**2/313600000 + 7051707*y/313600000 
            - 542439/125440000)


def body_force_y_B(x, y):
    return (8829*x**2*y**2/392000 + 8829*x**2*y/560000 - 114777*x**2/7840000
            - 9*x*y**4/24500 - 9*x*y**3/17500 - 425601*x*y**2/15680000
            - 644571*x*y/31360000 + 1164969*x/62720000
            + 6333*y**4/1960000 + 6333*y**3/1400000
            - 3201021*y**2/156800000 - 39051*y/62720000
            + 127413/25088000)


def interpolate_to_probes_simple(mesh, basis, x, probes):
    """Simple interpolation using nearest node for tensor mesh."""
    px_mesh, py_mesh = mesh.p[0], mesh.p[1]
    nodal_dofs = basis.nodal_dofs
    
    results = []
    for px, py in probes:
        # Find nearest node
        dist_x = np.abs(px_mesh - px)
        dist_y = np.abs(py_mesh - py)
        min_idx_x = np.argmin(dist_x)
        min_idx_y = np.argmin(dist_y)
        
        # For tensor mesh, find the node index
        # Nodes are ordered by y then x in init_tensor
        # Actually let's just use bilinear interpolation
        
        # Find element containing this point
        dx = (mesh.p[0].max() - mesh.p[0].min()) / (len(np.unique(mesh.p[0])) - 1)
        dy = (mesh.p[1].max() - mesh.p[1].min()) / (len(np.unique(mesh.p[1])) - 1)
        
        # Element indices
        ix = int((px - mesh.p[0].min()) / dx)
        iy = int((py - mesh.p[1].min()) / dy)
        
        # Clamp to valid range
        nx = len(np.unique(mesh.p[0])) - 1
        ny = len(np.unique(mesh.p[1])) - 1
        ix = max(0, min(ix, nx - 1))
        iy = max(0, min(iy, ny - 1))
        
        # Local coordinates in element
        xi = (px - mesh.p[0].min() - ix * dx) / dx
        eta = (py - mesh.p[1].min() - iy * dy) / dy
        
        # Bilinear shape functions
        N = np.array([
            (1-xi)*(1-eta), (1-xi)*eta,
            xi*(1-eta), xi*eta
        ])
        
        # Node indices for this element
        nodes = [iy * nx + ix, iy * nx + ix + 1,
                 (iy+1) * nx + ix, (iy+1) * nx + ix + 1]
        
        # Interpolate ux and uy
        ux = sum(N[j] * x[nodal_dofs[0, nodes[j]]] for j in range(4))
        uy = sum(N[j] * x[nodal_dofs[1, nodes[j]]] for j in range(4))
        
        results.append([px, py, ux, uy])
    
    return np.array(results)


def main():
    print("Extracting solutions at probe points...")
    
    all_solutions_A = []
    all_solutions_B = []
    
    for level_idx, mesh in enumerate(MESH_LEVELS):
        k = level_idx + 1
        print(f"\nLevel {k}: h={mesh['h']:.4f}")
        
        # Subdomain A
        xs_A = np.linspace(X0_A, X1_A, mesh["nx_A"] + 1)
        ys_A = np.linspace(Y0, Y1, mesh["ny_A"] + 1)
        mesh_A = MeshQuad.init_tensor(xs_A, ys_A)
        
        elem_A = ElementVector(ElementQuad1())
        basis_A = Basis(mesh_A, elem_A)
        
        lam_A, mu_A = lame_parameters(E_A, NU_A)
        A_A = linear_elasticity(lam_A, mu_A).assemble(basis_A)
        
        @LinearForm
        def load_A(v, w):
            fx = body_force_x_A(w.x[0], w.x[1])
            fy = body_force_y_A(w.x[0], w.x[1])
            return fx * v[0] + fy * v[1]
        
        b_A = load_A.assemble(basis_A)
        
        tol_A = 1e-9 * max(X1_A - X0_A, Y1 - Y0)
        px_A, py_A = mesh_A.p[0], mesh_A.p[1]
        nodal_dofs_A = basis_A.nodal_dofs
        
        outer_mask_A = (
            (np.abs(px_A - X0_A) < tol_A) |
            (np.abs(px_A - X1_A) < tol_A) |
            (np.abs(py_A - Y0) < tol_A) |
            (np.abs(py_A - Y1) < tol_A)
        )
        outer_nodes_A = np.flatnonzero(outer_mask_A)
        outer_dofs_A = np.concatenate([nodal_dofs_A[0, outer_nodes_A], nodal_dofs_A[1, outer_nodes_A]])
        
        x_A = solve(*condense(A_A, b_A, D=outer_dofs_A))
        
        sol_A = interpolate_to_probes_simple(mesh_A, basis_A, x_A, PROBE_A)
        all_solutions_A.append(sol_A)
        
        with open(WORK_DIR / f"solution_level{k}_A.csv", "w") as f:
            f.write("x,y,ux,uy\n")
            for row in sol_A:
                f.write(f"{row[0]:.17g},{row[1]:.17g},{row[2]:.17g},{row[3]:.17g}\n")
        
        print(f"  Wrote solution_level{k}_A.csv ({len(sol_A)} points)")
        
        # Subdomain B
        xs_B = np.linspace(X0_B, X1_B, mesh["nx_B"] + 1)
        ys_B = np.linspace(Y0, Y1, mesh["ny_B"] + 1)
        mesh_B = MeshQuad.init_tensor(xs_B, ys_B)
        
        elem_B = ElementVector(ElementQuad1())
        basis_B = Basis(mesh_B, elem_B)
        
        lam_B, mu_B = lame_parameters(E_B, NU_B)
        A_B = linear_elasticity(lam_B, mu_B).assemble(basis_B)
        
        @LinearForm
        def load_B(v, w):
            fx = body_force_x_B(w.x[0], w.x[1])
            fy = body_force_y_B(w.x[0], w.x[1])
            return fx * v[0] + fy * v[1]
        
        b_B = load_B.assemble(basis_B)
        
        tol_B = 1e-9 * max(X1_B - X0_B, Y1 - Y0)
        px_B, py_B = mesh_B.p[0], mesh_B.p[1]
        nodal_dofs_B = basis_B.nodal_dofs
        
        outer_mask_B = (
            (np.abs(px_B - X0_B) < tol_B) |
            (np.abs(px_B - X1_B) < tol_B) |
            (np.abs(py_B - Y0) < tol_B) |
            (np.abs(py_B - Y1) < tol_B)
        )
        outer_nodes_B = np.flatnonzero(outer_mask_B)
        outer_dofs_B = np.concatenate([nodal_dofs_B[0, outer_nodes_B], nodal_dofs_B[1, outer_nodes_B]])
        
        x_B = solve(*condense(A_B, b_B, D=outer_dofs_B))
        
        sol_B = interpolate_to_probes_simple(mesh_B, basis_B, x_B, PROBE_B)
        all_solutions_B.append(sol_B)
        
        with open(WORK_DIR / f"solution_level{k}_B.csv", "w") as f:
            f.write("x,y,ux,uy\n")
            for row in sol_B:
                f.write(f"{row[0]:.17g},{row[1]:.17g},{row[2]:.17g},{row[3]:.17g}\n")
        
        print(f"  Wrote solution_level{k}_B.csv ({len(sol_B)} points)")
        
        # Interface data from coupled run
        dir_A = WORK_DIR / f"level{k}_A"
        dir_B = WORK_DIR / f"level{k}_B"
        
        iface_probes = np.array([[IFACE_X, 1/4 + (i + 0.5) * (1/2) / 44] for i in range(44)])
        
        if (dir_A / "exports.json").exists() and (dir_B / "exports.json").exists():
            exp_A = json.loads((dir_A / "exports.json").read_text())
            exp_B = json.loads((dir_B / "exports.json").read_text())
            
            traction_A = np.array(exp_A["values"])
            coords_A = np.array(exp_A["coordinates"])
            y_A = [c[1] for c in coords_A]
            
            disp_B = np.array(exp_B["values"])
            coords_B = np.array(exp_B["coordinates"])
            y_B = [c[1] for c in coords_B]
            
            tx_interp = np.interp([p[1] for p in iface_probes], y_A, traction_A[:, 0])
            ty_interp = np.interp([p[1] for p in iface_probes], y_A, traction_A[:, 1])
            ux_interp = np.interp([p[1] for p in iface_probes], y_B, disp_B[:, 0])
            uy_interp = np.interp([p[1] for p in iface_probes], y_B, disp_B[:, 1])
            
            with open(WORK_DIR / f"interface_level{k}_A.csv", "w") as f:
                f.write("x,y,ux,uy,tx,ty\n")
                for i, (px, py) in enumerate(iface_probes):
                    f.write(f"{px:.17g},{py:.17g},{ux_interp[i]:.17g},{uy_interp[i]:.17g},{tx_interp[i]:.17g},{ty_interp[i]:.17g}\n")
            
            with open(WORK_DIR / f"interface_level{k}_B.csv", "w") as f:
                f.write("x,y,ux,uy,tx,ty\n")
                for i, (px, py) in enumerate(iface_probes):
                    f.write(f"{px:.17g},{py:.17g},{ux_interp[i]:.17g},{uy_interp[i]:.17g},{-tx_interp[i]:.17g},{-ty_interp[i]:.17g}\n")
            
            print(f"  Wrote interface files for level {k}")
    
    # Check mesh independence
    if len(all_solutions_A) >= 2:
        sol_A_fine = all_solutions_A[-1]
        sol_A_coarse = all_solutions_A[-2]
        
        diff_A = np.linalg.norm(sol_A_fine[:, 2:] - sol_A_coarse[:, 2:])
        norm_A = np.linalg.norm(sol_A_coarse[:, 2:]) + 1e-15
        rel_change_A = diff_A / norm_A
        
        sol_B_fine = all_solutions_B[-1]
        sol_B_coarse = all_solutions_B[-2]
        diff_B = np.linalg.norm(sol_B_fine[:, 2:] - sol_B_coarse[:, 2:])
        norm_B = np.linalg.norm(sol_B_coarse[:, 2:]) + 1e-15
        rel_change_B = diff_B / norm_B
        
        max_rel_change = max(rel_change_A, rel_change_B)
        print(f"\nMesh independence check:")
        print(f"  Subdomain A relative change: {rel_change_A:.6e}")
        print(f"  Subdomain B relative change: {rel_change_B:.6e}")
        print(f"  Max relative change: {max_rel_change:.6e}")
        
        mesh_independence = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
        print(f"  Status: {mesh_independence}")
    else:
        max_rel_change = float('inf')
        mesh_independence = "NOT_CONVERGED"


if __name__ == "__main__":
    main()
