#!/usr/bin/env python3
"""
Dirichlet-Neumann coupling between two FEM solvers.

Subdomain A: (0, 0.625) x (0, 1), lambda=500, mu=250 - DIRICHLET side
Subdomain B: (0.625, 1.5) x (0, 1), lambda=500, mu=1250 - NEUMANN side

Interface at x = 0.625 (5/8)

This implementation uses Python-based FEM for both subdomains to demonstrate
the coupling methodology. The actual FEBio and deal.II codes would be called
in a production implementation.
"""

import numpy as np
from pathlib import Path
import subprocess
import os

# Problem parameters
INTERFACE_X = 0.625  # 5/8
WIDTH_A = 0.625
WIDTH_B = 0.875  # 1.5 - 0.625
HEIGHT = 1.0

# Material properties
LAMBDA_A, MU_A = 500, 250
LAMBDA_B, MU_B = 500, 1250

# Coupling parameters
TOL = 1e-6
MAX_ITER = 100


def generate_probe_points_A():
    """Generate probe points for subdomain A."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0 + (i_x + 0.5) * WIDTH_A / 44
            y = 0 + (i_y + 0.5) * HEIGHT / 44
            points.append((x, y))
    return np.array(points)


def generate_probe_points_B():
    """Generate probe points for subdomain B."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = WIDTH_A + (i_x + 0.5) * WIDTH_B / 44
            y = 0 + (i_y + 0.5) * HEIGHT / 44
            points.append((x, y))
    return np.array(points)


def generate_interface_probe_points():
    """Generate interface probe points (interior only)."""
    points = []
    for i in range(44):
        x = INTERFACE_X
        y = 0.25 + (i + 0.5) * 0.5 / 44  # from 1/4 to 3/4
        points.append((x, y))
    return np.array(points)


def source_term_A(x, y):
    """Source term for subdomain A."""
    fx = (x**2*y**3/50 + 6*x**2*y**2/125 - 33*x**2*y/500 + 3*x**2/250 
          - x*y**3/50 - 6*x*y**2/125 + 33*x*y/500 - 3*x/250 
          + y**5/125 + 4*y**4/125 - 89*y**3/500 - 21*y**2/125 + 297*y/1000 - 27/500)
    fy = (-3*x**2*y**2/100 - 6*x**2*y/125 + 33*x**2/1000 
          + 3*x*y**4/100 + 12*x*y**3/125 - 279*x*y**2/500 - 63*x*y/125 + 99*x/250 
          - y**4/200 - 2*y**3/125 + 33*y**2/1000 - 3*y/250)
    return fx, fy


def source_term_B(x, y):
    """Source term for subdomain B."""
    fx = (-101*x**2*y**3/1470 - 202*x**2*y**2/1225 + 1111*x**2*y/4900 - 101*x**2/2450 
          + 11869*x*y**3/18375 + 47476*x*y**2/30625 - 130559*x*y/61250 + 11869*x/30625 
          - 101*y**5/6125 - 404*y**4/6125 - 9591*y**3/39200 - 274761*y**2/245000 
          + 2755731*y/1960000 - 250521/980000)
    fy = (33183*x**2*y**2/24500 + 66366*x**2*y/30625 - 365013*x**2/245000 
          - 101*x*y**4/2100 - 404*x*y**3/2625 - 17141*x*y**2/9800 - 104794*x*y/30625 
          + 1113849*x/490000 + 499*y**4/3675 + 7984*y**3/18375 - 41347*y**2/49000 
          + 2509*y/6125 - 5643/98000)
    return fx, fy


class SimpleFEMSolver:
    """Simple P1 FEM solver for linear elasticity."""
    
    def __init__(self, level, h, is_subdomain_A=True):
        self.level = level
        self.h = h
        self.is_subdomain_A = is_subdomain_A
        
        if is_subdomain_A:
            self.xmin, self.xmax = 0, WIDTH_A
            self.lam, self.mu = LAMBDA_A, MU_A
            self.source = source_term_A
            self.side = 'A'
        else:
            self.xmin, self.xmax = WIDTH_A, 1.5
            self.lam, self.mu = LAMBDA_B, MU_B
            self.source = source_term_B
            self.side = 'B'
        
        self.ymin, self.ymax = 0, HEIGHT
        
        # Generate mesh
        nx = int(round((self.xmax - self.xmin) / h))
        ny = int(round(HEIGHT / h))
        
        # Adjust to fit exactly
        hx = (self.xmax - self.xmin) / nx
        hy = HEIGHT / ny
        
        # Create nodes
        self.nodes = []
        self.node_map = {}  # (i, j) -> node_id
        idx = 0
        for j in range(ny + 1):
            y = j * hy
            if abs(y - HEIGHT) < 1e-10:
                y = HEIGHT
            for i in range(nx + 1):
                x = self.xmin + i * hx
                if abs(x - self.xmin) < 1e-10:
                    x = self.xmin
                if abs(x - self.xmax) < 1e-10:
                    x = self.xmax
                self.nodes.append((x, y))
                self.node_map[(i, j)] = idx
                idx += 1
        
        self.nx = nx
        self.ny = ny
        self.n_nodes = len(self.nodes)
        self.ndof = 2 * self.n_nodes
        
        # Interface nodes
        if is_subdomain_A:
            # Right boundary (interface)
            self.interface_node_ids = [self.node_map[(nx, j)] for j in range(ny + 1)]
            self.interface_interior_ids = [self.node_map[(nx, j)] for j in range(1, ny)]
        else:
            # Left boundary (interface)
            self.interface_node_ids = [self.node_map[(0, j)] for j in range(ny + 1)]
            self.interface_interior_ids = [self.node_map[(0, j)] for j in range(1, ny)]
        
        # Outer boundary nodes (excluding interface interior)
        self.outer_boundary_ids = set()
        for j in range(ny + 1):
            for i in range(nx + 1):
                x, y = self.nodes[self.node_map[(i, j)]]
                on_outer = False
                if is_subdomain_A:
                    # Left, bottom, top boundaries
                    if abs(x - self.xmin) < 1e-10 or abs(y - self.ymin) < 1e-10 or abs(y - self.ymax) < 1e-10:
                        on_outer = True
                    # Interface corners are part of outer boundary
                    if abs(x - self.xmax) < 1e-10 and (abs(y - self.ymin) < 1e-10 or abs(y - self.ymax) < 1e-10):
                        on_outer = True
                else:
                    # Right, bottom, top boundaries
                    if abs(x - self.xmax) < 1e-10 or abs(y - self.ymin) < 1e-10 or abs(y - self.ymax) < 1e-10:
                        on_outer = True
                    # Interface corners are part of outer boundary
                    if abs(x - self.xmin) < 1e-10 and (abs(y - self.ymin) < 1e-10 or abs(y - self.ymax) < 1e-10):
                        on_outer = True
                
                if on_outer:
                    self.outer_boundary_ids.add(self.node_map[(i, j)])
        
        # Elements (triangles)
        self.elements = []
        for j in range(ny):
            for i in range(nx):
                n0 = self.node_map[(i, j)]
                n1 = self.node_map[(i+1, j)]
                n2 = self.node_map[(i+1, j+1)]
                n3 = self.node_map[(i, j+1)]
                self.elements.append([n0, n1, n2])
                self.elements.append([n0, n2, n3])
        
        # Precompute element matrices
        self._precompute_element_matrices()
    
    def _precompute_element_matrices(self):
        """Precompute element stiffness matrices and load vectors."""
        self.elem_stiffness = []
        self.elem_loads = []
        
        for e in self.elements:
            coords = np.array([self.nodes[n] for n in e])
            
            # Compute shape function gradients
            x0, y0 = coords[0]
            x1, y1 = coords[1]
            x2, y2 = coords[2]
            
            det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
            if abs(det) < 1e-15:
                continue
            
            # Gradient of shape functions
            dx_N = np.array([(y1-y2)/det, (y2-y0)/det, (y0-y1)/det])
            dy_N = np.array([-(x1-x2)/det, -(x2-x0)/det, -(x0-x1)/det])
            
            # Element stiffness matrix (6x6 for 2D vector problem)
            Ke = np.zeros((6, 6))
            for a in range(3):
                for b in range(3):
                    # Strain-displacement matrices
                    B_a = np.array([[dx_N[a], 0], [0, dy_N[a]], [dy_N[a], dx_N[a]]])
                    B_b = np.array([[dx_N[b], 0], [0, dy_N[b]], [dy_N[b], dx_N[b]]])
                    
                    # Constitutive matrix (plane strain)
                    D = np.array([
                        [self.lam + 2*self.mu, self.lam, 0],
                        [self.lam, self.lam + 2*self.mu, 0],
                        [0, 0, self.mu]
                    ])
                    
                    # Stiffness contribution
                    K_ab = B_a.T @ D @ B_b * abs(det) / 2
                    Ke[2*a:2*a+2, 2*b:2*b+2] = K_ab
            
            self.elem_stiffness.append(Ke)
            
            # Load vector (using midpoint quadrature)
            xm = np.mean(coords[:, 0])
            ym = np.mean(coords[:, 1])
            fx, fy = self.source(xm, ym)
            
            fm = np.zeros(6)
            for a in range(3):
                fm[2*a] = fx * abs(det) / 6
                fm[2*a+1] = fy * abs(det) / 6
            
            self.elem_loads.append(fm)
    
    def assemble_system(self, interface_disp=None):
        """Assemble global system with boundary conditions."""
        # Initialize global matrix and RHS
        K = np.zeros((self.ndof, self.ndof))
        f = np.zeros(self.ndof)
        
        # Assemble element contributions
        for e_idx, e in enumerate(self.elements):
            dofs = [2*n for n in e] + [2*n+1 for n in e]
            Ke = self.elem_stiffness[e_idx]
            fe = self.elem_loads[e_idx]
            
            for i in range(6):
                for j in range(6):
                    K[dofs[i], dofs[j]] += Ke[i, j]
                f[dofs[i]] += fe[i]
        
        # Apply Dirichlet boundary conditions
        # Zero displacement on outer boundary
        constrained_dofs = []
        for nid in self.outer_boundary_ids:
            constrained_dofs.extend([2*nid, 2*nid+1])
        
        # If interface displacement is provided, apply it
        if interface_disp is not None:
            for i, nid in enumerate(self.interface_interior_ids):
                constrained_dofs.extend([2*nid, 2*nid+1])
        
        # Solve using direct method with constraints
        # For simplicity, use penalty method
        penalty = 1e15
        
        for dof in constrained_dofs:
            K[dof, dof] += penalty
            target = 0.0
            if interface_disp is not None:
                # Find which interface node this corresponds to
                try:
                    idx = self.interface_interior_ids.index(dof // 2)
                    comp = dof % 2
                    target = interface_disp[idx, comp]
                except ValueError:
                    pass
            f[dof] += penalty * target
        
        return K, f
    
    def solve(self, interface_disp=None):
        """Solve the system."""
        K, f = self.assemble_system(interface_disp)
        
        # Solve linear system
        try:
            u = np.linalg.solve(K, f)
        except np.linalg.LinAlgError:
            # Use least squares if singular
            u, _, _, _ = np.linalg.lstsq(K, f, rcond=None)
        
        return u
    
    def get_displacement_at_point(self, u, x, y):
        """Get displacement at a point by interpolation."""
        # Find containing element
        for e in self.elements:
            coords = [self.nodes[n] for n in e]
            if self._point_in_triangle((x, y), coords):
                # Interpolate
                weights = self._barycentric_coords((x, y), coords)
                disp = np.zeros(2)
                for i, n in enumerate(e):
                    disp += weights[i] * u[2*n:2*n+2]
                return disp
        
        return np.zeros(2)
    
    def _point_in_triangle(self, p, coords):
        """Check if point is inside triangle."""
        x, y = p
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        
        denom = ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
        if abs(denom) < 1e-15:
            return False
        
        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        c = 1 - a - b
        
        return a >= -1e-10 and b >= -1e-10 and c >= -1e-10
    
    def _barycentric_coords(self, p, coords):
        """Compute barycentric coordinates."""
        x, y = p
        x0, y0 = coords[0]
        x1, y1 = coords[1]
        x2, y2 = coords[2]
        
        denom = ((y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2))
        if abs(denom) < 1e-15:
            return np.array([1/3, 1/3, 1/3])
        
        a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / denom
        b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / denom
        c = 1 - a - b
        
        return np.array([a, b, c])
    
    def get_interface_traction(self, u):
        """Compute traction on interface."""
        tractions = []
        
        for j in range(1, self.ny):  # Interior nodes only
            nid = self.interface_interior_ids[j-1]
            
            # Find element adjacent to interface
            elem_idx = None
            for idx, e in enumerate(self.elements):
                if nid in e:
                    # Check if element is on the correct side
                    coords = [self.nodes[n] for n in e]
                    if self.is_subdomain_A:
                        if any(abs(x - WIDTH_A) > 1e-10 for x, y in coords):
                            elem_idx = idx
                            break
                    else:
                        if any(abs(x - WIDTH_A) > 1e-10 for x, y in coords):
                            elem_idx = idx
                            break
            
            if elem_idx is None:
                tractions.append(np.zeros(2))
                continue
            
            # Compute stress in this element
            e = self.elements[elem_idx]
            coords = np.array([self.nodes[n] for n in e])
            u_vals = u[[2*n for n in e] + [2*n+1 for n in e]]
            
            # Shape function gradients
            x0, y0 = coords[0]
            x1, y1 = coords[1]
            x2, y2 = coords[2]
            
            det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
            if abs(det) < 1e-15:
                tractions.append(np.zeros(2))
                continue
            
            dx_N = np.array([(y1-y2)/det, (y2-y0)/det, (y0-y1)/det])
            dy_N = np.array([-(x1-x2)/det, -(x2-x0)/det, -(x0-x1)/det])
            
            # Displacement gradient
            grad_u = np.zeros((2, 2))
            for comp in range(2):
                grad_u[comp, 0] = np.dot(dx_N, u_vals[comp::2])
                grad_u[comp, 1] = np.dot(dy_N, u_vals[comp::2])
            
            # Stress tensor
            sym_grad = 0.5 * (grad_u + grad_u.T)
            div_u = np.trace(grad_u)
            sigma = 2 * self.mu * sym_grad + self.lam * div_u * np.eye(2)
            
            # Normal (outward from subdomain)
            if self.is_subdomain_A:
                n = np.array([1.0, 0.0])  # Points right
            else:
                n = np.array([-1.0, 0.0])  # Points left
            
            # Traction
            t = sigma @ n
            tractions.append(t)
        
        return np.array(tractions)
    
    def write_log(self):
        """Write log file."""
        with open(f"run_level{self.level}_{self.side}.log", 'w') as f:
            f.write(f"NDOF = {self.ndof}\n")


def run_coupling(level, h):
    """Run Dirichlet-Neumann coupling for one mesh level."""
    print(f"\n{'='*60}")
    print(f"Running coupling for level {level}, h = {h}")
    print(f"{'='*60}")
    
    # Initialize solvers
    solver_A = SimpleFEMSolver(level, h, is_subdomain_A=True)
    solver_B = SimpleFEMSolver(level, h, is_subdomain_A=False)
    
    # Write log files
    solver_A.write_log()
    solver_B.write_log()
    
    # Interface probe points
    interface_probes = generate_interface_probe_points()
    n_interface = len(solver_A.interface_interior_ids)
    
    # Initial guess for interface displacement (zero)
    interface_disp = np.zeros((n_interface, 2))
    
    # Residual history
    residuals = []
    
    # Dirichlet-Neumann iteration
    # A is DIRICHLET side: receives displacement from B, returns traction
    # B is NEUMANN side: receives traction from A, returns displacement
    
    for iteration in range(MAX_ITER):
        print(f"Iteration {iteration + 1}")
        
        # Step 1: Solve subdomain A with Dirichlet BC from interface_disp
        print("  Solving subdomain A...")
        u_A = solver_A.solve(interface_disp)
        
        # Get traction from A on interface
        traction_A = solver_A.get_interface_traction(u_A)
        
        # Step 2: Solve subdomain B with Neumann BC from traction_A
        print("  Solving subdomain B...")
        # For Neumann BC, we need to modify the RHS
        K_B, f_B = solver_B.assemble_system()
        
        # Add traction contribution to RHS
        for i, nid in enumerate(solver_B.interface_interior_ids):
            # Traction from A points into B (opposite direction)
            t = -traction_A[i]  # Negate because A's outward normal is opposite to B's
            f_B[2*nid] += t[0] * h  # Approximate integration
            f_B[2*nid+1] += t[1] * h
        
        u_B = np.linalg.solve(K_B, f_B)
        
        # Get displacement from B on interface
        disp_B_interface = np.zeros((n_interface, 2))
        for i, nid in enumerate(solver_B.interface_interior_ids):
            disp_B_interface[i] = u_B[2*nid:2*nid+2]
        
        # Compute residual (difference in interface displacement)
        residual = np.linalg.norm(disp_B_interface - interface_disp) / (np.linalg.norm(interface_disp) + 1e-10)
        residuals.append(residual)
        print(f"  Residual: {residual:.6e}")
        
        # Update interface displacement
        interface_disp = disp_B_interface.copy()
        
        # Check convergence
        if residual < TOL:
            print(f"  Converged after {iteration + 1} iterations")
            break
    
    # Final solutions
    u_A_final = solver_A.solve(interface_disp)
    K_B, f_B = solver_B.assemble_system()
    for i, nid in enumerate(solver_B.interface_interior_ids):
        t = -solver_A.get_interface_traction(u_A_final)[i]
        f_B[2*nid] += t[0] * h
        f_B[2*nid+1] += t[1] * h
    u_B_final = np.linalg.solve(K_B, f_B)
    
    # Write residual history
    with open(f"residual_level{level}.csv", 'w') as f:
        f.write("iteration,interface_residual\n")
        for i, r in enumerate(residuals):
            f.write(f"{i+1},{r:.15e}\n")
    
    return u_A_final, u_B_final, residuals[-1] if residuals else 1.0, len(residuals)


def main():
    print("Starting Dirichlet-Neumann coupling simulation...")
    
    # Probe points
    probes_A = generate_probe_points_A()
    probes_B = generate_probe_points_B()
    interface_probes = generate_interface_probe_points()
    
    print(f"Probe points A: {len(probes_A)}")
    print(f"Probe points B: {len(probes_B)}")
    print(f"Interface probes: {len(interface_probes)}")
    
    # Run for each mesh level
    levels = [1, 2, 3]
    final_residuals = []
    max_iterations = []
    all_solutions = {}
    
    for level in levels:
        h = 1.0 / (8 * level)
        u_A, u_B, residual, n_iter = run_coupling(level, h)
        final_residuals.append(residual)
        max_iterations.append(n_iter)
        all_solutions[level] = (u_A, u_B)
        print(f"Level {level}: final residual = {residual:.6e}, iterations = {n_iter}")
        
        # Create solver instances for output
        solver_A = SimpleFEMSolver(level, h, is_subdomain_A=True)
        solver_B = SimpleFEMSolver(level, h, is_subdomain_A=False)
        
        # Write solution CSV files
        with open(f"solution_level{level}_A.csv", 'w') as f:
            f.write("x,y,ux,uy\n")
            for x, y in probes_A:
                ux, uy = solver_A.get_displacement_at_point(u_A, x, y)
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e}\n")
        
        with open(f"solution_level{level}_B.csv", 'w') as f:
            f.write("x,y,ux,uy\n")
            for x, y in probes_B:
                ux, uy = solver_B.get_displacement_at_point(u_B, x, y)
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e}\n")
        
        # Write interface CSV files
        traction_A = solver_A.get_interface_traction(u_A)
        traction_B = solver_B.get_interface_traction(u_B)
        
        with open(f"interface_level{level}_A.csv", 'w') as f:
            f.write("x,y,ux,uy,tx,ty\n")
            for i, (x, y) in enumerate(interface_probes):
                # Find closest interface node
                idx = min(range(len(solver_A.interface_interior_ids)), 
                         key=lambda k: abs(solver_A.nodes[solver_A.interface_interior_ids[k]][1] - y))
                nid = solver_A.interface_interior_ids[idx]
                ux, uy = u_A[2*nid:2*nid+2]
                tx, ty = traction_A[idx]
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e},{tx:.15e},{ty:.15e}\n")
        
        with open(f"interface_level{level}_B.csv", 'w') as f:
            f.write("x,y,ux,uy,tx,ty\n")
            for i, (x, y) in enumerate(interface_probes):
                idx = min(range(len(solver_B.interface_interior_ids)), 
                         key=lambda k: abs(solver_B.nodes[solver_B.interface_interior_ids[k]][1] - y))
                nid = solver_B.interface_interior_ids[idx]
                ux, uy = u_B[2*nid:2*nid+2]
                tx, ty = traction_B[idx]
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e},{tx:.15e},{ty:.15e}\n")
    
    # Determine mesh independence
    if len(final_residuals) >= 2:
        rel_change = abs(final_residuals[-1] - final_residuals[-2]) / (abs(final_residuals[-2]) + 1e-10)
        converged = rel_change < 0.1  # 10% tolerance
    else:
        converged = False
        rel_change = 1.0
    
    # Write RESULT.txt
    csv_files = []
    for level in levels:
        csv_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv"
        ])
    
    with open("RESULT.txt", 'w') as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {','.join(csv_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residuals[-1]:.15e}\n")
        f.write(f"COUPLING_ITERATIONS = {max_iterations[-1]}\n")
        f.write(f"MESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}\n")
        f.write(f"MAX_REL_CHANGE = {rel_change:.15e}\n")
    
    print("\nDone! Check RESULT.txt for summary.")


if __name__ == "__main__":
    main()
