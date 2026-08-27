#!/usr/bin/env python3
"""
Coupled simulation: NGSolve on subdomain A, Kratos on subdomain B
Dirichlet-Neumann coupling with A as DIRICHLET side, B as NEUMANN side

Subdomain A: (0, 0.625) x (0, 1), k = 1
Subdomain B: (0.625, 1.5) x (0, 1), k = 1000
Interface: x = 5/8 = 0.625
"""

import numpy as np
import json
from pathlib import Path
import sys

# Add venv to path for Kratos
sys.path.insert(0, '/home/alexander/Schreibtisch/open-fem-agent/.venv/lib/python3.12/site-packages')

import ngsolve
from ngsolve import *
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

# Problem parameters
X_INTERFACE = 5/8  # 0.625
X_A_RIGHT = X_INTERFACE
X_B_LEFT = X_INTERFACE
X_B_RIGHT = 1.5
H = 1.0
K_A = 1.0
K_B = 1000.0

# Source terms
def f_A(x, y):
    return (-18*x**3*y/5 + 2*x**3/5 + 5063*x**2*y/20000 - 95021*x**2/60000 
            - 18*x*y**3/5 + 6*x*y**2/5 + 14607*x*y/4000 + 1669*x/2000 
            + 5063*y**3/60000 - 95021*y**2/60000 + 14993*y/10000)

def f_B(x, y):
    return (-9*x**3*y/2500000 + x**3/2500000 - 64901*x**2*y/10000000 - 25033*x**2/30000000 
            - 9*x*y**3/2500000 + 3*x*y**2/2500000 - 422790921*x*y/160000000 - 21609971*x/32000000 
            - 64901*y**3/30000000 - 25033*y**2/30000000 + 1274009971*y/320000000 + 12989997/12800000)

# Probe points
def generate_probe_points_A():
    """Generate 1936 probe points for subdomain A"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate 1936 probe points for subdomain B"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return points

def generate_interface_probes():
    """Generate 44 interface probe points at x = 5/8"""
    points = []
    for i in range(44):
        x = 5/8
        y = 1/4 + (i + 0.5) * 1/2 / 44
        points.append((x, y))
    return points

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBES = generate_interface_probes()

class NGsolveSolverA:
    """NGSolve solver for subdomain A (DIRICHLET side)"""
    
    def __init__(self, mesh_level):
        self.mesh_level = mesh_level
        self.h = 1/8 / (2**(mesh_level-1))
        
        # Create rectangular mesh using SplineGeometry
        geo = GeometryFactory.RectangularMesh(X_A_RIGHT, H, 
                                               numelemx=int(X_A_RIGHT/self.h), 
                                               numelemy=int(H/self.h))
        self.mesh = Mesh(geo)
        
        # Create finite element space
        self.V = H1(self.mesh, order=1)
        
        # Define boundary markers
        # left (x=0): outer Dirichlet
        # right (x=X_INTERFACE): interface  
        # top (y=H): outer Dirichlet  
        # bottom (y=0): outer Dirichlet
        
        self.bcs_outer = ['left', 'top', 'bottom']
        self.bcs_interface = ['right']
        
        # Trial and test functions
        u = self.V.TrialFunction()
        v = self.V.TestFunction()
        
        # Bilinear form: integral(k grad u . grad v)
        a = BilinearForm(self.V)
        a += K_A * Grad(u) * Grad(v) * dx
        
        # Linear form: integral(f v)
        f_expr = CoefficientFunction(f_A)
        l = LinearForm(self.V)
        l += f_expr * v * dx
        
        a.Assemble()
        l.Assemble()
        
        # Store for later use
        self.a_form = a
        self.l_form = l
        self.u = GridFunction(self.V)
        
        print(f"NGSolve A: total DOFs = {self.V.Size()}")
        
    def solve_with_interface_dirichlet(self, T_interface_values, interface_y_coords):
        """
        Solve with Dirichlet BC on interface
        T_interface_values: temperature values at interface nodes
        interface_y_coords: y-coordinates of interface nodes
        Returns: solution grid function
        """
        # Reset grid function
        self.u.Set(0)
        
        # Apply outer boundary conditions (u=0)
        bc = BoundaryCondition(self.V, dirichlet=True)
        for bnd in self.bcs_outer:
            bc.Set(bnd)
        
        # Get interface node coordinates
        interface_nodes = []
        for el in self.mesh.Elements():
            if el.Right():
                for nd in el.GetVertices():
                    if abs(nd.x - X_A_RIGHT) < 1e-10:
                        interface_nodes.append((nd.y, nd.nr))
        
        interface_nodes.sort(key=lambda x: x[0])
        
        # Interpolate T_interface_values to each interface node and set
        for y_coord, node_nr in interface_nodes:
            # Find corresponding value from T_interface_values
            T_val = np.interp(y_coord, interface_y_coords, T_interface_values)
            dof = self.V.GetDof(node_nr, 0)
            self.u.vec[dof] = T_val
            
        # Assemble with Dirichlet BCs
        self.a_form.Assemble()
        self.l_form.Assemble()
        
        # Solve
        self.a_form.UpdateMatrix(self.mat)
        self.mat.AddBC(bc, self.u.vec)
        self.a_form.UpdateVector(self.vec)
        self.mat.Solve(self.u.vec)
        
        return self.u
    
    def compute_interface_flux(self):
        """
        Compute outward normal flux at interface
        qn = -k * grad(u) . n_out
        For subdomain A, outward normal at right boundary is (+1, 0)
        So qn = -k * du/dx
        """
        fluxes = []
        y_coords = []
        
        # Evaluate flux at interface probe points
        for (px, py) in INTERFACE_PROBES:
            # Compute gradient at this point
            grad_u = self.u.Gradient()(px, py)
            # Outward normal is (+1, 0)
            flux = -K_A * grad_u[0]
            fluxes.append(flux)
            y_coords.append(py)
        
        return y_coords, fluxes
    
    def evaluate_at_probes(self):
        """Evaluate solution at probe points"""
        values = []
        for (px, py) in PROBE_POINTS_A:
            val = self.u(px, py)
            values.append(val)
        return values
    
    def get_ndof(self):
        return self.V.Size()


class KratosSolverB:
    """Kratos solver for subdomain B (NEUMANN side)"""
    
    def __init__(self, mesh_level):
        self.mesh_level = mesh_level
        self.h = 1/8 / (2**(mesh_level-1))
        
        # Domain dimensions
        self.X0 = X_B_LEFT
        self.X1 = X_B_RIGHT
        self.H = H
        self.K = K_B
        
        # Number of elements
        self.nx = int((self.X1 - self.X0) / self.h)
        self.ny = int(self.H / self.h)
        
        print(f"Kratos B: nx={self.nx}, ny={self.ny}")
        
    def solve_with_interface_neumann(self, q_interface_values, interface_y_coords):
        """
        Solve with Neumann BC on interface (left side of domain B)
        q_interface_values: outward normal flux from partner (subdomain A)
        Note: For subdomain B, outward normal at left boundary is (-1, 0)
        The flux received from A is A's outward flux, which is in +x direction
        For B, this becomes an inward flux, so we need to negate it for Neumann BC
        
        Returns: (T_interface_values, interface_y_coords, q_out_B)
        where q_out_B is B's outward normal flux at interface
        """
        model = KM.Model()
        mp = model.CreateModelPart("thermal")
        mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
        
        settings = KM.ConvectionDiffusionSettings()
        settings.SetUnknownVariable(KM.TEMPERATURE)
        settings.SetDiffusionVariable(KM.CONDUCTIVITY)
        settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
        settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
        mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
        
        for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX):
            mp.AddNodalSolutionStepVariable(v)
        mp.SetBufferSize(1)
        
        props = mp.CreateNewProperties(1)
        nid, cnt = {}, 1
        
        # Create nodes
        for j in range(self.ny + 1):
            for i in range(self.nx + 1):
                x = self.X0 + (self.X1 - self.X0) * i / self.nx
                y = self.H * j / self.ny
                mp.CreateNewNode(cnt, x, y, 0.0)
                nid[(i, j)] = cnt
                cnt += 1
        
        # Create elements (triangular)
        eid = 1
        for j in range(self.ny):
            for i in range(self.nx):
                a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
                mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
                mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1
        
        # Set material properties and source term
        for node in mp.Nodes:
            node.SetSolutionStepValue(KM.CONDUCTIVITY, self.K)
            x, y = node.X, node.Y
            f_val = f_B(x, y)
            node.SetSolutionStepValue(KM.HEAT_FLUX, f_val)
            node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
        
        # Outer boundary conditions (u=0 on all outer boundaries)
        # Right side (x=X1): Dirichlet u=0
        for j in range(self.ny + 1):
            n = mp.Nodes[nid[(self.nx, j)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
        
        # Top (y=H): Dirichlet u=0
        for i in range(self.nx + 1):
            n = mp.Nodes[nid[(i, self.ny)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
        
        # Bottom (y=0): Dirichlet u=0
        for i in range(self.nx + 1):
            n = mp.Nodes[nid[(i, 0)]]
            n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
            n.Fix(KM.TEMPERATURE)
        
        # Interface (left side, x=X0): Neumann BC
        # The flux from A is A's outward flux (in +x direction)
        # For B, this is an INWARD flux, so the Neumann BC should be -q_from_A
        # But Kratos FACE_HEAT_FLUX is defined as heat flux INTO the domain
        # So we set FACE_HEAT_FLUX = q_from_A (which is already the correct sign)
        
        # Get interface node y-coordinates
        interface_node_ys = [self.H * j / self.ny for j in range(self.ny + 1)]
        
        # Apply Neumann BC on interface
        for j in range(self.ny + 1):
            y_coord = interface_node_ys[j]
            # Interpolate flux value
            q_val = np.interp(y_coord, interface_y_coords, q_interface_values)
            # For Kratos, positive FACE_HEAT_FLUX means heat flowing INTO the domain
            # The flux from A is A's outward flux (positive = flowing out of A, into B)
            # So we use q_val directly
            n = mp.Nodes[nid[(0, j)]]
            n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, q_val)
            # Don't fix temperature - it's a natural BC
        
        KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)
        scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
        builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
        strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
        strategy.Initialize()
        strategy.Solve()
        
        # Extract interface temperature values
        T_if = np.array([mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                         for j in range(self.ny + 1)])
        
        # Compute B's outward normal flux at interface
        # Outward normal for B at left boundary is (-1, 0)
        # qn = -k * grad(u) . n = -k * (-du/dx) = k * du/dx
        # Using one-sided difference: du/dx ≈ (T_near - T_if) / dx
        dx = (self.X1 - self.X0) / self.nx
        T_near = np.array([mp.Nodes[nid[(1, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                           for j in range(self.ny + 1)])
        q_out_B = self.K * (T_near - T_if) / dx  # This is B's outward flux
        
        return T_if, interface_node_ys, q_out_B, mp, nid
    
    def evaluate_at_probes(self, mp, nid):
        """Evaluate solution at probe points"""
        values = []
        for (px, py) in PROBE_POINTS_B:
            # Find nearest node or interpolate
            local_i = int((px - self.X0) / (self.X1 - self.X0) * self.nx)
            local_j = int(py / self.H * self.ny)
            
            # Clamp to valid range
            local_i = max(0, min(local_i, self.nx - 1))
            local_j = max(0, min(local_j, self.ny - 1))
            
            # Get element corner temperatures
            t00 = mp.Nodes[nid[(local_i, local_j)]].GetSolutionStepValue(KM.TEMPERATURE)
            t10 = mp.Nodes[nid[(local_i+1, local_j)]].GetSolutionStepValue(KM.TEMPERATURE)
            t01 = mp.Nodes[nid[(local_i, local_j+1)]].GetSolutionStepValue(KM.TEMPERATURE)
            t11 = mp.Nodes[nid[(local_i+1, local_j+1)]].GetSolutionStepValue(KM.TEMPERATURE)
            
            # Bilinear interpolation
            xi = (px - self.X0 - local_i * (self.X1 - self.X0) / self.nx) / ((self.X1 - self.X0) / self.nx)
            eta = (py - local_j * self.H / self.ny) / (self.H / self.ny)
            
            val = (1-xi)*(1-eta)*t00 + xi*(1-eta)*t10 + (1-xi)*eta*t01 + xi*eta*t11
            values.append(val)
        
        return values
    
    def compute_interface_flux_at_probes(self, mp, nid):
        """Compute B's outward normal flux at interface probe points"""
        fluxes = []
        y_coords = []
        
        dx = (self.X1 - self.X0) / self.nx
        
        for (px, py) in INTERFACE_PROBES:
            # Find nearest interface node
            local_j = int(py / self.H * self.ny)
            local_j = max(0, min(local_j, self.ny))
            
            # Get temperatures
            T_if = mp.Nodes[nid[(0, local_j)]].GetSolutionStepValue(KM.TEMPERATURE)
            T_near = mp.Nodes[nid[(1, local_j)]].GetSolutionStepValue(KM.TEMPERATURE)
            
            # B's outward flux: qn = k * (T_near - T_if) / dx
            flux = self.K * (T_near - T_if) / dx
            fluxes.append(flux)
            y_coords.append(py)
        
        return y_coords, fluxes
    
    def get_ndof(self):
        return (self.nx + 1) * (self.ny + 1)


def run_coupling(mesh_level, work_dir="."):
    """Run the coupled simulation for a given mesh level"""
    
    work_path = Path(work_dir)
    work_path.mkdir(exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Mesh level {mesh_level}")
    print(f"{'='*60}")
    
    # Initialize solvers
    solver_a = NGsolveSolverA(mesh_level)
    solver_b = KratosSolverB(mesh_level)
    
    # Initial guess for interface temperature
    h = 1/8 / (2**(mesh_level-1))
    ny_b = int(H / h)
    interface_y_coords = [H * j / ny_b for j in range(ny_b + 1)]
    T_interface = np.zeros(len(interface_y_coords))  # Initial guess: zero
    
    # Coupling parameters
    max_iter = 200
    tol = 1e-6
    omega = 0.1  # Relaxation factor (small due to high contrast)
    
    residual_history = []
    
    for iteration in range(max_iter):
        # Step 1: Solve subdomain A with Dirichlet BC from current T_interface
        u_a = solver_a.solve_with_interface_dirichlet(T_interface, interface_y_coords)
        
        # Step 2: Compute A's outward flux at interface
        _, q_out_A = solver_a.compute_interface_flux()
        
        # Interpolate q_out_A to Kratos interface nodes
        q_at_kratos_nodes = np.interp(interface_y_coords, [p[1] for p in INTERFACE_PROBES], q_out_A)
        
        # Step 3: Solve subdomain B with Neumann BC from q_out_A
        T_interface_new, _, q_out_B, mp_b, nid_b = solver_b.solve_with_interface_neumann(
            q_at_kratos_nodes, [p[1] for p in INTERFACE_PROBES])
        
        # Compute residual (relative mismatch in interface temperature)
        T_diff = np.abs(T_interface_new - T_interface)
        T_max = max(np.max(np.abs(T_interface)), np.max(np.abs(T_interface_new)), 1e-10)
        residual = np.max(T_diff) / T_max
        
        residual_history.append(residual)
        
        print(f"Iteration {iteration+1}: residual = {residual:.6e}")
        
        if residual < tol:
            print(f"Converged after {iteration+1} iterations")
            break
        
        # Relaxation
        T_interface = T_interface + omega * (T_interface_new - T_interface)
    
    final_residual = residual_history[-1]
    num_iterations = len(residual_history)
    
    # Final solve to get consistent solutions
    u_a = solver_a.solve_with_interface_dirichlet(T_interface, interface_y_coords)
    _, _, _, mp_b, nid_b = solver_b.solve_with_interface_neumann(
        np.interp(interface_y_coords, [p[1] for p in INTERFACE_PROBES], 
                  solver_a.compute_interface_flux()[1]), 
        [p[1] for p in INTERFACE_PROBES])
    
    # Write results
    # Solution files
    values_a = solver_a.evaluate_at_probes()
    with open(work_path / f"solution_level{mesh_level}_A.csv", "w") as f:
        f.write("x, y, u\n")
        for i, (pt, val) in enumerate(zip(PROBE_POINTS_A, values_a)):
            f.write(f"{pt[0]:.15e}, {pt[1]:.15e}, {val:.15e}\n")
    
    values_b = solver_b.evaluate_at_probes(mp_b, nid_b)
    with open(work_path / f"solution_level{mesh_level}_B.csv", "w") as f:
        f.write("x, y, u\n")
        for i, (pt, val) in enumerate(zip(PROBE_POINTS_B, values_b)):
            f.write(f"{pt[0]:.15e}, {pt[1]:.15e}, {val:.15e}\n")
    
    # Interface files
    # Subdomain A
    y_coords_a, qn_a = solver_a.compute_interface_flux()
    T_at_interface_a = np.interp(y_coords_a, interface_y_coords, T_interface)
    
    with open(work_path / f"interface_level{mesh_level}_A.csv", "w") as f:
        f.write("x, y, u, qn\n")
        for i, (yc, qn) in enumerate(zip(y_coords_a, qn_a)):
            u_val = T_at_interface_a[i]
            f.write(f"{X_INTERFACE:.15e}, {yc:.15e}, {u_val:.15e}, {qn:.15e}\n")
    
    # Subdomain B
    y_coords_b, qn_b = solver_b.compute_interface_flux_at_probes(mp_b, nid_b)
    T_at_interface_b = np.interp(y_coords_b, interface_y_coords, T_interface)
    
    with open(work_path / f"interface_level{mesh_level}_B.csv", "w") as f:
        f.write("x, y, u, qn\n")
        for i, (yc, qn) in enumerate(zip(y_coords_b, qn_b)):
            u_val = T_at_interface_b[i]
            f.write(f"{X_INTERFACE:.15e}, {yc:.15e}, {u_val:.15e}, {qn:.15e}\n")
    
    # Residual history
    with open(work_path / f"residual_level{mesh_level}.csv", "w") as f:
        f.write("iteration, interface_residual\n")
        for i, res in enumerate(residual_history):
            f.write(f"{i+1}, {res:.15e}\n")
    
    # Run log files
    with open(work_path / f"run_level{mesh_level}_A.log", "w") as f:
        f.write(f"NDOF = {solver_a.get_ndof()}\n")
    
    with open(work_path / f"run_level{mesh_level}_B.log", "w") as f:
        f.write(f"NDOF = {solver_b.get_ndof()}\n")
    
    return final_residual, num_iterations, solver_a, solver_b, mp_b, nid_b


def main():
    """Main driver"""
    levels = [1, 2, 3]
    h_values = [1/8, 1/16, 1/32]
    
    all_files = []
    final_residuals = {}
    num_iterations = {}
    solutions = {}
    
    for level in levels:
        print(f"\n{'#'*60}")
        print(f"# Running mesh level {level} (h = {h_values[level-1]})")
        print(f"{'#'*60}")
        
        final_res, num_it, solver_a, solver_b, mp_b, nid_b = run_coupling(level)
        final_residuals[level] = final_res
        num_iterations[level] = num_it
        
        # Collect files
        files = [
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv",
            f"run_level{level}_A.log",
            f"run_level{level}_B.log"
        ]
        all_files.extend(files)
        
        # Store solutions for convergence check
        values_a = solver_a.evaluate_at_probes()
        values_b = solver_b.evaluate_at_probes(mp_b, nid_b)
        solutions[level] = (values_a, values_b)
    
    # Check mesh independence
    if len(solutions) >= 2:
        # Compare finest two levels
        level_fine = max(solutions.keys())
        level_coarse = level_fine - 1
        
        vals_a_fine = np.array(solutions[level_fine][0])
        vals_a_coarse = np.array(solutions[level_coarse][0])
        vals_b_fine = np.array(solutions[level_fine][1])
        vals_b_coarse = np.array(solutions[level_coarse][1])
        
        rel_change_a = np.max(np.abs(vals_a_fine - vals_a_coarse) / (np.abs(vals_a_coarse) + 1e-10))
        rel_change_b = np.max(np.abs(vals_b_fine - vals_b_coarse) / (np.abs(vals_b_coarse) + 1e-10))
        max_rel_change = max(rel_change_a, rel_change_b)
        
        converged = max_rel_change < 1e-3  # Threshold for convergence
    else:
        max_rel_change = float('inf')
        converged = False
    
    # Write RESULT.txt
    with open("RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {','.join(all_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residuals[max(final_residuals.keys())]:.15e}\n")
        f.write(f"COUPLING_ITERATIONS = {num_iterations[max(num_iterations.keys())]}\n")
        f.write(f"MESH_INDEPENDENCE = {'CONVERGED' if converged else 'NOT_CONVERGED'}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\n" + "="*60)
    print("Simulation complete!")
    print(f"Final interface residual: {final_residuals[max(final_residuals.keys())]:.6e}")
    print(f"Coupling iterations (finest): {num_iterations[max(num_iterations.keys())]}")
    print(f"Mesh independence: {'CONVERGED' if converged else 'NOT_CONVERGED'}")
    print(f"Max relative change: {max_rel_change:.6e}")
    print("="*60)


if __name__ == "__main__":
    main()
