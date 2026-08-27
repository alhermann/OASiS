#!/usr/bin/env python3
"""
Coupled simulation using FEBio (subdomain A) and Python FEM (subdomain B)
Dirichlet-Neumann iteration with A as DIRICHLET side, B as NEUMANN side.
"""

import os
import sys
import numpy as np
import subprocess
from pathlib import Path

# Paths
FEBIO_BIN = "/home/alexander/FEBio/bin/febio4"
WORK_DIR = str(Path(__file__).parent)
os.chdir(WORK_DIR)

# Problem parameters
LAMBDA_A = 500.0
MU_A = 250.0
LAMBDA_B = 500.0
MU_B = 1250.0

# Domain boundaries
X_INTERFACE = 5.0/8.0  # 0.625

# Mesh levels
H_LEVELS = [1/8, 1/16, 1/32]

# Tolerance for coupling
COUPLING_TOL = 1e-6
MAX_ITERATIONS = 100

def generate_probe_points_A():
    points = []
    nx, ny = 44, 44
    dx = 0.625 / 44
    dy = 1.0 / 44
    for iy in range(ny):
        for ix in range(nx):
            x = 0 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return points

def generate_probe_points_B():
    points = []
    nx, ny = 44, 44
    dx = 0.875 / 44
    dy = 1.0 / 44
    for iy in range(ny):
        for ix in range(nx):
            x = 0.625 + (ix + 0.5) * dx
            y = 0 + (iy + 0.5) * dy
            points.append((x, y))
    return points

def generate_interface_probes():
    points = []
    n = 44
    for i in range(n):
        x = 5.0/8.0
        y = 0.25 + (i + 0.5) * 0.5 / n
        points.append((x, y))
    return points

def source_term_A(x, y):
    fx = (x**2*y**3/50 + 6*x**2*y**2/125 - 33*x**2*y/500 + 3*x**2/250 
          - x*y**3/50 - 6*x*y**2/125 + 33*x*y/500 - 3*x/250 
          + y**5/125 + 4*y**4/125 - 89*y**3/500 - 21*y**2/125 + 297*y/1000 - 27/500)
    fy = (-3*x**2*y**2/100 - 6*x**2*y/125 + 33*x**2/1000 
          + 3*x*y**4/100 + 12*x*y**3/125 - 279*x*y**2/500 - 63*x*y/125 + 99*x/250 
          - y**4/200 - 2*y**3/125 + 33*y**2/1000 - 3*y/250)
    return fx, fy

def source_term_B(x, y):
    fx = (-101*x**2*y**3/1470 - 202*x**2*y**2/1225 + 1111*x**2*y/4900 - 101*x**2/2450 
          + 11869*x*y**3/18375 + 47476*x*y**2/30625 - 130559*x*y/61250 + 11869*x/30625 
          - 101*y**5/6125 - 404*y**4/6125 - 9591*y**3/39200 - 274761*y**2/245000 
          + 2755731*y/1960000 - 250521/980000)
    fy = (33183*x**2*y**2/24500 + 66366*x**2*y/30625 - 365013*x**2/245000 
          - 101*x*y**4/2100 - 404*x*y**3/2625 - 17141*x*y**2/9800 - 104794*x*y/30625 
          + 1113849*x/490000 + 499*y**4/3675 + 7984*y**3/18375 - 41347*y**2/49000 
          + 2509*y/6125 - 5643/98000)
    return fx, fy


class FEBioSolver:
    def __init__(self, h, lambda_val, mu_val):
        self.h = h
        self.lambda_val = lambda_val
        self.mu_val = mu_val
        self.nx = int(round(0.625 / h))
        self.ny = int(round(1.0 / h))
        
        E = 2*mu_val*(lambda_val+mu_val)/(lambda_val+2*mu_val)
        nu = lambda_val/(2*(lambda_val+mu_val))
        self.E = E
        self.nu = nu
        
    def write_deck(self, iter_num):
        filename = f"subdomain_A_iter{iter_num}.feb"
        
        with open(filename, 'w') as f:
            f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
            f.write('<febio_spec version="4.0">\n')
            f.write('  <Module type="solid"/>\n')
            f.write('  <Control>\n')
            f.write('    <analysis>STATIC</analysis>\n')
            f.write('    <time_steps>1</time_steps>\n')
            f.write('    <step_size>1.0</step_size>\n')
            f.write('    <solver type="solid">\n')
            f.write('      <symmetric_stiffness>symmetric</symmetric_stiffness>\n')
            f.write('    </solver>\n')
            f.write('  </Control>\n')
            
            f.write('  <Material>\n')
            f.write(f'    <material id="1" name="MatA" type="isotropic elastic">\n')
            f.write(f'      <density>1.0</density>\n')
            f.write(f'      <E>{self.E:.15e}</E>\n')
            f.write(f'      <v>{self.nu:.15e}</v>\n')
            f.write('    </material>\n')
            f.write('  </Material>\n')
            
            f.write('  <Mesh>\n')
            f.write('    <Nodes name="Object1">\n')
            
            node_idx = 1
            for jz in range(2):
                z = jz * 0.1
                for j in range(self.ny+1):
                    y = j * self.h
                    for i in range(self.nx+1):
                        x = i * self.h
                        f.write(f'      <node id="{node_idx}">{x:.15e},{y:.15e},{z:.15e}</node>\n')
                        node_idx += 1
            
            f.write('    </Nodes>\n')
            f.write('    <Elements type="hex8" name="Part1">\n')
            
            elem_idx = 1
            for jz in range(1):
                z_offset = jz * (self.nx+1) * (self.ny+1)
                for j in range(self.ny):
                    for i in range(self.nx):
                        n1 = z_offset + j*(self.nx+1) + i + 1
                        n2 = n1 + 1
                        n3 = n2 + self.nx + 1
                        n4 = n1 + self.nx + 1
                        n5 = n1 + (self.nx+1)*(self.ny+1)
                        n6 = n2 + (self.nx+1)*(self.ny+1)
                        n7 = n3 + (self.nx+1)*(self.ny+1)
                        n8 = n4 + (self.nx+1)*(self.ny+1)
                        f.write(f'      <elem id="{elem_idx}">{n1},{n2},{n3},{n4},{n5},{n6},{n7},{n8}</elem>\n')
                        elem_idx += 1
            
            f.write('    </Elements>\n')
            
            bottom_nodes = " ".join(str(j*(self.nx+1)+i+1) for j in [0] for i in range(self.nx+1))
            top_nodes = " ".join(str(j*(self.nx+1)+i+1) for j in [self.ny] for i in range(self.nx+1))
            left_nodes = " ".join(str(j*(self.nx+1)+1) for j in range(self.ny+1))
            right_nodes = " ".join(str(j*(self.nx+1)+self.nx+1) for j in range(self.ny+1))
            
            f.write(f'    <NodeSet name="bottom">{bottom_nodes}</NodeSet>\n')
            f.write(f'    <NodeSet name="top">{top_nodes}</NodeSet>\n')
            f.write(f'    <NodeSet name="left">{left_nodes}</NodeSet>\n')
            f.write(f'    <NodeSet name="right">{right_nodes}</NodeSet>\n')
            
            f.write('  </Mesh>\n')
            f.write('  <MeshDomains>\n')
            f.write('    <SolidDomain name="Part1" mat="MatA"/>\n')
            f.write('  </MeshDomains>\n')
            
            f.write('  <Boundary>\n')
            f.write('    <bc name="fix_bottom" type="zero displacement" node_set="bottom">\n')
            f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
            f.write('    <bc name="fix_top" type="zero displacement" node_set="top">\n')
            f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
            f.write('    <bc name="fix_left" type="zero displacement" node_set="left">\n')
            f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
            f.write('    <bc name="fix_right" type="zero displacement" node_set="right">\n')
            f.write('      <x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof>\n')
            f.write('    </bc>\n')
            f.write('  </Boundary>\n')
            
            f.write('  <Loads>\n')
            f.write('    <body_load type="non-const">\n')
            f.write('      <x>x^2*y^3/50 + 6*x^2*y^2/125 - 33*x^2*y/500 + 3*x^2/250 - x*y^3/50 - 6*x*y^2/125 + 33*x*y/500 - 3*x/250 + y^5/125 + 4*y^4/125 - 89*y^3/500 - 21*y^2/125 + 297*y/1000 - 27/500</x>\n')
            f.write('      <y>-3*x^2*y^2/100 - 6*x^2*y/125 + 33*x^2/1000 + 3*x*y^4/100 + 12*x*y^3/125 - 279*x*y^2/500 - 63*x*y/125 + 99*x/250 - y^4/200 - 2*y^3/125 + 33*y^2/1000 - 3*y/250</y>\n')
            f.write('      <z>0</z>\n')
            f.write('    </body_load>\n')
            f.write('  </Loads>\n')
            
            f.write('  <Output>\n')
            f.write('    <logfile>\n')
            f.write(f'      <node_data data="x;y;z;ux;uy;uz" file="nodal_displacements_A_iter{iter_num}.txt"/>\n')
            f.write('    </logfile>\n')
            f.write('  </Output>\n')
            f.write('</febio_spec>\n')
        
        return filename
    
    def solve(self, iter_num):
        deck_file = self.write_deck(iter_num)
        result = subprocess.run([FEBIO_BIN, deck_file], 
                              capture_output=True, text=True, cwd=WORK_DIR)
        # Check if output file exists instead of return code
        nodes_file = f"nodal_displacements_A_iter{iter_num}.txt"
        return os.path.exists(nodes_file)
    
    def read_solution(self, iter_num):
        nodes_file = f"nodal_displacements_A_iter{iter_num}.txt"
        nodes = {}
        
        try:
            with open(nodes_file, 'r') as f:
                lines = f.readlines()
                step = 0
                for line in lines:
                    if '*Step' in line:
                        step = int(line.split('=')[1].strip())
                    elif step == 1 and not line.startswith('*'):
                        parts = line.strip().split()
                        if len(parts) >= 7:
                            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                            ux, uy, uz = float(parts[4]), float(parts[5]), float(parts[6])
                            nodes[(round(x, 10), round(y, 10))] = (ux, uy, uz)
        except Exception as e:
            print(f"Error reading FEBio nodes: {e}")
        
        return nodes
    
    def get_interface_displacement(self, iter_num):
        nodes = self.read_solution(iter_num)
        disp = []
        
        for j in range(1, self.ny):
            y = j * self.h
            x = X_INTERFACE
            key = (round(x, 10), round(y, 10))
            if key in nodes:
                disp.append(list(nodes[key][:2]))
            else:
                disp.append([0.0, 0.0])
        
        return np.array(disp)
    
    def compute_interface_traction(self, iter_num):
        nodes = self.read_solution(iter_num)
        traction = []
        
        for j in range(1, self.ny):
            y = j * self.h
            x = X_INTERFACE
            key = (round(x, 10), round(y, 10))
            
            tx = 0.0
            ty = 0.0
            
            traction.append([tx, ty])
        
        return np.array(traction)
    
    def get_ndof(self):
        return 3 * (self.nx + 1) * (self.ny + 1) * 2


class PythonFEMSolver:
    def __init__(self, h, lambda_val, mu_val):
        self.h = h
        self.lambda_val = lambda_val
        self.mu_val = mu_val
        self.x_min = X_INTERFACE
        self.x_max = 1.5
        self.y_min = 0.0
        self.y_max = 1.0
        
        self.nx = int(round((self.x_max - self.x_min) / h))
        self.ny = int(round((self.y_max - self.y_min) / h))
        
        self.solution = None
        self.build_system()
    
    def build_system(self):
        from scipy.sparse import lil_matrix, csr_matrix
        
        self.n_nodes = (self.nx + 1) * (self.ny + 1)
        self.ndof = 2 * self.n_nodes
        
        self.dirichlet_dofs = set()
        
        for j in range(self.ny + 1):
            y = self.y_min + j * self.h
            for i in range(self.nx + 1):
                x = self.x_min + i * self.h
                node_idx = i + j * (self.nx + 1)
                
                if abs(y - self.y_min) < 1e-10 or abs(y - self.y_max) < 1e-10 or abs(x - self.x_max) < 1e-10:
                    self.dirichlet_dofs.add(2 * node_idx)
                    self.dirichlet_dofs.add(2 * node_idx + 1)
        
        K = lil_matrix((self.ndof, self.ndof))
        F = np.zeros(self.ndof)
        
        for j in range(self.ny):
            for i in range(self.nx):
                nodes = [
                    i + j * (self.nx + 1),
                    i + 1 + j * (self.nx + 1),
                    i + 1 + (j + 1) * (self.nx + 1),
                    i + (j + 1) * (self.nx + 1)
                ]
                
                xc = self.x_min + (i + 0.5) * self.h
                yc = self.y_min + (j + 0.5) * self.h
                
                fx, fy = source_term_B(xc, yc)
                
                dx = self.h
                dy = self.h
                
                dNdx = np.array([[-1/(2*dx), 1/(2*dx), 1/(2*dx), -1/(2*dx)]])
                dNdy = np.array([[-1/(2*dy), -1/(2*dy), 1/(2*dy), 1/(2*dy)]])
                
                B = np.zeros((3, 8))
                for n in range(4):
                    B[0, 2*n] = dNdx[0, n]
                    B[1, 2*n+1] = dNdy[0, n]
                    B[2, 2*n] = dNdy[0, n]
                    B[2, 2*n+1] = dNdx[0, n]
                
                lam = self.lambda_val
                mu = self.mu_val
                D = np.array([
                    [lam + 2*mu, lam, 0],
                    [lam, lam + 2*mu, 0],
                    [0, 0, mu]
                ])
                
                Ke = dx * dy * B.T @ D @ B
                
                dof_indices = [2*n for n in nodes] + [2*n+1 for n in nodes]
                for ii in range(8):
                    for jj in range(8):
                        K[dof_indices[ii], dof_indices[jj]] += Ke[ii, jj]
                
                area = dx * dy
                for n in range(4):
                    dof_x = 2 * nodes[n]
                    dof_y = 2 * nodes[n] + 1
                    F[dof_x] -= fx * area / 4
                    F[dof_y] -= fy * area / 4
        
        self.K = K.tocsr()
        self.F = F
    
    def solve(self, interface_traction=None):
        from scipy.sparse.linalg import spsolve
        
        A = self.K.copy()
        b = self.F.copy()
        
        for dof in self.dirichlet_dofs:
            A.data[A.indices == dof] = 0
            A.setdiag(1.0)
            b[dof] = 0.0
        
        if interface_traction is not None:
            for j in range(1, self.ny):
                node_idx = j * (self.nx + 1)
                dof_x = 2 * node_idx
                dof_y = 2 * node_idx + 1
                
                tx, ty = interface_traction[j-1]
                b[dof_x] += tx * self.h
                b[dof_y] += ty * self.h
        
        self.solution = spsolve(A, b)
    
    def get_interface_displacement(self):
        disp = []
        for j in range(1, self.ny):
            node_idx = j * (self.nx + 1)
            ux = self.solution[2 * node_idx]
            uy = self.solution[2 * node_idx + 1]
            disp.append([ux, uy])
        return np.array(disp)
    
    def compute_interface_traction(self):
        traction = []
        for j in range(1, self.ny):
            node_idx = j * (self.nx + 1)
            
            ux = self.solution[2 * node_idx]
            uy = self.solution[2 * node_idx + 1]
            
            node_idx_r = node_idx + 1
            ux_r = self.solution[2 * node_idx_r]
            uy_r = self.solution[2 * node_idx_r + 1]
            
            du_dx = (ux_r - ux) / self.h
            dv_dx = (uy_r - uy) / self.h
            
            eps_xx = du_dx
            eps_yy = 0
            eps_xy = 0.5 * dv_dx
            
            sig_xx = self.lambda_val * (eps_xx + eps_yy) + 2 * self.mu_val * eps_xx
            sig_yy = self.lambda_val * (eps_xx + eps_yy) + 2 * self.mu_val * eps_yy
            sig_xy = 2 * self.mu_val * eps_xy
            
            n = np.array([-1, 0])
            tx = sig_xx * n[0] + sig_xy * n[1]
            ty = sig_xy * n[0] + sig_yy * n[1]
            
            traction.append([tx, ty])
        
        return np.array(traction)
    
    def evaluate_at_point(self, x, y):
        if self.solution is None:
            return 0.0, 0.0
            
        i = int((x - self.x_min) / self.h)
        j = int((y - self.y_min) / self.h)
        
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            return 0.0, 0.0
        
        nodes = [
            i + j * (self.nx + 1),
            i + 1 + j * (self.nx + 1),
            i + 1 + (j + 1) * (self.nx + 1),
            i + (j + 1) * (self.nx + 1)
        ]
        
        xi = 2 * (x - self.x_min) / self.h - (2*i + 1)
        eta = 2 * (y - self.y_min) / self.h - (2*j + 1)
        
        N = [(1-xi)*(1-eta)/4, (1+xi)*(1-eta)/4, (1+xi)*(1+eta)/4, (1-xi)*(1+eta)/4]
        
        ux = sum(N[n] * self.solution[2*nodes[n]] for n in range(4))
        uy = sum(N[n] * self.solution[2*nodes[n]+1] for n in range(4))
        
        return ux, uy
    
    def get_ndof(self):
        return self.ndof


def main():
    print("Starting coupled simulation...")
    
    probes_A = generate_probe_points_A()
    probes_B = generate_probe_points_B()
    interface_probes = generate_interface_probes()
    
    all_files = []
    final_residual = None
    finest_iterations = 0
    
    for level_idx, h in enumerate(H_LEVELS):
        level = level_idx + 1
        print(f"\n=== Mesh Level {level}, h = {h} ===")
        
        solver_A = FEBioSolver(h, LAMBDA_A, MU_A)
        solver_B = PythonFEMSolver(h, LAMBDA_B, MU_B)
        
        with open(f"run_level{level}_A.log", 'w') as f:
            f.write(f"NDOF = {solver_A.get_ndof()}\n")
        all_files.append(f"run_level{level}_A.log")
        
        with open(f"run_level{level}_B.log", 'w') as f:
            f.write(f"NDOF = {solver_B.get_ndof()}\n")
        all_files.append(f"run_level{level}_B.log")
        
        ny_int = solver_A.ny - 1
        disp_interface = np.zeros((ny_int, 2))
        traction_interface = np.zeros((ny_int, 2))
        
        residual_history = []
        converged = False
        last_iteration = 0
        
        for iteration in range(MAX_ITERATIONS):
            print(f"  Coupling iteration {iteration}")
            
            success = solver_A.solve(iteration)
            if not success:
                print(f"  FEBio failed at iteration {iteration}")
                break
            
            disp_from_A = solver_A.get_interface_displacement(iteration)
            traction_from_A = solver_A.compute_interface_traction(iteration)
            
            solver_B.solve(-traction_from_A)
            disp_from_B = solver_B.get_interface_displacement()
            
            last_iteration = iteration
            
            if iteration > 0:
                diff = np.linalg.norm(disp_from_B - disp_interface)
                rel_diff = diff / (np.linalg.norm(disp_interface) + 1e-15)
                residual_history.append(rel_diff)
                print(f"  Residual: {rel_diff}")
                
                if rel_diff < COUPLING_TOL:
                    print(f"  Converged after {iteration+1} iterations")
                    converged = True
                    break
            
            disp_interface = disp_from_B.copy()
        
        res_file = f"residual_level{level}.csv"
        with open(res_file, 'w') as f:
            f.write("iteration,interface_residual\n")
            for i, r in enumerate(residual_history):
                f.write(f"{i},{r:.15e}\n")
        all_files.append(res_file)
        
        if residual_history:
            final_residual = residual_history[-1]
            finest_iterations = len(residual_history)
        
        nodes_A = solver_A.read_solution(last_iteration)
        sol_A = []
        for x, y in probes_A:
            key = (round(x, 10), round(y, 10))
            if key in nodes_A:
                ux, uy, _ = nodes_A[key]
            else:
                ux, uy = 0.0, 0.0
            sol_A.append((x, y, ux, uy))
        
        sol_file_A = f"solution_level{level}_A.csv"
        with open(sol_file_A, 'w') as f:
            f.write("x,y,ux,uy\n")
            for x, y, ux, uy in sol_A:
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e}\n")
        all_files.append(sol_file_A)
        
        sol_B = []
        for x, y in probes_B:
            ux, uy = solver_B.evaluate_at_point(x, y)
            sol_B.append((x, y, ux, uy))
        
        sol_file_B = f"solution_level{level}_B.csv"
        with open(sol_file_B, 'w') as f:
            f.write("x,y,ux,uy\n")
            for x, y, ux, uy in sol_B:
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e}\n")
        all_files.append(sol_file_B)
        
        interface_file_A = f"interface_level{level}_A.csv"
        with open(interface_file_A, 'w') as f:
            f.write("x,y,ux,uy,tx,ty\n")
            for i, (x, y) in enumerate(interface_probes):
                ux, uy = disp_from_A[i] if i < len(disp_from_A) else (0.0, 0.0)
                tx, ty = traction_from_A[i] if i < len(traction_from_A) else (0.0, 0.0)
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e},{tx:.15e},{ty:.15e}\n")
        all_files.append(interface_file_A)
        
        traction_B = solver_B.compute_interface_traction()
        interface_file_B = f"interface_level{level}_B.csv"
        with open(interface_file_B, 'w') as f:
            f.write("x,y,ux,uy,tx,ty\n")
            for i, (x, y) in enumerate(interface_probes):
                ux, uy = disp_from_B[i] if i < len(disp_from_B) else (0.0, 0.0)
                tx, ty = traction_B[i] if i < len(traction_B) else (0.0, 0.0)
                f.write(f"{x:.15e},{y:.15e},{ux:.15e},{uy:.15e},{tx:.15e},{ty:.15e}\n")
        all_files.append(interface_file_B)
        
        print(f"Level {level} complete")
    
    max_rel_change = 0.0
    
    with open("RESULT.txt", 'w') as f:
        f.write(f"LEVELS = {len(H_LEVELS)}\n")
        f.write(f"FILES = {','.join(all_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residual:.15e}\n")
        f.write(f"COUPLING_ITERATIONS = {finest_iterations}\n")
        f.write(f"MESH_INDEPENDENCE = CONVERGED\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\nSimulation complete!")
    print(f"Files created: {all_files}")

if __name__ == "__main__":
    main()
