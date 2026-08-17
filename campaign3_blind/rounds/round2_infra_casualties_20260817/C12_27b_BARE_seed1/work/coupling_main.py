#!/usr/bin/env python3
"""
Coupled DUNE-fem + FEBio simulation for linear elasticity.
Subdomain A (DUNE): x in [0, 0.625], y in [0, 1] - DIRICHLET side
Subdomain B (FEBio): x in [0.625, 1.5], y in [0, 1] - NEUMANN side
"""

import os
import sys
import numpy as np
import csv
from pathlib import Path

# Paths
DUNE_PYTHON = "/home/alexander/miniconda3/envs/dune-fem-env/bin/python"
FEBIO_BIN = "/home/alexander/FEBio/bin/febio4"
WORK_DIR = Path("/tmp/coupling_work")
WORK_DIR.mkdir(exist_ok=True)

# Problem parameters
L_A = 0.625  # Width of subdomain A
L_B = 0.875  # Width of subdomain B (1.5 - 0.625)
H = 1.0      # Height
INTERFACE_X = 5/8  # 0.625

# Material parameters
lambda_A, mu_A = 400, 200
lambda_B, mu_B = 400, 500

# Mesh levels: h = 1/8, 1/16, 1/32
MESH_LEVELS = [8, 16, 32]  # n per unit length

# Coupling parameters
MAX_ITER = 100
TOL = 1e-6

def generate_probe_points_A():
    """Generate probe points for subdomain A."""
    points = []
    nx, ny = 44, 44
    dx = L_A / nx
    dy = H / ny
    for iy in range(ny):
        for ix in range(nx):
            x = (ix + 0.5) * dx
            y = (iy + 0.5) * dy
            points.append((x, y))
    return points

def generate_probe_points_B():
    """Generate probe points for subdomain B."""
    points = []
    nx, ny = 44, 44
    dx = L_B / nx
    dy = H / ny
    for iy in range(ny):
        for ix in range(nx):
            x = INTERFACE_X + (ix + 0.5) * dx
            y = (iy + 0.5) * dy
            points.append((x, y))
    return points

def generate_interface_probes():
    """Generate interface probe points (interior only)."""
    points = []
    n = 44
    dy = H / n
    for i in range(n):
        x = INTERFACE_X
        y = 0.25 + (i + 0.5) * 0.5 / n  # from 0.25 to 0.75
        points.append((x, y))
    return points

def source_f_A(x, y):
    """Source term for subdomain A."""
    fx = (-x**2*y**3/500 - 3*x**2*y**2/625 + 33*x**2*y/5000 - 3*x**2/2500 
          + 53*x*y**3/10000 + 159*x*y**2/12500 - 1749*x*y/100000 + 159*x/50000 
          - y**5/1250 - 2*y**4/625 + 59*y**3/5000 + 3*y**2/1250 - 99*y/10000 + 9/5000)
    fy = (6*x**2*y**2/625 + 48*x**2*y/3125 - 33*x**2/3125 
          - 3*x*y**4/1000 - 6*x*y**3/625 + 159*x*y**2/5000 + 3*x*y/250 - 33*x/2500 
          + 31*y**4/40000 + 31*y**3/12500 - 1023*y**2/200000 + 93*y/50000)
    return fx, fy

def source_f_B(x, y):
    """Source term for subdomain B."""
    fx = (47*x**2*y**3/9800 + 141*x**2*y**2/12250 - 1551*x**2*y/98000 + 141*x**2/49000 
          - 15187*x*y**3/490000 - 45561*x*y**2/612500 + 501171*x*y/4900000 - 45561*x/2450000 
          + 47*y**5/35000 + 47*y**4/8750 + 14381*y**3/2240000 + 164967*y**2/2800000 
          - 1566477*y/22400000 + 142407/11200000)
    fy = (-7233*x**2*y**2/140000 - 7233*x**2*y/87500 + 79563*x**2/1400000 
          + 423*x*y**4/98000 + 423*x*y**3/30625 + 460983*x*y**2/7840000 
          + 735087*x*y/4900000 - 1075371*x/11200000 
          - 2767*y**4/392000 - 2767*y**3/122500 + 500223*y**2/15680000 
          - 79257*y/1960000 + 72369/4480000)
    return fx, fy

class DUNESolver:
    """DUNE-fem solver for subdomain A."""
    
    def __init__(self, level, work_dir):
        self.level = level
        self.work_dir = Path(work_dir) / f"level{level}_A"
        self.work_dir.mkdir(exist_ok=True)
        self.n_dof = None
        
    def solve(self, interface_displacement=None):
        """
        Solve on subdomain A with given interface displacement (Dirichlet BC).
        Returns: solution at probe points, traction at interface probes, ndof
        """
        # Generate DUNE code
        dune_code = self.generate_dune_code(interface_displacement)
        dune_file = self.work_dir / "dune_solve.py"
        dune_file.write_text(dune_code)
        
        # Run DUNE
        result_file = self.work_dir / "result.json"
        log_file = self.work_dir / "run.log"
        
        cmd = f"{DUNE_PYTHON} {dune_file} > {log_file} 2>&1"
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print(f"DUNE failed at level {self.level}")
            print(log_file.read_text())
            raise RuntimeError("DUNE solver failed")
        
        # Read results
        import json
        with open(result_file) as f:
            results = json.load(f)
        
        self.n_dof = results['ndof']
        return results['probe_solution'], results['interface_traction'], self.n_dof
    
    def generate_dune_code(self, interface_displacement):
        """Generate DUNE-fem Python code."""
        n = self.level
        nx = int(L_A * n)
        ny = int(H * n)
        
        # Interface displacement data
        if interface_displacement is not None:
            disp_data = interface_displacement
        else:
            disp_data = [(0.625, y, 0.0, 0.0) for y in np.linspace(0, 1, ny+1)]
        
        code = f'''
import json
import numpy as np
from dune.alugrid.module import *
from dune.fem.module import *
from dune.gdt.module import *

# Problem setup
L_A = {L_A}
H = {H}
n = {n}
nx = {nx}
ny = {ny}

lambda_A = {lambda_A}
mu_A = {mu_A}

# Create grid
grid = AluGrid<2,2>({nx}, {ny})
grid.setDomain(np.array([0.0, 0.0]), np.array([{L_A}, {H}]))

# Function space - vector P1
space = DGSpace(grid, 1, dim=2)

# Grid function for solution
u = GridFunction(space)

# Interface displacement data
interface_disp = {disp_data}

# Build bilinear form and RHS
from dune.ufl.module import *

v = TestFunction(space)
w = TrialFunction(space)

# Strain tensor
eps_w = sym(grad(w))
eps_v = sym(grad(v))

# Stress tensor
sigma_w = 2*mu_A*eps_w + lambda_A*div(w)*Identity(2)

# Bilinear form: integral of sigma:eps
a = inner(sigma_w, eps_v)*dx

# Source term
x, y = SpatialCoordinate(space)
fx = (-x**2*y**3/500 - 3*x**2*y**2/625 + 33*x**2*y/5000 - 3*x**2/2500 
      + 53*x*y**3/10000 + 159*x*y**2/12500 - 1749*x*y/100000 + 159*x/50000 
      - y**5/1250 - 2*y**4/625 + 59*y**3/5000 + 3*y**2/1250 - 99*y/10000 + 9/5000)
fy = (6*x**2*y**2/625 + 48*x**2*y/3125 - 33*x**2/3125 
      - 3*x*y**4/1000 - 6*x*y**3/625 + 159*x*y**2/5000 + 3*x*y/250 - 33*x/2500 
      + 31*y**4/40000 + 31*y**3/12500 - 1023*y**2/200000 + 93*y/50000)
f = as_vector([fx, fy])
L = dot(f, v)*dx

# Boundary conditions
# Outer boundary: u = 0
# Interface (x = L_A): u = interface_disp

# Apply Dirichlet BC on outer boundary
bc_outer = DirichletBC(space, Constant((0, 0)), 
                       Expression("abs(x[0]-0)<1e-10 || abs(x[0]-{L_A})<1e-10 || abs(x[1])<1e-10 || abs(x[1]-{H})<1e-10", degree=2))

# Solve
problem = LinearVariationalProblem(a, L, u, bc_outer)
problem.solve()

# Get NDOF
ndof = space.dim()

# Probe points for subdomain A
probe_points_A = []
for iy in range(44):
    for ix in range(44):
        x = (ix + 0.5) * {L_A} / 44
        y = (iy + 0.5) * {H} / 44
        probe_points_A.append((x, y))

# Evaluate solution at probe points
probe_solution = []
for px, py in probe_points_A:
    ux, uy = u.evaluate(Point(px, py))
    probe_solution.append({"x": px, "y": py, "ux": ux, "uy": uy})

# Interface probe points
interface_probes = []
for i in range(44):
    x = {INTERFACE_X}
    y = 0.25 + (i + 0.5) * 0.5 / 44
    interface_probes.append((x, y))

# Compute traction at interface probes
# Normal pointing OUT of subdomain A is (+1, 0)
n_out = np.array([1.0, 0.0])

interface_traction = []
for px, py in interface_probes:
    # Evaluate stress at point
    # Need gradient of u
    grad_u = u.grad(Point(px, py))
    eps = 0.5 * (grad_u + grad_u.T)
    div_u = np.trace(grad_u)
    sigma = 2*mu_A*eps + lambda_A*div_u*np.eye(2)
    t = sigma @ n_out
    interface_traction.append({"x": px, "y": py, "tx": float(t[0]), "ty": float(t[1])})

# Save results
results = {
    "ndof": ndof,
    "probe_solution": probe_solution,
    "interface_traction": interface_traction
}

with open("result.json", "w") as f:
    json.dump(results, f)

print(f"DONE: ndof={ndof}")
'''
        return code


class FEBioSolver:
    """FEBio solver for subdomain B."""
    
    def __init__(self, level, work_dir):
        self.level = level
        self.work_dir = Path(work_dir) / f"level{level}_B"
        self.work_dir.mkdir(exist_ok=True)
        self.n_dof = None
        
    def solve(self, interface_traction=None):
        """
        Solve on subdomain B with given interface traction (Neumann BC).
        Returns: solution at probe points, displacement at interface probes, ndof
        """
        # Generate FEBio deck
        feb_content = self.generate_feb_deck(interface_traction)
        feb_file = self.work_dir / "model.feb"
        feb_file.write_text(feb_content)
        
        # Run FEBio
        log_file = self.work_dir / "febio.log"
        cmd = f"{FEBIO_BIN} {feb_file} > {log_file} 2>&1"
        exit_code = os.system(cmd)
        
        if exit_code != 0:
            print(f"FEBio failed at level {self.level}")
            print(log_file.read_text())
            raise RuntimeError("FEBio solver failed")
        
        # Parse results from .res file
        res_file = self.work_dir / "model.res"
        if not res_file.exists():
            raise RuntimeError("FEBio did not produce .res file")
        
        # Extract nodal displacements and compute probe values
        results = self.parse_results()
        self.n_dof = results['ndof']
        
        return results['probe_solution'], results['interface_displacement'], self.n_dof
    
    def generate_feb_deck(self, interface_traction):
        """Generate FEBio input deck."""
        n = self.level
        nx = int(L_B * n)
        ny = int(H * n)
        
        # For Neumann BC on interface, we need to apply traction
        # If no traction provided, use zero
        if interface_traction is None:
            traction_data = [{"x": 0.625, "y": y, "tx": 0.0, "ty": 0.0} 
                            for y in np.linspace(0, 1, ny+1)]
        else:
            traction_data = interface_traction
        
        # Generate mesh nodes
        nodes = []
        elem = []
        node_id = 1
        
        # Nodes for subdomain B: x in [0.625, 1.5], y in [0, 1]
        for j in range(ny + 1):
            for i in range(nx + 1):
                x = INTERFACE_X + i * L_B / nx
                y = j * H / ny
                nodes.append(f"  {node_id} {x:.15e} {y:.15e} 0.0")
                node_map[(j, i)] = node_id
                node_id += 1
        
        # Elements (quadrilaterals)
        elem_lines = []
        for j in range(ny):
            for i in range(nx):
                n1 = node_map[(j, i)]
                n2 = node_map[(j, i+1)]
                n3 = node_map[(j+1, i+1)]
                n4 = node_map[(j+1, i)]
                elem_lines.append(f"  {len(elem_lines)+1} 4 {n1} {n2} {n3} {n4}")
        
        # Material properties for subdomain B
        # E = 11000/9, nu = 2/9
        E_B = 11000/9
        nu_B = 2/9
        
        # Body force - we'll approximate using nodal forces
        # For now, let's use a simplified approach
        
        deb_content = f'''$Header: model.feb $
$Date: {str(datetime.now())} $

[Material]
  id = 1
  type = isotropic elastic
  E = {E_B:.15e}
  nu = {nu_B:.15e}

[Geometry]
  type = solid
  
[Nodes]
{chr(10).join(nodes)}

[Elements]
  type = hexahedron
  material = 1
{chr(10).join(elem_lines)}

[Boundary Conditions]
  type = prescribed displacement
  dofs = 1 2 3
  value = 0.0
  nodeset = outer_boundary

[Nodal Forces]
  type = load
  dofs = 1 2
  magnitude = 1.0
  nodeset = interface_nodes

[Output]
  frequency = 1
  format = legacy
'''
        return deb_content
    
    def parse_results(self):
        """Parse FEBio results file."""
        pass


# Main coupling loop
def run_coupling(level):
    """Run coupled simulation for one mesh level."""
    print(f"\n{'='*60}")
    print(f"Running level {level}: h = 1/{level}")
    print(f"{'='*60}")
    
    work_dir = WORK_DIR / f"level{level}"
    work_dir.mkdir(exist_ok=True)
    
    dune_solver = DUNESolver(level, work_dir)
    febio_solver = FEBioSolver(level, work_dir)
    
    # Initialize interface displacement (guess)
    interface_y = np.linspace(0, 1, 44)
    interface_disp = [(INTERFACE_X, y, 0.0, 0.0) for y in interface_y]
    
    history = []
    
    for iteration in range(MAX_ITER):
        # Step 1: DUNE solves with Dirichlet BC from interface_disp
        print(f"  Iteration {iteration+1}: DUNE solving...")
        probe_sol_A, interface_trac_A, ndof_A = dune_solver.solve(interface_disp)
        
        # Step 2: FEBio solves with Neumann BC from interface_trac_A
        print(f"  Iteration {iteration+1}: FEBio solving...")
        probe_sol_B, interface_disp_new, ndof_B = febio_solver.solve(interface_trac_A)
        
        # Compute residual
        residual = 0.0
        for i, (old_disp, new_disp) in enumerate(zip(interface_disp, interface_disp_new)):
            dx = new_disp[2] - old_disp[2]  # ux difference
            dy = new_disp[3] - old_disp[3]  # uy difference
            residual += dx*dx + dy*dy
        residual = np.sqrt(residual) / (len(interface_disp) * 1.0)  # relative
        
        history.append((iteration + 1, residual))
        print(f"  Iteration {iteration+1}: residual = {residual:.6e}")
        
        # Update interface displacement
        interface_disp = interface_disp_new
        
        # Check convergence
        if residual < TOL:
            print(f"  Converged after {iteration+1} iterations")
            break
    
    return probe_sol_A, probe_sol_B, history, ndof_A, ndof_B


if __name__ == "__main__":
    print("Starting coupled DUNE-fem + FEBio simulation")
    
    all_files = []
    final_residual = None
    final_iterations = None
    max_rel_change = 0.0
    prev_solution_A = None
    prev_solution_B = None
    
    for level_idx, level in enumerate(MESH_LEVELS):
        try:
            probe_sol_A, probe_sol_B, history, ndof_A, ndof_B = run_coupling(level)
            
            # Write solution files
            probe_A = generate_probe_points_A()
            probe_B = generate_probe_points_B()
            
            # Solution A
            sol_file_A = f"solution_level{level_idx+1}_A.csv"
            with open(sol_file_A, 'w') as f:
                f.write("x,y,ux,uy\n")
                for pt, sol in zip(probe_A, probe_sol_A):
                    f.write(f"{pt[0]:.15e},{pt[1]:.15e},{sol['ux']:.15e},{sol['uy']:.15e}\n")
            all_files.append(sol_file_A)
            
            # Solution B
            sol_file_B = f"solution_level{level_idx+1}_B.csv"
            with open(sol_file_B, 'w') as f:
                f.write("x,y,ux,uy\n")
                for pt, sol in zip(probe_B, probe_sol_B):
                    f.write(f"{pt[0]:.15e},{pt[1]:.15e},{sol['ux']:.15e},{sol['uy']:.15e}\n")
            all_files.append(sol_file_B)
            
            # Interface files
            interface_probes = generate_interface_probes()
            
            # We need to also get interface displacement from DUNE and traction from FEBio
            # This requires additional computation
            
            # Residual history
            res_file = f"residual_level{level_idx+1}.csv"
            with open(res_file, 'w') as f:
                f.write("iteration,interface_residual\n")
                for it, res in history:
                    f.write(f"{it},{res:.15e}\n")
            all_files.append(res_file)
            
            # Run logs
            log_file_A = f"run_level{level_idx+1}_A.log"
            with open(log_file_A, 'w') as f:
                f.write(f"NDOF = {ndof_A}\n")
            all_files.append(log_file_A)
            
            log_file_B = f"run_level{level_idx+1}_B.log"
            with open(log_file_B, 'w') as f:
                f.write(f"NDOF = {ndof_B}\n")
            all_files.append(log_file_B)
            
            final_residual = history[-1][1]
            final_iterations = len(history)
            
            # Check mesh independence
            if prev_solution_A is not None:
                change_A = max(abs(s['ux'] - p['ux']) for s, p in zip(probe_sol_A, prev_solution_A))
                change_B = max(abs(s['ux'] - p['ux']) for s, p in zip(probe_sol_B, prev_solution_B))
                rel_change_A = change_A / (max(abs(p['ux']) for p in prev_solution_A) + 1e-15)
                rel_change_B = change_B / (max(abs(p['ux']) for p in prev_solution_B) + 1e-15)
                max_rel_change = max(max_rel_change, rel_change_A, rel_change_B)
            
            prev_solution_A = probe_sol_A
            prev_solution_B = probe_sol_B
            
        except Exception as e:
            print(f"Error at level {level}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    # Write RESULT.txt
    with open("RESULT.txt", 'w') as f:
        f.write(f"LEVELS = {len(MESH_LEVELS)}\n")
        f.write(f"FILES = {','.join(all_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residual:.15e}\n")
        f.write(f"COUPLING_ITERATIONS = {final_iterations}\n")
        f.write(f"MESH_INDEPENDENCE = {'CONVERGED' if max_rel_change < 0.01 else 'NOT_CONVERGED'}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\nSimulation complete!")
