#!/usr/bin/env python3
"""
Main coupling script for Dirichlet-Neumann iteration between NGSolve (A) and Kratos (B).
"""
import sys
import json
from pathlib import Path
import subprocess
import numpy as np

def get_interface_probes():
    """Get interface probe points."""
    probes = []
    for i in range(44):
        x = 5/8  # 0.625
        y = 1/4 + (i + 0.5) * 1/2 / 44
        probes.append((x, y))
    return probes

def run_ngsolve_A(level, work_dir, interface_field=None):
    """Run NGSolve solver for subdomain A."""
    imports_path = Path(work_dir) / "imports.json"
    
    if interface_field is not None:
        # Write imports.json with field from B
        imports = {"B_to_A": {"field": interface_field}}
        with open(imports_path, 'w') as f:
            json.dump(imports, f)
    else:
        # First iteration - no imports needed
        with open(imports_path, 'w') as f:
            json.dump({}, f)
    
    result = subprocess.run(
        [sys.executable, "ngsolve_A.py", str(level), work_dir],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("NGSolve STDERR:", result.stderr[:2000])
    return result.returncode == 0

def run_kratos_B(level, work_dir, interface_flux=None):
    """Run Kratos solver for subdomain B."""
    imports_path = Path(work_dir) / "imports.json"
    
    if interface_flux is not None:
        # Write imports.json with flux from A
        imports = {"A_to_B": {"flux": interface_flux}}
        with open(imports_path, 'w') as f:
            json.dump(imports, f)
    else:
        # First iteration - no imports needed
        with open(imports_path, 'w') as f:
            json.dump({}, f)
    
    result = subprocess.run(
        [sys.executable, "kratos_B.py", str(level), work_dir],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Kratos STDERR:", result.stderr[:2000])
    return result.returncode == 0

def evaluate_solution_A(level, work_dir):
    """Evaluate solution at probe points for subdomain A."""
    script = f'''
import sys
sys.path.insert(0, '/home/alexander/Schreibtisch/open-fem-agent/.venv/lib/python3.12/site-packages')

import ngsolve as ng
from ngsolve import *
import numpy as np
import json
from pathlib import Path

level = {level}
work_dir = "{work_dir}"

# Domain parameters for subdomain A
x_max_A = 0.625
y_max = 1.0
k_A = 1.0

# Mesh size
h = 1.0 / (8 * level)
nx = int(x_max_A / h + 0.5)
ny = int(y_max / h + 0.5)

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

# Read interface field from exports.json (last computed by B)
exports_path = Path(work_dir) / "exports.json"
with open(exports_path) as f:
    exports = json.load(f)

if "B_to_A" in exports and "field" in exports["B_to_A"]:
    partner_y = [pt[0] for pt in exports["B_to_A"]["field"]]
    partner_u = [pt[1] for pt in exports["B_to_A"]["field"]]
    
    for i, dof in enumerate(interface_dofs_sorted):
        y_pos = interface_y_coords[i]
        u_val = np.interp(y_pos, partner_y, partner_u)
        gfu.vec[dof] = u_val

# Apply outer BCs
for dof in outer_dofs:
    gfu.vec[dof] = 0

# Solve
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

# Evaluate at probe points
probes = []
for i_x in range(44):
    for i_y in range(44):
        x = 0 + (i_x + 0.5) * 0.625 / 44
        y = 0 + (i_y + 0.5) * 1.0 / 44
        probes.append((x, y))

results = []
for x, y in probes:
    u_val = gfu.Evaluate(Point(x, y), mesh).x
    results.append((x, y, u_val))

# Write CSV
csv_path = Path(work_dir) / f"solution_level{level}_A.csv"
with open(csv_path, 'w') as f:
    f.write("x, y, u\\n")
    for px, py, u in results:
        f.write(f"{px:.15e}, {py:.15e}, {u:.15e}\\n")

print(f"Evaluated {len(results)} probe points for subdomain A")
'''
    
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])
    return result.returncode == 0

def evaluate_solution_B(level, work_dir):
    """Evaluate solution at probe points for subdomain B."""
    script = f'''
import sys
sys.path.insert(0, '/home/alexander/Schreibtisch/open-fem-agent/.venv/lib/python3.12/site-packages')

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication
import json
from pathlib import Path

level = {level}
work_dir = "{work_dir}"

# Domain parameters for subdomain B
x_min_B = 0.625
x_max_B = 1.5
y_max = 1.0
k_B = 1000.0

# Mesh size
h = 1.0 / (8 * level)
nx = int((x_max_B - x_min_B) / h + 0.5)
ny = int(y_max / h + 0.5)

dx = (x_max_B - x_min_B) / nx
dy = y_max / ny

# Create model
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

nid = {}
cnt = 1
for j in range(ny + 1):
    for i in range(nx + 1):
        x = x_min_B + i * dx
        y = j * dy
        mp.CreateNewNode(cnt, x, y, 0.0)
        nid[(i, j)] = cnt
        cnt += 1

eid = 1
for j in range(ny):
    for i in range(nx):
        a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
        mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
        mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1

def source_B(x, y):
    return (-9*x**3*y/2500000 + x**3/2500000 - 64901*x**2*y/10000000 - 25033*x**2/30000000 
            - 9*x*y**3/2500000 + 3*x*y**2/2500000 - 422790921*x*y/160000000 - 21609971*x/32000000 
            - 64901*y**3/30000000 - 25033*y**2/30000000 + 1274009971*y/320000000 + 12989997/12800000)

for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, k_B)
    node.SetSolutionStepValue(KM.HEAT_FLUX, source_B(node.X, node.Y))
    node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

# Outer Dirichlet BCs
for i in range(nx + 1):
    n = mp.Nodes[nid[(i, ny)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)
    n = mp.Nodes[nid[(i, 0)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

for j in range(ny + 1):
    n = mp.Nodes[nid[(nx, j)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Interface Neumann BC
exports_path = Path(work_dir) / "exports.json"
with open(exports_path) as f:
    exports = json.load(f)

if "A_to_B" in exports and "flux" in exports["A_to_B"]:
    partner_y = [pt[0] for pt in exports["A_to_B"]["flux"]]
    partner_qn = [pt[1] for pt in exports["A_to_B"]["flux"]]
    
    for j in range(ny + 1):
        n = mp.Nodes[nid[(0, j)]]
        y_pos = n.Y
        qn_A = np.interp(y_pos, partner_y, partner_qn)
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, -qn_A)

KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)

scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
strategy.Initialize()
strategy.Solve()

# Evaluate at probe points using linear interpolation over elements
probes = []
for i_x in range(44):
    for i_y in range(44):
        x = 0.625 + (i_x + 0.5) * 0.875 / 44
        y = 0 + (i_y + 0.5) * 1.0 / 44
        probes.append((x, y))

def interpolate_at_point(x, y, mp, nid, nx, ny, dx, dy):
    """Linear interpolation at point (x, y) using surrounding nodes."""
    # Find element containing point
    i = int((x - x_min_B) / dx)
    j = int(y / dy)
    
    if i >= nx: i = nx - 1
    if j >= ny: j = ny - 1
    if i < 0: i = 0
    if j < 0: j = 0
    
    # Get corner nodes of element
    n00 = mp.Nodes[nid[(i, j)]]
    n10 = mp.Nodes[nid[(i+1, j)]]
    n11 = mp.Nodes[nid[(i+1, j+1)]]
    n01 = mp.Nodes[nid[(i, j+1)]]
    
    u00 = n00.GetSolutionStepValue(KM.TEMPERATURE)
    u10 = n10.GetSolutionStepValue(KM.TEMPERATURE)
    u11 = n11.GetSolutionStepValue(KM.TEMPERATURE)
    u01 = n01.GetSolutionStepValue(KM.TEMPERATURE)
    
    # Bilinear interpolation
    xi = (x - (x_min_B + i*dx)) / dx
    eta = (y - j*dy) / dy
    
    u = (1-xi)*(1-eta)*u00 + xi*(1-eta)*u10 + xi*eta*u11 + (1-xi)*eta*u01
    return u

results = []
for x, y in probes:
    u_val = interpolate_at_point(x, y, mp, nid, nx, ny, dx, dy)
    results.append((x, y, u_val))

# Write CSV
csv_path = Path(work_dir) / f"solution_level{level}_B.csv"
with open(csv_path, 'w') as f:
    f.write("x, y, u\\n")
    for px, py, u in results:
        f.write(f"{px:.15e}, {py:.15e}, {u:.15e}\\n")

print(f"Evaluated {len(results)} probe points for subdomain B")
'''
    
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])
    return result.returncode == 0

def write_interface_data(level, work_dir, side, field_data, flux_data):
    """Write interface data to CSV."""
    csv_path = Path(work_dir) / f"interface_level{level}_{side}.csv"
    
    with open(csv_path, 'w') as f:
        f.write("x, y, u, qn\n")
        for i in range(len(field_data)):
            y = field_data[i][0]
            u = field_data[i][1]
            qn = flux_data[i][1] if i < len(flux_data) else 0
            x = 5/8  # Interface x coordinate
            f.write(f"{x:.15e}, {y:.15e}, {u:.15e}, {qn:.15e}\n")

def main():
    work_dir = "."
    
    levels = [1, 2, 3]  # h = 1/8, 1/16, 1/32
    all_files = []
    final_residual = None
    final_iterations = None
    max_rel_change = 0
    
    prev_solution_A = None
    prev_solution_B = None
    
    for level in levels:
        print(f"\n{'='*60}")
        print(f"Mesh level {level}: h = 1/{8*level}")
        print(f"{'='*60}")
        
        # Coupling iteration parameters
        max_iter = 500
        tol = 1e-6
        relaxation = 0.01  # Small relaxation for high contrast
        
        # Initial guess for interface field (from B to A)
        interface_probes = get_interface_probes()
        interface_field = [[y, 0.0] for x, y in interface_probes]
        
        residual_history = []
        
        for iteration in range(max_iter):
            # Step 1: Solve subdomain A (Dirichlet side) with interface field from B
            success_A = run_ngsolve_A(level, work_dir, interface_field)
            if not success_A:
                print("ERROR: NGSolve failed")
                break
            
            # Read exports from A
            with open(Path(work_dir) / "exports.json") as f:
                exports = json.load(f)
            
            flux_from_A = exports["A_to_B"]["flux"]  # [y, qn]
            field_from_A = exports["A_to_B"]["field"]  # [y, u]
            
            # Step 2: Solve subdomain B (Neumann side) with flux from A
            success_B = run_kratos_B(level, work_dir, flux_from_A)
            if not success_B:
                print("ERROR: Kratos failed")
                break
            
            # Read exports from B
            with open(Path(work_dir) / "exports.json") as f:
                exports = json.load(f)
            
            field_from_B = exports["B_to_A"]["field"]  # [y, u]
            flux_from_B = exports["B_to_A"]["flux"]  # [y, qn]
            
            # Compute residual (relative difference in interface field)
            old_field = interface_field
            new_field = field_from_B
            
            residual = 0
            for i in range(len(new_field)):
                diff = abs(new_field[i][1] - old_field[i][1])
                ref = max(abs(old_field[i][1]), abs(new_field[i][1]), 1e-10)
                residual += diff / ref
            residual /= len(new_field)
            
            residual_history.append((iteration + 1, residual))
            
            # Relaxation update
            interface_field = []
            for i in range(len(new_field)):
                y = new_field[i][0]
                u_new = new_field[i][1]
                u_old = old_field[i][1]
                u_relaxed = u_old + relaxation * (u_new - u_old)
                interface_field.append([y, u_relaxed])
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration+1}: residual = {residual:.6e}")
            
            if residual < tol:
                print(f"Converged after {iteration+1} iterations")
                break
        
        # Write residual history
        csv_path = Path(work_dir) / f"residual_level{level}.csv"
        with open(csv_path, 'w') as f:
            f.write("iteration, interface_residual\n")
            for it, res in residual_history:
                f.write(f"{it}, {res:.15e}\n")
        all_files.append(f"residual_level{level}.csv")
        
        final_residual = residual_history[-1][1]
        final_iterations = len(residual_history)
        
        # Re-run solvers one more time to get final solution
        run_ngsolve_A(level, work_dir, interface_field)
        with open(Path(work_dir) / "exports.json") as f:
            exports = json.load(f)
        flux_from_A = exports["A_to_B"]["flux"]
        
        run_kratos_B(level, work_dir, flux_from_A)
        with open(Path(work_dir) / "exports.json") as f:
            exports = json.load(f)
        field_from_B = exports["B_to_A"]["field"]
        
        # Write interface data
        # For side A: field and flux from A's perspective
        with open(Path(work_dir) / "exports.json") as f:
            exports = json.load(f)
        
        # Run A again to get its interface data
        run_ngsolve_A(level, work_dir, field_from_B)
        with open(Path(work_dir) / "exports.json") as f:
            exports_A = json.load(f)
        field_A = exports_A["A_to_B"]["field"]
        flux_A = exports_A["A_to_B"]["flux"]
        
        # Run B again to get its interface data
        run_kratos_B(level, work_dir, flux_A)
        with open(Path(work_dir) / "exports.json") as f:
            exports_B = json.load(f)
        field_B = exports_B["B_to_A"]["field"]
        flux_B = exports_B["B_to_A"]["flux"]
        
        write_interface_data(level, work_dir, "A", field_A, flux_A)
        write_interface_data(level, work_dir, "B", field_B, flux_B)
        all_files.append(f"interface_level{level}_A.csv")
        all_files.append(f"interface_level{level}_B.csv")
        
        # Evaluate solutions at probe points
        evaluate_solution_A(level, work_dir)
        evaluate_solution_B(level, work_dir)
        all_files.append(f"solution_level{level}_A.csv")
        all_files.append(f"solution_level{level}_B.csv")
        
        # Check mesh independence
        if prev_solution_A is not None:
            # Read previous and current solutions
            with open(f"solution_level{levels[level-1]}_A.csv") as f:
                lines_A_prev = f.readlines()[1:]  # Skip header
            with open(f"solution_level{level}_A.csv") as f:
                lines_A_curr = f.readlines()[1:]
            
            max_change_A = 0
            for i in range(min(len(lines_A_prev), len(lines_A_curr))):
                parts_prev = lines_A_prev[i].strip().split(',')
                parts_curr = lines_A_curr[i].strip().split(',')
                u_prev = float(parts_prev[2])
                u_curr = float(parts_curr[2])
                rel_change = abs(u_curr - u_prev) / max(abs(u_prev), abs(u_curr), 1e-10)
                max_change_A = max(max_change_A, rel_change)
            
            max_rel_change = max(max_rel_change, max_change_A)
        
        prev_solution_A = level
        
        print(f"Level {level} complete, residual = {final_residual:.6e}")
    
    # Determine mesh independence
    mesh_independent = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
    
    # Write RESULT.txt
    with open(Path(work_dir) / "RESULT.txt", 'w') as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {','.join(all_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residual:.15e}\n")
        f.write(f"COUPLING_ITERATIONS = {final_iterations}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independent}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print("\n" + "="*60)
    print("COUPLING COMPLETE")
    print("="*60)
    print(f"Levels: {len(levels)}")
    print(f"Final residual: {final_residual:.6e}")
    print(f"Coupling iterations (finest): {final_iterations}")
    print(f"Mesh independence: {mesh_independent}")
    print(f"Max relative change: {max_rel_change:.6e}")


if __name__ == "__main__":
    main()
