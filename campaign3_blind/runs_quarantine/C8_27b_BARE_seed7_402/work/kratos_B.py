#!/usr/bin/env python3
"""
Kratos solver for subdomain B (Neumann side in Dirichlet-Neumann coupling).
Subdomain B: (0.625, 1.5) x (0, 1), k = 1000
Interface at x = 0.625 receives flux from partner and returns field.
"""
import sys
import json
from pathlib import Path
import subprocess

def run_kratos(level, work_dir, interface_flux=None):
    """Run Kratos solver for subdomain B."""
    
    # Convert interface_flux to string format for embedding
    if interface_flux is None:
        interface_flux_str = "None"
    else:
        interface_flux_str = str(interface_flux)
    
    script = f'''
import sys
sys.path.insert(0, '/home/alexander/Schreibtisch/open-fem-agent/.venv/lib/python3.12/site-packages')

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

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

print(f"Kratos B: Creating mesh with nx={nx}, ny={ny}")

# Create model
model = KM.Model()
mp = model.CreateModelPart("thermal")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2

# Settings
settings = KM.ConvectionDiffusionSettings()
settings.SetUnknownVariable(KM.TEMPERATURE)
settings.SetDiffusionVariable(KM.CONDUCTIVITY)
settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)

for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX):
    mp.AddNodalSolutionStepVariable(v)

mp.SetBufferSize(1)

# Create properties
props = mp.CreateNewProperties(1)

# Create nodes
nid = {}
cnt = 1
dx = (x_max_B - x_min_B) / nx
dy = y_max / ny

for j in range(ny + 1):
    for i in range(nx + 1):
        x = x_min_B + i * dx
        y = j * dy
        mp.CreateNewNode(cnt, x, y, 0.0)
        nid[(i, j)] = cnt
        cnt += 1

num_nodes = cnt - 1
print(f"Kratos B: Created {num_nodes} nodes")

# Create elements (triangular P1)
eid = 1
for j in range(ny):
    for i in range(nx):
        a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
        mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
        mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1

num_elements = eid - 1
print(f"Kratos B: Created {num_elements} elements")

# Set material properties and source term
def source_B(x, y):
    return (-9*x**3*y/2500000 + x**3/2500000 - 64901*x**2*y/10000000 - 25033*x**2/30000000 
            - 9*x*y**3/2500000 + 3*x*y**2/2500000 - 422790921*x*y/160000000 - 21609971*x/32000000 
            - 64901*y**3/30000000 - 25033*y**2/30000000 + 1274009971*y/320000000 + 12989997/12800000)

for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, k_B)
    node.SetSolutionStepValue(KM.HEAT_FLUX, source_B(node.X, node.Y))
    node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

# Boundary conditions
# Outer boundaries: u = 0 on right (x=x_max_B), top (y=y_max), bottom (y=0)
# Interface (left, x=x_min_B): Neumann BC from partner

# Apply outer Dirichlet BCs
for j in range(ny + 1):
    # Right boundary (x = x_max_B)
    n = mp.Nodes[nid[(nx, j)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)
    
    # Top boundary (y = y_max), except corners already handled
    if j == ny and i < nx:
        n = mp.Nodes[nid[(i, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
        n.Fix(KM.TEMPERATURE)
    
    # Bottom boundary (y = 0), except corners already handled  
    if j == 0 and i < nx:
        n = mp.Nodes[nid[(i, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
        n.Fix(KM.TEMPERATURE)

# Fix top and bottom edges properly
for i in range(nx + 1):
    # Top
    n = mp.Nodes[nid[(i, ny)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)
    # Bottom
    n = mp.Nodes[nid[(i, 0)]]
    n.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    n.Fix(KM.TEMPERATURE)

# Interface Neumann BC (left side, x = x_min_B)
# Outward normal from B is -x direction
# qn = -k * grad(u) . n = -k * (-du/dx) = k * du/dx
# We receive flux from A, which is A's outward flux (+x direction)
# At interface: A's outward flux = -B's outward flux (continuity)
# So B's Neumann BC should use negative of A's flux

interface_flux_data = {interface_flux_str}
if interface_flux_data is not None and len(interface_flux_data) > 0:
    partner_y = [pt[0] for pt in interface_flux_data]
    partner_qn = [pt[1] for pt in interface_flux_data]
    
    for j in range(ny + 1):
        n = mp.Nodes[nid[(0, j)]]
        y_pos = n.Y
        
        # Interpolate flux
        qn_A = np.interp(y_pos, partner_y, partner_qn)
        
        # A's outward flux is in +x direction
        # For continuity: -k_A * duA/dx = -k_B * duB/dx
        # B's outward normal is -x, so B's outward flux = -k_B * (-duB/dx) = k_B * duB/dx
        # The Neumann BC in Kratos expects the flux in the direction of the outward normal
        # So we set FACE_HEAT_FLUX = -qn_A (since A's outward = -B's outward for continuity)
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, -qn_A)

# Add DOFs
KM.VariableUtils().AddDof(KM.TEMPERATURE, mp)

# Solve
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, False, False, False, False)
strategy.Initialize()
strategy.Solve()

print(f"Kratos B: Solution complete")

# Extract interface data
# Interface is at x = x_min_B (left side)
# Outward normal from B is -x direction
# qn = -k * grad(u) . n = -k * (-du/dx) = k * du/dx

interface_results = []
for j in range(ny + 1):
    n_if = mp.Nodes[nid[(0, j)]]
    n_near = mp.Nodes[nid[(1, j)]]
    
    y_pos = n_if.Y
    u_if = n_if.GetSolutionStepValue(KM.TEMPERATURE)
    u_near = n_near.GetSolutionStepValue(KM.TEMPERATURE)
    
    # du/dx approximated
    du_dx = (u_near - u_if) / dx
    
    # Outward normal from B is -x, so qn = -k * (-du/dx) = k * du/dx
    qn = k_B * du_dx
    
    interface_results.append([y_pos, u_if, qn])

# Write run log
log_path = Path(work_dir) / f"run_level{level}_B.log"
with open(log_path, 'w') as f:
    f.write(f"NDOF = {num_nodes}\\n")
    f.write(f"Mesh level: {level}\\n")
    f.write(f"Elements: {num_elements}\\n")

# Write exports.json
exports = {
    "B_to_A": {
        "flux": [[r[0], r[2]] for r in interface_results],  # [y, qn]
        "field": [[r[0], r[1]] for r in interface_results]   # [y, u]
    }
}
with open(Path(work_dir) / "exports.json", 'w') as f:
    json.dump(exports, f, indent=2)

print(f"Kratos B: level={level}, ndof={num_nodes}, done")
'''
    
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[:2000])
    return result.returncode == 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python kratos_B.py <level> <work_dir>")
        sys.exit(1)
    
    level = int(sys.argv[1])
    work_dir = sys.argv[2]
    
    # Read imports.json for interface data
    imports_path = Path(work_dir) / "imports.json"
    interface_flux = None
    if imports_path.exists():
        imports = json.loads(imports_path.read_text())
        if "A_to_B" in imports and "flux" in imports["A_to_B"]:
            interface_flux = imports["A_to_B"]["flux"]
    
    success = run_kratos(level, work_dir, interface_flux)
    sys.exit(0 if success else 1)
