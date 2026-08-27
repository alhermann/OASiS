"""Test the coupling between FEBio (A) and scikit-fem (B)."""
import json
import subprocess
import numpy as np
from pathlib import Path

SKFEM_PYTHON = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
WORK_DIR = Path("/tmp/coupled_elasticity")

# Mesh parameters for level 1
NX_A, NY_A = 5, 8
NX_B, NY_B = 7, 8

# Interface y-coordinates - use a common fine grid for exchange
IFACE_X = 0.625
iface_y = [1/4 + (i + 0.5) * (1/2) / 44 for i in range(44)]

dir_A = WORK_DIR / "level1_A"
dir_B = WORK_DIR / "level1_B"

# Copy participant scripts
import shutil
shutil.copy(WORK_DIR / "participant_A_febio.py", dir_A / "participant_A.py")
shutil.copy(WORK_DIR / "participant_B_skfem.py", dir_B / "participant_B.py")

# Coupling parameters
max_iter = 50
tol = 1e-6
theta = 0.74  # Optimal for rho ≈ 0.35

# Initialize with zero displacement at interface
u_iface_prev = np.zeros((len(iface_y), 2))

residual_history = []

print(f"Starting coupling: {max_iter} max iterations, tol={tol}, theta={theta}")
print(f"Interface has {len(iface_y)} points")

for iteration in range(max_iter):
    # Write imports for A (displacement from B)
    imports_A = {
        "B": {
            "field_name": "displacement",
            "coordinates": [[IFACE_X, y] for y in iface_y],
            "values": u_iface_prev.tolist(),
            "normal_fluxes": [[0.0, 0.0]] * len(iface_y)
        }
    }
    (dir_A / "imports.json").write_text(json.dumps(imports_A))
    
    # Run A
    result_A = subprocess.run(
        [SKFEM_PYTHON, "participant_A.py", str(NX_A), str(NY_A)],
        cwd=str(dir_A), capture_output=True, text=True
    )
    if result_A.returncode != 0:
        print(f"Participant A failed at iteration {iteration+1}")
        print(result_A.stderr[-1000:])
        break
    
    # Read traction from A
    exports_A = json.loads((dir_A / "exports.json").read_text())
    traction_A = np.array(exports_A["values"])
    traction_coords = np.array(exports_A["coordinates"])
    traction_y = [c[1] for c in traction_coords]
    
    print(f"  A exported {len(traction_A)} traction values at y={traction_y[0]:.4f}..{traction_y[-1]:.4f}")
    
    # Interpolate traction to our common interface grid
    tx_interp = np.interp(iface_y, traction_y, traction_A[:, 0])
    ty_interp = np.interp(iface_y, traction_y, traction_A[:, 1])
    traction_for_B = np.column_stack([tx_interp, ty_interp])
    
    # Write imports for B (traction from A)
    imports_B = {
        "A": {
            "field_name": "traction",
            "coordinates": [[IFACE_X, y] for y in iface_y],
            "values": traction_for_B.tolist(),
            "normal_fluxes": traction_for_B.tolist()
        }
    }
    (dir_B / "imports.json").write_text(json.dumps(imports_B))
    
    # Run B
    result_B = subprocess.run(
        [SKFEM_PYTHON, "participant_B.py", str(NX_B), str(NY_B)],
        cwd=str(dir_B), capture_output=True, text=True
    )
    if result_B.returncode != 0:
        print(f"Participant B failed at iteration {iteration+1}")
        print(result_B.stderr[-1000:])
        break
    
    # Read displacement from B
    exports_B = json.loads((dir_B / "exports.json").read_text())
    u_iface_new_raw = np.array(exports_B["values"])
    u_coords = np.array(exports_B["coordinates"])
    u_y = [c[1] for c in u_coords]
    
    print(f"  B exported {len(u_iface_new_raw)} displacement values at y={u_y[0]:.4f}..{u_y[-1]:.4f}")
    
    # Interpolate displacement to our common interface grid
    ux_interp = np.interp(iface_y, u_y, u_iface_new_raw[:, 0])
    uy_interp = np.interp(iface_y, u_y, u_iface_new_raw[:, 1])
    u_iface_new = np.column_stack([ux_interp, uy_interp])
    
    # Compute residual (relative change in displacement)
    diff = np.linalg.norm(u_iface_new - u_iface_prev)
    norm = np.linalg.norm(u_iface_prev) + 1e-15
    residual = diff / norm
    residual_history.append(residual)
    
    print(f"Iteration {iteration+1}: residual = {residual:.6e}, ||u|| = {np.linalg.norm(u_iface_new):.6e}")
    
    # Relaxation
    u_iface_relaxed = (1 - theta) * u_iface_prev + theta * u_iface_new
    
    # Check convergence
    if residual < tol:
        print(f"Converged at iteration {iteration+1}")
        u_iface_final = u_iface_relaxed
        break
    
    u_iface_prev = u_iface_relaxed
else:
    print(f"Did not converge after {max_iter} iterations")
    u_iface_final = u_iface_relaxed

# Save residual history
with open(WORK_DIR / "residual_level1.csv", "w") as f:
    f.write("iteration,interface_residual\n")
    for i, res in enumerate(residual_history, 1):
        f.write(f"{i},{res:.17g}\n")

print(f"\nFinal residual: {residual_history[-1]:.6e}")
print(f"Iterations: {len(residual_history)}")
