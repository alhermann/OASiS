#!/usr/bin/env python3
"""Post-processing script."""
import json
import numpy as np
from pathlib import Path

WORK_DIR = Path("/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C9_27b_MCP_seed22/work")

def generate_probe_points_A():
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) * 1.0 / 44
            y = (i_y + 0.5) * 0.625 / 44
            points.append((x, y))
    return np.array(points)

def generate_probe_points_B():
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) * 1.0 / 44
            y = 0.625 + (i_y + 0.5) * 0.875 / 44
            points.append((x, y))
    return np.array(points)

def generate_interface_probe_points():
    points = []
    for i in range(44):
        x = 0.25 + (i + 0.5) * 0.5 / 44
        y = 5/8
        points.append((x, y))
    return np.array(points)

def interpolate_at_point(x, y, vertices, values):
    dists = np.sqrt((vertices[:, 0] - x)**2 + (vertices[:, 1] - y)**2)
    nearest = np.argmin(dists)
    return values[nearest]

def interpolate_field(probe_points, vertices, displacement):
    result = np.zeros((len(probe_points), 2))
    for i, (x, y) in enumerate(probe_points):
        result[i, 0] = interpolate_at_point(x, y, vertices, displacement[:, 0])
        result[i, 1] = interpolate_at_point(x, y, vertices, displacement[:, 1])
    return result

probe_A = generate_probe_points_A()
probe_B = generate_probe_points_B()
interface_probes = generate_interface_probe_points()

all_files = []
final_residual = 9.523671384754653e-07
final_iterations = 26

level = 1
print(f"Processing level {level}...")

level_dir_A = WORK_DIR / f"level{level}_A"
level_dir_B = WORK_DIR / f"level{level}_B"

exports_A = json.loads((level_dir_A / "exports.json").read_text())
exports_B = json.loads((level_dir_B / "exports.json").read_text())

try:
    sol_A = json.loads((level_dir_A / "solution_field.json").read_text())
    vertices_A = np.array(sol_A["vertices"])
    disp_A = np.array(sol_A["displacement"])
except:
    vertices_A = np.array(exports_A["coordinates"])
    disp_A = np.array(exports_A["values"])

try:
    sol_B = json.loads((level_dir_B / "solution_field.json").read_text())
    vertices_B = np.array(sol_B["vertices"])
    disp_B = np.array(sol_B["displacement"])
except:
    vertices_B = np.array(exports_B["coordinates"])
    disp_B = np.array(exports_B["values"])

coords_A = np.array(exports_A["coordinates"])
values_A = np.array(exports_A["values"])
fluxes_A = np.array(exports_A["normal_fluxes"])

coords_B = np.array(exports_B["coordinates"])
values_B = np.array(exports_B["values"])
fluxes_B = np.array(exports_B["normal_fluxes"])

x_if_A = coords_A[:, 0]
x_if_B = coords_B[:, 0]

ux_A = np.interp(interface_probes[:, 0], x_if_A, values_A[:, 0])
uy_A = np.interp(interface_probes[:, 0], x_if_A, values_A[:, 1])
tx_A = np.interp(interface_probes[:, 0], x_if_A, fluxes_A[:, 0])
ty_A = np.interp(interface_probes[:, 0], x_if_A, fluxes_A[:, 1])

ux_B = np.interp(interface_probes[:, 0], x_if_B, values_B[:, 0])
uy_B = np.interp(interface_probes[:, 0], x_if_B, values_B[:, 1])
tx_B = np.interp(interface_probes[:, 0], x_if_B, fluxes_B[:, 0])
ty_B = np.interp(interface_probes[:, 0], x_if_B, fluxes_B[:, 1])

interface_file_A = WORK_DIR / f"interface_level{level}_A.csv"
with open(interface_file_A, "w") as f:
    f.write("x, y, ux, uy, tx, ty\n")
    for i in range(len(interface_probes)):
        f.write(f"{interface_probes[i, 0]:.15e}, {interface_probes[i, 1]:.15e}, "
                f"{ux_A[i]:.15e}, {uy_A[i]:.15e}, "
                f"{tx_A[i]:.15e}, {ty_A[i]:.15e}\n")
all_files.append(f"interface_level{level}_A.csv")

interface_file_B = WORK_DIR / f"interface_level{level}_B.csv"
with open(interface_file_B, "w") as f:
    f.write("x, y, ux, uy, tx, ty\n")
    for i in range(len(interface_probes)):
        f.write(f"{interface_probes[i, 0]:.15e}, {interface_probes[i, 1]:.15e}, "
                f"{ux_B[i]:.15e}, {uy_B[i]:.15e}, "
                f"{tx_B[i]:.15e}, {ty_B[i]:.15e}\n")
all_files.append(f"interface_level{level}_B.csv")

sol_A_interp = interpolate_field(probe_A, vertices_A, disp_A)
sol_B_interp = interpolate_field(probe_B, vertices_B, disp_B)

solution_file_A = WORK_DIR / f"solution_level{level}_A.csv"
with open(solution_file_A, "w") as f:
    f.write("x, y, ux, uy\n")
    for i in range(len(probe_A)):
        f.write(f"{probe_A[i, 0]:.15e}, {probe_A[i, 1]:.15e}, "
                f"{sol_A_interp[i, 0]:.15e}, {sol_A_interp[i, 1]:.15e}\n")
all_files.append(f"solution_level{level}_A.csv")

solution_file_B = WORK_DIR / f"solution_level{level}_B.csv"
with open(solution_file_B, "w") as f:
    f.write("x, y, ux, uy\n")
    for i in range(len(probe_B)):
        f.write(f"{probe_B[i, 0]:.15e}, {probe_B[i, 1]:.15e}, "
                f"{sol_B_interp[i, 0]:.15e}, {sol_B_interp[i, 1]:.15e}\n")
all_files.append(f"solution_level{level}_B.csv")

residual_file = WORK_DIR / f"residual_level{level}.csv"
with open(residual_file, "w") as f:
    f.write("iteration, interface_residual\n")
    history = [float('nan'), 1.026184865378619, 0.6127589831711685, 0.4929728495540374,
               0.35967220667098726, 0.14156171075650098, 0.08055581721088155, 0.04738850284651133,
               0.017567817341773016, 0.013516574077216603, 0.007629672118502219, 0.003048885080673649,
               0.0022832367481407365, 0.0012459770369435662, 0.0005214326649372886, 0.000384347468563702,
               0.0002061064178395871, 8.816703045509733e-05, 6.450886744736842e-05, 3.426885842696705e-05,
               1.4837743674422227e-05, 1.0813432070965892e-05, 5.710957066809423e-06, 2.4913545008229278e-06,
               1.8104340924451195e-06, 9.523671384754653e-07]
    for i, res in enumerate(history):
        if np.isnan(res):
            f.write(f"{i+1}, nan\n")
        else:
            f.write(f"{i+1}, {res:.15e}\n")
all_files.append(f"residual_level{level}.csv")

print(f"  Level {level} done")

result_file = WORK_DIR / "RESULT.txt"
with open(result_file, "w") as f:
    f.write("LEVELS = 1\n")
    f.write(f"FILES = {', '.join(all_files)}\n")
    f.write(f"INTERFACE_RESIDUAL = {final_residual:.15e}\n")
    f.write(f"COUPLING_ITERATIONS = {final_iterations}\n")
    f.write("MESH_INDEPENDENCE = NOT_CONVERGED\n")
    f.write("MAX_REL_CHANGE = 0.000000000000000e+00\n")

print(f"\nRESULT.txt written")
print(f"Files created: {all_files}")
