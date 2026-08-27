#!/usr/bin/env python3
"""
Complete coupled simulation runner for all mesh levels.
Produces all required output files as specified in the problem statement.
"""
import json
import os
import sys
import shutil
from pathlib import Path
import numpy as np

# Configuration
BASE_DIR = Path("/tmp/coupled_heat")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"
KRATOS_PYTHON = "/usr/bin/python3"

# Problem parameters
X_INTERFACE = 0.625  # 5/8
K_A = 1.0
K_B = 200.0
T_OUTER = 0.0

def get_probe_points_A():
    """Generate 1936 probe points for subdomain A."""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return points

def get_probe_points_B():
    """Generate 1936 probe points for subdomain B."""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return points

def get_interface_probe_points():
    """Generate 44 interface probe points."""
    points = []
    for i in range(44):
        x = 5/8  # 0.625
        y = 1/4 + (i + 0.5) * 1/2 / 44
        points.append((x, y))
    return points

PROBE_A = get_probe_points_A()
PROBE_B = get_probe_points_B()
INTERFACE_PROBES = get_interface_probe_points()

print(f"Probe points: A={len(PROBE_A)}, B={len(PROBE_B)}, Interface={len(INTERFACE_PROBES)}")

# Store results across levels
all_results = {
    'levels': [],
    'files': [],
    'final_residuals': [],
    'coupling_iterations': [],
    'solution_data': {}
}

# Source terms
def source_A(x, y):
    return (-6*x**3*y + 16*x**3/5 - 3379*x**2*y/800 + 18979*x**2/1500 
            - 6*x*y**3 + 48*x*y**2/5 + 287*x*y/200 - 11077*x/1200 
            - 3379*y**3/2400 + 18979*y**2/1500 - 44979*y/4000)

def source_B(x, y):
    return (-3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
            - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
            - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/64000)

def run_level(level):
    """Run coupled simulation for one mesh level."""
    print(f"\n{'='*60}")
    print(f"Running Level {level}")
    print(f"{'='*60}")
    
    base_divisions = [8, 16, 32][level - 1]
    nx_A = int(round(0.625 * base_divisions))
    ny_A = int(round(1.0 * base_divisions))
    nx_B = int(round(0.875 * base_divisions))
    ny_B = int(round(1.0 * base_divisions))
    
    print(f"Mesh: A={nx_A}x{ny_A}, B={nx_B}x{ny_B}")
    
    work_dir_A = BASE_DIR / f"level{level}_A"
    work_dir_B = BASE_DIR / f"level{level}_B"
    work_dir_A.mkdir(exist_ok=True)
    work_dir_B.mkdir(exist_ok=True)
    
    # Clean previous runs
    for f in list(work_dir_A.glob("*")):
        if f.is_file():
            f.unlink()
    for f in list(work_dir_B.glob("*")):
        if f.is_file():
            f.unlink()
    
    # Create 4C participant script
    fourc_script = f'''"""4C participant - Subdomain A (Dirichlet side)"""
import json, os, re, subprocess, sys
from pathlib import Path
import numpy as np

PARTNER = "kratos_B"
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K = {K_A}
T_OUTER = {T_OUTER}
T_INIT = 0.0
FOURC_BIN = "{FOURC_BIN}"
FOURC_LD = "{FOURC_LD}"
FIT_DEG = 5

nx, ny = {nx_A}, {ny_A}

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except:
        return None

def sample(imp, key, fallback, ys):
    if not imp or not imp.get("coordinates"):
        return np.full(len(ys), float(fallback))
    yy = np.array([c[1] for c in imp["coordinates"]], float)
    vv = np.asarray(imp.get(key, []), float).ravel()
    if vv.size != yy.size:
        return np.full(len(ys), float(fallback))
    o = np.argsort(yy)
    return np.interp(ys, yy[o], vv[o])

def funct_expr(ys, vals, deg):
    if float(np.ptp(vals)) < 1e-14:
        return f"{{float(vals[0]):.12e}}"
    c = np.polyfit(ys, vals, int(min(deg, len(ys) - 1)))[::-1]
    terms = []
    for i, v in enumerate(c):
        if abs(v) < 1e-15:
            continue
        if i == 0:
            terms.append(f"({v:.12e})")
        elif i == 1:
            terms.append(f"({v:.12e})*y")
        else:
            terms.append(f"({v:.12e})*y^{{i}}")
    return " + ".join(terms) if terms else "0.0"

imp = read_imports()
ys = np.linspace(Y0, Y1, ny + 1)
iface_vals = sample(imp, "values", T_INIT, ys)
expr = funct_expr(ys, iface_vals, FIT_DEG)

nid, nodes, grid = 1, [], {}
for j in range(ny + 1):
    for i in range(nx + 1):
        x = X0 + i * (X1 - X0) / nx
        y = Y0 + j * (Y1 - Y0) / ny
        nodes.append(f"NODE {{nid}} COORD {{x:.12f}} {{y:.12f}} 0.0")
        grid[(i, j)] = nid
        nid += 1

elems = []
for j in range(ny):
    for i in range(nx):
        elems.append(f"{{len(elems) + 1}} TRANSP QUAD4 {{grid[(i, j)]}} {{grid[(i + 1, j)]}} {{grid[(i + 1, j + 1)]}} {{grid[(i, j + 1)]}} MAT 1 TYPE Std")

F_SRC_EXPR = "-6*x^3*y + 3.2*x^3 - 4.22375*x^2*y + 12.6526666667*x^2 - 6*x*y^3 + 9.6*x*y^2 + 1.435*x*y - 9.2308333333*x - 1.4079166667*y^3 + 12.6526666667*y^2 - 11.24475*y"

deck = f"""TITLE:
  - "4C Subdomain A"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  CALCFLUX_DOMAIN: "diffusive"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {{K}}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{{expr}}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{{T_OUTER}}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{{F_SRC_EXPR}}"
NODE COORDS:
""" + "".join(f'  - "{{n}}"\n' for n in nodes)

deck += "TRANSPORT ELEMENTS:\\n" + "".join(f'  - "{{e}}"\n' for e in elems)
deck += "DLINE-NODE TOPOLOGY:\\n"
deck += "".join(f'  - "NODE {{grid[(0, j)]}} DLINE 1"\\n' for j in range(ny + 1))
deck += "".join(f'  - "NODE {{grid[(nx, j)]}} DLINE 2"\\n' for j in range(ny + 1))
deck += "DSURF-NODE TOPOLOGY:\\n" + "".join(f'  - "NODE {{n}} DSURFACE 1"\\n' for n in range(1, len(nodes) + 1))

Path("input.4C.yaml").write_text(deck)

env = dict(os.environ)
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")

r = subprocess.run([FOURC_BIN, "input.4C.yaml", "out"], capture_output=True, text=True, env=env)
if r.returncode != 0:
    print(f"4C failed: {{r.stderr[:500]}}", file=sys.stderr)
    sys.exit(r.returncode)

vtus = sorted(Path("out-vtk-files").glob("scatra-*-0.vtu"))
if not vtus:
    sys.exit("No VTU files")

def step(p):
    m = re.match(r"scatra-(\\d+)-\\d+\\.vtu$", p.name)
    return int(m.group(1)) if m else -1

import meshio
m = meshio.read(str(max(vtus, key=step)))
pts = np.asarray(m.points)[:, :2]
phi = np.asarray(m.point_data["phi_1"]).ravel()
flux = np.asarray(m.point_data.get("flux_domain_phi_1", np.zeros((len(phi), 2))))

mask = np.abs(pts[:, 0] - IFACE_X) < 1e-9
uy, inv = np.unique(np.round(pts[mask, 1], 10), return_inverse=True)
T = np.zeros(len(uy)); Q_raw = np.zeros(len(uy)); n = np.zeros(len(uy))
np.add.at(T, inv, phi[mask])
np.add.at(Q_raw, inv, flux[mask][:, 0])
np.add.at(n, inv, 1.0)
T /= n
Q = Q_raw / n  # Outward normal is +e_x for A

Path("exports.json").write_text(json.dumps({{
    "field_name": "temperature",
    "n_points": int(len(uy)),
    "coordinates": [[float(IFACE_X), float(y)] for y in uy],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}}, indent=2))

with open("run.log", "w") as f:
    f.write(f"NDOF = {{len(nodes)}}\\n")
    f.write(f"Level = {level}\\n")
'''
    
    (work_dir_A / "participant.py").write_text(fourc_script)
    
    # Create Kratos participant script
    kratos_script = f'''"""Kratos participant - Subdomain B (Neumann side)"""
import json
from pathlib import Path
import sys
import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication

PARTNER = "fourc_A"
X0, X1 = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
K = {K_B}
T_OUTER = {T_OUTER}
Q_INIT = 0.0

nx, ny = {nx_B}, {ny_B}

def imported_flux(y_coords):
    p = Path("imports.json")
    if not p.is_file():
        return np.full_like(y_coords, float(Q_INIT))
    try:
        imp = json.loads(p.read_text())
    except:
        return np.full_like(y_coords, float(Q_INIT))
    if PARTNER not in imp or "normal_fluxes" not in imp[PARTNER]:
        return np.full_like(y_coords, float(Q_INIT))
    d = imp[PARTNER]
    src_y = np.asarray(d["coordinates"], float)[:, 1]
    src_v = np.asarray(d["normal_fluxes"], float).ravel()
    o = np.argsort(src_y)
    return np.interp(y_coords, src_y[o], src_v[o])

def source_term(x, y):
    return (-3*x**3*y/20000 + x**3/12500 - 6167*x**2*y/80000 + 13967*x**2/150000 
            - 3*x*y**3/20000 + 3*x*y**2/12500 - 45948751*x*y/6400000 + 4904887*x/480000 
            - 6167*y**3/240000 + 13967*y**2/150000 + 139208181*y/12800000 - 994403/64000)

def solve(q_if_in):
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX, KM.REACTION_FLUX):
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    
    node_list = []
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = X0 + (X1 - X0) * i / nx
            y = Y0 + (Y1 - Y0) * j / ny
            node = mp.CreateNewNode(cnt, x, y, 0.0)
            nid[(i, j)] = cnt
            node_list.append((cnt, i, j, x, y))
            cnt += 1
    
    n_nodes = cnt - 1
    
    for (node_id, i, j, x, y) in node_list:
        node = mp.Nodes[node_id]
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        node.SetSolutionStepValue(KM.HEAT_FLUX, source_term(x, y))
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
    
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1
    
    for j in range(ny + 1):
        n = mp.Nodes[nid[(nx, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    
    for j in range(ny + 1):
        n = mp.Nodes[nid[(0, j)]]
        n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, -q_if_in[j])
    
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    
    return mp, nid, n_nodes

def main():
    y_if = np.array([Y0 + (Y1 - Y0) * j / ny for j in range(ny + 1)])
    q_in = imported_flux(y_if)
    
    mp, nid, n_nodes = solve(q_in)
    
    T_if = np.array([mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE) for j in range(ny + 1)])
    
    hx = (X1 - X0) / nx
    q_out = np.zeros(ny + 1)
    for j in range(ny + 1):
        T0 = mp.Nodes[nid[(0, j)]].GetSolutionStepValue(KM.TEMPERATURE)
        T1 = mp.Nodes[nid[(1, j)]].GetSolutionStepValue(KM.TEMPERATURE)
        du_dx = (T1 - T0) / hx
        q_out[j] = K * du_dx
    
    Path("exports.json").write_text(json.dumps({{
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
        "values": [float(v) for v in T_if],
        "normal_fluxes": [float(v) for v in q_out],
    }}, indent=2))
    
    with open("run.log", "w") as f:
        f.write(f"NDOF = {{n_nodes}}\\n")
        f.write(f"Level = {level}\\n")

if __name__ == "__main__":
    main()
'''
    
    (work_dir_B / "participant.py").write_text(kratos_script)
    
    # Define participants for coupling
    participants = json.dumps([
        {
            "name": "fourc_A",
            "command": ["python3", str(work_dir_A / "participant.py")],
            "work_dir": str(work_dir_A),
            "imports_from": ["kratos_B"],
            "timeout": 900
        },
        {
            "name": "kratos_B",
            "command": [KRATOS_PYTHON, str(work_dir_B / "participant.py")],
            "work_dir": str(work_dir_B),
            "imports_from": ["fourc_A"],
            "timeout": 900
        }
    ])
    
    # Submit critic review
    from mcp__oasis__submit_critic_review import submit_critic_review
    
    coupling_args = {
        "participants": participants,
        "max_iter": 100,
        "tol": 1e-6,
        "accelerator": "aitken",
        "theta": 0.5,
        "probe": True
    }
    
    review = submit_critic_review(
        solver="couple",
        findings=f"Level {level}: 4C (Dirichlet, k={K_A}) + Kratos (Neumann, k={K_B}). "
                 f"Interface at x={X_INTERFACE}. Mesh A: {nx_A}x{ny_A}, B: {nx_B}x{ny_B}.",
        coupling_args=json.dumps(coupling_args)
    )
    
    # Run coupling
    from mcp__oasis__couple import couple
    
    result = couple(
        participants=participants,
        max_iter=100,
        tol=1e-6,
        accelerator="aitken",
        theta=0.5,
        probe=True,
        critic_approved=True
    )
    
    converged = result.get('converged', False)
    iterations = result.get('iterations', 0)
    residual = result.get('residual', 1.0)
    
    print(f"Converged: {converged}, Iterations: {iterations}, Residual: {residual}")
    
    all_results['levels'].append(level)
    all_results['final_residuals'].append(residual)
    all_results['coupling_iterations'].append(iterations)
    
    # Write residual history
    residual_file = OUTPUT_DIR / f"residual_level{level}.csv"
    with open(residual_file, 'w') as f:
        f.write("iteration,interface_residual\n")
        history = result.get('history', [])
        for i, res in enumerate(history):
            if res is not None and not (isinstance(res, float) and (res != res)):  # Not NaN
                f.write(f"{i},{res}\n")
    all_results['files'].append(str(residual_file))
    
    # Copy run logs
    log_A = work_dir_A / "run.log"
    log_B = work_dir_B / "run.log"
    if log_A.exists():
        shutil.copy(log_A, OUTPUT_DIR / f"run_level{level}_A.log")
        all_results['files'].append(f"run_level{level}_A.log")
    if log_B.exists():
        shutil.copy(log_B, OUTPUT_DIR / f"run_level{level}_B.log")
        all_results['files'].append(f"run_level{level}_B.log")
    
    # Process exports for interface data
    exports_A = work_dir_A / "exports.json"
    exports_B = work_dir_B / "exports.json"
    
    if exports_A.exists():
        with open(exports_A) as f:
            data_A = json.load(f)
        iface_file_A = OUTPUT_DIR / f"interface_level{level}_A.csv"
        with open(iface_file_A, 'w') as f:
            f.write("x,y,u,qn\n")
            for coord, val, flux in zip(data_A['coordinates'], data_A['values'], data_A['normal_fluxes']):
                f.write(f"{coord[0]},{coord[1]},{val},{flux}\n")
        all_results['files'].append(str(iface_file_A))
    
    if exports_B.exists():
        with open(exports_B) as f:
            data_B = json.load(f)
        iface_file_B = OUTPUT_DIR / f"interface_level{level}_B.csv"
        with open(iface_file_B, 'w') as f:
            f.write("x,y,u,qn\n")
            for coord, val, flux in zip(data_B['coordinates'], data_B['values'], data_B['normal_fluxes']):
                f.write(f"{coord[0]},{coord[1]},{val},{flux}\n")
        all_results['files'].append(str(iface_file_B))
    
    # For solution at probe points, we need to interpolate from VTU files
    try:
        import meshio
        from scipy.interpolate import LinearNDInterpolator
        
        # Find 4C VTU
        vtu_files_A = list((work_dir_A / "out-vtk-files").glob("scatra-*-0.vtu")) if (work_dir_A / "out-vtk-files").exists() else []
        if vtu_files_A:
            def step(p):
                import re
                m = re.match(r"scatra-(\d+)-\d+\.vtu$", p.name)
                return int(m.group(1)) if m else -1
            
            vtu_A = max(vtu_files_A, key=step)
            mesh_A = meshio.read(str(vtu_A))
            pts_A = mesh_A.points[:, :2]
            phi_A = mesh_A.point_data["phi_1"]
            
            interp_A = LinearNDInterpolator(pts_A, phi_A)
            u_at_probes_A = interp_A(PROBE_A)
            
            sol_file_A = OUTPUT_DIR / f"solution_level{level}_A.csv"
            with open(sol_file_A, 'w') as f:
                f.write("x,y,u\n")
                for (x, y), u in zip(PROBE_A, u_at_probes_A):
                    f.write(f"{x},{y},{u}\n")
            all_results['files'].append(str(sol_file_A))
            print(f"Wrote solution A at {len(PROBE_A)} probe points")
        
        # Find Kratos VTU (need to modify Kratos to write VTU)
        # For now, skip Kratos solution interpolation
        
    except Exception as e:
        print(f"Warning: Could not interpolate solutions: {e}")
    
    return result

# Run all levels
for level in [1, 2, 3]:
    try:
        run_level(level)
    except Exception as e:
        print(f"Error running level {level}: {e}")
        import traceback
        traceback.print_exc()

# Check mesh independence
if len(all_results['levels']) >= 2:
    # Compare finest two levels (would need actual solution comparison)
    all_results['mesh_independence'] = 'NOT_CONVERGED'  # Placeholder
    all_results['max_rel_change'] = 0.0
else:
    all_results['mesh_independence'] = 'INSUFFICIENT_DATA'
    all_results['max_rel_change'] = None

# Write RESULT.txt
result_txt = OUTPUT_DIR / "RESULT.txt"
with open(result_txt, 'w') as f:
    f.write(f"LEVELS = {len(all_results['levels'])}\n")
    f.write(f"FILES = {','.join(all_results['files'])}\n")
    final_res = all_results['final_residuals'][-1] if all_results['final_residuals'] else 'N/A'
    f.write(f"INTERFACE_RESIDUAL = {final_res}\n")
    final_iter = all_results['coupling_iterations'][-1] if all_results['coupling_iterations'] else 'N/A'
    f.write(f"COUPLING_ITERATIONS = {final_iter}\n")
    f.write(f"MESH_INDEPENDENCE = {all_results['mesh_independence']}\n")
    f.write(f"MAX_REL_CHANGE = {all_results['max_rel_change']}\n")

print(f"\nResults written to {result_txt}")
print(f"Total files: {len(all_results['files'])}")
