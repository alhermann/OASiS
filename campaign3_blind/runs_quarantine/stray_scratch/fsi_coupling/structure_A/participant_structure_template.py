"""4C STRUCTURE participant for OASiS `couple` driver — FSI (mesh independence template)."""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PARTNER     = "fluid"
LX          = 1.0
Y0          = 0.2
HS          = 0.05
# __RESOLUTION__ will be replaced by verify_mesh_independence
NXS, NYS    = __RESOLUTION__, 4
E_MOD       = 3.0e6
NU          = 0.3
T_INIT      = 0.0
FOURC_BIN   = "/home/alexander/4C/build/4C"
FOURC_LD    = "/opt/4C-dependencies/lib"


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample_traction(imp, x_targets, fallback):
    if not imp or not imp.get("coordinates"):
        return np.full((len(x_targets), 2), float(fallback))
    
    xs_src = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs_src), -1)
    
    out = np.zeros((len(x_targets), 2))
    order = np.argsort(xs_src)
    for c in range(min(2, vals.shape[1])):
        out[:, c] = np.interp(x_targets, xs_src[order], vals[order, c])
    return out


def main():
    imp = read_imports()
    
    x_if = np.linspace(0.0, LX, NXS + 1)
    ref_coords = np.column_stack([x_if, np.full_like(x_if, Y0)])
    
    traction_at_nodes = sample_traction(imp, x_if, T_INIT)
    
    def make_piecewise_linear_expr(xs, vals_x, vals_y):
        if len(xs) < 2:
            return f"{float(vals_x[0]):.12e}"
        
        expr_parts = []
        for i in range(len(xs) - 1):
            x0, x1 = xs[i], xs[i+1]
            v0x, v1x = vals_x[i], vals_x[i+1]
            v0y, v1y = vals_y[i], vals_y[i+1]
            
            slope_x = (v1x - v0x) / (x1 - x0) if abs(x1 - x0) > 1e-14 else 0.0
            slope_y = (v1y - v0y) / (x1 - x0) if abs(x1 - x0) > 1e-14 else 0.0
            
            intercept_x = v0x - slope_x * x0
            intercept_y = v0y - slope_y * x0
            
            hx = f"(heaviside(x-{x0:.12e}) - heaviside(x-{x1:.12e}))"
            expr_parts.append(f"(({slope_x:.12e}*x + {intercept_x:.12e})*{hx})")
            expr_parts.append(f"(({slope_y:.12e}*x + {intercept_y:.12e})*{hx})")
        
        return " + ".join(expr_parts)
    
    tx_expr = make_piecewise_linear_expr(x_if, traction_at_nodes[:, 0], 
                                          np.zeros_like(x_if))
    ty_expr = make_piecewise_linear_expr(x_if, np.zeros_like(x_if),
                                          traction_at_nodes[:, 1])
    
    nid = 1
    nodes = []
    grid = {}
    
    for j in range(NYS + 1):
        for i in range(NXS + 1):
            x = i * LX / NXS
            y = Y0 + j * HS / NYS
            nodes.append(f"NODE {nid} COORD {x:.12f} {y:.12f} 0.0")
            grid[(i, j)] = nid
            nid += 1
    
    elems = []
    for j in range(NYS):
        for i in range(NXS):
            n1 = grid[(i, j)]
            n2 = grid[(i+1, j)]
            n3 = grid[(i+1, j+1)]
            n4 = grid[(i, j+1)]
            elems.append(f"{len(elems)+1} WALL QUAD4 {n1} {n2} {n3} {n4} MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2")
    
    dline_left = [grid[(0, j)] for j in range(NYS + 1)]
    dline_right = [grid[(NXS, j)] for j in range(NYS + 1)]
    dline_iface = [grid[(i, 0)] for i in range(NXS + 1)]
    dline_top = [grid[(i, NYS)] for i in range(NXS + 1)]
    
    deck = f'''TITLE:
  - "FSI Structure Participant (4C)"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Structure"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 50
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E_MOD}
      NUE: {NU}
      DENS: 0.0
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{tx_expr}"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{ty_expr}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 2
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN LINE NEUMANN CONDITIONS:
  - E: 3
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [1.0, 1.0]
    FUNCT: [1, 2]
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
NODE COORDS:
'''
    for n in nodes:
        deck += f'  - "{n}"\n'
    
    deck += "STRUCTURE ELEMENTS:\n"
    for e in elems:
        deck += f'  - "{e}"\n'
    
    deck += "DLINE-NODE TOPOLOGY:\n"
    for n in dline_left:
        deck += f'  - "NODE {n} DLINE 1"\n'
    for n in dline_right:
        deck += f'  - "NODE {n} DLINE 2"\n'
    for n in dline_iface:
        deck += f'  - "NODE {n} DLINE 3"\n'
    for n in dline_top:
        deck += f'  - "NODE {n} DLINE 4"\n'
    
    Path("input.4C.yaml").write_text(deck)
    
    env = dict(os.environ)
    if FOURC_LD:
        env["LD_LIBRARY_PATH"] = FOURC_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    
    result = subprocess.run(
        [FOURC_BIN, "input.4C.yaml", "out"],
        capture_output=True, text=True, env=env, cwd=str(Path.cwd())
    )
    
    if result.returncode != 0:
        print(f"4C failed with rc={result.returncode}", flush=True)
        sys.exit(1)
    
    import meshio
    
    vtus = sorted(Path("out-vtk-files").glob("structure-*-0.vtu"))
    if not vtus:
        sys.exit("4C produced no VTU file")
    
    def step_num(p):
        m = re.match(r"structure-(\d+)-\d+\.vtu$", p.name)
        return int(m.group(1)) if m else -1
    
    last_vtu = max(vtus, key=step_num)
    mesh = meshio.read(str(last_vtu))
    
    pts = np.asarray(mesh.points)[:, :2]
    disp = np.asarray(mesh.point_data["displacement"]).reshape(-1, 2)
    
    mask = np.abs(pts[:, 1] - Y0) < 1e-9
    if not mask.any():
        sys.exit(f"No nodes found at interface y={Y0}")
    
    iface_pts = pts[mask]
    iface_disp = disp[mask]
    
    ux, inv = np.unique(np.round(iface_pts[:, 0], 10), return_inverse=True)
    disp_unique = np.zeros((len(ux), 2))
    n_counts = np.zeros(len(ux))
    np.add.at(disp_unique, inv, iface_disp)
    np.add.at(n_counts, inv, 1)
    disp_unique /= n_counts[:, None]
    
    order = np.argsort(ux)
    ux = ux[order]
    disp_unique = disp_unique[order]
    
    fx_net = np.sum(traction_at_nodes[:, 0] * (LX/NXS))
    fy_net = np.sum(traction_at_nodes[:, 1] * (LX/NXS))
    
    n_nodes_total = len(nodes)
    ndof = 2 * n_nodes_total
    
    log_content = f"NDOF = {ndof}\n"
    log_content += f"4C exit code: {result.returncode}\n"
    log_content += f"Net traction received: fx={fx_net:.6e}, fy={fy_net:.6e}\n"
    log_content += f"Max displacement: |u|_max = {np.max(np.linalg.norm(disp_unique, axis=1)):.6e}\n"
    Path("run_level1_A.log").write_text(log_content)
    
    export_data = {
        "field_name": "interface_displacement",
        "n_points": int(len(ux)),
        "coordinates": [[float(x), float(Y0)] for x in ux],
        "values": disp_unique.tolist(),
        "normal_fluxes": traction_at_nodes.tolist(),
        "meta": {
            "net_force_received": [float(fx_net), float(fy_net)],
            "max_abs_disp": float(np.max(np.abs(disp_unique))),
            "n_dofs": ndof,
        }
    }
    
    Path("exports.json").write_text(json.dumps(export_data, indent=2))
    
    print(f"[structure] recv_force=({fx_net:.6e},{fy_net:.6e}) "
          f"max|dy|={np.max(np.abs(disp_unique[:, 1])):.6e}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
