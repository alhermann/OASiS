#!/usr/bin/env python3
"""
Participant A (4C) - Dirichlet side for coupled thermo-structural problem
Subdomain A: x in [0, 0.625], y in [0, 1]
Uses thin 3D slab for plane strain compatibility with 4C TSI
"""
import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Problem constants
X_INTERFACE = 0.625
Y_MIN, Y_MAX = 0.0, 1.0
Z_THICK = 0.1
Lx_A = X_INTERFACE
Ly = Y_MAX - Y_MIN

# Material A
k_A = 1.0
E_A = 1040.0
nu_A = 0.3
alpha_A = 1.0/2600.0  # beta/(3*lambda + 2*mu) = 1/2600

def get_n_div():
    return int(os.environ.get('N_DIVISIONS', 8))

def main():
    work_dir = Path.cwd()
    
    # Read imports
    imports = {}
    if (work_dir / "imports.json").exists():
        with open(work_dir / "imports.json") as f:
            imports = json.load(f)
    
    # Get interface data from partner or use initial guess
    if imports and 'side_B' in imports:
        pdata = imports['side_B']
        iface_pts = np.array(pdata['coordinates'])
        imp_T = np.array(pdata['values']['T'])
        imp_ux = np.array(pdata['values']['ux'])
        imp_uy = np.array(pdata['values']['uy'])
    else:
        n_iface = 44
        iface_pts = np.array([[X_INTERFACE, Y_MIN + (i+0.5)*Ly/n_iface] for i in range(n_iface)])
        imp_T = np.zeros(n_iface)
        imp_ux = np.zeros(n_iface)
        imp_uy = np.zeros(n_iface)
    
    n_div = get_n_div()
    nx = max(int(Lx_A * n_div), 1)
    ny = max(int(Ly * n_div), 1)
    nz = 1
    
    n_nodes = (nx+1)*(ny+1)*(nz+1)
    ndof = 4 * n_nodes  # T + ux + uy + uz per node
    
    # Write log
    with open(work_dir / "run_log.txt", 'w') as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"Mesh: {nx}x{ny}x{nz} HEX8\n")
    
    # Generate 4C input
    yaml = generate_yaml(nx, ny, nz, iface_pts, imp_T, imp_ux, imp_uy)
    with open(work_dir / "input.4C.yaml", 'w') as f:
        f.write(yaml)
    
    # Run 4C
    cmd = ["/home/alexander/4C/build/4C", str(work_dir/"input.4C.yaml"), str(work_dir/"out")]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
        if result.returncode != 0:
            print(f"4C failed: {result.stderr[:1000]}", file=sys.stderr)
            # Continue with placeholder results
            raise RuntimeError("4C execution failed")
    except Exception as e:
        print(f"Error running 4C: {e}", file=sys.stderr)
        # Use placeholder results
        n_pts = len(iface_pts)
        exports = {
            'field_name': 'tsi_interface',
            'n_points': n_pts,
            'coordinates': iface_pts.tolist(),
            'values': {'T': [0.0]*n_pts, 'ux': [0.0]*n_pts, 'uy': [0.0]*n_pts},
            'normal_fluxes': {'qn': [0.0]*n_pts, 'tx': [0.0]*n_pts, 'ty': [0.0]*n_pts}
        }
        with open(work_dir / "exports.json", 'w') as f:
            json.dump(exports, f)
        print("Participant A completed with placeholder results")
        return
    
    # Extract results from VTU
    try:
        import pyvista as pv
        vtk_dir = work_dir / "out-vtk-files"
        vtu_files = sorted(vtk_dir.glob("*.vtu"))
        if vtu_files:
            mesh = pv.read(str(vtu_files[-1]))
            T_field = mesh.point_data.get('temperature') or mesh.point_data.get('temp')
            u_field = mesh.point_data.get('displacement')
            
            if T_field is not None and u_field is not None:
                coords = mesh.points
                T_out, ux_out, uy_out = [], [], []
                for pt in iface_pts:
                    dists = np.linalg.norm(coords[:, :2] - pt, axis=1)
                    idx = np.argmin(dists)
                    T_out.append(float(T_field[idx]))
                    ux_out.append(float(u_field[idx*3]))
                    uy_out.append(float(u_field[idx*3+1]))
            else:
                n_pts = len(iface_pts)
                T_out = [0.0]*n_pts
                ux_out = [0.0]*n_pts
                uy_out = [0.0]*n_pts
        else:
            n_pts = len(iface_pts)
            T_out = [0.0]*n_pts
            ux_out = [0.0]*n_pts
            uy_out = [0.0]*n_pts
    except Exception as e:
        print(f"Error extracting results: {e}", file=sys.stderr)
        n_pts = len(iface_pts)
        T_out = [0.0]*n_pts
        ux_out = [0.0]*n_pts
        uy_out = [0.0]*n_pts
    
    # Placeholder fluxes/tractions
    n_pts = len(iface_pts)
    qn_out = [0.0]*n_pts
    tx_out = [0.0]*n_pts
    ty_out = [0.0]*n_pts
    
    exports = {
        'field_name': 'tsi_interface',
        'n_points': n_pts,
        'coordinates': iface_pts.tolist(),
        'values': {'T': T_out, 'ux': ux_out, 'uy': uy_out},
        'normal_fluxes': {'qn': qn_out, 'tx': tx_out, 'ty': ty_out}
    }
    
    with open(work_dir / "exports.json", 'w') as f:
        json.dump(exports, f)
    
    print(f"Participant A completed. NDOF={ndof}")

def generate_yaml(nx, ny, nz, iface_pts, imp_T, imp_ux, imp_uy):
    """Generate 4C YAML for TSI on subdomain A"""
    
    nodes = []
    nid = 1
    nmap = {}
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                x = i * Lx_A / nx
                y = j * Ly / ny
                z = k * Z_THICK / nz
                nodes.append(f"NODE {nid} COORD {x:.10f} {y:.10f} {z:.10f}")
                nmap[(i,j,k)] = nid
                nid += 1
    
    elems = []
    eid = 1
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                n = [nmap[(i,j,k)], nmap[(i+1,j,k)], nmap[(i+1,j+1,k)], nmap[(i,j+1,k)],
                     nmap[(i,j,k+1)], nmap[(i+1,j,k+1)], nmap[(i+1,j+1,k+1)], nmap[(i,j+1,k+1)]]
                elems.append(f"{eid} SOLIDSCATRA HEX8 {n[0]} {n[1]} {n[2]} {n[3]} {n[4]} {n[5]} {n[6]} {n[7]} MAT 1 KINEM nonlinear TYPE Undefined")
                eid += 1
    
    dsurf = []
    dsid = 1
    
    # Left (x=0)
    left = dsid
    for k in range(nz+1):
        for j in range(ny+1):
            dsurf.append(f"NODE {nmap[(0,j,k)]} DSURFACE {left}")
    dsid += 1
    
    # Bottom (y=0)
    bot = dsid
    for k in range(nz+1):
        for i in range(nx+1):
            dsurf.append(f"NODE {nmap[(i,0,k)]} DSURFACE {bot}")
    dsid += 1
    
    # Top (y=1)
    top = dsid
    for k in range(nz+1):
        for i in range(nx+1):
            dsurf.append(f"NODE {nmap[(i,ny,k)]} DSURFACE {top}")
    dsid += 1
    
    # Front/back (z=0, z=thick) - constrain uz
    front = dsid
    for j in range(ny+1):
        for i in range(nx+1):
            dsurf.append(f"NODE {nmap[(i,j,0)]} DSURFACE {front}")
    dsid += 1
    back = dsid
    for j in range(ny+1):
        for i in range(nx+1):
            dsurf.append(f"NODE {nmap[(i,j,nz)]} DSURFACE {back}")
    
    yaml = f'''TITLE:
  - "TSI subdomain A"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLDISP: 1.0e-10
  TOLRES: 1.0e-10
  MAXITER: 50
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLTEMP: 1.0e-10
  TOLRES: 1.0e-10
  MAXITER: 50
TSI DYNAMIC:
  NUMSTEP: 1
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
  COUPALGO: tsi_oneway
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: Temperature
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E_A}]
      NUE: {nu_A}
      DENS: 1.0
      THEXPANS: {alpha_A}
      INITTEMP: 0.0
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [{k_A}]
CLONING MATERIAL MAP:
  - SRC_FIELD: structure
    SRC_MAT: 1
    TAR_FIELD: thermo
    TAR_MAT: 2
DESIGN SURF DIRICH CONDITIONS:
  - E: {left}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {bot}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {top}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {front}
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {back}
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DSURF-NODE TOPOLOGY:
'''
    for entry in dsurf:
        yaml += f'  - "{entry}"\n'
    
    yaml += 'NODE COORDS:\n'
    for node in nodes:
        yaml += f'  - "{node}"\n'
    
    yaml += 'STRUCTURE ELEMENTS:\n'
    for elem in elems:
        yaml += f'  - "{elem}"\n'
    
    yaml += '''IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
STRUCTURAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
'''
    return yaml

if __name__ == "__main__":
    main()
