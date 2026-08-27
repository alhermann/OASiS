#!/usr/bin/env python3
"""
Participant A (deal.II): Subdomain (0, 0.625) x (0, 1)
Dirichlet side - receives temperature from partner, returns flux
K = [[1, 0.5], [0.5, 2]]
Interface at x = 0.625 (right boundary)
"""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

# Problem parameters
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
PARTNER = "B"
DEALII_EXE = "/home/alexander/oasis_coupled_run/dealii_src/build/heat_iface_dealii"

def read_imports():
    """Read imports.json from coupling driver."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
        return d.get(PARTNER)
    except:
        return None

def sample_interface(imp, key, fallback, y_coords):
    """Interpolate partner's data onto our interface y-coordinates."""
    if imp and imp.get("coordinates"):
        ys = np.array([c[1] for c in imp["coordinates"]])
        vs = np.array(imp.get(key, []))
        if len(vs) == len(ys) and len(ys) > 0:
            idx = np.argsort(ys)
            return np.interp(y_coords, ys[idx], vs[idx])
    return np.full(len(y_coords), fallback)

def main():
    # Determine mesh resolution from NX, NY (will be set per level)
    # For now, use a default that will be overridden
    import os
    NX = int(os.environ.get('NX', 8))
    NY = int(os.environ.get('NY', 8))
    
    # Interface y-coordinates (excluding corners)
    n_interface_points = NY
    y_interface = np.linspace(Y0 + (Y1-Y0)/(2*NY), Y1 - (Y1-Y0)/(2*NY), n_interface_points)
    
    # Read imports
    imp = read_imports()
    
    # Get interface temperature from partner (or use initial guess)
    T_init = 0.0  # Initial guess
    T_interface = sample_interface(imp, "values", T_init, y_interface)
    
    # Write deal.II input file
    # Format: side_flag X0 X1 Y0 Y1 IFACE_X NX NY DEGREE n_samples
    #         y_0 T_0
    #         y_1 T_1
    #         ...
    with open("dealii_input.txt", "w") as f:
        f.write(f"0 {X0} {X1} {Y0} {Y1} {IFACE_X} {NX} {NY} 1 {len(y_interface)}\n")
        for y, T in zip(y_interface, T_interface):
            f.write(f"{y:.16g} {T:.16g}\n")
    
    # Remove old output if exists
    out_path = Path("dealii_output.txt")
    if out_path.exists():
        out_path.unlink()
    
    # Run deal.II solver
    result = subprocess.run([DEALII_EXE, "dealii_input.txt", "dealii_output.txt"],
                           capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"deal.II solver failed (rc={result.returncode})", file=sys.stderr)
        print(result.stdout, file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    
    # Read output
    if not out_path.exists():
        print("deal.II produced no output file", file=sys.stderr)
        sys.exit(1)
    
    coords, temps, fluxes = [], [], []
    with open(out_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 3:
                y, T, q = float(parts[0]), float(parts[1]), float(parts[2])
                coords.append([IFACE_X, y])
                temps.append(T)
                fluxes.append(q)
    
    if not coords:
        print("deal.II produced no interface points", file=sys.stderr)
        sys.exit(1)
    
    # Write exports.json
    exports = {
        "field_name": "temperature",
        "n_points": len(coords),
        "coordinates": coords,
        "values": temps,
        "normal_fluxes": fluxes
    }
    
    with open("exports.json", "w") as f:
        json.dump(exports, f, indent=2)
    
    # Also write NDOF to log file
    ndof_path = Path("ndof.txt")
    if ndof_path.exists():
        with open(ndof_path) as nf:
            content = nf.read()
            with open("run.log", "w") as lf:
                lf.write(content)
    else:
        with open("run.log", "w") as lf:
            lf.write("NDOF = unknown\n")
    
    print(f"Participant A: {len(coords)} interface points, exported")

if __name__ == "__main__":
    main()
