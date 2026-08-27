#!/usr/bin/env python3
"""
Participant A (scikit-fem): Subdomain (0, 0.625) x (0, 1)
Dirichlet side - receives temperature from partner B, returns flux
K = [[1, 0.5], [0.5, 2]]
Interface at x = 0.625 (right boundary)
"""
import json
from pathlib import Path
import numpy as np
import os

import skfem
from skfem import *
from skfem.helpers import condense

# Problem parameters
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625
PARTNER = "B"

# Anisotropic conductivity K = [[1, 0.5], [0.5, 2]]
K = np.array([[1.0, 0.5], [0.5, 2.0]])

def source_A(x, y):
    """Source term for subdomain A."""
    xx, yy = x[0], y[0]
    return (9*xx**3*yy + 2*xx**3 + 27*xx**2*yy**2/4 + 129*xx**2*yy/16 - 319*xx**2/24 
            + 9*xx*yy**3/2 + 177*xx*yy**2/32 - 631*xx*yy/24 + 323*xx/32
            + 27*yy**3/32 - 85*yy**2/12 + 673*yy/96 - 37/48)

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
        return d.get(PARTNER)
    except:
        return None

def interpolate_1d(y_query, y_data, v_data):
    """Linear interpolation with clamping."""
    if len(y_data) < 2:
        return np.full(len(y_query), v_data[0] if len(v_data) > 0 else 0.0)
    # Clamp to range
    y_min, y_max = min(y_data), max(y_data)
    y_clamped = np.clip(y_query, y_min, y_max)
    return np.interp(y_clamped, y_data, v_data)

def main():
    NX = int(os.environ.get('NX', 8))
    NY = int(os.environ.get('NY', 8))
    
    # Create mesh
    m = skfem.MeshQuad.init_rect(X0, X1, Y0, Y1, NX, NY)
    e = ElementQuad1()
    fe = Basis(m, e)
    
    # Identify interface nodes (right boundary, excluding corners)
    corner_tol = 1e-9
    iface_nodes = m.boundary_nodes(m.p[0] > X1 - 1e-10)
    iface_nodes = iface_nodes[(m.p[1, iface_nodes] > Y0 + corner_tol) & 
                               (m.p[1, iface_nodes] < Y1 - corner_tol)]
    
    # Sort by y coordinate
    iface_order = np.argsort(m.p[1, iface_nodes])
    iface_nodes_sorted = iface_nodes[iface_order]
    y_iface = m.p[1, iface_nodes_sorted]
    
    # Read imports from partner
    imp = read_imports()
    
    # Get interface temperature from partner (or use initial guess)
    if imp and imp.get("coordinates"):
        ys = np.array([c[1] for c in imp["coordinates"]])
        Ts = np.array(imp.get("values", []))
        T_iface = interpolate_1d(y_iface, ys, Ts)
    else:
        T_iface = np.zeros(len(iface_nodes_sorted))  # Initial guess
    
    # Assemble anisotropic Laplacian
    @BilinearForm
    def anisotropic_laplace(u, v, w):
        grad_u = np.array([w.grad(u)[0], w.grad(u)[1]])
        grad_v = np.array([w.grad(v)[0], w.grad(v)[1]])
        return (K[0,0]*grad_u[0]*grad_v[0] + K[0,1]*grad_u[0]*grad_v[1] +
                K[1,0]*grad_u[1]*grad_v[0] + K[1,1]*grad_u[1]*grad_v[1])
    
    @LinearForm  
    def source_form(v, w):
        return source_A(w.x[0], w.x[1]) * v
    
    A = anisotropic_laplace.assemble(fe)
    b = assemble(source_form, fe)
    
    # Dirichlet boundaries: all outer boundaries + interface
    D_outer = m.boundary_nodes()
    D = np.concatenate([D_outer, iface_nodes_sorted])
    D = np.unique(D)
    
    # Boundary values: 0 on outer, imported T on interface
    bnd_vals = np.zeros(len(D))
    for i, node in enumerate(iface_nodes_sorted):
        idx = np.where(D == node)[0][0]
        bnd_vals[idx] = T_iface[i]
    
    # Solve
    A_red, b_red = condense(A, b, D=D, bnd_values=bnd_vals)
    x_red = solve(*A_red, b_red)
    x = np.zeros(A.shape[0])
    x[np.setdiff1d(np.arange(A.shape[0]), D)] = x_red
    x[D] = bnd_vals
    
    # Compute outward normal flux on interface
    # qn = -(K grad u) . n where n = (1, 0) for right boundary
    # qn = -(K[0,0]*ux + K[0,1]*uy)
    
    feq = Basis(m, ElementQuad1())
    
    @BilinearForm
    def mass_form(u, v, w):
        return u * v
    
    @LinearForm  
    def flux_form(v, w):
        ux = w.grad(x)[0]
        uy = w.grad(x)[1]
        return -(K[0,0]*ux + K[0,1]*uy) * v
    
    M = mass_form.assemble(feq)
    f_flux = assemble(flux_form, feq)
    q_proj = solve(*M, f_flux)
    
    # Extract values at interface
    Q = q_proj[iface_nodes_sorted]
    T_vals = x[iface_nodes_sorted]
    
    # Write exports.json
    coords = [[IFACE_X, float(y)] for y in y_iface]
    
    exports = {
        "field_name": "temperature",
        "n_points": len(iface_nodes_sorted),
        "coordinates": coords,
        "values": T_vals.tolist(),
        "normal_fluxes": Q.tolist()
    }
    
    with open("exports.json", "w") as f:
        json.dump(exports, f, indent=2)
    
    # Write log file
    with open("run.log", "w") as lf:
        lf.write(f"NDOF = {fe.ndofs}\n")
    
    print(f"Participant A: {len(iface_nodes_sorted)} interface points, NDOF={fe.ndofs}")

if __name__ == "__main__":
    main()
