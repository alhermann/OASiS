#!/usr/bin/env python3
"""
Participant A: Subdomain (0, 0.625) x (0, 1)
Dirichlet side - receives temperature from partner, returns flux
K = [[1, 0.5], [0.5, 2]]
Interface at x = 0.625 (right boundary)
Uses scikit-fem for FEM solve
"""
import json
from pathlib import Path
import numpy as np
import os

import skfem
from skfem import *
from skfem.models.poisson import laplace, mass

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
    """Linear interpolation."""
    if len(y_data) < 2:
        return np.full(len(y_query), v_data[0] if len(v_data) > 0 else 0.0)
    return np.interp(y_query, y_data, v_data)

def main():
    NX = int(os.environ.get('NX', 8))
    NY = int(os.environ.get('NY', 8))
    
    # Create mesh
    m = skfem.MeshQuad.init_rect(X0, X1, Y0, Y1, NX, NY)
    e = ElementQuad1()
    fe = Basis(m, e)
    
    # Identify boundary edges
    # Right boundary (interface): x = X1
    D_interface = m.boundary_nodes(m.p[0] > X1 - 1e-10)
    # All boundaries for Dirichlet
    D_all = m.boundary_nodes()
    
    # Read imports
    imp = read_imports()
    
    # Get interface DOFs (excluding corners)
    iface_nodes = m.boundary_nodes(m.p[0] > X1 - 1e-10)
    # Exclude corners
    corner_tol = 1e-9
    iface_nodes = iface_nodes[(m.p[1, iface_nodes] > Y0 + corner_tol) & 
                               (m.p[1, iface_nodes] < Y1 - corner_tol)]
    
    # Sort by y coordinate
    iface_order = np.argsort(m.p[1, iface_nodes])
    iface_nodes_sorted = iface_nodes[iface_order]
    y_iface = m.p[1, iface_nodes_sorted]
    
    # For Dirichlet side: apply imported T on interface, u=0 elsewhere
    if imp and imp.get("coordinates"):
        ys = np.array([c[1] for c in imp["coordinates"]])
        Ts = np.array(imp.get("values", []))
        T_iface = interpolate_1d(y_iface, ys, Ts)
    else:
        T_iface = np.zeros(len(iface_nodes_sorted))
    
    # Assemble system
    @BilinearForm
    def anisotropic_laplace(u, v, w):
        grad_u = np.array([w.grad(u)[0], w.grad(u)[1]])
        grad_v = np.array([w.grad(v)[0], w.grad(v)[1]])
        # K_ij * grad_u_i * grad_v_j
        return (K[0,0]*grad_u[0]*grad_v[0] + K[0,1]*grad_u[0]*grad_v[1] +
                K[1,0]*grad_u[1]*grad_v[0] + K[1,1]*grad_u[1]*grad_v[1])
    
    @LinearForm
    def source_form(v, w):
        return source_form_func(w.x[0], w.x[1]) * v
    
    def source_form_func(x, y):
        return source_A(x, y)
    
    A = anisotropic_laplace.assemble()
    b = assemble(source_form, fe)
    
    # Apply Dirichlet BCs
    D = D_all.copy()  # All boundaries constrained
    
    # Set interface values
    I = identity(A.shape[0])
    I[:, iface_nodes_sorted] = 0
    I[iface_nodes_sorted, iface_nodes_sorted] = 1
    
    # Solve with Dirichlet conditions
    from skfem.helpers import condense
    
    # Interior DOFs
    i_D = np.concatenate([D, iface_nodes_sorted])
    i_D = np.unique(i_D)
    i_dof = np.setdiff1d(np.arange(A.shape[0]), i_D)
    
    # Boundary values
    bnd_vals = np.zeros(len(i_D))
    bnd_vals[np.searchsorted(i_D, iface_nodes_sorted)] = T_iface
    
    A_red, b_red = condense(A, b, D=i_D, bnd_values=bnd_vals)
    
    x_red = solve(*A_red, b_red)
    x = np.zeros(A.shape[0])
    x[i_dof] = x_red
    x[i_D] = bnd_vals
    
    # Compute outward normal flux on interface
    # qn = -(K grad u) . n where n = (1, 0) for right boundary
    # qn = -(K[0,0]*ux + K[0,1]*uy)
    
    # Project gradient onto H1
    feq = Basis(m, ElementQuad1())
    
    @BilinearForm
    def mass_form(u, v, w):
        return u * v
    
    @LinearForm  
    def flux_form_x(v, w):
        ux = w.grad(x)[0]
        uy = w.grad(x)[1]
        return -(K[0,0]*ux + K[0,1]*uy) * v
    
    M = mass_form.assemble(feq)
    f_flux = assemble(flux_form_x, feq)
    q_proj = solve(*M, f_flux)
    
    # Extract flux at interface nodes
    Q = q_proj[iface_nodes_sorted]
    T_vals = x[iface_nodes_sorted]
    
    # Write exports
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
    
    with open("run.log", "w") as lf:
        lf.write(f"NDOF = {fe.ndofs}\n")
    
    print(f"Participant A: {len(iface_nodes_sorted)} interface points")

if __name__ == "__main__":
    main()
