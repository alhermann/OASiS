#!/usr/bin/env python3
"""
Participant B (NGSolve): Subdomain (0.625, 1.5) x (0, 1)
Neumann side - receives flux from partner, returns temperature
K = [[6, 0.5], [0.5, 3]]
Interface at x = 0.625 (left boundary of this subdomain)
"""
import json
from pathlib import Path
import numpy as np
import os

# Import NGSolve
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, CoefficientFunction, GridFunction,
                     H1, LinearForm, Mesh, NodeId, TaskManager, ds, dx, grad, SetCoefficient)

# Problem parameters
X0, X1 = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
IFACE_X = 0.625  # Left boundary of subdomain B
PARTNER = "A"

# Anisotropic conductivity tensor K = [[6, 0.5], [0.5, 3]]
K00, K01, K10, K11 = 6.0, 0.5, 0.5, 3.0

def source_B(x, y):
    """Source term for subdomain B."""
    return (x**3*y/16 + x**3/72 + x**2*y**2/32 + 467*x**2*y/576 - 41*x**2/144 
            + x*y**3/8 + 67*x*y**2/192 + 12451*x*y/27648 - 7405*x/13824 
            + 17*y**3/32 - 7285*y**2/18432 - 189917*y/55296 + 77521/55296)

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
    # Determine mesh resolution from environment
    NX = int(os.environ.get('NX', 8))
    NY = int(os.environ.get('NY', 8))
    
    MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)
    ORDER = 1
    
    # Interface is on the LEFT (x = X0 = IFACE_X)
    ON_LEFT = True
    OUTER_X = X1  # Right boundary has Dirichlet BC
    S = -1.0  # Outward normal at interface points LEFT (negative x direction)
    
    TOL = 1e-9 * max(X1 - X0, Y1 - Y0)
    
    # Read imports
    imp = read_imports()
    
    # Create geometry with boundary names
    geo = SplineGeometry()
    # Rectangle with boundary names: bottom, left, top, right
    # For subdomain B, interface is on the LEFT
    geo.AddRectangle((X0, Y0), (X1, Y1),
                     bcs=("bottom", "interface", "top", "outer"))
    mesh = Mesh(geo.GenerateMesh(maxh=MAXH))
    
    # Create finite element space
    # Neumann side: Dirichlet only on outer boundary (right side)
    fes = H1(mesh, order=ORDER, dirichlet="outer")
    u, v = fes.TnT()
    
    # Get vertex-to-dof mapping
    vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, i))[0] for i in range(mesh.nv)], dtype=int)
    vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], dtype=float)
    
    # Find interface vertices (on left boundary, excluding corners)
    iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
    # Exclude corners
    iface_v = iface_v[(vxy[iface_v, 1] > Y0 + TOL) & (vxy[iface_v, 1] < Y1 - TOL)]
    iface_v = iface_v[np.argsort(vxy[iface_v, 1])]  # Sort by y
    y_if = vxy[iface_v, 1]
    iface_dofs = vdof[iface_v]
    
    # Find outer boundary vertices (right side)
    outer_v = np.where(np.abs(vxy[:, 0] - OUTER_X) < TOL)[0]
    outer_dofs = vdof[outer_v]
    
    # Define source term as coefficient function
    f_src = CoefficientFunction(source_B)
    
    # Bilinear form with anisotropic conductivity
    a = BilinearForm(fes)
    # -div(K grad u) . v = K grad u . grad v
    # For anisotropic K: K00*ux*vx + K01*ux*vy + K10*uy*vx + K11*uy*vy
    a += (K00 * grad(u)[0] * grad(v)[0] + 
          K01 * grad(u)[0] * grad(v)[1] + 
          K10 * grad(u)[1] * grad(v)[0] + 
          K11 * grad(u)[1] * grad(v)[1]) * dx
    
    # Linear form
    f = LinearForm(fes)
    f += f_src * v * dx
    
    # Add Neumann contribution from partner's flux
    if imp and imp.get("normal_fluxes"):
        q_if = sample_interface(imp, "normal_fluxes", 0.0, y_if)
        
        # Create trace function for flux
        gfun = GridFunction(fes)
        gfun.vec[:] = 0.0
        for d, q in zip(iface_dofs, q_if):
            gfun.vec[int(d)] = float(q)
        
        # Neumann BC: + integral(g * v) ds on interface
        # The flux from partner is already in the correct sign convention
        f += gfun * v * ds("interface")
    
    # Initialize solution with Dirichlet values
    gfu = GridFunction(fes)
    gfu.vec[:] = 0.0
    # Outer boundary: u = 0
    for d in outer_dofs:
        gfu.vec[int(d)] = 0.0
    
    # Solve
    with TaskManager():
        a.Assemble()
        f.Assemble()
        
        res = f.vec.CreateVector()
        res.data = f.vec - a.mat * gfu.vec
        
        gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * res
    
    # Compute outward normal flux on interface
    # qn = -(K grad u) . n_out where n_out = (-1, 0) for left boundary
    # qn = -(-K00*ux - K01*uy) = K00*ux + K01*uy
    
    # Use L2 projection to get gradient
    fesq = H1(mesh, order=ORDER)
    p, w = fesq.TnT()
    
    m = BilinearForm(fesq)
    m += p * w * dx
    m.Assemble()
    
    # Project -K grad u . n = K00*ux + K01*uy (outward flux for left boundary)
    fq = LinearForm(fesq)
    fq += (K00 * grad(gfu)[0] + K01 * grad(gfu)[1]) * w * dx
    fq.Assemble()
    
    qh = GridFunction(fesq)
    qh.vec.data = m.mat.Inverse(fesq.FreeDofs(), inverse="sparsecholesky") * fq.vec
    
    # Extract flux at interface nodes
    qdofs = np.array([fesq.GetDofNrs(NodeId(VERTEX, int(i)))[0] for i in iface_v], dtype=int)
    Q = np.array([qh.vec[int(d)] for d in qdofs], dtype=float)
    
    # Extract temperature at interface nodes
    T_vals = np.array([gfu.vec[int(d)] for d in iface_dofs], dtype=float)
    
    # Write exports.json
    coords = [[float(IFACE_X), float(yy)] for yy in y_if]
    
    exports = {
        "field_name": "temperature",
        "n_points": len(iface_v),
        "coordinates": coords,
        "values": T_vals.tolist(),
        "normal_fluxes": Q.tolist()
    }
    
    with open("exports.json", "w") as f:
        json.dump(exports, f, indent=2)
    
    # Write NDOF to log file
    with open("run.log", "w") as lf:
        lf.write(f"NDOF = {fes.ndof}\n")
    
    print(f"Participant B: {len(iface_v)} interface points, exported")

if __name__ == "__main__":
    main()
