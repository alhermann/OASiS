#!/usr/bin/env python3
"""
DUNE-fem participant for subdomain B (NEUMANN side) - 3D heat conduction
Interface at x = 0.625, subdomain B is [0.625, 1.5] x [0, 1] x [0, 1]
"""
import json
from pathlib import Path
import numpy as np

from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC, Constant
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 conditional, dot, ds, dx, grad, lt, abs as ufl_abs)

# Problem definition
PARTNER = "A"
X0, X1 = 0.625, 1.5
Y0, Y1 = 0.0, 1.0
Z0, Z1 = 0.0, 1.0
K = 4.0
IFACE_X = 0.625
T_OUTER = 0.0
Q_INIT = 0.0

# Mesh resolution (will be read from file)
NX = NY = NZ = 8

def source_B(x, y, z):
    """Source term for subdomain B"""
    return (-15*x**3*y**3/32 + 35*x**3*y**2/32 - 45*x**3*y*z**2/32 + 45*x**3*y*z/32 
            - 5*x**3*y/8 + 35*x**3*z**2/32 - 35*x**3*z/32 + 875*x**2*y**3/384 
            + 635*x**2*y**2/128 + 875*x**2*y*z**2/128 - 875*x**2*y*z/128 
            - 695*x**2*y/96 + 635*x**2*z**2/128 - 635*x**2*z/128 
            - 45*x*y**3*z**2/32 + 45*x*y**3*z/32 + 4585*x*y**3/2048 
            + 105*x*y**2*z**2/32 - 105*x*y**2*z/32 - 2805*x*y**2/2048 
            + 9915*x*y*z**2/2048 - 9915*x*y*z/2048 - 445*x*y/512 
            - 2805*x*z**2/2048 + 2805*x*z/2048 + 875*y**3*z**2/384 
            - 875*y**3*z/384 - 28275*y**3/4096 + 635*y**2*z**2/128 
            - 635*y**2*z/128 - 52425*y**2/4096 - 343435*y*z**2/12288 
            + 343435*y*z/12288 + 20175*y/1024 - 52425*z**2/4096 + 52425*z/4096)

def read_imports():
    """Read imported flux from partner A"""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER)
    except:
        return None

def interpolate_2d(imp, y_coords, z_coords):
    """Interpolate imported flux to (y, z) grid"""
    if not imp or "coordinates" not in imp:
        return np.full(len(y_coords) * len(z_coords), Q_INIT)
    
    coords = np.array(imp["coordinates"])
    values = np.array(imp["normal_fluxes"])
    
    result = []
    for y, z in zip(y_coords, z_coords):
        dists = np.sum((coords[:, 1:] - [y, z])**2, axis=1)
        result.append(values[np.argmin(dists)])
    return np.array(result)

def main():
    global NX, NY, NZ
    
    # Read mesh resolution
    try:
        with open("mesh_resolution.txt") as f:
            NX, NY, NZ = map(int, f.read().strip().split())
    except:
        pass
    
    # Create grid and space
    gridView = structuredGrid([X0, Y0, Z0], [X1, Y1, Z1], [NX, NY, NZ])
    space = lagrange(gridView, order=1)
    x = SpatialCoordinate(space)
    
    # Get dof coordinates
    xd = np.array(space.interpolate(x[0]).as_numpy)
    yd = np.array(space.interpolate(x[1]).as_numpy)
    zd = np.array(space.interpolate(x[2]).as_numpy)
    
    EPS = 1e-8
    
    # Find interface dofs (x = IFACE_X, interior only)
    iface_mask = np.abs(xd - IFACE_X) < EPS
    outer_y = (np.abs(yd - Y0) < EPS) | (np.abs(yd - Y1) < EPS)
    outer_z = (np.abs(zd - Z0) < EPS) | (np.abs(zd - Z1) < EPS)
    iface_interior = iface_mask & ~outer_y & ~outer_z
    iface_dofs = np.where(iface_interior)[0]
    
    # Sort by y then z
    y_if = yd[iface_dofs]
    z_if = zd[iface_dofs]
    sort_idx = np.lexsort((z_if, y_if))
    iface_dofs = iface_dofs[sort_idx]
    y_if = y_if[sort_idx]
    z_if = z_if[sort_idx]
    
    # Create weak form
    u, v = TrialFunction(space), TestFunction(space)
    a = K * dot(grad(u), grad(v)) * dx
    
    # Source term via discrete function
    src_fun = space.interpolate(0, name="source")
    src_vals = np.array([source_B(xd[i], yd[i], zd[i]) for i in range(len(xd))])
    src_fun.as_numpy[:] = src_vals
    b = src_fun * v * dx
    
    # Outer boundary conditions: u = 0 on all outer faces
    bcs = [
        DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[0] - X1), EPS), 1, 0)),  # x = 1.5
        DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[1] - Y0), EPS), 1, 0)),  # y = 0
        DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[1] - Y1), EPS), 1, 0)),  # y = 1
        DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[2] - Z0), EPS), 1, 0)),  # z = 0
        DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[2] - Z1), EPS), 1, 0)),  # z = 1
    ]
    
    # Read imported flux
    imp = read_imports()
    
    # Interface data carrier
    gfun = space.interpolate(0, name="flux_data")
    gdofs = gfun.as_numpy
    gdofs[:] = 0.0
    
    # Interpolate imported flux to interface dofs
    Q_imported = interpolate_2d(imp, y_if, z_if)
    gdofs[iface_dofs] = Q_imported
    
    # Apply Neumann BC on interface
    # For subdomain B, outward normal at interface is -x direction
    # The imported flux is from A's perspective (outward from A = +x)
    # So we need to negate it for B's natural BC
    # Natural BC: + integral(g * v) ds where g is the flux density
    b = b + conditional(lt(ufl_abs(x[0] - IFACE_X), EPS), -gfun * v, 0.0) * ds
    
    # Solve
    scheme = galerkin([a == b] + bcs, solver="cg")
    uh = space.interpolate(0, name="temperature")
    info = scheme.solve(target=uh)
    
    # Extract interface data
    T_dofs = np.array(uh.as_numpy)
    
    iface_coords = [[float(IFACE_X), float(yd[d]), float(zd[d])] for d in iface_dofs]
    iface_values = [float(T_dofs[d]) for d in iface_dofs]
    
    # Compute outward normal flux for subdomain B
    # Outward normal at interface is -x direction
    # q = -k * grad(T) . n = -k * dT/dx * (-1) = k * dT/dx
    p, w = TrialFunction(space), TestFunction(space)
    proj_form = p * w * dx == K * grad(uh)[0] * w * dx
    proj = galerkin([proj_form], solver="cg")
    qh = space.interpolate(0, name="flux")
    proj.solve(target=qh)
    Q_dofs = np.array(qh.as_numpy)
    iface_fluxes = [float(Q_dofs[d]) for d in iface_dofs]
    
    # Write exports
    exports = {
        "field_name": "temperature",
        "n_points": len(iface_coords),
        "coordinates": iface_coords,
        "values": iface_values,
        "normal_fluxes": iface_fluxes
    }
    Path("exports.json").write_text(json.dumps(exports))
    
    # Write log
    with open("run.log", "w") as f:
        f.write(f"NDOF = {len(T_dofs)}\n")
    
    print(f"DUNE B: {len(iface_coords)} iface pts, NDOF={len(T_dofs)}")

if __name__ == "__main__":
    main()
