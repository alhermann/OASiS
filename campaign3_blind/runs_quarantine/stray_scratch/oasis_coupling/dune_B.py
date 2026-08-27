"""DUNE-fem participant for subdomain B (NEUMANN side) - 3D heat conduction.

CONTRACT: runs in work_dir with no arguments, reads imports.json, writes exports.json LAST.
Neumann side: imports flux values, exports temperature values.
"""
import json
from pathlib import Path
import numpy as np
from dune.grid import structuredGrid
from dune.fem import assemble
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.fem.operator import galerkin as operator_galerkin
from dune.ufl import DirichletBC, Constant
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 conditional, dot, ds, dx, grad, lt, abs as ufl_abs)

# Problem parameters for subdomain B
SIDE = "neumann"
PARTNER = "A"
X0, X1 = 0.625, 1.5  # Subdomain B: x in [0.625, 1.5]
Y0, Y1 = 0.0, 1.0
Z0, Z1 = 0.0, 1.0
K = 4.0  # thermal conductivity
IFACE_X = 0.625  # interface location (left side of subdomain B)
T_OUTER = 0.0  # outer boundary condition
T_INIT = 0.0
Q_INIT = 0.0

# Mesh resolution - will be set based on level
NX, NY, NZ = 12, 8, 8  # default, will be overridden

# Source term for subdomain B
F_SRC = Constant(0.0, name="f_src")  # Will be set to polynomial

def source_B(x, y, z):
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
    """Read imported flux data from partner."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER)


def interpolate_imported(imp, y_coords, z_coords):
    """Interpolate imported flux to interface grid."""
    if not imp or "coordinates" not in imp:
        return np.full(len(y_coords) * len(z_coords), Q_INIT)
    
    coords = np.array(imp["coordinates"])
    values = np.array(imp["normal_fluxes"])
    
    result = []
    for y in y_coords:
        for z in z_coords:
            dists = np.sqrt((coords[:, 1] - y)**2 + **(coords[:, 2] - z)2)
            if len(dists) > 0:
                idx = np.argmin(dists)
                result.append(values[idx])
            else:
                result.append(Q_INIT)
    return np.array(result)


def main():
    global NX, NY, NZ
    
    # Read mesh resolution from file
    try:
        with open("mesh_resolution.txt", "r") as f:
            NX, NY, NZ = map(int, f.read().strip().split())
    except:
        pass  # Use defaults
    
    # Read imported data
    imp = read_imports()
    
    # Create grid
    gridView = structuredGrid([X0, Y0, Z0], [X1, Y1, Z1], [NX, NY, NZ])
    space = lagrange(gridView, order=1)
    x = SpatialCoordinate(space)
    
    # Get dof coordinates
    xd = np.array(space.interpolate(x[0], name="xcoord").as_numpy)
    yd = np.array(space.interpolate(x[1], name="ycoord").as_numpy)
    zd = np.array(space.interpolate(x[2], name="zcoord").as_numpy)
    
    # Find interface dofs (x = IFACE_X, interior only)
    EPS = 1e-8
    iface_mask = np.abs(xd - IFACE_X) < EPS
    # Exclude dofs on outer boundary (y=0, y=1, z=0, z=1)
    outer_y = (np.abs(yd - Y0) < EPS) | (np.abs(yd - Y1) < EPS)
    outer_z = (np.abs(zd - Z0) < EPS) | (np.abs(zd - Z1) < EPS)
    iface_interior = iface_mask & ~outer_y & ~outer_z
    iface_dofs = np.where(iface_interior)[0]
    iface_dofs = iface_dofs[np.argsort(yd[iface_dofs])]  # sort by y, then z
    
    y_if = yd[iface_dofs]
    z_if = zd[iface_dofs]
    
    # Find outer boundary dofs (x = X1)
    outer_x_dofs = np.where(np.abs(xd - X1) < EPS)[0]
    
    # Create weak form
    u, v = TrialFunction(space), TestFunction(space)
    a = K * dot(grad(u), grad(v)) * dx
    
    # Source term - use a discrete function for the polynomial source
    src_fun = space.interpolate(0, name="source")
    src_vals = np.array([source_B(xd[i], yd[i], zd[i]) for i in range(len(xd))])
    src_fun.as_numpy[:] = src_vals
    b = src_fun * v * dx
    
    # Outer boundary condition: u = 0 on x = X1
    bcs = [DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[0] - X1), EPS), 1, 0))]
    
    # Also constrain y and z boundaries
    bcs.append(DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[1] - Y0), EPS), 1, 0)))
    bcs.append(DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[1] - Y1), EPS), 1, 0)))
    bcs.append(DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[2] - Z0), EPS), 1, 0)))
    bcs.append(DirichletBC(space, T_OUTER, conditional(lt(ufl_abs(x[2] - Z1), EPS), 1, 0)))
    
    # Interface data carrier
    gfun = space.interpolate(0, name="iface_data")
    gdofs = gfun.as_numpy
    gdofs[:] = 0.0
    
    # Neumann side: import flux and apply as natural BC
    Q_imported = interpolate_imported(imp, y_if, z_if)
    gdofs[iface_dofs] = Q_imported
    
    # Apply flux on interface: + integral(g * v) ds
    # The flux is applied with respect to the outward normal of subdomain B
    # Outward normal at x = IFACE_X for subdomain B is -x direction
    # So we need to negate the imported flux (which is from A's perspective)
    b = b + conditional(lt(ufl_abs(x[0] - IFACE_X), EPS), -gfun * v, 0.0) * ds
    
    # Solve
    scheme = galerkin([a == b] + bcs, solver="cg")
    uh = space.interpolate(0, name="temperature")
    info = scheme.solve(target=uh)
    
    # Extract interface data for export
    T_dofs = np.array(uh.as_numpy)
    
    iface_coords = []
    iface_values = []
    iface_fluxes = []
    
    for dof in iface_dofs:
        iface_coords.append([float(IFACE_X), float(yd[dof]), float(zd[dof])])
        iface_values.append(float(T_dofs[dof]))
        
        # Compute outward normal flux for subdomain B
        # Outward normal at interface is -x direction
        # q = -k * grad(T) . n = -k * dT/dx * (-1) = k * dT/dx
        # Use projection to get gradient
        pass
    
    # Compute flux via projection
    p, w = TrialFunction(space), TestFunction(space)
    # q = -k * grad(T) . n_out, where n_out = -e_x at interface
    # So q = k * dT/dx
    proj_form = p * w * dx == K * grad(uh)[0] * w * dx
    proj = galerkin([proj_form], solver="cg")
    qh = space.interpolate(0, name="normal_flux")
    proj.solve(target=qh)
    Q_dofs = np.array(qh.as_numpy)
    
    for dof in iface_dofs:
        iface_fluxes.append(float(Q_dofs[dof]))
    
    # Write exports
    exports = {
        "field_name": "temperature",
        "n_points": len(iface_coords),
        "coordinates": iface_coords,
        "values": iface_values,
        "normal_fluxes": iface_fluxes
    }
    
    Path("exports.json").write_text(json.dumps(exports, indent=2))
    
    # Write NDOF to log file
    ndof = len(T_dofs)
    with open("run.log", "w") as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(f"Interface points = {len(iface_coords)}\n")
    
    print(f"DUNE B (Neumann): {len(iface_coords)} interface points, NDOF = {ndof}")


if __name__ == "__main__":
    main()
