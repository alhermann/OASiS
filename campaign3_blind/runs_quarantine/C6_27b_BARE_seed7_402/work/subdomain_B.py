#!/usr/bin/env python3
"""
Subdomain B solver using NGSolve - Simplified version.
Solves -div(K grad u) = f on (0.625, 1.5) x (0, 1)
with K = [[6, 1/2], [1/2, 3]]
Dirichlet BC on outer boundary, Neumann BC on interface (x=0.625)
"""

import sys
import numpy as np
from ngsolve import *
from netgen.csg import CSGeometry, OrthoBrick, Pnt

def main():
    if len(sys.argv) < 7:
        print("Usage: python subdomain_B.py <level> <input_flux> <output_solution> <output_interface> <output_log> <output_values>")
        sys.exit(1)
    
    level = int(sys.argv[1])
    input_flux = sys.argv[2]
    output_solution = sys.argv[3]
    output_interface = sys.argv[4]
    output_log = sys.argv[5]
    output_values = sys.argv[6]
    
    # Domain parameters
    x_left = 0.625
    x_right = 1.5
    y_bottom = 0.0
    y_top = 1.0
    
    # Conductivity tensor K = [[6, 1/2], [1/2, 3]]
    K11 = 6.0
    K12 = 0.5
    K21 = 0.5
    K22 = 3.0
    
    # Create mesh
    h = 1.0 / (8 * (2 ** (level - 1)))
    
    # Create structured mesh using CSGeometry
    geom = CSGeometry()
    rect = OrthoBrick(Pnt(x_left, y_bottom, 0), Pnt(x_right, y_top, 0))
    geom.Add(rect)
    mesh = Mesh(geom.GenerateMesh(maxh=h))
    
    # Create finite element space with Dirichlet on all boundaries except left (interface)
    V = H1(mesh, order=1, dirichlet="bottom|top|right")
    
    # Define trial and test functions
    u, v = V.TnT()
    
    # Read flux values from input file
    flux_qn = []
    with open(input_flux, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                flux_qn.append(float(parts[2]))
    
    avg_flux = np.mean(flux_qn) if flux_qn else 0.0
    
    # Weak form: integral(K grad u . grad v) = integral(g v) on interface
    a = BilinearForm(V, symmetric=True)
    a += (K11*grad(u)[0]*grad(v)[0] + K12*grad(u)[1]*grad(v)[0] + 
          K21*grad(u)[0]*grad(v)[1] + K22*grad(u)[1]*grad(v)[1]) * dx
    
    f = LinearForm(V)
    # Add Neumann term on interface (left boundary of B)
    f += avg_flux * v * ds(left=True)
    
    a.Assemble()
    f.Assemble()
    
    gridfunc = GridFunction(V)
    inv_a = a.mat.Inverse()
    gridfunc.vec.data = inv_a * f.vec
    
    # Write log file
    ndof = V.ndof
    with open(output_log, 'w') as f:
        f.write(f"NDOF = {ndof}\n")
    
    # Generate probe points for subdomain B
    probe_points = []
    for i_y in range(44):
        for i_x in range(44):
            x = x_left + (i_x + 0.5) * (x_right - x_left) / 44.0
            y = y_bottom + (i_y + 0.5) * (y_top - y_bottom) / 44.0
            probe_points.append((x, y))
    
    # Interpolate solution at probe points
    with open(output_solution, 'w') as f:
        f.write("x,y,u\n")
        for (px, py) in probe_points:
            pt = Point(px, py)
            u_val = gridfunc(pt)
            f.write("{:.15e},{:.15e},{:.15e}\n".format(px, py, u_val))
    
    # Interface probe points
    interface_probe_points = []
    for i in range(44):
        y = 0.25 + (i + 0.5) * 0.5 / 44.0
        interface_probe_points.append((x_left, y))
    
    # Compute interface values and flux
    with open(output_interface, 'w') as f:
        f.write("x,y,u,qn\n")
        for (px, py) in interface_probe_points:
            pt = Point(px, py)
            u_val = gridfunc(pt)
            
            # Compute gradient at this point
            grad_u = grad(gridfunc)(pt)
            
            # Outward normal from subdomain B at x=0.625 is (-1, 0)
            n = Vector(-1.0, 0.0)
            
            # qn = -(K grad u) . n
            Kgrad_u_0 = K11 * grad_u[0] + K12 * grad_u[1]
            Kgrad_u_1 = K21 * grad_u[0] + K22 * grad_u[1]
            qn = -(Kgrad_u_0 * n[0] + Kgrad_u_1 * n[1])
            
            f.write("{:.15e},{:.15e},{:.15e},{:.15e}\n".format(px, py, u_val, qn))
    
    # Write interface values for coupling
    with open(output_values, 'w') as f:
        f.write("x,y,u\n")
        for (px, py) in interface_probe_points:
            pt = Point(px, py)
            u_val = gridfunc(pt)
            f.write("{:.15e},{:.15e},{:.15e}\n".format(px, py, u_val))

if __name__ == "__main__":
    main()
