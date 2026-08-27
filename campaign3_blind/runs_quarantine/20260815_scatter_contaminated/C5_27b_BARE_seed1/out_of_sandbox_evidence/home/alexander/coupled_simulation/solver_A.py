#!/usr/bin/env python3
"""
Subdomain A solver using scikit-fem.
Domain A = unit square minus (0.5, 1) x (0, 0.5) minus notch (0.75, 1) x (0.75, 1)
"""

import numpy as np
from skfem import MeshTri, asm, CellBasis, condense, ElementTriP1, LinearForm
from skfem.models.poisson import laplace
import sys

def get_k(x, y):
    """Get conductivity k at point (x, y)."""
    if x <= 0.5 and y <= 0.5:
        return 1.0
    elif x > 0.5 and y <= 0.5:
        return 2.5
    elif x > 0.5 and y > 0.5:
        return 5.0
    else:
        return 2.0

def get_f(x, y):
    """Get source term f at point (x, y)."""
    if x <= 0.5 and y <= 0.5:
        return (3*x**4*y - 11*x**4/8 + 12*x**3*y**2 - 123*x**3*y/20 - x**3/40 
                + 6*x**2*y**3 - 477*x**2*y**2/20 + 2799*x**2*y/400 + 1423*x**2/800 
                + 6*x*y**4 - 123*x*y**3/20 + 993*x*y**2/200 + 63*x*y/1600 - 609*x/800 
                - 13*y**4/5 + 279*y**3/200 + 1423*y**2/800 - 327*y/320)
    elif x <= 0.5 and y > 0.5:
        return (3*x**4*y/4 - 5*x**4/16 + 3*x**3*y**2/2 - 3*x**3*y/80 - 13*x**3/32 
                + 3*x**2*y**3/2 - 153*x**2*y**2/40 - 873*x**2*y/800 + 119*x**2/80 
                + 3*x*y**4/4 - 3*x*y**3/80 - 471*x*y**2/800 + 9*x*y/16 - 3*x/800 
                - 13*y**4/40 - 241*y**3/800 + 37*y**2/40 - 107*y/3200 - 849/3200)
    elif x > 0.5 and y <= 0.5:
        return (24*x**4*y/125 - 11*x**4/125 + 48*x**3*y**2/25 - 51*x**3*y/125 - 67*x**3/250 
                + 48*x**2*y**3/125 - 306*x**2*y**2/125 + 36*x**2*y/125 + 811*x**2/2000 
                + 24*x*y**4/25 - 51*x*y**3/125 - 471*x*y**2/250 + 657*x*y/1000 + 603*x/4000 
                - 8*y**4/25 + 9*y**3/250 + 2971*y**2/2000 - 2487*y/8000 - 801/4000)
    else:
        return (6*x**4*y/125 - x**4/50 + 6*x**3*y**2/25 + 69*x**3*y/500 - x**3/8 
                + 12*x**2*y**3/125 - 9*x**2*y**2/25 - 9*x**2*y/40 + 769*x**2/4000 
                + 3*x*y**4/25 + 69*x*y**3/500 - 51*x*y**2/100 - 549*x*y/8000 + 2853*x/16000 
                - y**4/25 - 71*y**3/1000 + 233*y**2/800 + 419*y/4000 - 2031/16000)


class SubdomainA:
    def __init__(self, h):
        self.h = h
        self.n_elem = int(1/h)
        self.mesh = None
        self.basis = None
        self.interface_nodes = []
        self.outer_boundary_mask = None
        self.ndof = 0
        self.u = None
        
    def create_mesh(self):
        """Create triangular mesh for subdomain A."""
        nx = self.n_elem
        ny = self.n_elem
        h = self.h
        
        # Build node dictionary
        nodes_dict = {}
        
        # Left rectangle [0, 0.5] x [0, 1]
        for j in range(ny + 1):
            for i in range(nx // 2 + 1):
                x = i * h
                y = j * h
                key = (round(x, 12), round(y, 12))
                if key not in nodes_dict:
                    nodes_dict[key] = len(nodes_dict)
        
        # Top-right rectangle [0.5, 1] x [0.5, 1], excluding notch
        for j in range(ny // 2, ny + 1):
            for i in range(nx // 2, nx + 1):
                x = i * h
                y = j * h
                # Skip notch region (x > 0.75 and y > 0.75)
                if x > 0.75 and y > 0.75:
                    continue
                key = (round(x, 12), round(y, 12))
                if key not in nodes_dict:
                    nodes_dict[key] = len(nodes_dict)
        
        self.all_nodes = nodes_dict
        n_nodes = len(nodes_dict)
        
        # Create node array
        p = np.zeros((2, n_nodes))
        for (x, y), idx in nodes_dict.items():
            p[0, idx] = x
            p[1, idx] = y
        
        # Create elements
        elements = []
        
        # Left rectangle elements
        for j in range(ny):
            for i in range(nx // 2):
                p00 = (i * h, j * h)
                p10 = ((i + 1) * h, j * h)
                p01 = (i * h, (j + 1) * h)
                p11 = ((i + 1) * h, (j + 1) * h)
                
                k00 = (round(p00[0], 12), round(p00[1], 12))
                k10 = (round(p10[0], 12), round(p10[1], 12))
                k01 = (round(p01[0], 12), round(p01[1], 12))
                k11 = (round(p11[0], 12), round(p11[1], 12))
                
                if all(k in nodes_dict for k in [k00, k10, k01, k11]):
                    n00 = nodes_dict[k00]
                    n10 = nodes_dict[k10]
                    n01 = nodes_dict[k01]
                    n11 = nodes_dict[k11]
                    # Two triangles per quad
                    elements.append([n00, n10, n01])
                    elements.append([n10, n11, n01])
        
        # Top-right rectangle elements
        for j in range(ny // 2, ny):
            for i in range(nx // 2, nx):
                x_min = i * h
                x_max = (i + 1) * h
                y_min = j * h
                y_max = (j + 1) * h
                
                # Skip if entirely in notch
                if x_min > 0.75 and y_min > 0.75:
                    continue
                
                p00 = (x_min, y_min)
                p10 = (x_max, y_min)
                p01 = (x_min, y_max)
                p11 = (x_max, y_max)
                
                k00 = (round(p00[0], 12), round(p00[1], 12))
                k10 = (round(p10[0], 12), round(p10[1], 12))
                k01 = (round(p01[0], 12), round(p01[1], 12))
                k11 = (round(p11[0], 12), round(p11[1], 12))
                
                # Check which corners exist (not in notch)
                valid_corners = [k in nodes_dict for k in [k00, k10, k01, k11]]
                
                if sum(valid_corners) >= 3:
                    n00 = nodes_dict.get(k00)
                    n10 = nodes_dict.get(k10)
                    n01 = nodes_dict.get(k01)
                    n11 = nodes_dict.get(k11)
                    
                    # Handle partial quads near notch
                    if valid_corners == [True, True, True, False]:
                        elements.append([n00, n10, n01])
                    elif valid_corners == [True, False, True, True]:
                        elements.append([n00, n11, n01])
                    elif valid_corners == [False, True, True, True]:
                        elements.append([n10, n11, n01])
                    elif valid_corners == [True, True, False, True]:
                        elements.append([n00, n10, n11])
                    else:
                        elements.append([n00, n10, n01])
                        elements.append([n10, n11, n01])
        
        t = np.array(elements).T
        
        self.mesh = MeshTri(p, t)
        return self.mesh
    
    def setup(self):
        """Set up FEM problem."""
        self.create_mesh()
        
        basis = CellBasis(self.mesh, ElementTriP1())
        self.basis = basis
        
        tol = 1e-10
        
        # Identify outer boundary nodes
        outer_boundary = np.zeros(len(self.mesh.p[0]), dtype=bool)
        interface_nodes = []
        
        for idx, (x, y) in enumerate(self.mesh.p.T):
            # Outer boundary of unit square
            on_outer = (abs(x) < tol or abs(x - 1) < tol or 
                       abs(y) < tol or abs(y - 1) < tol)
            
            # Notch boundaries
            on_notch = (x > 0.75 - tol and y > 0.75 - tol and 
                       (abs(x - 0.75) < tol or abs(y - 0.75) < tol))
            
            if on_outer or on_notch:
                outer_boundary[idx] = True
            # Interface: x = 0.5 (y < 0.5) or y = 0.5 (x > 0.5)
            elif (abs(x - 0.5) < tol and y < 0.5 - tol) or \
                 (abs(y - 0.5) < tol and x > 0.5 + tol):
                interface_nodes.append(idx)
        
        self.outer_boundary_mask = outer_boundary
        self.interface_nodes = np.array(interface_nodes)
        
        # Get cell centers for coefficient evaluation
        cell_centers = self.mesh.p[:, self.mesh.t].mean(axis=1)
        
        # Define subdomains based on material regions
        elem_markers = {}
        for i, cc in enumerate(cell_centers.T):
            xc, yc = cc[0], cc[1]
            if xc <= 0.5 and yc <= 0.5:
                elem_markers.setdefault('k1', []).append(i)
            elif xc > 0.5 and yc <= 0.5:
                # This region is in subdomain B, not A
                pass
            elif xc > 0.5 and yc > 0.5:
                elem_markers.setdefault('k5', []).append(i)
            else:  # xc <= 0.5 and yc > 0.5
                elem_markers.setdefault('k2', []).append(i)
        
        for k in elem_markers:
            elem_markers[k] = np.array(elem_markers[k])
        
        m_with_subdomains = self.mesh.with_subdomains(elem_markers)
        
        # Assemble Laplacian on each subdomain with appropriate k value
        K_total = None
        F_total = None
        
        k_values = {'k1': 1.0, 'k2': 2.0, 'k5': 5.0}
        
        @LinearForm
        def source(v, w):
            return v
        
        for region, k_val in k_values.items():
            if region in m_with_subdomains.subdomains:
                basis_region = CellBasis(m_with_subdomains, ElementTriP1(), 
                                         elements=m_with_subdomains.subdomains[region])
                K_region = asm(laplace, basis_region)
                F_region = asm(source, basis_region)
                
                if K_total is None:
                    K_total = k_val * K_region
                    F_total = F_region
                else:
                    K_total = K_total + k_val * K_region
                    F_total = F_total + F_region
        
        self.K_total = K_total
        self.F_total = F_total
        
        # Condense outer boundary
        K_red, F_red, _, _ = condense(K_total, F_total, D=outer_boundary)
        
        self.A_reduced = K_red
        self.F_reduced = F_red
        
        # Map from reduced to full DOFs
        free_dofs = ~outer_boundary
        self.free_indices = np.where(free_dofs)[0]
        
        # Map interface nodes to reduced indices
        self.interface_reduced = []
        for node in self.interface_nodes:
            if node in self.free_indices:
                idx = np.where(self.free_indices == node)[0][0]
                self.interface_reduced.append(idx)
        self.interface_reduced = np.array(self.interface_reduced)
        
        self.ndof = self.A_reduced.shape[0]
        
        return self.ndof
    
    def solve_dirichlet(self, u_interface):
        """Solve with Dirichlet BC on interface."""
        A = self.A_reduced.copy()
        F = self.F_reduced.copy()
        
        # Impose Dirichlet on interface
        for i, dof in enumerate(self.interface_reduced):
            val = u_interface[i]
            for j in range(A.shape[1]):
                if j != dof:
                    F[j] -= A[j, dof] * val
            A[dof, :] = 0
            A[:, dof] = 0
            A[dof, dof] = 1
            F[dof] = val
        
        u_reduced = np.linalg.solve(A, F)
        
        # Expand to full solution
        u_full = np.zeros(len(self.mesh.p[0]))
        u_full[self.free_indices] = u_reduced
        
        return u_full
    
    def compute_flux(self, u):
        """Compute outward normal flux on interface."""
        from skfem.helpers import grad
        
        basis = self.basis
        u_vec = basis.interpolate(u)
        grad_u = grad(basis)(u_vec)
        
        flux = []
        tol = 1e-10
        
        for node_idx in self.interface_nodes:
            x, y = self.mesh.p[:, node_idx]
            
            # Determine which leg and get k value
            if abs(x - 0.5) < tol and y < 0.5:
                # Vertical leg: outward normal from A is (-1, 0)
                k_val = 1.0
                n = np.array([-1.0, 0.0])
            else:
                # Horizontal leg: outward normal from A is (0, -1)
                k_val = 2.0
                n = np.array([0.0, -1.0])
            
            # Get gradient at this node (average over adjacent cells)
            cells = self.mesh.find_nodes(x, y)
            g_avg = np.zeros(2)
            count = 0
            for cell_idx in cells:
                if cell_idx < len(grad_u):
                    g_avg += grad_u[cell_idx]
                    count += 1
            
            if count > 0:
                g_avg /= count
            
            qn = -k_val * np.dot(g_avg, n)
            flux.append(qn)
        
        return np.array(flux)
    
    def interpolate_at_point(self, x, y):
        """Interpolate solution at arbitrary point."""
        try:
            cell_idx = self.mesh.point_locator.locate(np.array([[x], [y]]))[0]
            if cell_idx >= 0:
                cell_nodes = self.mesh.t[:, cell_idx]
                # Simple barycentric interpolation
                return np.mean(self.u[cell_nodes])
        except:
            pass
        
        # Nearest neighbor fallback
        min_dist = float('inf')
        closest_val = 0.0
        for i, (xi, yi) in enumerate(self.mesh.p.T):
            dist = (xi - x)**2 + (yi - y)**2
            if dist < min_dist:
                min_dist = dist
                closest_val = self.u[i]
        
        return closest_val


if __name__ == "__main__":
    # Test
    h = 1/8
    solver = SubdomainA(h)
    ndof = solver.setup()
    print(f"Subdomain A: {ndof} DOFs")
    print(f"Interface nodes: {len(solver.interface_nodes)}")
