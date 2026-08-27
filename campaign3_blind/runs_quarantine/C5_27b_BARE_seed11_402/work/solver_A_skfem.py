#!/usr/bin/env python3
"""
Subdomain A solver using scikit-fem
Domain: L-shaped domain minus (0.5,1)x(0,0.5)
"""

import numpy as np
from skfem import *
from skfem.models.poisson import laplace
import sys
import pickle
import csv

# Material coefficient on subdomain A - vectorized version
def k_A_vec(x, y):
    """Vectorized conductivity function."""
    result = np.zeros_like(x)
    mask1 = (x < 0.5) & (y < 0.5)
    result[mask1] = 1.0
    mask2 = (x < 0.5) & (y >= 0.5)
    result[mask2] = 2.0
    mask3 = (x >= 0.5) & (y >= 0.5)
    result[mask3] = 5.0
    return result

# Source term on subdomain A - vectorized version
def f_A_vec(x, y):
    """Vectorized source term."""
    result = np.zeros_like(x)
    
    mask1 = (x < 0.5) & (y < 0.5)
    result[mask1] = (3*x[mask1]**4*y[mask1] - 11*x[mask1]**4/8 + 12*x[mask1]**3*y[mask1]**2 
                     - 123*x[mask1]**3*y[mask1]/20 - x[mask1]**3/40 
                     + 6*x[mask1]**2*y[mask1]**3 - 477*x[mask1]**2*y[mask1]**2/20 
                     + 2799*x[mask1]**2*y[mask1]/400 + 1423*x[mask1]**2/800 
                     + 6*x[mask1]*y[mask1]**4 - 123*x[mask1]*y[mask1]**3/20 
                     + 993*x[mask1]*y[mask1]**2/200 + 63*x[mask1]*y[mask1]/1600 
                     - 609*x[mask1]/800 - 13*y[mask1]**4/5 + 279*y[mask1]**3/200 
                     + 1423*y[mask1]**2/800 - 327*y[mask1]/320)
    
    mask2 = (x < 0.5) & (y >= 0.5)
    result[mask2] = (3*x[mask2]**4*y[mask2]/4 - 5*x[mask2]**4/16 
                     + 3*x[mask2]**3*y[mask2]**2/2 - 3*x[mask2]**3*y[mask2]/80 
                     - 13*x[mask2]**3/32 + 3*x[mask2]**2*y[mask2]**3/2 
                     - 153*x[mask2]**2*y[mask2]**2/40 - 873*x[mask2]**2*y[mask2]/800 
                     + 119*x[mask2]**2/80 + 3*x[mask2]*y[mask2]**4/4 
                     - 3*x[mask2]*y[mask2]**3/80 - 471*x[mask2]*y[mask2]**2/800 
                     + 9*x[mask2]*y[mask2]/16 - 3*x[mask2]/800 - 13*y[mask2]**4/40 
                     - 241*y[mask2]**3/800 + 37*y[mask2]**2/40 - 107*y[mask2]/3200 
                     - 849/3200)
    
    mask3 = (x >= 0.5) & (y >= 0.5)
    result[mask3] = (6*x[mask3]**4*y[mask3]/125 - x[mask3]**4/50 
                     + 6*x[mask3]**3*y[mask3]**2/25 + 69*x[mask3]**3*y[mask3]/500 
                     - x[mask3]**3/8 + 12*x[mask3]**2*y[mask3]**3/125 
                     - 9*x[mask3]**2*y[mask3]**2/25 - 9*x[mask3]**2*y[mask3]/40 
                     + 769*x[mask3]**2/4000 + 3*x[mask3]*y[mask3]**4/25 
                     + 69*x[mask3]*y[mask3]**3/500 - 51*x[mask3]*y[mask3]**2/100 
                     - 549*x[mask3]*y[mask3]/8000 + 2853*x[mask3]/16000 
                     - y[mask3]**4/25 - 71*y[mask3]**3/1000 + 233*y[mask3]**2/800 
                     + 419*y[mask3]/4000 - 2031/16000)
    
    return result

class SubdomainA:
    def __init__(self, h_inv):
        self.h_inv = h_inv
        self._build_mesh()
    
    def _build_mesh(self):
        h_inv = self.h_inv
        
        n_refinements = int(np.log2(h_inv))
        
        # Part 1: Left rectangle (0, 0.5) x (0, 1)
        mesh1 = MeshTri().refined(n_refinements)
        mesh1.p[0] *= 0.5
        
        # Part 2: Top-right L-shape
        mesh2a = MeshTri().refined(n_refinements)
        mesh2a.p[0] *= 0.25
        mesh2a.p[1] *= 0.5
        mesh2a.p[0] += 0.5
        mesh2a.p[1] += 0.5
        
        mesh2b = MeshTri().refined(n_refinements)
        mesh2b.p[0] *= 0.25
        mesh2b.p[1] *= 0.25
        mesh2b.p[0] += 0.75
        mesh2b.p[1] += 0.5
        
        self.mesh = mesh1 + mesh2a + mesh2b
        
        element = ElementTriP1()
        self.basis = Basis(self.mesh, element, intorder=2)
        
        self.edge_midpoints = self.mesh.p[:, self.mesh.edges].mean(axis=1)
        self.interface_edges = self._find_interface_edges()
        self.outer_boundary_nodes = self._find_outer_boundary_nodes()
        self.interface_nodes = self._find_interface_nodes()
        
        D_outer = self.basis.get_dofs(self.outer_boundary_nodes).flatten()
        self.D_outer = D_outer
        D_interface = self.basis.get_dofs(self.interface_nodes).flatten()
        self.D_interface = D_interface
        
        total_dofs = len(self.basis.dofs.nodal_dofs.flatten())
        self.ndof = total_dofs - len(D_outer) - len(D_interface)
        print(f"Subdomain A: ndof = {self.ndof}, total dofs = {total_dofs}")
        print(f"Mesh has {self.mesh.nelements} elements")
    
    def _is_outer_boundary_node(self, x, y):
        eps = 1e-10
        if abs(x) < eps:
            return True
        if abs(y) < eps and x < 0.5 - eps:
            return True
        if abs(y - 1.0) < eps:
            return True
        if abs(x - 1.0) < eps and y > 0.5 + eps:
            return True
        if abs(x - 0.75) < eps and y > 0.75 + eps:
            return True
        if abs(y - 0.75) < eps and x > 0.75 + eps:
            return True
        return False
    
    def _is_interface_node(self, x, y):
        eps = 1e-10
        if abs(x - 0.5) < eps and y > eps and y < 0.5 - eps:
            return True
        if abs(y - 0.5) < eps and x > 0.5 + eps and x < 1.0 - eps:
            return True
        return False
    
    def _find_interface_edges(self):
        interface_mask = []
        for i, ec in enumerate(self.edge_midpoints.T):
            eps = 1e-10
            if abs(ec[0] - 0.5) < eps and ec[1] > eps and ec[1] < 0.5 - eps:
                interface_mask.append(True)
            elif abs(ec[1] - 0.5) < eps and ec[0] > 0.5 + eps and ec[0] < 1.0 - eps:
                interface_mask.append(True)
            else:
                interface_mask.append(False)
        return np.where(interface_mask)[0]
    
    def _find_outer_boundary_nodes(self):
        nodes = []
        for i in range(self.mesh.nvertices):
            x, y = self.mesh.p[0, i], self.mesh.p[1, i]
            if self._is_outer_boundary_node(x, y):
                nodes.append(i)
        return np.array(nodes)
    
    def _find_interface_nodes(self):
        nodes = []
        for i in range(self.mesh.nvertices):
            x, y = self.mesh.p[0, i], self.mesh.p[1, i]
            if self._is_interface_node(x, y):
                nodes.append(i)
        return np.array(nodes)
    
    def solve(self, interface_u=None):
        # Use InteriorBasis for variable coefficient assembly
        int_basis = InteriorBasis(self.mesh, self.basis.elem, intorder=self.basis.intorder)
        
        # Assemble stiffness matrix with variable k
        K = asm(lambda x, y, z: k_A_vec(x, y) * laplace(x, y, z), int_basis)
        F = asm(lambda x, y, z: f_A_vec(x, y), int_basis)
        
        I = self.basis.complement(self.D_outer)
        if interface_u is not None:
            I = I.intersection(self.basis.complement(self.D_interface))
        
        A = project_matrix(K, I, I)
        b = project_vector(F, I)
        
        if interface_u is not None:
            u_interface_dofs = self.interpolate_interface_u(interface_u)
            b -= K[I, :][:, self.D_interface] @ u_interface_dofs
        
        u_I = solve(A.toarray(), b.toarray())
        
        u = np.zeros(len(self.basis.dofs.nodal_dofs.flatten()))
        u[I] = u_I
        if interface_u is not None:
            u[self.D_interface] = self.interpolate_interface_u(interface_u)
        
        return u
    
    def interpolate_interface_u(self, interface_data):
        u_at_nodes = np.zeros(len(self.interface_nodes))
        for i, node_idx in enumerate(self.interface_nodes):
            x, y = self.mesh.p[0, node_idx], self.mesh.p[1, node_idx]
            min_dist = float('inf')
            closest_u = 0.0
            for (ix, iy, iu) in interface_data:
                dist = (x - ix)**2 + (y - iy)**2
                if dist < min_dist:
                    min_dist = dist
                    closest_u = iu
            u_at_nodes[i] = closest_u
        return u_at_nodes
    
    def evaluate_at_points(self, u, points):
        results = []
        interp_func = self.basis.interpolate(u)
        for x, y in points:
            val = interp_func(x, y)
            results.append((x, y, val))
        return results


def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    h_inv = 8 * level
    
    print(f"=== Subdomain A, Level {level}, h_inv = {h_inv} ===")
    
    solver = SubdomainA(h_inv)
    
    with open(f"run_level{level}_A.log", "w") as f:
        f.write(f"NDOF = {solver.ndof}\n")
    
    u = solver.solve()
    print(f"Solution computed, max |u| = {np.max(np.abs(u))}")
    
    print("Done")


if __name__ == "__main__":
    main()
