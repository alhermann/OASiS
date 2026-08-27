#!/usr/bin/env python3
"""
Subdomain A solver using scikit-fem
Domain: L-shaped domain minus (1/2,1)x(0,1/2)
This is everything except the rectangle B
"""

import numpy as np
from skfem import *
from skfem.models.poisson import laplace, mass
import os
import pickle

# Material coefficient on subdomain A
def k_A(x, y):
    if x < 0.5 and y < 0.5:
        return 1.0
    elif x < 0.5 and y >= 0.5:
        return 2.0
    elif x >= 0.5 and y >= 0.5:
        return 5.0
    else:
        return 0.0

# Source term on subdomain A
def f_A(x, y):
    if x < 0.5 and y < 0.5:
        return (3*x**4*y - 11*x**4/8 + 12*x**3*y**2 - 123*x**3*y/20 - x**3/40 
                + 6*x**2*y**3 - 477*x**2*y**2/20 + 2799*x**2*y/400 + 1423*x**2/800 
                + 6*x*y**4 - 123*x*y**3/20 + 993*x*y**2/200 + 63*x*y/1600 - 609*x/800 
                - 13*y**4/5 + 279*y**3/200 + 1423*y**2/800 - 327*y/320)
    elif x < 0.5 and y >= 0.5:
        return (3*x**4*y/4 - 5*x**4/16 + 3*x**3*y**2/2 - 3*x**3*y/80 - 13*x**3/32 
                + 3*x**2*y**3/2 - 153*x**2*y**2/40 - 873*x**2*y/800 + 119*x**2/80 
                + 3*x*y**4/4 - 3*x*y**3/80 - 471*x*y**2/800 + 9*x*y/16 - 3*x/800 
                - 13*y**4/40 - 241*y**3/800 + 37*y**2/40 - 107*y/3200 - 849/3200)
    elif x >= 0.5 and y >= 0.5:
        return (6*x**4*y/125 - x**4/50 + 6*x**3*y**2/25 + 69*x**3*y/500 - x**3/8 
                + 12*x**2*y**3/125 - 9*x**2*y**2/25 - 9*x**2*y/40 + 769*x**2/4000 
                + 3*x*y**4/25 + 69*x*y**3/500 - 51*x*y**2/100 - 549*x*y/8000 + 2853*x/16000 
                - y**4/25 - 71*y**3/1000 + 233*y**2/800 + 419*y/4000 - 2031/16000)
    else:
        return 0.0

class SubdomainA:
    def __init__(self, h_inv):
        self.h_inv = h_inv
        self.mesh = None
        self.basis = None
        self.element_basis = None
        self.ndof = 0
        
        self._build_mesh()
    
    def _build_mesh(self):
        h_inv = self.h_inv
        
        # Part 1: Left rectangle (0, 0.5) x (0, 1)
        nx1 = h_inv // 2
        ny1 = h_inv
        mesh1 = MeshTri().refined(nx1).refined(ny1)
        mesh1.p[0] *= 0.5
        
        # Part 2: Top-right L-shape
        nx2a = h_inv // 4
        ny2a = h_inv // 2
        mesh2a = MeshTri().refined(nx2a).refined(ny2a)
        mesh2a.p[0] *= 0.25
        mesh2a.p[1] *= 0.5
        mesh2a.p[0] += 0.5
        mesh2a.p[1] += 0.5
        
        nx2b = h_inv // 4
        ny2b = h_inv // 4
        mesh2b = MeshTri().refined(nx2b).refined(ny2b)
        mesh2b.p[0] *= 0.25
        mesh2b.p[1] *= 0.25
        mesh2b.p[0] += 0.75
        mesh2b.p[1] += 0.5
        
        self.mesh = mesh1 + mesh2a + mesh2b
        
        element = ElementTriP1()
        self.basis = Basis(self.mesh, element, intorder=2)
        self.element_basis = InteriorBasis(self.mesh, element, intorder=2)
        
        centroids = self.mesh.p[:, self.mesh.t].mean(axis=1)
        self.k_vals = np.array([k_A(cx, cy) for cx, cy in zip(centroids[0], centroids[1])])
        self.f_vals = np.array([f_A(cx, cy) for cx, cy in zip(centroids[0], centroids[1])])
        
        self.interface_edges = self._find_interface_edges()
        self.outer_boundary_edges = self._find_outer_boundary_edges()
        
        D_outer = self.basis.find_dofs(lambda x: self._is_outer_boundary(x))
        self.D_outer = D_outer
        self.D_interface = self.basis.find_dofs(lambda x: self._is_interface(x))
        
        self.ndof = len(self.basis.dofs.flat) - len(D_outer) - len(self.D_interface)
        print(f"Subdomain A: ndof = {self.ndof}, total dofs = {len(self.basis.dofs.flat)}")
    
    def _is_outer_boundary(self, x):
        eps = 1e-10
        if abs(x[0]) < eps:
            return True
        if abs(x[1]) < eps and x[0] < 0.5 - eps:
            return True
        if abs(x[1] - 1.0) < eps:
            return True
        if abs(x[0] - 1.0) < eps and x[1] > 0.5 + eps:
            return True
        if abs(x[0] - 0.75) < eps and x[1] > 0.75 + eps:
            return True
        if abs(x[1] - 0.75) < eps and x[0] > 0.75 + eps:
            return True
        return False
    
    def _is_interface(self, x):
        eps = 1e-10
        if abs(x[0] - 0.5) < eps and x[1] > eps and x[1] < 0.5 - eps:
            return True
        if abs(x[1] - 0.5) < eps and x[0] > 0.5 + eps and x[0] < 1.0 - eps:
            return True
        return False
    
    def _find_interface_edges(self):
        # Compute edge midpoints manually
        edge_midpoints = self.mesh.p[:, self.mesh.edges].mean(axis=1)
        interface_mask = []
        for i, ec in enumerate(edge_midpoints.T):
            eps = 1e-10
            if abs(ec[0] - 0.5) < eps and ec[1] > eps and ec[1] < 0.5 - eps:
                interface_mask.append(True)
            elif abs(ec[1] - 0.5) < eps and ec[0] > 0.5 + eps and ec[0] < 1.0 - eps:
                interface_mask.append(True)
            else:
                interface_mask.append(False)
        return np.where(interface_mask)[0]
    
    def _find_outer_boundary_edges(self):
        edge_midpoints = self.mesh.p[:, self.mesh.edges].mean(axis=1)
        outer_mask = [self._is_outer_boundary(ec) for ec in edge_midpoints.T]
        return np.where(outer_mask)[0]
    
    def solve(self, interface_u=None):
        K = asm(lambda x, y, z: k_A(x, y) * laplace(x, y, z), self.basis)
        F = asm(lambda x, y, z: f_A(x, y), self.basis)
        
        I = self.basis.complement(self.D_outer)
        if interface_u is not None:
            I = I.intersection(self.basis.complement(self.D_interface))
            interface_dofs = self.D_interface
            u_interface_dofs = self.interpolate_interface_u(interface_u)
        
        A = project_matrix(K, I, I)
        b = project_vector(F, I)
        
        if interface_u is not None:
            interface_dofs = self.D_interface
            u_interface_dofs = self.interpolate_interface_u(interface_u)
            b -= K[I, :][:, interface_dofs] @ u_interface_dofs
        
        u_I = solve(A.toarray(), b.toarray())
        
        u = np.zeros(len(self.basis.dofs.flat))
        u[I] = u_I
        if interface_u is not None:
            u[self.D_interface] = self.interpolate_interface_u(interface_u)
        
        return u
    
    def interpolate_interface_u(self, interface_data):
        interface_nodes = self.basis.get_dofs(self.D_interface).flatten()
        node_coords = self.mesh.p[:, interface_nodes]
        
        u_at_nodes = np.zeros(len(interface_nodes))
        for i, node_idx in enumerate(interface_nodes):
            x, y = node_coords[0, i], node_coords[1, i]
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
        for x, y in points:
            val = self.basis.interpolate(u)(x, y)
            results.append((x, y, val))
        return results


def main():
    print("Testing Subdomain A solver...")
    
    for h_inv in [8, 16, 32]:
        print(f"\n=== Mesh level h_inv = {h_inv} ===")
        solver = SubdomainA(h_inv)
        print(f"Number of elements: {solver.mesh.ne}")
        print(f"Number of vertices: {solver.mesh.nv}")
        print(f"Interface edges: {len(solver.interface_edges)}")
        print(f"Outer boundary edges: {len(solver.outer_boundary_edges)}")
        
        u = solver.solve()
        print(f"Solution computed, max |u| = {np.max(np.abs(u))}")


if __name__ == "__main__":
    main()
