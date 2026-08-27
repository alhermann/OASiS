#!/usr/bin/env python3
"""
Subdomain B solver using dolfinx.
Domain B = rectangle (0.5, 1) x (0, 0.5)
"""

import numpy as np
from dolfinx import mesh, fem
from mpi4py import MPI
from dolfinx.fem import functionspace, Function, dirichletbc
from dolfinx.mesh import locate_entities_boundary
from ufl import TrialFunction, TestFunction, inner, grad, dx, ds, Constant
import basix
from petsc4py import PETSc

def get_f(x, y):
    """Get source term f at point (x, y) for subdomain B region."""
    return (24*x**4*y/125 - 11*x**4/125 + 48*x**3*y**2/25 - 51*x**3*y/125 - 67*x**3/250 
            + 48*x**2*y**3/125 - 306*x**2*y**2/125 + 36*x**2*y/125 + 811*x**2/2000 
            + 24*x*y**4/25 - 51*x*y**3/125 - 471*x*y**2/250 + 657*x*y/1000 + 603*x/4000 
            - 8*y**4/25 + 9*y**3/250 + 2971*y**2/2000 - 2487*y/8000 - 801/4000)


class SubdomainB:
    def __init__(self, h):
        self.h = h
        self.n_elem_x = int(0.5 / h)
        self.n_elem_y = int(0.5 / h)
        self.mesh = None
        self.V = None
        self.u = None
        self.ndof = 0
        
    def create_mesh(self):
        """Create mesh for subdomain B: rectangle (0.5, 1) x (0, 0.5)."""
        nx = self.n_elem_x
        ny = self.n_elem_y
        
        domain = ((0.5, 1.0), (0.0, 0.5))
        self.mesh = mesh.create_rectangle(
            comm=MPI.COMM_WORLD,
            points=(np.array([0.5, 0.0]), np.array([1.0, 0.5])),
            n=(nx, ny),
            cell_type=mesh.CellType.triangle,
            diagonal=mesh.DiagonalType.right
        )
        
        return self.mesh
    
    def setup(self):
        """Set up FEM problem."""
        self.create_mesh()
        
        # Create function space
        element = basix.element("Lagrange", self.mesh.topology.cell_name(), 1)
        self.V = functionspace(self.mesh, element)
        
        tol = 1e-10
        
        def outer_boundary(x):
            """Outer boundary of subdomain B: x=1 or y=0."""
            return (np.abs(x[0] - 1.0) < tol) | (np.abs(x[1]) < tol)
        
        def interface_boundary(x):
            """Interface boundary of subdomain B: x=0.5 or y=0.5."""
            return (np.abs(x[0] - 0.5) < tol) | (np.abs(x[1] - 0.5) < tol)
        
        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, fdim)
        
        outer_facets = locate_entities_boundary(self.mesh, fdim, outer_boundary)
        interface_facets = locate_entities_boundary(self.mesh, fdim, interface_boundary)
        
        # Dirichlet BC on outer boundary
        u_zero = Function(self.V)
        u_zero.x.array[:] = 0.0
        self.bc = dirichletbc(u_zero, outer_facets, self.V)
        
        self.interface_facets = interface_facets
        
        # Find interface nodes for flux computation
        interface_nodes = []
        x_coords = self.mesh.geometry.x
        for i, (xi, yi) in enumerate(x_coords):
            if abs(xi - 0.5) < tol or abs(yi - 0.5) < tol:
                interface_nodes.append(i)
        self.interface_nodes_global = np.array(interface_nodes)
        
        # Assemble stiffness matrix
        u_trial = TrialFunction(self.V)
        v_test = TestFunction(self.V)
        k = Constant(self.mesh, 2.5)
        
        a_form = inner(k * grad(u_trial), grad(v_test)) * dx
        self.a_mat = fem.petsc.assemble_matrix(fem.Form(a_form), bcs=[self.bc])
        self.a_mat.assemble()
        
        # Source term
        W = fem.functionspace(self.mesh, ("CG", 1))
        self.f_func = Function(W)
        self.f_func.interpolate(lambda x: np.array([get_f(x[0,i], x[1,i]) for i in range(x.shape[1])] ))
        
        L_form = self.f_func * v_test * dx
        self.L_form = fem.Form(L_form)
        
        self.ndof = self.V.dofmap.index_map.size_local
        
        return self.ndof
    
    def solve_neumann(self, qn_interface, interface_node_order):
        """Solve with Neumann BC on interface."""
        from dolfinx.fem import assemble_vector
        from petsc4py.PETSc import InsertMode, ScatterMode
        
        # Assemble RHS
        L_vec = fem.petsc.assemble_vector(self.L_form)
        
        # Add Neumann contribution
        dofs_on_interface = []
        for node_idx in self.interface_nodes_global:
            dofs = self.V.dofmap.list[node_idx]
            dofs_on_interface.extend(dofs)
        dofs_on_interface = list(set(dofs_on_interface))
        
        for i, dof in enumerate(dofs_on_interface):
            if i < len(qn_interface):
                L_vec.array[dof] += qn_interface[i] * self.h
        
        fem.petsc.apply_lifting(L_vec, [fem.Form(inner(Constant(self.mesh, 1.0)*grad(TestFunction(self.V)), Constant(self.mesh, 0.0))*dx)], [self.bc])
        L_vec.ghostUpdate(addv=InsertMode.ADD_VALUES, mode=ScatterMode.REVERSE)
        fem.set_bc(L_vec, [self.bc])
        
        # Solve
        self.u = Function(self.V)
        KSP = PETSc.KSP().create(self.mesh.comm)
        KSP.setOperators(self.a_mat)
        KSP.setType("preonly")
        KSP.getPC().setType("lu")
        KSP.getPC().setFactorSolverType("mumps")
        
        KSP.solve(L_vec, self.u.x.vec)
        
        return self.u.x.numpy()
    
    def get_interface_values(self):
        """Extract solution values at interface nodes."""
        values = []
        x_coords = self.mesh.geometry.x
        tol = 1e-10
        
        vertical_nodes = []
        horizontal_nodes = []
        
        for i, (xi, yi) in enumerate(x_coords):
            if abs(xi - 0.5) < tol and yi < 0.5 - tol:
                vertical_nodes.append((i, xi, yi))
            elif abs(yi - 0.5) < tol and xi > 0.5 + tol:
                horizontal_nodes.append((i, xi, yi))
        
        vertical_nodes.sort(key=lambda x: x[2])
        horizontal_nodes.sort(key=lambda x: x[1])
        
        for node_idx, xi, yi in vertical_nodes:
            dofs = self.V.dofmap.list[node_idx]
            if len(dofs) > 0:
                values.append((xi, yi, self.u.x.array[dofs[0]]))
        
        for node_idx, xi, yi in horizontal_nodes:
            dofs = self.V.dofmap.list[node_idx]
            if len(dofs) > 0:
                values.append((xi, yi, self.u.x.array[dofs[0]]))
        
        return values
    
    def compute_flux(self):
        """Compute outward normal flux on interface from subdomain B."""
        from dolfinx.fem import Expression
        
        grad_u = fem.Expression(grad(self.u), self.V.element.interpolation_points())
        
        flux = []
        x_coords = self.mesh.geometry.x
        tol = 1e-10
        
        vertical_nodes = []
        horizontal_nodes = []
        
        for i, (xi, yi) in enumerate(x_coords):
            if abs(xi - 0.5) < tol and yi < 0.5 - tol:
                vertical_nodes.append((i, xi, yi))
            elif abs(yi - 0.5) < tol and xi > 0.5 + tol:
                horizontal_nodes.append((i, xi, yi))
        
        vertical_nodes.sort(key=lambda x: x[2])
        horizontal_nodes.sort(key=lambda x: x[1])
        
        k_val = 2.5
        
        for node_idx, xi, yi in vertical_nodes:
            n = np.array([1.0, 0.0])
            cells = self.mesh.topology.connectivity[self.mesh.topology.dim, 0].transpose()[node_idx]
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
        
        for node_idx, xi, yi in horizontal_nodes:
            n = np.array([0.0, 1.0])
            cells = self.mesh.topology.connectivity[self.mesh.topology.dim, 0].transpose()[node_idx]
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
        x_coords = self.mesh.geometry.x
        min_dist = float('inf')
        closest_val = 0.0
        
        for i, (xb, yb) in enumerate(x_coords):
            dist = (xb - x)**2 + (yb - y)**2
            if dist < min_dist:
                min_dist = dist
                dofs = self.V.dofmap.list[i]
                if len(dofs) > 0:
                    closest_val = self.u.x.array[dofs[0]]
        
        return closest_val


if __name__ == "__main__":
    # Test
    h = 1/8
    solver = SubdomainB(h)
    ndof = solver.setup()
    print(f"Subdomain B: {ndof} DOFs")
    print(f"Interface facets: {len(solver.interface_facets)}")
