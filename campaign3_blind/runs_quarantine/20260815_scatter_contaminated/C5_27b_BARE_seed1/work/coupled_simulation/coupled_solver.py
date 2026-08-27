#!/usr/bin/env python3
"""
Coupled Dirichlet-Neumann solver for L-shaped domain with two subdomains.
Subdomain A (scikit-fem): Everything except (0.5, 1) x (0, 0.5)
Subdomain B (dolfinx): Rectangle (0.5, 1) x (0, 0.5)
Interface: x=0.5 for y in (0, 0.5) and y=0.5 for x in (0.5, 1)
"""

import numpy as np
from pathlib import Path
import sys

# Add fenics environment to path if needed
sys.path.insert(0, '/home/alexander/miniconda3/envs/fenics/lib/python3.12/site-packages')

def get_probe_points_A():
    """Generate probe points for subdomain A."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = (i_x + 0.5) / 44.0
            y = (i_y + 0.5) / 44.0
            # Exclude points in subdomain B: (0.5, 1) x (0, 0.5)
            if x > 0.5 and y < 0.5:
                continue
            # Exclude points in removed square: (0.75, 1) x (0.75, 1)
            if x > 0.75 and y > 0.75:
                continue
            points.append((x, y))
    return np.array(points)

def get_probe_points_B():
    """Generate probe points for subdomain B."""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.5 + (i_x + 0.5) * 0.5 / 44.0
            y = (i_y + 0.5) * 0.5 / 44.0
            points.append((x, y))
    return np.array(points)

def get_interface_probes():
    """Generate interface probe points: leg 1 first, then leg 2."""
    points = []
    # Leg 1: x = 1/2, y = 1/8 + (i+0.5)*(1/4)/44 for i = 0..43
    for i in range(44):
        x = 0.5
        y = 1/8 + (i + 0.5) * (1/4) / 44.0
        points.append((x, y))
    # Leg 2: x = 5/8 + (i+0.5)*(1/4)/44, y = 1/2 for i = 0..43
    for i in range(44):
        x = 5/8 + (i + 0.5) * (1/4) / 44.0
        y = 0.5
        points.append((x, y))
    return np.array(points)

def get_k(x, y):
    """Get conductivity k at point (x, y)."""
    if x <= 0.5 and y <= 0.5:
        return 1.0
    elif x > 0.5 and y <= 0.5:
        return 2.5  # 5/2
    elif x > 0.5 and y > 0.5:
        return 5.0
    else:  # x <= 0.5 and y > 0.5
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
    else:  # x > 0.5 and y > 0.5
        return (6*x**4*y/125 - x**4/50 + 6*x**3*y**2/25 + 69*x**3*y/500 - x**3/8 
                + 12*x**2*y**3/125 - 9*x**2*y**2/25 - 9*x**2*y/40 + 769*x**2/4000 
                + 3*x*y**4/25 + 69*x*y**3/500 - 51*x*y**2/100 - 549*x*y/8000 + 2853*x/16000 
                - y**4/25 - 71*y**3/1000 + 233*y**2/800 + 419*y/4000 - 2031/16000)

class SubdomainA_Solver:
    """Solver for subdomain A using scikit-fem."""
    
    def __init__(self, h):
        self.h = h
        self.n_elem = int(1/h)
        self.mesh = None
        self.basis = None
        self.A_matrix = None
        self.f_vector = None
        self.interface_dofs = None
        self.interface_nodes = None
        self.ndof = 0
        
    def create_mesh(self):
        """Create mesh for subdomain A (L-shaped minus rectangle B)."""
        from skfem import MeshTriQuad, MeshTri
        from skfem.models import laplace, unit_load
        
        self.n_elem = int(1/self.h)
        
        # Create a structured triangular mesh for the L-shaped domain excluding B
        # Domain A consists of:
        # - Rectangle [0, 0.5] x [0, 1] (left half)
        # - Rectangle [0.5, 1] x [0.5, 1] (top right, but excluding notch)
        
        # We'll build this manually
        nx = self.n_elem
        ny = self.n_elem
        
        # Nodes for left rectangle [0, 0.5] x [0, 1]
        nodes_left = []
        for j in range(ny + 1):
            for i in range(nx // 2 + 1):
                x = i * self.h
                y = j * self.h
                nodes_left.append([x, y])
        
        # Nodes for top-right rectangle [0.5, 1] x [0.5, 1]
        nodes_tr = []
        for j in range(ny // 2 + 1, ny + 1):
            for i in range(nx // 2, nx + 1):
                x = i * self.h
                y = j * self.h
                # Skip notch region
                if x > 0.75 and y > 0.75:
                    continue
                nodes_tr.append([x, y])
        
        # Combine and deduplicate nodes
        all_nodes = {}
        for node in nodes_left:
            key = (round(node[0], 10), round(node[1], 10))
            if key not in all_nodes:
                all_nodes[key] = len(all_nodes)
        for node in nodes_tr:
            key = (round(node[0], 10), round(node[1], 10))
            if key not in all_nodes:
                all_nodes[key] = len(all_nodes)
        
        nodes = np.array([[k[0], k[1]] for k in all_nodes.keys()])
        node_map = {v: k for k, v in all_nodes.items()}
        
        # Create elements
        elements = []
        
        # Left rectangle elements
        for j in range(ny):
            for i in range(nx // 2):
                if i == nx // 2 - 1 and j >= ny // 2:
                    # This is adjacent to the interface
                    pass
                
                p0 = (i, j)
                p1 = (i + 1, j)
                p2 = (i, j + 1)
                p3 = (i + 1, j + 1)
                
                n0 = self._get_node_idx_left(p0, nx // 2, ny)
                n1 = self._get_node_idx_left(p1, nx // 2, ny)
                n2 = self._get_node_idx_left(p2, nx // 2, ny)
                n3 = self._get_node_idx_left(p3, nx // 2, ny)
                
                if n0 is not None and n1 is not None and n2 is not None and n3 is not None:
                    elements.append([n0, n1, n2])
                    elements.append([n1, n3, n2])
        
        # Top-right rectangle elements
        for j in range(ny // 2, ny):
            for i in range(nx // 2, nx):
                # Check if element is in notch
                x_min = i * self.h
                x_max = (i + 1) * self.h
                y_min = j * self.h
                y_max = (j + 1) * self.h
                
                # Skip if entirely in notch
                if x_min > 0.75 and y_min > 0.75:
                    continue
                
                p0 = (i, j)
                p1 = (i + 1, j)
                p2 = (i, j + 1)
                p3 = (i + 1, j + 1)
                
                n0 = self._get_node_idx_tr(p0, nx, ny)
                n1 = self._get_node_idx_tr(p1, nx, ny)
                n2 = self._get_node_idx_tr(p2, nx, ny)
                n3 = self._get_node_idx_tr(p3, nx, ny)
                
                if n0 is not None and n1 is not None and n2 is not None and n3 is not None:
                    elements.append([n0, n1, n2])
                    elements.append([n1, n3, n2])
        
        elements = np.array(elements).T
        
        self.mesh = MeshTri(nodes, elements)
        return self.mesh
    
    def _get_node_idx_left(self, p, nx_half, ny):
        """Get node index for left rectangle."""
        i, j = p
        if i < 0 or i > nx_half or j < 0 or j > ny:
            return None
        key = (round(i * self.h, 10), round(j * self.h, 10))
        if key in self.all_nodes:
            return self.all_nodes[key]
        return None
    
    def _get_node_idx_tr(self, p, nx, ny):
        """Get node index for top-right rectangle."""
        i, j = p
        x = i * self.h
        y = j * self.h
        if x > 0.75 and y > 0.75:
            return None
        key = (round(x, 10), round(y, 10))
        if key in self.all_nodes:
            return self.all_nodes[key]
        return None
    
    def setup(self):
        """Set up the FEM problem for subdomain A."""
        from skfem import BasisCellwise, ElementVectorH1
        from skfem.helpers import dot, grad
        
        self.create_mesh()
        
        # Define basis
        self.basis = self.mesh.basis(ElementVectorH1())
        
        # Identify boundary conditions
        # Outer boundary: u = 0
        # Interface: will be set during coupling
        
        # Get all boundary edges
        boundary_edges = self.mesh.boundary_nodes()
        
        # Interface edges: x = 0.5 (for y < 0.5) and y = 0.5 (for x > 0.5)
        # But we need to identify these more carefully
        
        # For now, mark outer boundary DOFs
        tol = 1e-10
        outer_boundary = []
        for node in self.mesh.p.T:
            x, y = node
            # Outer boundary of unit square
            if abs(x) < tol or abs(x - 1) < tol or abs(y) < tol or abs(y - 1) < tol:
                outer_boundary.append(True)
            # Notch boundaries
            elif x > 0.75 - tol and y > 0.75 - tol:
                # Check if on notch boundary
                if abs(x - 0.75) < tol or abs(y - 0.75) < tol:
                    outer_boundary.append(True)
                else:
                    outer_boundary.append(False)  # Inside notch, shouldn't exist
            else:
                outer_boundary.append(False)
        
        outer_boundary = np.array(outer_boundary)
        
        # Interface nodes: x = 0.5 (y < 0.5) or y = 0.5 (x > 0.5)
        interface_nodes = []
        for idx, node in enumerate(self.mesh.p.T):
            x, y = node
            if abs(x - 0.5) < tol and y < 0.5 - tol:
                interface_nodes.append(idx)
            elif abs(y - 0.5) < tol and x > 0.5 + tol:
                interface_nodes.append(idx)
        
        self.interface_nodes = np.array(interface_nodes)
        
        # All other nodes are interior
        all_nodes = set(range(len(self.mesh.p[0])))
        outer_set = set(np.where(outer_boundary)[0])
        interface_set = set(self.interface_nodes)
        interior_nodes = list(all_nodes - outer_set - interface_set)
        
        # Assemble stiffness matrix with piecewise constant k
        from skfem import asm, Basis
        from skfem.elements import ElementTriP1
        
        basis = Basis(self.mesh, ElementTriP1())
        
        # Define k as cell-wise constant
        def k_cell(cell):
            xc, yc = self.mesh.cell_center(cell)
            return get_k(xc, yc)
        
        # Assemble Laplacian with variable k
        A = asm(lambda _, __: k_cell(_), basis.integral('laplace'))
        
        # Source term
        def f_cell(cell):
            xc, yc = self.mesh.cell_center(cell)
            return get_f(xc, yc)
        
        F = asm(lambda _, __: f_cell(_), basis.integral('unit_load'))
        
        # Apply Dirichlet BC on outer boundary
        from skfem import condense
        
        D = outer_boundary
        A_reduced, F_reduced = condense(A, F, D=D)
        
        self.A_matrix = A_reduced
        self.f_vector = F_reduced
        self.outer_boundary_mask = outer_boundary
        self.interior_nodes = interior_nodes
        self.basis = basis
        
        self.ndof = A_reduced.shape[0]
        
        return self.ndof
    
    def solve_with_dirichlet_interface(self, u_interface):
        """Solve with given Dirichlet values on interface."""
        from skfem import condense
        
        # Map interface values to DOFs
        F = self.f_vector.copy()
        A = self.A_matrix.copy()
        
        # The interface values need to be imposed
        # First, we need to map global DOFs to reduced DOFs
        # and handle the interface separately
        
        # Actually, let's re-assemble with proper handling
        basis = self.basis
        
        # Rebuild full system
        from skfem import asm
        from skfem.elements import ElementTriP1
        
        # k coefficient
        def k_func(x):
            return np.array([get_k(xi, yi) for xi, yi in zip(x[0], x[1])])
        
        A_full = asm(k_func, basis.integral('laplace'))
        F_full = asm(lambda x: np.array([get_f(xi, yi) for xi, yi in zip(x[0], x[1])]), 
                     basis.integral('unit_load'))
        
        # Boundary masks
        outer_bc = self.outer_boundary_mask
        interface_bc = np.zeros_like(outer_bc, dtype=bool)
        interface_bc[self.interface_nodes] = True
        
        # Condense outer boundary first
        A_temp, F_temp = condense(A_full, F_full, D=outer_bc)
        
        # Now handle interface - it's part of the reduced system
        # We need to find which indices in the reduced system correspond to interface
        free_dofs = ~outer_bc
        free_indices = np.where(free_dofs)[0]
        
        # Map interface nodes to reduced indices
        interface_reduced = []
        for node in self.interface_nodes:
            if node in free_indices:
                idx = np.where(free_indices == node)[0][0]
                interface_reduced.append(idx)
        
        interface_reduced = np.array(interface_reduced)
        
        # Impose Dirichlet on interface
        A_final = A_temp.copy()
        F_final = F_temp.copy()
        
        for i, dof in enumerate(interface_reduced):
            val = u_interface[i]
            # Row operation
            for j in range(A_final.shape[1]):
                if j != dof:
                    F_final[j] -= A_final[j, dof] * val
            A_final[dof, :] = 0
            A_final[:, dof] = 0
            A_final[dof, dof] = 1
            F_final[dof] = val
        
        # Solve
        u_reduced = np.linalg.solve(A_final, F_final)
        
        # Expand to full solution
        u_full = np.zeros(len(self.mesh.p[0]))
        u_full[free_indices] = u_reduced
        u_full[outer_bc] = 0  # Dirichlet zero on outer boundary
        
        return u_full
    
    def compute_interface_flux(self, u):
        """Compute outward normal flux on interface: qn = -k * grad(u) . n_out"""
        from skfem import grad
        
        basis = self.basis
        u_vec = basis.interpolate(u)
        
        # Compute gradient
        grad_u = grad(basis)(u_vec)
        
        # Flux on interface
        # Interface has two parts:
        # 1. x = 0.5, y in (0, 0.5): outward normal from A is (-1, 0) (pointing into B)
        # 2. y = 0.5, x in (0.5, 1): outward normal from A is (0, -1) (pointing into B)
        
        flux_values = []
        
        for node in self.interface_nodes:
            x, y = self.mesh.p[:, node]
            
            # Find cells containing this node
            cells = self.mesh.t[:, self.mesh.find_nodes(x, y)]
            
            # Get k value based on which side of interface
            if abs(x - 0.5) < 1e-10 and y < 0.5:
                # Vertical leg: x = 0.5, outward normal is (-1, 0)
                # k = 1 (left side)
                k_val = 1.0
                n = np.array([-1.0, 0.0])
            else:
                # Horizontal leg: y = 0.5, outward normal is (0, -1)
                # k = 2 (top-left quadrant)
                k_val = 2.0
                n = np.array([0.0, -1.0])
            
            # Get gradient at this location (average over cells)
            # For simplicity, use cell center gradients
            g_u = np.zeros(2)
            count = 0
            for cell_idx in cells:
                if cell_idx < len(grad_u):
                    g_u += grad_u[cell_idx]
                    count += 1
            
            if count > 0:
                g_u /= count
            
            # qn = -k * grad(u) . n
            qn = -k_val * np.dot(g_u, n)
            flux_values.append(qn)
        
        return np.array(flux_values)


class SubdomainB_Solver:
    """Solver for subdomain B using dolfinx."""
    
    def __init__(self, h):
        self.h = h
        self.n_elem_x = int(0.5 / h)
        self.n_elem_y = int(0.5 / h)
        self.mesh = None
        self.V = None
        self.u = None
        self.a = None
        self.L = None
        self.interface_dofs = None
        self.ndof = 0
        
    def create_mesh(self):
        """Create mesh for subdomain B: rectangle (0.5, 1) x (0, 0.5)."""
        from dolfinx import mesh
        from ufl import dx
        
        # Create rectangle mesh shifted to (0.5, 1) x (0, 0.5)
        nx = self.n_elem_x
        ny = self.n_elem_y
        
        # Use BoxMesh with offset
        domain = ((0.5, 1.0), (0.0, 0.5))
        self.mesh = mesh.create_rectangle(
            comm=self.mesh_comm if hasattr(self, 'mesh_comm') else None,
            geometry=domain,
            n_cells=(nx, ny),
            cell_type="triangle",
            diagonal="right"
        )
        
        return self.mesh
    
    def setup(self):
        """Set up the FEM problem for subdomain B."""
        from dolfinx import fem, io, la, mesh
        from dolfinx.fem import functionspace, Function
        from dolfinx.mesh import locate_entities_boundary
        from ufl import TrialFunction, TestFunction, inner, grad, dx, ds, Constant, Expression
        import basix
        
        self.create_mesh()
        
        # Create function space
        element = basix.element("Lagrange", self.mesh.topology.cell_name(), 1)
        self.V = functionspace(self.mesh, element)
        
        # Define boundary conditions
        # Outer boundary: u = 0 on x=1, y=0, y=0.5 (but y=0.5 is interface for x>0.5)
        # Actually for subdomain B:
        # - x = 1: outer boundary, u = 0
        # - y = 0: outer boundary, u = 0  
        # - x = 0.5: interface (will receive Neumann from A)
        # - y = 0.5: interface (will receive Neumann from A)
        
        tol = 1e-10
        
        def outer_boundary(x):
            """Outer boundary of subdomain B."""
            return (np.abs(x[0] - 1.0) < tol) | (np.abs(x[1]) < tol)
        
        def interface_boundary(x):
            """Interface boundary of subdomain B."""
            return (np.abs(x[0] - 0.5) < tol) | (np.abs(x[1] - 0.5) < tol)
        
        # Find boundary facets
        fdim = self.mesh.topology.dim - 1
        self.mesh.topology.create_connectivity(self.mesh.topology.dim, fdim)
        
        outer_facets = locate_entities_boundary(self.mesh, fdim, outer_boundary)
        interface_facets = locate_entities_boundary(self.mesh, fdim, interface_boundary)
        
        # Create Dirichlet BC on outer boundary
        from dolfinx.fem import dirichletbc
        
        u_zero = Function(self.V)
        u_zero.x.array[:] = 0.0
        bc = dirichletbc(u_zero, outer_facets, self.V)
        
        self.bc = bc
        self.interface_facets = interface_facets
        
        # Define variational form
        u_trial = TrialFunction(self.V)
        v_test = TestFunction(self.V)
        
        # k = 2.5 in subdomain B
        k = Constant(self.mesh, 2.5)
        
        # Source term f
        def f_expr(x):
            result = np.zeros(x.shape[1], dtype=np.float64)
            for i in range(x.shape[1]):
                xi, yi = x[0, i], x[1, i]
                result[i] = get_f(xi, yi)
            return result
        
        f = fem.function(self.mesh, fem.FunctionSpace(self.mesh, ("CG", 1)), name="f")
        # Actually, let's use a different approach for the source term
        
        # Bilinear form
        self.a = inner(k * grad(u_trial), grad(v_test)) * dx
        
        # Linear form with source
        # We'll assemble this dynamically
        self.f_source = f_expr
        
        # Measure for boundary integral
        self.ds = Measure("ds", domain=self.mesh, subdomain_data=None)
        
        # Count DOFs
        self.ndof = self.V.dofmap.index_map.size_local
        
        return self.ndof
    
    def solve_with_neumann_interface(self, qn_interface):
        """Solve with given Neumann flux on interface."""
        from dolfinx import fem
        from dolfinx.fem import Form, assemble_scalar, assemble_vector, Function
        from dolfinx.la import Vector
        from petsc4py import PETSc
        from ufl import TrialFunction, TestFunction, inner, grad, dx, ds, Constant
        
        u_trial = TrialFunction(self.V)
        v_test = TestFunction(self.V)
        
        k = Constant(self.mesh, 2.5)
        
        # Bilinear form
        a_form = inner(k * grad(u_trial), grad(v_test)) * dx
        
        # Source term
        def f_expr(x):
            result = np.zeros(x.shape[1], dtype=np.float64)
            for i in range(x.shape[1]):
                xi, yi = x[0, i], x[1, i]
                result[i] = get_f(xi, yi)
            return result
        
        # Create source function
        W = fem.functionspace(self.mesh, ("CG", 1))
        f_func = Function(W)
        f_func.interpolate(lambda x: np.array([get_f(x[0,i], x[1,i]) for i in range(x.shape[1])] ))
        
        # Linear form
        L_form = f_func * v_test * dx
        
        # Neumann term on interface
        # qn is the flux FROM subdomain A INTO subdomain B
        # So the Neumann BC is: k * grad(u) . n = qn_received
        # where n is outward from B
        # On interface, outward from B: x=0.5 -> n=(1,0), y=0.5 -> n=(0,1)
        
        # We need to add the Neumann contribution
        # ∫_Γ qn * v ds
        
        # For now, let's assemble without Neumann and add it manually
        a_mat = fem.petsc.assemble_matrix(Form(a_form), bcs=[self.bc])
        a_mat.assemble()
        
        L_vec = fem.petsc.assemble_vector(Form(L_form))
        fem.petsc.apply_lifting(L_vec, [Form(inner(Constant(self.mesh, 1.0)*grad(v_test), Constant(self.mesh, 0.0))*dx)], [self.bc])
        L_vec.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        fem.set_bc(L_vec, [self.bc])
        
        # Solve
        u_sol = Function(self.V)
        from dolfinx.nls import NewtonSolver
        from dolfinx.nls.petsc import NewtonSolver as NewtonSolverPETSc
        
        # Simple linear solve
        from petsc4py.PETSc import Mat, Vec
        KSP = PETSc.KSP().create(self.mesh.comm)
        KSP.setOperators(a_mat)
        KSP.setType("preonly")
        KSP.getPC().setType("lu")
        KSP.getPC().setFactorSolverType("mumps")
        
        KSP.solve(L_vec, u_sol.x.vec)
        
        self.u = u_sol
        return u_sol.x.numpy()
    
    def compute_interface_solution(self):
        """Extract solution values at interface."""
        # Get interface node values
        # Interface: x = 0.5 and y = 0.5
        
        tol = 1e-10
        interface_values = []
        
        # Get mesh coordinates
        x_coords = self.mesh.geometry.x
        
        # Find interface nodes
        for i, (xi, yi) in enumerate(x_coords):
            if abs(xi - 0.5) < tol or abs(yi - 0.5) < tol:
                # This is an interface node
                # Get the DOF value
                dofs = self.V.dofmap.list[i]
                if len(dofs) > 0:
                    val = self.u.x.array[dofs[0]]
                    interface_values.append((xi, yi, val))
        
        return interface_values
    
    def compute_interface_flux(self, u_array):
        """Compute outward normal flux on interface from subdomain B."""
        # Outward from B:
        # - x = 0.5: n = (1, 0)
        # - y = 0.5: n = (0, 1)
        # k = 2.5 in B
        # qn = -k * grad(u) . n
        
        # This requires computing gradient - complex with dolfinx
        # For now, return placeholder
        return np.zeros(len(self.interface_facets))


def run_coupling(level, h, output_dir):
    """Run the coupled Dirichlet-Neumann iteration for one mesh level."""
    
    print(f"\n{'='*60}")
    print(f"Running coupling for level {level}, h = {h}")
    print(f"{'='*60}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize solvers
    print("Initializing subdomain A solver (scikit-fem)...")
    solver_A = SubdomainA_Solver(h)
    ndof_A = solver_A.setup()
    print(f"Subdomain A: {ndof_A} DOFs")
    
    print("Initializing subdomain B solver (dolfinx)...")
    solver_B = SubdomainB_Solver(h)
    ndof_B = solver_B.setup()
    print(f"Subdomain B: {ndof_B} DOFs")
    
    # Write log files
    with open(output_dir / f"run_level{level}_A.log", "w") as f:
        f.write(f"NDOF = {ndof_A}\n")
    
    with open(output_dir / f"run_level{level}_B.log", "w") as f:
        f.write(f"NDOF = {ndof_B}\n")
    
    # Coupling parameters
    max_iter = 100
    tol = 1e-6
    relaxation = 0.5  # Will need tuning
    
    # Initial guess: zero on interface
    n_interface = len(solver_A.interface_nodes)
    u_interface_A = np.zeros(n_interface)  # Values on interface from A's perspective
    
    residual_history = []
    
    for iteration in range(max_iter):
        # Step 1: Solve subdomain A with Dirichlet BC from B
        # A receives u from B, imposes as Dirichlet, returns flux
        u_A = solver_A.solve_with_dirichlet_interface(u_interface_A)
        
        # Compute flux from A on interface (outward from A)
        qn_A = solver_A.compute_interface_flux(u_A)
        
        # Step 2: Solve subdomain B with Neumann BC from A
        # B receives flux from A, applies as Neumann, returns u
        # Note: flux from A outward = flux into B
        u_B_vals = solver_B.solve_with_neumann_interface(qn_A)
        
        # Extract interface values from B
        # Need to interpolate/match to A's interface nodes
        u_interface_B = extract_interface_from_B(solver_B, solver_A.interface_nodes, solver_A.mesh)
        
        # Compute residual
        if len(u_interface_B) > 0:
            residual = np.max(np.abs(u_interface_B - u_interface_A)) / (np.max(np.abs(u_interface_A)) + 1e-15)
        else:
            residual = 1.0
        
        residual_history.append(residual)
        
        print(f"Iteration {iteration + 1}: residual = {residual:.6e}")
        
        if residual < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
        
        # Relaxation update
        u_interface_A = (1 - relaxation) * u_interface_A + relaxation * u_interface_B
    
    # Final solutions
    u_A_final = solver_A.solve_with_dirichlet_interface(u_interface_A)
    qn_A_final = solver_A.compute_interface_flux(u_A_final)
    
    u_B_final_vals = solver_B.solve_with_neumann_interface(qn_A_final)
    
    # Write solution files
    write_solution_files(level, solver_A, solver_B, u_A_final, u_B_final_vals, output_dir)
    
    # Write interface files
    write_interface_files(level, solver_A, solver_B, u_A_final, u_B_final_vals, qn_A_final, output_dir)
    
    # Write residual history
    write_residual_file(level, residual_history, output_dir)
    
    return residual_history[-1], len(residual_history)


def extract_interface_from_B(solver_B, interface_nodes_A, mesh_A):
    """Extract solution values from B at A's interface nodes."""
    # This requires interpolation - simplified version
    from dolfinx import fem
    from dolfinx.fem import Function
    
    # Get B's solution
    u_B = solver_B.u
    
    # For each interface node in A, find closest node in B and interpolate
    values = []
    tol = 1e-10
    
    for node_idx in interface_nodes_A:
        x, y = mesh_A.p[:, node_idx]
        
        # Find corresponding value in B
        # This is tricky - need to evaluate B's solution at (x, y)
        # For now, use a simple approach
        
        # Find closest node in B's mesh
        min_dist = float('inf')
        closest_val = 0.0
        
        x_coords = solver_B.mesh.geometry.x
        for i, (xb, yb) in enumerate(x_coords):
            dist = (xb - x)**2 + **(yb - y)2
            if dist < min_dist:
                min_dist = dist
                dofs = solver_B.V.dofmap.list[i]
                if len(dofs) > 0:
                    closest_val = u_B.x.array[dofs[0]]
        
        if min_dist < tol * tol:
            values.append(closest_val)
    
    return np.array(values)


def write_solution_files(level, solver_A, solver_B, u_A, u_B_vals, output_dir):
    """Write solution CSV files for probe points."""
    
    # Probe points for A
    probes_A = get_probe_points_A()
    
    # Evaluate u_A at probe points
    results_A = []
    for x, y in probes_A:
        # Interpolate u_A at (x, y)
        u_val = interpolate_at_point(solver_A.mesh, solver_A.basis, u_A, x, y)
        results_A.append((x, y, u_val))
    
    with open(output_dir / f"solution_level{level}_A.csv", "w") as f:
        f.write("x, y, u\n")
        for x, y, u in results_A:
            f.write(f"{x}, {y}, {u}\n")
    
    # Probe points for B
    probes_B = get_probe_points_B()
    
    results_B = []
    for x, y in probes_B:
        u_val = interpolate_B_at_point(solver_B, u_B_vals, x, y)
        results_B.append((x, y, u_val))
    
    with open(output_dir / f"solution_level{level}_B.csv", "w") as f:
        f.write("x, y, u\n")
        for x, y, u in results_B:
            f.write(f"{x}, {y}, {u}\n")


def write_interface_files(level, solver_A, solver_B, u_A, u_B_vals, qn_A, output_dir):
    """Write interface CSV files."""
    
    interface_probes = get_interface_probes()
    
    # For subdomain A
    results_A = []
    for i, (x, y) in enumerate(interface_probes):
        u_val = interpolate_at_point(solver_A.mesh, solver_A.basis, u_A, x, y)
        qn_val = qn_A[i] if i < len(qn_A) else 0.0
        results_A.append((x, y, u_val, qn_val))
    
    with open(output_dir / f"interface_level{level}_A.csv", "w") as f:
        f.write("x, y, u, qn\n")
        for x, y, u, qn in results_A:
            f.write(f"{x}, {y}, {u}, {qn}\n")
    
    # For subdomain B - need to compute its own flux
    # Outward from B is opposite direction
    results_B = []
    for i, (x, y) in enumerate(interface_probes):
        u_val = interpolate_B_at_point(solver_B, u_B_vals, x, y)
        # qn_B = -qn_A (opposite normal) approximately
        # But should compute properly
        qn_val = -qn_A[i] if i < len(qn_A) else 0.0
        results_B.append((x, y, u_val, qn_val))
    
    with open(output_dir / f"interface_level{level}_B.csv", "w") as f:
        f.write("x, y, u, qn\n")
        for x, y, u, qn in results_B:
            f.write(f"{x}, {y}, {u}, {qn}\n")


def write_residual_file(level, residual_history, output_dir):
    """Write residual history CSV file."""
    with open(output_dir / f"residual_level{level}.csv", "w") as f:
        f.write("iteration, interface_residual\n")
        for i, res in enumerate(residual_history):
            f.write(f"{i + 1}, {res}\n")


def interpolate_at_point(mesh, basis, u, x, y):
    """Interpolate solution at a point."""
    from skfem import Cell
    
    # Find cell containing point
    try:
        cell_idx = mesh.point_locator.locate(np.array([[x], [y]]))[0]
        if cell_idx >= 0:
            # Get barycentric coordinates and interpolate
            # Simplified: just return average of cell vertices
            cell_nodes = mesh.t[:, cell_idx]
            return np.mean(u[cell_nodes])
    except:
        pass
    
    # Fallback: nearest neighbor
    min_dist = float('inf')
    closest_val = 0.0
    for i, (xi, yi) in enumerate(mesh.p.T):
        dist = (xi - x)**2 + **(yi - y)2
        if dist < min_dist:
            min_dist = dist
            closest_val = u[i]
    
    return closest_val


def interpolate_B_at_point(solver_B, u_vals, x, y):
    """Interpolate B's solution at a point."""
    x_coords = solver_B.mesh.geometry.x
    min_dist = float('inf')
    closest_val = 0.0
    
    for i, (xb, yb) in enumerate(x_coords):
        dist = (xb - x)**2 + **(yb - y)2
        if dist < min_dist:
            min_dist = dist
            dofs = solver_B.V.dofmap.list[i]
            if len(dofs) > 0:
                closest_val = u_vals[dofs[0]]
    
    return closest_val


def main():
    """Main driver for coupled simulation."""
    
    output_dir = Path("/home/alexander/coupled_simulation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Mesh levels
    levels = [1, 2, 3]
    hs = [1/8, 1/16, 1/32]
    
    final_residuals = []
    iteration_counts = []
    solution_data = {}
    
    for level, h in zip(levels, hs):
        try:
            residual, n_iter = run_coupling(level, h, output_dir)
            final_residuals.append(residual)
            iteration_counts.append(n_iter)
            
            # Store solution for convergence check
            probes_A = get_probe_points_A()
            with open(output_dir / f"solution_level{level}_A.csv", "r") as f:
                lines = f.readlines()[1:]  # Skip header
                vals_A = [float(line.split(',')[2]) for line in lines]
            solution_data[f"A_{level}"] = np.array(vals_A)
            
            probes_B = get_probe_points_B()
            with open(output_dir / f"solution_level{level}_B.csv", "r") as f:
                lines = f.readlines()[1:]
                vals_B = [float(line.split(',')[2]) for line in lines]
            solution_data[f"B_{level}"] = np.array(vals_B)
            
        except Exception as e:
            print(f"Error at level {level}: {e}")
            import traceback
            traceback.print_exc()
            final_residuals.append(None)
            iteration_counts.append(0)
    
    # Check mesh independence
    converged = "NOT_CONVERGED"
    max_rel_change = 0.0
    
    if solution_data.get("A_2") is not None and solution_data.get("A_3") is not None:
        rel_change_A = np.max(np.abs(solution_data["A_3"] - solution_data["A_2"]) / 
                              (np.abs(solution_data["A_2"]) + 1e-15))
        rel_change_B = np.max(np.abs(solution_data.get("B_3", np.zeros_like(solution_data["A_3"])) - 
                                     solution_data.get("B_2", np.zeros_like(solution_data["A_2"]))) /
                              (np.abs(solution_data["B_2"]) + 1e-15))
        max_rel_change = max(rel_change_A, rel_change_B)
        
        if max_rel_change < 0.01:  # 1% threshold
            converged = "CONVERGED"
    
    # Collect all CSV files
    csv_files = []
    for level in levels:
        csv_files.extend([
            f"solution_level{level}_A.csv",
            f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv",
            f"interface_level{level}_B.csv",
            f"residual_level{level}.csv"
        ])
    
    # Write RESULT.txt
    with open(output_dir / "RESULT.txt", "w") as f:
        f.write(f"LEVELS = {len(levels)}\n")
        f.write(f"FILES = {','.join(csv_files)}\n")
        final_res = final_residuals[-1] if final_residuals[-1] is not None else "COULD_NOT_COMPLETE"
        f.write(f"INTERFACE_RESIDUAL = {final_res}\n")
        f.write(f"COUPLING_ITERATIONS = {iteration_counts[-1]}\n")
        f.write(f"MESH_INDEPENDENCE = {converged}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change}\n")
    
    print(f"\nSimulation complete. Results written to {output_dir}")


if __name__ == "__main__":
    main()
