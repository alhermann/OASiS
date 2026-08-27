"""FEniCSx (dolfinx) FLUID participant for OASiS `couple` driver — FSI.

CONTRACT: runs in work_dir with no arguments, reads imports.json ({} on iter 1),
writes exports.json LAST.

Physics: steady incompressible Navier-Stokes on channel (0,1) x (0,0.2).
- Parabolic inflow at x=0 with mean U=1
- Traction-free outflow at x=1
- No-slip at y=0
- FSI interface at y=0.2: ALE mesh motion from imported displacement,
  no-slip u=0 on DEFORMED interface
- Exports TRACTION on structure: t = sigma_f . n_s where n_s = -n_f

SIGN CONVENTION:
  exported traction t = sigma_f . n_s = -sigma_f . n_f
  n_f = fluid's outward normal at interface = +e_y (at y=0.2)
  n_s = structure's outward normal = -e_y
  For static fluid at p>0: sigma_f = -pI, so t = -(-pI).(-e_y) = +p*e_y
  This pushes wall AWAY from fluid (+y direction). ✓

Interface parametrisation: LAGRANGIAN (reference positions).
"""
import json
import sys
from pathlib import Path

import numpy as np
import ufl
import basix.ufl
from dolfinx import fem, mesh as dmesh
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem
from mpi4py import MPI
from petsc4py import PETSc

# ── PROBLEM PARAMETERS ───────────────────────────────────────────────────────
PARTNER     = "structure"  # partner's name in couple(...) call
LX          = 1.0          # channel length
HY          = 0.2          # channel height
IFACE_Y     = HY           # FSI interface at top
NX, NY      = 48, 10       # fluid mesh divisions
MU          = 1.0          # dynamic viscosity
RHO_F       = 1.0          # fluid density
U_MEAN      = 1.0          # mean inflow velocity
ALE_STIFF   = 0.0          # Jacobian stiffening (0 = plain harmonic)
D_INIT      = 0.0          # iteration-1 fallback displacement
MOVE_MESH   = True         # SET False to disable structure->fluid coupling
# ─────────────────────────────────────────────────────────────────────────────


def read_imports():
    """imports.json is {partner_name: InterfaceData}; {} on iteration 1."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        return data.get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample_displacement(imp, x_targets, fallback):
    """Map partner's displacement samples onto THIS participant's interface nodes.
    
    Interpolates component-by-component along x coordinate.
    Returns (n_nodes, 2) array of displacement values.
    """
    if not imp or not imp.get("coordinates"):
        return np.full((len(x_targets), 2), float(fallback))
    
    xs_src = np.asarray(imp["coordinates"], float)[:, 0]
    vals = np.asarray(imp["values"], float).reshape(len(xs_src), -1)
    
    out = np.zeros((len(x_targets), 2))
    order = np.argsort(xs_src)
    for c in range(min(2, vals.shape[1])):
        out[:, c] = np.interp(x_targets, xs_src[order], vals[order, c])
    return out


def _signed_areas(msh):
    """Twice the signed area of every triangle."""
    cells = np.asarray(msh.geometry.dofmap).reshape(-1, 3)
    px = msh.geometry.x[:, :2]
    v0 = px[cells[:, 1]] - px[cells[:, 0]]
    v1 = px[cells[:, 2]] - px[cells[:, 0]]
    return v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]


def main():
    comm = MPI.COMM_SELF
    
    # Create reference mesh
    msh = dmesh.create_rectangle(
        comm, [np.array([0.0, 0.0]), np.array([LX, HY])], [NX, NY],
        dmesh.CellType.triangle)
    gdim = msh.geometry.dim
    
    # P1 vector space for ALE lift and traction projection
    V1 = fem.functionspace(msh, ("Lagrange", 1, (gdim,)))
    dofc = V1.tabulate_dof_coordinates()[:, :gdim]
    
    # Reference interface node coordinates (Lagrangian)
    tol = 1e-9
    iface_nodes = np.where(np.abs(dofc[:, 1] - IFACE_Y) < tol)[0]
    x_iface = dofc[iface_nodes, 0]
    order = np.argsort(x_iface)
    iface_nodes = iface_nodes[order]
    x_iface = x_iface[order]
    ref_coords = np.column_stack([x_iface, np.full_like(x_iface, IFACE_Y)])
    
    # Read imported displacement
    imp = read_imports()
    d_iface = sample_displacement(imp, x_iface, D_INIT)
    
    # Facet tags: 1=inflow, 2=outflow, 3=bottom wall, 4=FSI interface
    def _inflow(x):
        return np.isclose(x[0], 0.0)
    
    def _outflow(x):
        return np.isclose(x[0], LX)
    
    def _iface(x):
        return np.isclose(x[1], IFACE_Y)
    
    def _wall(x):
        return np.isclose(x[1], 0.0)
    
    fdim = msh.topology.dim - 1
    facets_list = []
    marks_list = []
    for tag, fn in ((1, _inflow), (2, _outflow), (3, _wall), (4, _iface)):
        f = dmesh.locate_entities_boundary(msh, fdim, fn)
        facets_list.append(f)
        marks_list.append(np.full(len(f), tag, dtype=np.int32))
    
    facets = np.concatenate(facets_list)
    marks = np.concatenate(marks_list)
    srt = np.argsort(facets)
    ft = dmesh.meshtags(msh, fdim, facets[srt], marks[srt])
    
    # ── ALE mesh motion ──────────────────────────────────────────────────────
    area0 = _signed_areas(msh)
    d_ale = fem.Function(V1, name="ale_displacement")
    
    if MOVE_MESH and np.any(np.abs(d_iface) > 1e-14):
        u_, v_ = ufl.TrialFunction(V1), ufl.TestFunction(V1)
        
        # Harmonic extension (Laplace equation)
        a = ufl.inner(ufl.grad(u_), ufl.grad(v_)) * ufl.dx
        L = ufl.inner(fem.Constant(msh, np.zeros(gdim)), v_) * ufl.dx
        
        # Dirichlet BC: prescribed displacement at interface, zero elsewhere
        g_iface = fem.Function(V1)
        g_iface.x.array.reshape(-1, gdim)[iface_nodes] = d_iface
        
        zero = fem.Function(V1)
        bcs_ale = [
            fem.dirichletbc(g_iface, fem.locate_dofs_topological(
                V1, fdim, ft.find(4))),
            fem.dirichletbc(zero, fem.locate_dofs_topological(
                V1, fdim, np.concatenate([ft.find(1), ft.find(2), ft.find(3)])))
        ]
        
        pr = LinearProblem(a, L, bcs=bcs_ale, u=d_ale,
                          petsc_options_prefix="ale_",
                          petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        pr.solve()
        
        # Move mesh
        msh.geometry.x[:, :gdim] += d_ale.x.array.reshape(-1, gdim)
        
        # Check for inverted elements
        moved = _signed_areas(msh)
        bad = int(np.sum(moved * area0 <= 0.0))
        if bad > 0:
            raise RuntimeError(
                f"ALE mesh INVERTED: {bad} cells changed orientation. "
                f"Max |d_iface| = {np.max(np.abs(d_iface)):.3e}")
    
    # ── Taylor-Hood P2/P1 Navier-Stokes ─────────────────────────────────────
    Ve = basix.ufl.element("Lagrange", msh.basix_cell(), 2, shape=(gdim,))
    Qe = basix.ufl.element("Lagrange", msh.basix_cell(), 1)
    W = fem.functionspace(msh, basix.ufl.mixed_element([Ve, Qe]))
    
    w = fem.Function(W)
    u, p = ufl.split(w)
    v, q = ufl.TestFunctions(W)
    
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    n = ufl.FacetNormal(msh)
    
    # Weak form: steady NS
    F = (RHO_F * ufl.inner(ufl.dot(ufl.grad(u), u), v) * ufl.dx
         + MU * ufl.inner(ufl.grad(u) + ufl.grad(u).T, ufl.grad(v)) * ufl.dx
         - ufl.inner(p, ufl.div(v)) * ufl.dx
         + ufl.inner(ufl.div(u), q) * ufl.dx)
    
    # Collapsed velocity space for BCs
    W0 = W.sub(0)
    Vsub, _ = W0.collapse()
    
    # Parabolic inflow profile
    def _parabolic(x):
        vals = np.zeros((gdim, x.shape[1]))
        vals[0] = 6.0 * U_MEAN * x[1] * (HY - x[1]) / HY**2
        return vals
    
    u_in = fem.Function(Vsub)
    u_in.interpolate(_parabolic)
    u_zero = fem.Function(Vsub)
    
    # Boundary conditions
    bcs = [
        # Inflow: parabolic profile
        fem.dirichletbc(u_in, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(1)), W0),
        # Bottom wall: no-slip
        fem.dirichletbc(u_zero, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(3)), W0),
        # FSI interface: no-slip on DEFORMED interface (u=0)
        fem.dirichletbc(u_zero, fem.locate_dofs_topological(
            (W0, Vsub), fdim, ft.find(4)), W0),
    ]
    
    # Newton solver
    problem = NonlinearProblem(
        F, w, bcs=bcs, petsc_options_prefix="ns_",
        petsc_options={
            "snes_type": "newtonls", "snes_rtol": 1e-10, "snes_atol": 1e-11,
            "snes_max_it": 40, "ksp_type": "preonly", "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "snes_error_if_not_converged": True,
            "ksp_error_if_not_converged": True
        })
    problem.solve()
    
    reason = problem.solver.getConvergedReason()
    nit = problem.solver.getIterationNumber()
    if reason <= 0:
        raise RuntimeError(f"Fluid Newton did not converge (reason={reason})")
    
    # ── Extract variationally consistent traction ───────────────────────────
    uh, ph = w.sub(0).collapse(), w.sub(1).collapse()
    
    # Cauchy stress: sigma = -p*I + mu*(grad u + grad u^T)
    sigma = -ph * ufl.Identity(gdim) + MU * (ufl.grad(uh) + ufl.grad(uh).T)
    
    # Traction on structure: t = sigma_f . n_s = -sigma_f . n_f
    # n_f = facet normal (points OUT of fluid = +e_y at top)
    # So we compute: t = -sigma . n (negative because n_s = -n_f)
    
    tt, vv = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    
    # L2 projection of traction onto interface trace of P1
    a_m = ufl.inner(tt, vv) * ds(4)
    L_t = ufl.inner(-ufl.dot(sigma, n), vv) * ds(4)
    
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    
    A = assemble_matrix(fem.form(a_m))
    A.assemble()
    ai, aj, av = A.getValuesCSR()
    M = sp.csr_matrix((av, aj, ai), shape=A.getSize())
    
    b = assemble_vector(fem.form(L_t))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    bvec = b.array.copy()
    
    # Fix singular rows (off-interface DOFs)
    diag = M.diagonal().copy()
    dead = np.where(np.abs(diag) < 1e-14)[0]
    Mfix = M.tolil()
    for i in dead:
        Mfix[i, i] = 1.0
    bvec[dead] = 0.0
    
    t_all = spla.spsolve(Mfix.tocsc(), bvec).reshape(-1, gdim)
    traction = t_all[iface_nodes]
    
    # Net interface force
    fx = fem.assemble_scalar(fem.form(-ufl.dot(sigma, n)[0] * ds(4)))
    fy = fem.assemble_scalar(fem.form(-ufl.dot(sigma, n)[1] * ds(4)))
    
    # Count DOFs
    ndof = W.dofmap.index_map.size_global
    
    # Write log file
    log_content = f"NDOF = {ndof}\n"
    log_content += f"Newton iterations: {nit}\n"
    log_content += f"Net interface force: fx={fx:.6e}, fy={fy:.6e}\n"
    log_content += f"Max |d_iface| = {np.max(np.abs(d_iface)):.6e}\n"
    Path("run_level1_B.log").write_text(log_content)
    
    # ── Write exports.json ──────────────────────────────────────────────────
    export_data = {
        "field_name": "traction_on_structure",
        "n_points": int(len(x_iface)),
        "coordinates": ref_coords.tolist(),
        "values": traction.tolist(),
        # normal_fluxes: traction w.r.t. fluid's OWN outward normal n_f
        # Since values = t = sigma . n_s = -sigma . n_f,
        # normal_fluxes should be sigma . n_f = -t
        "normal_fluxes": (-traction).tolist(),
        "meta": {
            "sign_convention": "values = sigma_f . n_s (load ON structure); "
                               "normal_fluxes = sigma_f . n_f (own outward)",
            "net_force": [float(fx), float(fy)],
            "mesh_moved": bool(MOVE_MESH),
            "newton_iterations": int(nit),
            "displacement_imposed": d_iface.tolist(),
        }
    }
    
    Path("exports.json").write_text(json.dumps(export_data, indent=2))
    
    print(f"[fluid] newton={nit} net_force=({fx:.6e},{fy:.6e}) "
          f"|d_iface|max={np.max(np.abs(d_iface)):.4e}", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
