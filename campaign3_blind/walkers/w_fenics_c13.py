"""FEniCSx path-walk participant for C13: the conducting solid.

NEUMANN-type role (forced by the partner: SPARTA can only receive a wall
temperature). Steady conduction on (0, LS) x (0, H): T = T_HOT at x = 0,
insulated y-faces (natural), and the interface x = LS carries the gas's tallied
energy flux as the natural datum — applied UNCHANGED, because the gas exports
etot with positive = energy leaving the gas, which is exactly this side's
inflow with the sign the weak form wants (measured negative here: the solid is
the hot side and loses heat to the gas).

FLUX_SOLID for the conservation identity is NOT read at the interface — those
dofs are free and their residual vanishes. It is the CONSISTENT reaction at the
x = 0 Dirichlet face: total heat entering the solid there divided by the
height, which at steady state with no source equals the interface flux exactly
at the discrete level. The identity therefore tests the exchange and the
steadiness of the DSMC tally, not the solid's internal conservation.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

import ufl                                                       # noqa: E402
from dolfinx import default_scalar_type, fem, mesh as dmesh      # noqa: E402
from dolfinx.fem import petsc as _fp                             # noqa: E402
from dolfinx.fem.petsc import LinearProblem                      # noqa: E402
from mpi4py import MPI                                           # noqa: E402

cfg = W.load_cfg()
if cfg["side"] != "neumann":
    sys.exit("the C13 solid is the Neumann-type side; SPARTA cannot take a "
             "flux")
LS, HH = cfg["ls"], cfg["h"]
KS, T_HOT = cfg["ks"], cfg["t_hot"]
NX, NY = cfg.get("n", [40, 24])

domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[0.0, 0.0], [LS, HH]],
                                [NX, NY], dmesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 1))
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
XY = V.tabulate_dof_coordinates()[:, :2]

inode = np.where(np.abs(XY[:, 0] - LS) < 1e-12)[0]
inode = inode[np.argsort(XY[inode, 1])]
ipts = XY[inode]
hot = np.where(np.abs(XY[:, 0]) < 1e-12)[0]

imp = W.read_imports(cfg["partner"])
# the gas's etot at the interface wall, applied UNCHANGED
q = W.sample(imp, "normal_fluxes", ipts, 0.0, 1, [1]).ravel()

facets = dmesh.locate_entities_boundary(domain, fdim,
                                        lambda x: np.isclose(x[0], LS))
tags = dmesh.meshtags(domain, fdim, np.sort(facets),
                      np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)

u_, v_ = ufl.TrialFunction(V), ufl.TestFunction(V)
gq = fem.Function(V)
gq.x.array[:] = 0.0
gq.x.array[inode] = q
a_form = KS * ufl.inner(ufl.grad(u_), ufl.grad(v_)) * ufl.dx
L_form = gq * v_ * ds_if
bcs = [fem.dirichletbc(default_scalar_type(T_HOT), hot.astype(np.int32), V)]
uh = LinearProblem(a_form, L_form, bcs=bcs, petsc_options_prefix="c13",
                   petsc_options={"ksp_type": "preonly",
                                  "pc_type": "lu"}).solve()

# consistent reaction at the x = 0 face: r = A u - b, unconstrained
Amat = _fp.assemble_matrix(fem.form(a_form))
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_form))
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)
# residual on the x=0 rows = -int_{x=0} (k grad T . n_out) phi ds; heat flows
# IN through x=0, so k grad T . n_out < 0 there and r > 0. Total heat entering
# = sum r_i; the mean flux density divides by the height.
flux_solid = float(np.sum(r.array[hot])) / HH

T_if = uh.x.array[inode]
theta = (float(np.mean(T_if)) - cfg["t_cold"]) / (T_HOT - cfg["t_cold"])

W.write_nodes("nodes.csv", XY, uh.x.array.reshape(-1, 1))
W.write_log(cfg, V.dofmap.index_map.size_global,
            f"num_cells = "
            f"{domain.topology.index_map(domain.topology.dim).size_local}\n"
            f"FLUX_SOLID = {flux_solid:.8g}\n"
            f"THETA_INTERFACE = {theta:.8g}")
print(f"[fenics {cfg['sidename']} neumann] NDOF="
      f"{V.dofmap.index_map.size_global} T_if_mean={np.mean(T_if):.4f} "
      f"theta={theta:.5f} flux_solid={flux_solid:.6g} "
      f"q_applied_mean={np.mean(q):.6g}")
W.write_exports(ipts, T_if.reshape(-1, 1), np.zeros((len(inode), 1)),
                "temperature")
