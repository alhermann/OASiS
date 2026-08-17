"""scikit-fem VECTOR participant for OASiS couple driver - Subdomain B (Neumann side).

Plane-strain linear elasticity on [0.625, 1.5] x [0, 1].
Exchanges displacement and traction at interface x = 0.625.
"""
import json
from pathlib import Path

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, ElementVector,
                   FacetBasis, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

# ── PROBLEM PARAMETERS ─
SIDE      = "neumann"     # B is Neumann side
PARTNER   = "A"           # partner name in couple()
X0, X1    = 0.625, 1.5    # subdomain B extent
Y0, Y1    = 0.0, 1.0      # y extent
IFACE_X   = 0.625         # shared interface
E_MOD     = 2100.0        # Young's modulus (lambda=450, mu=900)
NU        = 1.0/6.0       # Poisson ratio
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))   # plane strain
MU = E_MOD / (2.0 * (1.0 + NU))

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-min?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, y):
    """Map the partner's VECTOR samples onto this participant's y-coordinates,
    COMPONENT BY COMPONENT."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, c])
                            for c in range(vs.shape[1])])


def u_dirichlet(x, y):
    """The prescribed displacement on the non-interface boundary (u=0)."""
    return (np.zeros_like(x), np.zeros_like(y))


@BilinearForm
def stiffness(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return 2.0 * MU * ddot(eu, ev) + LAM * trace(eu) * trace(ev)


@BilinearForm
def mass(u, v, w):
    return u[0] * v[0] + u[1] * v[1]


@LinearForm
def traction(v, w):
    return w["t"][0] * v[0] + w["t"][1] * v[1]


@LinearForm
def proj_rhs(v, w):
    """L2 projection of q_out = -(sigma . n_own) onto the vector P1 space."""
    g = w["uh"].grad                       # g[i][j] = du_i/dx_j
    exx, eyy = g[0][0], g[1][1]
    exy = 0.5 * (g[0][1] + g[1][0])
    sxx = 2.0 * MU * exx + LAM * (exx + eyy)
    sxy = 2.0 * MU * exy
    return (-S) * (sxx * v[0] + sxy * v[1])


@LinearForm
def unit_load(v, w):
    """w_i = int_Gamma phi_i ds."""
    return 1.0 * v[0] + 1.0 * v[1]


if __name__ == "__main__":
    # Get NX, NY from environment or defaults
    nx_env = int(Path("NX.txt").read_text().strip()) if Path("NX.txt").exists() else 7
    ny_env = int(Path("NY.txt").read_text().strip()) if Path("NY.txt").exists() else 8
    
    imp = read_imports()

    mesh = MeshTri.init_tensor(np.linspace(X0, X1, nx_env + 1),
                               np.linspace(Y0, Y1, ny_env + 1))
    elem = ElementVector(ElementTriP1())
    basis = Basis(mesh, elem)
    nd = basis.nodal_dofs                      # (2, nnodes): node -> (x, y) dof

    px, py = mesh.p[0], mesh.p[1]
    iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
    iface_n = iface_n[np.argsort(py[iface_n])]             # sorted by y
    y_if = py[iface_n]
    
    # Outer boundary: all nodes NOT on interface
    outer_n = np.where((np.abs(px - OUTER_X) < TOL) |
                       (np.abs(py - Y0) < TOL) | (np.abs(py - Y1) < TOL))[0]
    
    # THE TWO INTERFACE CORNERS BELONG TO THE OUTER BOUNDARY, ON BOTH SIDES.
    iface_bc_n = iface_n[(np.abs(py[iface_n] - Y0) > TOL) &
                         (np.abs(py[iface_n] - Y1) > TOL)]
    iface_bc_dofs = np.concatenate([nd[0, iface_bc_n], nd[1, iface_bc_n]])
    outer_dofs = np.concatenate([nd[0, outer_n], nd[1, outer_n]])

    A = stiffness.assemble(basis)          # UNCONSTRAINED
    b = basis.zeros()
    fbi = FacetBasis(mesh, elem,
                     facets=mesh.facets_satisfying(
                         lambda p: np.abs(p[0] - IFACE_X) < TOL))

    sol = basis.zeros()
    ux_d, uy_d = u_dirichlet(px[outer_n], py[outer_n])
    sol[nd[0, outer_n]] = ux_d
    sol[nd[1, outer_n]] = uy_d
    D = outer_dofs

    if SIDE == "dirichlet":
        u_if = sample(imp, "values", (UI_X, UI_Y), y_if)
        keep = (np.abs(y_if - Y0) > TOL) & (np.abs(y_if - Y1) > TOL)
        sol[nd[0, iface_bc_n]] = u_if[keep, 0]
        sol[nd[1, iface_bc_n]] = u_if[keep, 1]
        D = np.unique(np.concatenate([outer_dofs, iface_bc_dofs]))
    else:
        t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), y_if)
        gnod = basis.zeros()                   # P1 trace of the partner's samples
        gnod[nd[0, iface_n]] = t_if[:, 0]
        gnod[nd[1, iface_n]] = t_if[:, 1]
        # APPLY the partner's numbers UNCHANGED (+ integral(g . v) ds_interface)
        b = b + asm(traction, fbi, t=fbi.interpolate(gnod))

    sol = solve(*condense(A, b, x=sol, D=D))

    # Interface traction export q_out = -(sigma . n_own).
    if SIDE == "dirichlet":
        r = A @ sol - b                        # r = A u_h - b, no bc applied
        wgt = unit_load.assemble(fbi)          # w_i = int_Gamma phi_i ds

        idx = np.column_stack([nd[0, iface_n], nd[1, iface_n]])
        wi = wgt[idx]
        Q = np.zeros_like(wi)
        ok = np.abs(wi) > 1e-14
        Q[ok] = -r[idx][ok] / wi[ok]

        suspect = np.isin(iface_n, outer_n) | ~ok.all(axis=1)
        good = np.where(~suspect)[0]
        if len(good):
            for i in np.where(suspect)[0]:
                Q[i] = Q[good[np.argmin(np.abs(good - i))]]
    else:
        # NEUMANN SIDE: use stress projection
        qh = solve(mass.assemble(basis),
                   proj_rhs.assemble(basis, uh=basis.interpolate(sol)))
        Q = np.column_stack([qh[nd[0, iface_n]], qh[nd[1, iface_n]]])

    # Write NDOF to log file
    ndof = len(mesh.p[0]) * 2  # 2 DOFs per node
    with open("run.log", "w") as f:
        f.write(f"NDOF = {ndof}\n")

    print(f"[skfem B] interface n={len(iface_n)} "
          f"ux=[{sol[nd[0,iface_n]].min():.6g},{sol[nd[0,iface_n]].max():.6g}] "
          f"uy=[{sol[nd[1,iface_n]].min():.6g},{sol[nd[1,iface_n]].max():.6g}] "
          f"tx=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}] "
          f"ty=[{Q[:,1].min():.6g},{Q[:,1].max():.6g}]")

    Path("exports.json").write_text(json.dumps({
        "field_name": "displacement",
        "n_points": int(len(iface_n)),
        "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
        "values": [[float(sol[nd[0, i]]), float(sol[nd[1, i]])] for i in iface_n],
        "normal_fluxes": [[float(q0), float(q1)] for q0, q1 in Q],
    }, indent=2))
