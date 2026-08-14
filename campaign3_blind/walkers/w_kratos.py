"""Kratos Multiphysics path-walk participant: scalar conduction, EITHER role, 2-D or 3-D.

WHAT DOES NOT SHIP. ``data/coupling_participants/participant_kratos.py`` is
Dirichlet-only. Two cells need Kratos on the NEUMANN side, and one of them (the
1:1000 contrast) cannot converge in the other role at all -- measured on D6, 60
iterations at the theoretically optimal relaxation factor with the residual stuck
at 0.988. The Neumann branch below is written for that, with ThermalFace
conditions carrying FACE_HEAT_FLUX; the condition is registered in this Kratos
build in both 2-D and 3-D.

Also new here: the 3-D branch. LaplacianElement3D4N exists and the 3-D interface
is a PLANE, so the partner must be sampled on a two-dimensional set -- see
wcommon.sample, which falls back to bilinear interpolation on the partner's
structured interface grid. The 1-D interpolation the shipped participant uses
would return numbers of the right length and the wrong values, silently.

THE SOURCE TERM. Kratos's ConvectionDiffusion takes a volumetric source as the
NODAL variable HEAT_FLUX, so the manufactured source enters as its P1
interpolant. That is a consistent variational crime of the same order as the
discretisation and leaves the L2 error O(h^2).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

import KratosMultiphysics as KM                                  # noqa: E402
import KratosMultiphysics.ConvectionDiffusionApplication         # noqa: E402,F401

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
if cfg["physics"] != "scalar":
    sys.exit("the Kratos walk participant serves scalar conduction only")
K = np.asarray(cfg["K"], float)
if not np.allclose(K, K[0, 0] * np.eye(dim)):
    sys.exit("Kratos ConvectionDiffusion takes a SCALAR conductivity; this cell "
             "carries a tensor")
kval = float(K[0, 0])
free_axes = [i for i in range(dim) if i != axis]
S = W.outward_sign(ext, axis, xi)
fsrc = W.make_fun(cfg["source"], dim)

model = KM.Model()
mp = model.CreateModelPart("thermal")
mp.ProcessInfo[KM.DOMAIN_SIZE] = dim
st = KM.ConvectionDiffusionSettings()
st.SetUnknownVariable(KM.TEMPERATURE)
st.SetDiffusionVariable(KM.CONDUCTIVITY)
st.SetVolumeSourceVariable(KM.HEAT_FLUX)
st.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, st)
for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
          KM.REACTION_FLUX):
    mp.AddNodalSolutionStepVariable(v)
mp.SetBufferSize(1)
props = mp.CreateNewProperties(1)

axes = [np.linspace(lo, hi, k + 1) for (lo, hi), k in zip(ext, n)]
shape = tuple(len(a) for a in axes)
nid = np.zeros(shape, int)
P = []
c = 1
if dim == 2:
    for i in range(shape[0]):
        for j in range(shape[1]):
            mp.CreateNewNode(c, float(axes[0][i]), float(axes[1][j]), 0.0)
            nid[i, j] = c
            P.append((axes[0][i], axes[1][j]))
            c += 1
    e = 1
    for i in range(n[0]):
        for j in range(n[1]):
            a, b = nid[i, j], nid[i + 1, j]
            d_, f_ = nid[i + 1, j + 1], nid[i, j + 1]
            mp.CreateNewElement("LaplacianElement2D3N", e, [a, b, f_], props)
            e += 1
            mp.CreateNewElement("LaplacianElement2D3N", e, [b, d_, f_], props)
            e += 1
else:
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                mp.CreateNewNode(c, float(axes[0][i]), float(axes[1][j]),
                                 float(axes[2][k]))
                nid[i, j, k] = c
                P.append((axes[0][i], axes[1][j], axes[2][k]))
                c += 1
    # A cube split into six tetrahedra by the main diagonal (0,7): the standard
    # Kuhn triangulation, which is conforming across every face.
    KUHN = [(0, 1, 3, 7), (0, 1, 7, 5), (0, 5, 7, 4),
            (0, 3, 2, 7), (0, 6, 4, 7), (0, 2, 6, 7)]
    CORN = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0),
            (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
    e = 1
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                cn = [nid[i + a, j + b, k + d] for (a, b, d) in CORN]
                for t in KUHN:
                    mp.CreateNewElement("LaplacianElement3D4N", e,
                                        [int(cn[q]) for q in t], props)
                    e += 1
P = np.array(P, float)

for node in mp.Nodes:
    node.SetSolutionStepValue(KM.CONDUCTIVITY, kval)
    p = [node.X, node.Y, node.Z][:dim]
    node.SetSolutionStepValue(KM.HEAT_FLUX, float(fsrc(*p)))
    node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)

# ── interface and outer node sets ─────────────────────────────────────
ifm = np.abs(P[:, axis] - xi) < 1e-9
outer = np.zeros(len(P), bool)
for a in range(dim):
    for val in ext[a]:
        if a == axis and abs(val - xi) < 1e-9:
            continue
        outer |= np.abs(P[:, a] - val) < 1e-9
inode = np.where(ifm)[0]
inode = inode[np.lexsort(tuple(P[inode, a] for a in reversed(free_axes)))]
ipts = P[inode]

imp = W.read_imports(cfg["partner"])

for i in np.where(outer)[0]:                       # u = 0 on the whole outside
    node = mp.Nodes[i + 1]
    node.SetSolutionStepValue(KM.TEMPERATURE, 0.0)
    node.Fix(KM.TEMPERATURE)

if cfg["side"] == "dirichlet":
    g = W.sample(imp, "values", ipts, 0.0, 1, free_axes).ravel()
    for i, val in zip(inode, g):
        if outer[i]:
            continue          # the interface ends keep the OUTER datum, on both
        node = mp.Nodes[i + 1]                     # sides; see wcommon
        node.SetSolutionStepValue(KM.TEMPERATURE, float(val))
        node.Fix(KM.TEMPERATURE)
else:
    # NEUMANN: ThermalFace conditions on the interface facets, carrying the
    # partner's number UNCHANGED as FACE_HEAT_FLUX. Kratos adds
    # +int_Gamma q v ds to the residual, which is the natural term for this
    # side because the two participants export with opposite outward normals.
    q = W.sample(imp, "normal_fluxes", ipts, 0.0, 1, free_axes).ravel()
    qn = np.zeros(len(P))
    qn[inode] = q
    for i in range(len(P)):
        mp.Nodes[i + 1].SetSolutionStepValue(KM.FACE_HEAT_FLUX, float(qn[i]))
    cid = 1
    if dim == 2:
        ii = 0 if abs(ext[0][0] - xi) < 1e-9 else n[0]
        jj = 0 if abs(ext[1][0] - xi) < 1e-9 else n[1]
        for m in range(n[1 - axis]):
            if axis == 0:
                a, b = nid[ii, m], nid[ii, m + 1]
            else:
                a, b = nid[m, jj], nid[m + 1, jj]
            mp.CreateNewCondition("ThermalFace2D2N", cid, [int(a), int(b)],
                                  props)
            cid += 1
    else:
        ii = 0 if abs(ext[axis][0] - xi) < 1e-9 else n[axis]
        f0, f1 = free_axes
        for m in range(n[f0]):
            for l in range(n[f1]):
                idx = [0, 0, 0]
                idx[axis] = ii

                def _nd(du, dv):
                    q_ = list(idx)
                    q_[f0] = m + du
                    q_[f1] = l + dv
                    return int(nid[q_[0], q_[1], q_[2]])
                mp.CreateNewCondition("ThermalFace3D3N", cid,
                                      [_nd(0, 0), _nd(1, 0), _nd(1, 1)], props)
                cid += 1
                mp.CreateNewCondition("ThermalFace3D3N", cid,
                                      [_nd(0, 0), _nd(1, 1), _nd(0, 1)], props)
                cid += 1

KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
lin = (KM.SkylineLUFactorizationSolver() if len(P) < 30000
       else KM.SkylineLUFactorizationSolver())
builder = KM.ResidualBasedBlockBuilderAndSolver(lin)
# arg 4 is CalculateReactionsFlag: it must be True, the interface flux below is
# read out of the reactions.
strat = KM.ResidualBasedLinearStrategy(mp, scheme, builder, True, False, False,
                                       False)
strat.Initialize()
strat.Solve()

T = np.array([mp.Nodes[i + 1].GetSolutionStepValue(KM.TEMPERATURE)
              for i in range(len(P))], float)

if cfg["side"] == "dirichlet":
    # CONSISTENT (reaction) flux. Kratos's builder recomputes the RHS with NO
    # Dirichlet condition applied and stores (A u - b)_i at every fixed dof --
    # exactly the functional int_Gamma qn phi_i ds, up to sign. Dividing by
    # w_i = int_Gamma phi_i ds turns it into a density the partner can sample.
    # A one-sided difference quotient or an L2 projection of the gradient is
    # only O(h) on the boundary, and the boundary trace is what the coupling
    # reads.
    h = [(ext[a][1] - ext[a][0]) / n[a] for a in range(dim)]
    if dim == 2:
        hf = h[free_axes[0]]
        w = np.full(len(inode), hf)
        w[0] = w[-1] = 0.5 * hf
    else:
        w = np.ones(len(inode))
        for a in free_axes:
            e_ = np.abs(P[inode, a] - ext[a][0]) < 1e-9
            e_ |= np.abs(P[inode, a] - ext[a][1]) < 1e-9
            w *= np.where(e_, 0.5 * h[a], h[a])
    r = np.array([mp.Nodes[i + 1].GetSolutionStepValue(KM.REACTION_FLUX)
                  for i in inode], float)
    Q = -r / w
    bad = outer[inode]
    good = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]
else:
    # The Dirichlet partner reads this side's VALUES, not its flux, so a
    # difference quotient here is harmless; the reaction formula must NOT be
    # used, these dofs are free and it would export zero.
    step = 1
    off = np.array(P[inode])
    off[:, axis] -= S * (ext[axis][1] - ext[axis][0]) / n[axis]
    key = {tuple(np.round(p, 10)): i for i, p in enumerate(P)}
    near = np.array([key[tuple(np.round(p, 10))] for p in off], int)
    dxh = (ext[axis][1] - ext[axis][0]) / n[axis]
    Q = -kval * S * (T[inode] - T[near]) / dxh

W.write_nodes("nodes.csv", P, T.reshape(-1, 1))
W.write_log(cfg, len(P), f"number of elements = {mp.NumberOfElements()}")
print(f"[kratos {cfg['sidename']} {cfg['side']}] NDOF={len(P)} "
      f"iface_n={len(inode)} T=[{T.min():.6g},{T.max():.6g}] "
      f"q=[{Q.min():.6g},{Q.max():.6g}]")
W.write_exports(ipts, T[inode].reshape(-1, 1), Q.reshape(-1, 1), "temperature")
