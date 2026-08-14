"""4C thermo-structure path-walk participant (C1): the DIRICHLET side.

WHAT DOES NOT SHIP. participant_fourc.py is scalar conduction only. C1's
subdomain solves BOTH a heat equation and an elasticity equation whose stress
carries a thermal term, and its interface transmits temperature, heat flux,
displacement and traction together. 4C has the machinery -- PROBLEMTYPE
Thermo_Structure_Interaction with MAT_Struct_ThermoStVenantK, whose stress is
C : (eps - alpha (T - T0) I), i.e. sigma_el - beta T I with
beta = (3 lambda + 2 mu) alpha and INITTEMP = 0. That is C1's constitutive law
exactly; the cell's shared beta with jumping mu means alpha differs per
subdomain, which the task states.

TWO 4C RUNS PER COUPLING ITERATION, for two reasons that were measured.

Run 1 -- Scalar_Transport, 2-D, for the temperature and its interface flux.
   The heat equation in this cell is independent of u, so it can be solved by
   the verified scalar path, and 4C's OWN consistent boundary flux is available
   there: CALCFLUX_BOUNDARY "diffusive" plus a SCATRA FLUX CALC LINE condition
   on the interface. That is the same reaction construction every participant
   here uses -- NOT the projected `flux_domain`, which was measured on C2 to
   drift the coupled field order to 1.593, below the 1.6 band edge.

Run 2 -- Thermo_Structure_Interaction for the displacement.
   COUPALGO tsi_oneway (the coupling in this cell IS one-way), Statics both
   fields, KINEM linear. Every TSI test 4C ships is 3-D SOLIDSCATRA HEX8; a
   2-D TSI path exists in the grammar but no test exercises it, so this runs
   the well-trodden road: a ONE-ELEMENT-THICK 3-D slab with u_z = 0 on every
   node and z-invariant data, which reduces the trilinear hex EXACTLY to
   bilinear plane strain (the same construction the FEBio participant uses).

TRACTION EXPORT. 4C exposes no mechanical reactions in its VTU, so the
consistent traction is recovered the way the C2/FEBio walkers do it: the 2-D
coupled thermoelastic Q1 system (K_uu, the -beta*T*div(v) coupling block, and
the load) is re-assembled in numpy on 4C's OWN mesh and evaluated at 4C's OWN
solution -- post-processing, not a second solve -- and the interior residual is
required to vanish and is LOGGED, so a modelling mismatch shows up as a number
rather than as a quietly wrong traction. The thermal flux from the same
re-assembly is cross-checked against 4C's boundary-flux field and their
difference is logged too.

This participant serves the DIRICHLET role only: C1 prescribes A = 4C as the
Dirichlet side, and no cell puts 4C on the Neumann side of a thermoelastic
exchange.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
if cfg["physics"] != "thermoelastic" or dim != 2:
    sys.exit("w_fourc_tsi serves the 2-D thermoelastic pair only")
if cfg["side"] != "dirichlet":
    sys.exit("w_fourc_tsi serves the DIRICHLET role only; no cell puts 4C on "
             "the Neumann side of a thermoelastic exchange")
kval = float(cfg["k"])
lam, mu, beta = float(cfg["lam"]), float(cfg["mu"]), float(cfg["beta"])
E_ = mu * (3 * lam + 2 * mu) / (lam + mu)
nu_ = lam / (2 * (lam + mu))
alpha = beta / (3 * lam + 2 * mu)
free_axis = 1 - axis
S = W.outward_sign(ext, axis, xi)
BIN = cfg.get("fourc_bin", "/home/alexander/4C/build/4C")
LD = cfg.get("fourc_ld", "/opt/4C-dependencies/lib")
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = LD + os.pathsep + ENV.get("LD_LIBRARY_PATH", "")

# ── the 2-D node/element layout, shared by both runs and the re-assembly ──
ax = [np.linspace(lo, hi, k + 1) for (lo, hi), k in zip(ext, n)]
nx1, ny1 = len(ax[0]), len(ax[1])
nid = np.arange(nx1 * ny1).reshape(ny1, nx1).T + 1        # (i, j) -> 1-based
P = np.array([(ax[0][i], ax[1][j]) for j in range(ny1) for i in range(nx1)])
hx = (ext[0][1] - ext[0][0]) / n[0]
hy = (ext[1][1] - ext[1][0]) / n[1]
TZ = min(hx, hy)                                          # slab thickness

ifm = np.abs(P[:, axis] - xi) < 1e-9
outer = np.zeros(len(P), bool)
for a in range(2):
    for val in ext[a]:
        if a == axis and abs(val - xi) < 1e-9:
            continue
        outer |= np.abs(P[:, a] - val) < 1e-9
inode = np.where(ifm)[0]
inode = inode[np.argsort(P[inode, free_axis])]
ipts = P[inode]

imp = W.read_imports(cfg["partner"])
g = W.sample(imp, "values", ipts, 0.0, 3, [free_axis])    # (T, ux, uy)


def _f4(e):
    return str(e).replace("**", "^")


def run4c(deck: str, tag: str):
    Path(f"input_{tag}.4C.yaml").write_text(deck)
    # stdbuf is not optional with 4C: without line buffering an MPI_Abort eats
    # the error message and the failure arrives as an empty log.
    r = subprocess.run(["stdbuf", "-oL", "-eL", BIN, f"input_{tag}.4C.yaml",
                       f"out_{tag}"], capture_output=True, text=True, env=ENV,
                       timeout=3600)
    Path(f"fourc_{tag}.log").write_text(r.stdout[-200000:] + "\n"
                                        + r.stderr[-20000:])
    return r


def read_vtu(pattern: str, tag: str):
    import meshio
    vtus = sorted(Path(f"out_{tag}-vtk-files").glob(pattern))
    if not vtus:
        sys.exit(f"4C run {tag!r} produced no VTU matching {pattern!r}; see "
                 f"fourc_{tag}.log")

    def _step(p):
        # <field>-00001-0.vtu -> 1. The TRAILING number is the MPI RANK, not
        # the step: matching the last number returns the initial condition.
        m = re.match(r".*-(\d+)-\d+\.vtu$", p.name)
        return int(m.group(1)) if m else -1
    return meshio.read(str(max(vtus, key=_step)))


def collapse(m, field, comps, zlayer=None):
    """Nodal field from a 4C VTU onto the 2-D node layout.

    4C repeats every node once per element; a 3-D slab additionally has two
    z-layers. Average duplicates; for the slab take the z ~ 0 layer (the field
    is z-invariant, which the re-assembly's interior check would expose if it
    were not)."""
    pts = np.asarray(m.points)
    val = np.asarray(m.point_data[field])
    if val.ndim == 1:
        val = val.reshape(-1, 1)
    sel = np.ones(len(pts), bool)
    if zlayer is not None:
        sel = np.abs(pts[:, 2] - zlayer) < 1e-9
    key = {}
    for row in np.where(sel)[0]:
        key.setdefault(tuple(np.round(pts[row, :2], 9)), []).append(row)
    out = np.zeros((len(P), comps))
    for i, p in enumerate(P):
        rows = key.get(tuple(np.round(p, 9)))
        if not rows:
            sys.exit(f"4C VTU field {field!r} has no node at {p}")
        out[i] = val[rows, :comps].mean(axis=0)
    return out


# ══════════════════════════════════════════════════════════════════════
# RUN 1 — Scalar_Transport for T, with 4C's own consistent boundary flux
# ══════════════════════════════════════════════════════════════════════
nodes2d = [f"NODE {nid[i, j]} COORD {ax[0][i]:.15f} {ax[1][j]:.15f} 0.0"
           for j in range(ny1) for i in range(nx1)]
elems2d = []
for j in range(n[1]):
    for i in range(n[0]):
        elems2d.append(f"{len(elems2d) + 1} TRANSP QUAD4 {nid[i, j]} "
                       f"{nid[i + 1, j]} {nid[i + 1, j + 1]} {nid[i, j + 1]} "
                       f"MAT 1 TYPE Std")

point_dirich, dnode_top = [], []
d = 1
for k_, i in enumerate(inode):
    if outer[i]:
        continue          # the interface ends keep the OUTER datum, both sides
    point_dirich.append(f"  - E: {d}\n    NUMDOF: 1\n    ONOFF: [1]\n"
                        f"    VAL: [{g[k_, 0]:.17g}]\n    FUNCT: [0]\n")
    dnode_top.append(f'  - "NODE {i + 1} DNODE {d}"\n')
    d += 1

deck1 = f"""TITLE:
  - "C1 walk, run 1: scalar transport with consistent boundary flux"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  CALCFLUX_BOUNDARY: "diffusive"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: "ascii"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {kval}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{_f4(cfg['source_T'])}"
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN POINT DIRICH CONDITIONS:
{''.join(point_dirich)}"""
deck1 += ("DESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n"
          "    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]\n")
deck1 += ("SCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2\n")
deck1 += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in nodes2d)
deck1 += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{s}"\n' for s in elems2d)
deck1 += "DLINE-NODE TOPOLOGY:\n"
deck1 += "".join(f'  - "NODE {i + 1} DLINE 1"\n' for i in np.where(outer)[0])
deck1 += "".join(f'  - "NODE {i + 1} DLINE 2"\n' for i in inode)
deck1 += "DNODE-NODE TOPOLOGY:\n" + "".join(dnode_top)
deck1 += "DSURF-NODE TOPOLOGY:\n" + "".join(
    f'  - "NODE {i + 1} DSURFACE 1"\n' for i in range(len(P)))

r1 = run4c(deck1, "scatra")
m1 = read_vtu("scatra-*-0.vtu", "scatra")
T2d = collapse(m1, "phi_1", 1)[:, 0]
if "flux_boundary_phi_1" not in m1.point_data:
    sys.exit("run 1 wrote no flux_boundary_phi_1 field; CALCFLUX_BOUNDARY or "
             "the SCATRA FLUX CALC condition did not take")
QB = collapse(m1, "flux_boundary_phi_1", 2)
# The boundary-flux field is the consistent NORMAL flux as a vector along the
# outward normal, so its projection on n_out is the exported density.
q4c = QB[inode, axis] * S

# ══════════════════════════════════════════════════════════════════════
# RUN 2 — one-way TSI on the one-element-thick slab, for the displacement
# ══════════════════════════════════════════════════════════════════════
def nid3(i, j, k):
    return 1 + i + nx1 * j + nx1 * ny1 * k


nodes3d, coords3 = [], {}
for k in range(2):
    for j in range(ny1):
        for i in range(nx1):
            nodes3d.append(f"NODE {nid3(i, j, k)} COORD {ax[0][i]:.15f} "
                           f"{ax[1][j]:.15f} {k * TZ:.15f}")
elems3d = []
for j in range(n[1]):
    for i in range(n[0]):
        c = [nid3(i, j, 0), nid3(i + 1, j, 0), nid3(i + 1, j + 1, 0),
             nid3(i, j + 1, 0), nid3(i, j, 1), nid3(i + 1, j, 1),
             nid3(i + 1, j + 1, 1), nid3(i, j + 1, 1)]
        elems3d.append(f"{len(elems3d) + 1} SOLIDSCATRA HEX8 "
                       f"{' '.join(map(str, c))} MAT 1 KINEM linear "
                       f"TYPE Undefined")


def both_layers(i2d):
    i, j = int(i2d % nx1), int(i2d // nx1)
    return [nid3(i, j, 0), nid3(i, j, 1)]


pd_struct, pd_thermo, dnode3 = [], [], []
d = 1
for k_, i in enumerate(inode):
    if outer[i]:
        continue
    for n3 in both_layers(i):
        pd_struct.append(f"  - E: {d}\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n"
                         f"    VAL: [{g[k_, 1]:.17g}, {g[k_, 2]:.17g}, 0.0]\n"
                         f"    FUNCT: [0, 0, 0]\n")
        pd_thermo.append(f"  - E: {d}\n    NUMDOF: 1\n    ONOFF: [1]\n"
                         f"    VAL: [{g[k_, 0]:.17g}]\n    FUNCT: [0]\n")
        dnode3.append(f'  - "NODE {n3} DNODE {d}"\n')
        d += 1

outer3 = [n3 for i in np.where(outer)[0] for n3 in both_layers(i)]

deck2 = f"""TITLE:
  - "C1 walk, run 2: one-way TSI, statics, one-element-thick plane-strain slab"
PROBLEM SIZE:
  DIM: 3
IO:
  STRUCT_STRESS: "No"
  STRUCT_STRAIN: "No"
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1
  NUMSTEP: 1
  MAXTIME: 1
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  NUMSTEP: 1
  MAXTIME: 1
  TIMESTEP: 1
  ITEMAX: 1
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: "ascii"
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E_:.17g}]
      NUE: {nu_:.17g}
      DENS: 1
      THEXPANS: {alpha:.17g}
      INITTEMP: 0
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 1
      CONDUCT:
        constant: [{kval}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{_f4(cfg['source_u'][0])}"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{_f4(cfg['source_u'][1])}"
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{_f4(cfg['source_T'])}"
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1.0, 1.0, 0.0]
    FUNCT: [1, 2, 0]
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [3]
DESIGN VOL DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0]
    FUNCT: [0]
DESIGN POINT DIRICH CONDITIONS:
{''.join(pd_struct)}DESIGN POINT THERMO DIRICH CONDITIONS:
{''.join(pd_thermo)}"""
deck2 += "NODE COORDS:\n" + "".join(f'  - "{s}"\n' for s in nodes3d)
deck2 += "STRUCTURE ELEMENTS:\n" + "".join(f'  - "{s}"\n' for s in elems3d)
deck2 += "DNODE-NODE TOPOLOGY:\n" + "".join(dnode3)
deck2 += "DSURF-NODE TOPOLOGY:\n" + "".join(
    f'  - "NODE {n3} DSURFACE 1"\n' for n3 in outer3)
deck2 += "DVOL-NODE TOPOLOGY:\n" + "".join(
    f'  - "NODE {i} DVOL 1"\n' for i in range(1, 2 * nx1 * ny1 + 1))

r2 = run4c(deck2, "tsi")
ms = read_vtu("structure-*-0.vtu", "tsi")
U2d = collapse(ms, "displacement", 2, zlayer=0.0)
mt = read_vtu("thermo-*-0.vtu", "tsi")
Ttsi = collapse(mt, "temperature", 1, zlayer=0.0)[:, 0]

# The two runs solve the same discrete heat problem with the same direct
# solver; if their temperatures disagree, one deck is not the problem it
# claims to be. Checked, not assumed.
dT = float(np.max(np.abs(Ttsi - T2d)) / max(np.max(np.abs(T2d)), 1e-300))
if dT > 1e-8:
    print(f"[4C tsi] WARNING: run 1 and run 2 temperatures differ by "
          f"{dT:.3e} relative", file=sys.stderr)


# ══════════════════════════════════════════════════════════════════════
# Consistent traction by re-assembly (and the flux cross-check)
# ══════════════════════════════════════════════════════════════════════
def reassemble():
    import scipy.sparse as sp
    g2 = [-1 / np.sqrt(3.0), 1 / np.sqrt(3.0)]
    gl = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                   0.5384693101056831, 0.9061798459386640])
    wl = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                   0.4786286704993665, 0.2369268850561891])
    D = np.array([[lam + 2 * mu, lam, 0.0],
                  [lam, lam + 2 * mu, 0.0],
                  [0.0, 0.0, mu]])
    fT = W.make_fun(cfg["source_T"], 2)
    fu = W.make_vec_fun(cfg["source_u"], 2)
    NP = len(P)
    rowsK, colsK, valsK = [], [], []      # K_TT
    rowsU, colsU, valsU = [], [], []      # K_uu
    rowsC, colsC, valsC = [], [], []      # C_uT: -beta * int T div(v)
    bT, bu = np.zeros(NP), np.zeros(2 * NP)
    Ke_T = np.zeros((4, 4))
    Ke_u = np.zeros((8, 8))
    Ce = np.zeros((8, 4))
    for a in g2:
        for b in g2:
            dN = 0.25 * np.array([[-(1 - b), (1 - b), (1 + b), -(1 + b)],
                                  [-(1 - a), -(1 + a), (1 + a), (1 - a)]])
            Nsh = 0.25 * np.array([(1 - a) * (1 - b), (1 + a) * (1 - b),
                                   (1 + a) * (1 + b), (1 - a) * (1 + b)])
            J = np.diag([hx / 2, hy / 2])
            G = np.linalg.solve(J, dN)
            detJ = np.linalg.det(J)
            Ke_T += kval * (G.T @ G) * detJ
            B = np.zeros((3, 8))
            for q in range(4):
                B[0, 2 * q] = G[0, q]
                B[1, 2 * q + 1] = G[1, q]
                B[2, 2 * q] = G[1, q]
                B[2, 2 * q + 1] = G[0, q]
            Ke_u += B.T @ D @ B * detJ
            # -beta * int N_T,j * div(v_i): div picks row 0 of B for ux dofs
            # and row 1 for uy dofs
            divv = np.zeros(8)
            for q in range(4):
                divv[2 * q] = G[0, q]
                divv[2 * q + 1] = G[1, q]
            Ce += -beta * np.outer(divv, Nsh) * detJ
    for j in range(n[1]):
        for i in range(n[0]):
            loc = [i + nx1 * j, (i + 1) + nx1 * j,
                   (i + 1) + nx1 * (j + 1), i + nx1 * (j + 1)]
            gu = [2 * m + c for m in loc for c in (0, 1)]
            for r_ in range(4):
                for c_ in range(4):
                    rowsK.append(loc[r_])
                    colsK.append(loc[c_])
                    valsK.append(Ke_T[r_, c_])
            for r_ in range(8):
                for c_ in range(8):
                    rowsU.append(gu[r_])
                    colsU.append(gu[c_])
                    valsU.append(Ke_u[r_, c_])
                for c_ in range(4):
                    rowsC.append(gu[r_])
                    colsC.append(loc[c_])
                    valsC.append(Ce[r_, c_])
            x0, y0 = ax[0][i], ax[1][j]
            for a, wa in zip(gl, wl):
                for b, wb in zip(gl, wl):
                    Nsh = 0.25 * np.array([(1 - a) * (1 - b), (1 + a) * (1 - b),
                                           (1 + a) * (1 + b), (1 - a) * (1 + b)])
                    xg, yg = x0 + hx * (a + 1) / 2, y0 + hy * (b + 1) / 2
                    wq = wa * wb * hx * hy / 4
                    bT[loc] += Nsh * float(fT(xg, yg)) * wq
                    fv = fu(xg, yg)
                    for m in range(4):
                        bu[gu[2 * m]] += Nsh[m] * float(fv[0]) * wq
                        bu[gu[2 * m + 1]] += Nsh[m] * float(fv[1]) * wq
    NPp = len(P)
    K_TT = sp.coo_matrix((valsK, (rowsK, colsK)), shape=(NPp, NPp)).tocsr()
    K_uu = sp.coo_matrix((valsU, (rowsU, colsU)),
                         shape=(2 * NPp, 2 * NPp)).tocsr()
    C_uT = sp.coo_matrix((valsC, (rowsC, colsC)),
                         shape=(2 * NPp, NPp)).tocsr()
    rT = K_TT @ T2d - bT
    ru = (K_uu @ U2d.ravel() + C_uT @ T2d - bu).reshape(-1, 2)
    free = ~(outer | ifm)
    scT = max(float(np.max(np.abs(rT[ifm]))), 1e-300)
    scu = max(float(np.max(np.abs(ru[ifm]))), 1e-300)
    intT = float(np.max(np.abs(rT[free]))) / scT
    intu = float(np.max(np.abs(ru[free]))) / scu
    hf = hy if axis == 0 else hx
    w = np.full(len(inode), hf)
    w[0] = w[-1] = 0.5 * hf
    qT = -rT[inode] / w
    tu = -ru[inode] / w[:, None]
    bad = outer[inode]
    good = np.where(~bad)[0]
    for k_ in np.where(bad)[0]:
        qT[k_] = qT[good[np.argmin(np.abs(good - k_))]]
        tu[k_] = tu[good[np.argmin(np.abs(good - k_))]]
    return qT, tu, intT, intu


qre, tre, intT, intu = reassemble()
# 4C's own boundary flux vs the re-assembled reaction, on the graded interior
# of the interface (the ends carry the outer reaction too and are excluded).
inner_ = ~outer[inode]
dq = float(np.max(np.abs(q4c[inner_] - qre[inner_]))
           / max(np.max(np.abs(qre[inner_])), 1e-300))
note = (f"reassembly_interior_residual_rel_T = {intT:.3e}\n"
        f"reassembly_interior_residual_rel_u = {intu:.3e}\n"
        f"fourc_boundary_flux_vs_reassembly_rel = {dq:.3e}")
if intu > 1e-4:
    print(f"[4C tsi] WARNING: the re-assembled mechanical operator is not "
          f"4C's: interior residual {intu:.3e}", file=sys.stderr)

Q = np.column_stack([q4c, tre[:, 0], tre[:, 1]])
V = np.column_stack([T2d[inode], U2d[inode, 0], U2d[inode, 1]])
allV = np.column_stack([T2d, U2d[:, 0], U2d[:, 1]])

W.write_nodes("nodes.csv", P, allV)
W.write_log(cfg, 3 * len(P), f"number of nodes = {len(P)}\n" + note)
print(f"[4C tsi {cfg['sidename']} dirichlet] NDOF={3 * len(P)} "
      f"iface_n={len(inode)} T=[{T2d.min():.6g},{T2d.max():.6g}] "
      f"u=[{U2d.min():.6g},{U2d.max():.6g}] {note.replace(chr(10), '  ')}")
W.write_exports(ipts, V, Q, "thermoelastic")
