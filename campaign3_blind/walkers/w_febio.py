"""FEBio 4 path-walk participant: plane-strain elasticity, either role.

FEBio has no scripting API -- it is XML in, log files out -- so this is a wrapper
that writes a complete .feb deck each coupling iteration, runs febio4, parses the
ASCII logfile and writes exports.json.

WHY EVERY FEBio CELL IS MECHANICAL. FEBio 4 has no heat module (FEBioHeat was
removed upstream and exists only as a plugin), so a cell with FEBio on one side
cannot be a conduction problem at all. That is why C12, DUNE + FEBio, forced a
DUNE VECTOR participant to be written: neither shipped participant could serve it.

WHY THE MANUFACTURED DISPLACEMENT IS SMALL. FEBio has no small-strain material
either; ``isotropic elastic`` is a finite-deformation law solved by a Newton
iteration. The instances are scaled to strains of order 1e-5, so the geometric
terms are ~1e-10 relative -- seven orders below the discretisation error -- and
FEBio reproduces the linear solution the other codes compute. That scale is a
property of the problem, not a fix applied here: linear elasticity IS a
small-strain theory and a displacement of order 0.3 on a unit body is outside it.

PLANE STRAIN. A slab one element thick in z with every z-displacement
constrained. The trilinear hex then reduces exactly to bilinear plane strain, and
the z = 0 layer carries the 2-D field.

THE SPATIALLY VARYING BODY FORCE, which no shipped participant needs, is a
``body force`` load whose ``force`` is a ``math`` valuator -- FEBio evaluates the
expression in X, Y, Z at the integration points, so the source is integrated
rather than lumped. The material density is 1 because FEBio multiplies the body
force by density on assembly.

THE INTERFACE TRACTION. FEBio's nodal reaction is not exposed in a form whose
sign convention can be relied on, so the Dirichlet side recovers the CONSISTENT
traction the same way the 4C wrapper does: the plane-strain Q1 system is
re-assembled in numpy on FEBio's OWN mesh and evaluated at FEBio's OWN nodal
displacements, which is post-processing and not a second solve. Whether that
re-assembly IS FEBio's discretisation is CHECKED, not assumed: the residual must
vanish at every interior node, and the number is printed and logged.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
if dim != 2 or cfg["physics"] != "vector":
    sys.exit("the FEBio walk participant serves 2-D plane-strain elasticity only")
lam, mu = float(cfg["lam"]), float(cfg["mu"])
E = mu * (3 * lam + 2 * mu) / (lam + mu)
nu = lam / (2 * (lam + mu))
free_axis = 1 - axis
S = W.outward_sign(ext, axis, xi)
FEBIO = cfg.get("febio_bin",
                "/home/alexander/Schreibtisch/febio-src/cbuild/bin/febio4")
TZ = float(cfg.get("thickness", 0.125))
DECK, LOG_U = "cpl.feb", "cpl_node.csv"


def _n(v):
    """Full-precision XML number. NEVER repr()/!r: numpy 2 scalars stringify as
    'np.float64(0.0)' and FEBio rejects the deck."""
    return format(float(v), ".17g")


# ── mesh: one hex8 layer ──────────────────────────────────────────────
ax = [np.linspace(lo, hi, k + 1) for (lo, hi), k in zip(ext, n)]
nxp, nyp = len(ax[0]), len(ax[1])
P = np.array([(ax[0][i], ax[1][j]) for j in range(nyp) for i in range(nxp)],
             float)


def nid(i, j, k):
    return 1 + i + nxp * j + nxp * nyp * k


nodes = []
for k in range(2):
    for j in range(nyp):
        for i in range(nxp):
            nodes.append((nid(i, j, k), ax[0][i], ax[1][j], k * TZ))
elems = []
for j in range(n[1]):
    for i in range(n[0]):
        elems.append((len(elems) + 1,
                      [nid(i, j, 0), nid(i + 1, j, 0), nid(i + 1, j + 1, 0),
                       nid(i, j + 1, 0), nid(i, j, 1), nid(i + 1, j, 1),
                       nid(i + 1, j + 1, 1), nid(i, j + 1, 1)]))

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
# THE INTERFACE ENDS BELONG TO THE OUTER BOUNDARY ON BOTH SIDES.
inode_bc = np.array([i for i in inode if not outer[i]], int)


def _both(idx):
    """The 2-D node index as its two FEBio ids (z = 0 and z = TZ layers)."""
    return [int(idx) + 1, int(idx) + 1 + nxp * nyp]


imp = W.read_imports(cfg["partner"])
# THE BODY FORCE ENTERS FEBio's RESIDUAL WITH THE OPPOSITE SIGN.
#
# FEElasticSolidDomain::BodyForce assembles fa = -H * density * f * J0 into the
# load vector, so a `body force` of f solves div(sigma) = rho*f, i.e.
# -div(sigma) = -rho*f. The manufactured problem is -div(sigma) = f, so the deck
# must carry -f. MEASURED rather than argued: with the exact interface datum
# imposed and the sign as printed, the RMS error against the manufactured field
# is 5.14e-08 against a solution whose own RMS is 3.75e-08 -- 137% wrong, and
# the coupled run converges cleanly to it (C11 gave a flat error of 3.3e-08 at
# every level, observed order -0.14). Negated, the same run gives 5.82e-10,
# which is 1.6% at h = 1/8 and falls under refinement.
def _mexpr(e):
    return ("-(" + str(e).replace("**", "^") + ")").replace(
        "x", "X").replace("y", "Y")


fx, fy = (_mexpr(e) for e in cfg["source"])

L = ['<?xml version="1.0" encoding="ISO-8859-1"?>',
     '<febio_spec version="4.0">',
     '  <Module type="solid"/>',
     '  <Control><analysis>STATIC</analysis><time_steps>1</time_steps>'
     '<step_size>1</step_size>'
     '<solver type="solid"><symmetric_stiffness>symmetric</symmetric_stiffness>'
     '<dtol>1e-14</dtol><etol>1e-14</etol><rtol>0</rtol>'
     '<max_refs>25</max_refs></solver></Control>',
     '  <Globals><Constants><T>0</T><R>0</R><Fc>0</Fc></Constants></Globals>',
     f'  <Material><material id="1" name="Mat" type="isotropic elastic">'
     f'<density>1</density><E>{_n(E)}</E><v>{_n(nu)}</v></material></Material>',
     '  <Mesh>', '    <Nodes name="Object1">']
for (i, xx, yy, zz) in nodes:
    L.append(f'      <node id="{i}">{_n(xx)},{_n(yy)},{_n(zz)}</node>')
L.append('    </Nodes>')
L.append('    <Elements type="hex8" mat="1" name="Part1">')
for (e, conn) in elems:
    L.append(f'      <elem id="{e}">{",".join(str(c) for c in conn)}</elem>')
L.append('    </Elements>')
L.append('    <NodeSet name="all_nodes">'
         + ",".join(str(i) for (i, *_ ) in nodes) + '</NodeSet>')
outer_ids = [i for k in np.where(outer)[0] for i in _both(k)]
L.append('    <NodeSet name="outer">' + ",".join(map(str, outer_ids))
         + '</NodeSet>')
iface_ids = [i for k in inode_bc for i in _both(k)]
L.append('    <NodeSet name="iface">' + ",".join(map(str, iface_ids))
         + '</NodeSet>')
L.append('    <NodeSet name="iface_out">'
         + ",".join(str(_both(k)[0]) for k in inode) + '</NodeSet>')
# the interface facets, as quad4 on the slab
faces = []
for m in range(n[free_axis]):
    if axis == 0:
        i = 0 if abs(ext[0][0] - xi) < 1e-9 else n[0]
        faces.append([nid(i, m, 0), nid(i, m + 1, 0),
                      nid(i, m + 1, 1), nid(i, m, 1)])
    else:
        j = 0 if abs(ext[1][0] - xi) < 1e-9 else n[1]
        faces.append([nid(m, j, 0), nid(m + 1, j, 0),
                      nid(m + 1, j, 1), nid(m, j, 1)])
L.append('    <Surface name="iface_surf">')
for fi, f in enumerate(faces, 1):
    L.append(f'      <quad4 id="{fi}">{",".join(str(c) for c in f)}</quad4>')
L.append('    </Surface>')
L.append('  </Mesh>')
L.append('  <MeshDomains><SolidDomain name="Part1" mat="Mat"/></MeshDomains>')

if cfg["side"] == "dirichlet":
    g = W.sample(imp, "values", ipts, 0.0, 2, [free_axis])
    gmap = {int(k): g[a] for a, k in enumerate(inode)}
    L.append('  <MeshData>')
    for c, nm in ((0, "ux_map"), (1, "uy_map")):
        L.append(f'    <NodeData name="{nm}" node_set="iface" '
                 f'data_type="scalar">')
        lid = 1
        for k in inode_bc:
            for _ in _both(k):
                L.append(f'      <node lid="{lid}">{_n(gmap[int(k)][c])}</node>')
                lid += 1
        L.append('    </NodeData>')
    L.append('  </MeshData>')
else:
    q = W.sample(imp, "normal_fluxes", ipts, 0.0, 2, [free_axis])
    # per-FACE traction: FEBio's <traction> takes one vec3 per facet, so the
    # imported nodal values are averaged onto each interface quad4. The
    # partner's number is applied UNCHANGED -- both sides export with respect to
    # their own outward normal.
    tf = 0.5 * (q[:-1] + q[1:])
    L.append('  <MeshData>')
    L.append('    <SurfaceData name="t_map" surface="iface_surf" '
             'data_type="vec3">')
    for lid, tv in enumerate(tf, 1):
        L.append(f'      <face lid="{lid}">{_n(tv[0])},{_n(tv[1])},0</face>')
    L.append('    </SurfaceData>')
    L.append('  </MeshData>')

L.append('  <Boundary>')
L.append('    <bc name="planar" type="zero displacement" node_set="all_nodes">'
         '<x_dof>0</x_dof><y_dof>0</y_dof><z_dof>1</z_dof></bc>')
L.append('    <bc name="outer" type="zero displacement" node_set="outer">'
         '<x_dof>1</x_dof><y_dof>1</y_dof><z_dof>1</z_dof></bc>')
if cfg["side"] == "dirichlet":
    for dof, nm in (("x", "ux_map"), ("y", "uy_map")):
        L.append(f'    <bc name="iface_u{dof}" type="prescribed displacement" '
                 f'node_set="iface"><dof>{dof}</dof>'
                 f'<value lc="1" type="map">{nm}</value>'
                 f'<relative>0</relative></bc>')
L.append('  </Boundary>')

L.append('  <Loads>')
L.append('    <body_load type="body force">')
L.append(f'      <force type="math">{fx},{fy},0</force>')
L.append('    </body_load>')
if cfg["side"] == "neumann":
    L.append('    <surface_load name="iface_t" type="traction" '
             'surface="iface_surf">')
    L.append('      <traction type="map">t_map</traction>')
    L.append('      <scale lc="1">1.0</scale><linear>0</linear>')
    L.append('    </surface_load>')
L.append('  </Loads>')
L.append('  <LoadData><load_controller id="1" type="loadcurve">'
         '<interpolate>LINEAR</interpolate><extend>CONSTANT</extend>'
         '<points><pt>0,0</pt><pt>1,1</pt></points>'
         '</load_controller></LoadData>')
L.append('  <Output><logfile>')
L.append(f'    <node_data data="ux;uy" delim="," file="{LOG_U}" '
         f'node_set="all_nodes"/>')
L.append('  </logfile></Output>')
L.append('</febio_spec>')
Path(DECK).write_text("\n".join(L) + "\n")

Path(LOG_U).unlink(missing_ok=True)
r = subprocess.run([FEBIO, "-i", DECK], capture_output=True, text=True,
                   timeout=3600)
Path("febio_stdout.log").write_text((r.stdout or "")[-200000:])
if "N O R M A L   T E R M I N A T I O N" not in (r.stdout or ""):
    sys.exit(f"FEBio did not terminate normally (rc={r.returncode})\n"
             f"{(r.stdout or '')[-3000:]}")


def parse_log(path):
    """FEBio ASCII logfile: '*Step' / '*Data =' blocks, then 'id,v1,v2'.
    Returns {id: [values]} for the LAST step in the file."""
    out = {}
    txt = Path(path).read_text()
    body = txt.split("*Step")[-1]
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            out[int(float(parts[0]))] = [float(v) for v in parts[1:]]
        except ValueError:
            continue
    return out


ux = parse_log(LOG_U)
if not ux:
    sys.exit("empty FEBio logfile output")
U = np.array([ux[i + 1] for i in range(len(P))], float)[:, :2]


def _consistent_traction():
    """Q1 plane-strain reaction on FEBio's own mesh at FEBio's own solution."""
    import scipy.sparse as sp
    hx = (ext[0][1] - ext[0][0]) / n[0]
    hy = (ext[1][1] - ext[1][0]) / n[1]
    gl = np.array([-0.9061798459386640, -0.5384693101056831, 0.0,
                   0.5384693101056831, 0.9061798459386640])
    wl = np.array([0.2369268850561891, 0.4786286704993665, 0.5688888888888889,
                   0.4786286704993665, 0.2369268850561891])
    g2 = [-1 / np.sqrt(3.0), 1 / np.sqrt(3.0)]
    D = np.array([[lam + 2 * mu, lam, 0.0],
                  [lam, lam + 2 * mu, 0.0],
                  [0.0, 0.0, mu]])
    Ke = np.zeros((8, 8))
    for a in g2:
        for b in g2:
            dN = 0.25 * np.array([[-(1 - b), (1 - b), (1 + b), -(1 + b)],
                                  [-(1 - a), -(1 + a), (1 + a), (1 - a)]])
            J = np.diag([hx / 2, hy / 2])
            G = np.linalg.solve(J, dN)
            B = np.zeros((3, 8))
            for q_ in range(4):
                B[0, 2 * q_] = G[0, q_]
                B[1, 2 * q_ + 1] = G[1, q_]
                B[2, 2 * q_] = G[1, q_]
                B[2, 2 * q_ + 1] = G[0, q_]
            Ke += B.T @ D @ B * np.linalg.det(J)
    fun = W.make_vec_fun(cfg["source"], 2)
    N2 = len(P) * 2
    rows, cols, vals = [], [], []
    bvec = np.zeros(N2)
    for j in range(n[1]):
        for i in range(n[0]):
            loc = [i + nxp * j, (i + 1) + nxp * j,
                   (i + 1) + nxp * (j + 1), i + nxp * (j + 1)]
            gdof = [2 * m + c for m in loc for c in (0, 1)]
            for r_ in range(8):
                for c_ in range(8):
                    rows.append(gdof[r_]); cols.append(gdof[c_])
                    vals.append(Ke[r_, c_])
            for a, wa in zip(gl, wl):
                for b, wb in zip(gl, wl):
                    Nsh = 0.25 * np.array([(1 - a) * (1 - b), (1 + a) * (1 - b),
                                           (1 + a) * (1 + b), (1 - a) * (1 + b)])
                    xg = ax[0][i] + hx * (a + 1) / 2
                    yg = ax[1][j] + hy * (b + 1) / 2
                    fv = fun(xg, yg)
                    for m in range(4):
                        for c in (0, 1):
                            bvec[gdof[2 * m + c]] += (Nsh[m] * float(fv[c])
                                                      * wa * wb * hx * hy / 4)
    A = sp.coo_matrix((vals, (rows, cols)), shape=(N2, N2)).tocsr()
    res = A @ U.ravel() - bvec
    r2 = res.reshape(-1, 2)
    free = ~(outer | ifm)
    scale = max(float(np.max(np.abs(r2[ifm]))), 1e-300)
    interior = float(np.max(np.abs(r2[free]))) / scale
    hf = hy if axis == 0 else hx
    w = np.full(len(inode), hf)
    w[0] = w[-1] = 0.5 * hf
    Q = -r2[inode] / w[:, None]
    bad = outer[inode]
    good = np.where(~bad)[0]
    for i in np.where(bad)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]
    return Q, interior


note = ""
if cfg["side"] == "dirichlet":
    Q, interior = _consistent_traction()
    note = f"reassembly_interior_residual_rel = {interior:.3e}"
    if interior > 1e-4:
        print(f"[febio] WARNING: the re-assembled operator is not FEBio's: "
              f"interior residual {interior:.3e} of the interface scale.",
              file=sys.stderr)
else:
    # The Dirichlet partner reads this side's VALUES, not its traction.
    Q = np.zeros((len(inode), 2))

W.write_nodes("nodes.csv", P, U)
W.write_log(cfg, 2 * len(P), f"number of elements = {len(elems)}\n" + note)
print(f"[febio {cfg['sidename']} {cfg['side']}] NDOF={2 * len(P)} "
      f"iface_n={len(inode)} u=[{U.min():.6g},{U.max():.6g}] "
      f"t=[{Q.min():.6g},{Q.max():.6g}] {note}")
W.write_exports(ipts, U[inode], Q, "displacement")
