"""FEBio 4 VECTOR participant for OASiS couple driver - Subdomain A (Dirichlet side).

Plane-strain linear elasticity on [0, 0.625] x [0, 1].
Exchanges displacement and traction at interface x = 0.625.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

# ── PROBLEM PARAMETERS ─
SIDE      = "dirichlet"   # A is Dirichlet side
PARTNER   = "B"           # partner name in couple()
X0, X1    = 0.0, 0.625    # subdomain A extent
Y0, Y1    = 0.0, 1.0      # y extent
ZTHICK    = 0.05          # slab thickness for plane strain
IFACE_X   = 0.625         # shared interface
E_MOD     = 600.0         # Young's modulus (lambda=450, mu=225)
NU        = 1.0/3.0       # Poisson ratio
UI_X, UI_Y = 0.0, 0.0     # iteration-1 fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # iteration-1 fallback interface traction
FEBIO     = "/home/alexander/FEBio/bin/febio4"
LINSOLVE  = "skyline"

LAM = E_MOD * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E_MOD / (2.0 * (1.0 + NU))

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

DECK = "cpl.feb"
LOG_U = "cpl_u.csv"
LOG_R = "cpl_r.csv"


def _n(v):
    return format(float(v), ".17g")


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
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


class Mesh:
    def __init__(self, nx, ny):
        self.NX, self.NY = nx, ny
        self.xs = np.linspace(X0, X1, nx + 1)
        self.ys = np.linspace(Y0, Y1, ny + 1)
        self.zs = np.array([0.0, ZTHICK])

        fast_i = nx <= ny
        if fast_i:
            def nid(i, j, k):
                return 1 + k + 2 * i + 2 * (nx + 1) * j
        else:
            def nid(i, j, k):
                return 1 + k + 2 * j + 2 * (ny + 1) * i

        self.nid = nid
        self.nodes = [(nid(i, j, k), self.xs[i], self.ys[j], self.zs[k])
                      for k in range(2) for j in range(ny + 1)
                      for i in range(nx + 1)]
        self.nodes.sort()
        self.xyz = np.zeros((len(self.nodes) + 1, 3))
        for (n, x, y, z) in self.nodes:
            self.xyz[n] = (x, y, z)
        self.elems = []
        e = 1
        for j in range(ny):
            for i in range(nx):
                self.elems.append((e, [nid(i, j, 0), nid(i + 1, j, 0),
                                       nid(i + 1, j + 1, 0), nid(i, j + 1, 0),
                                       nid(i, j, 1), nid(i + 1, j, 1),
                                       nid(i + 1, j + 1, 1), nid(i, j + 1, 1)]))
                e += 1

        i_col = nx - 1 if ON_RIGHT else 0
        self.iface_elems = [1 + i_col + nx * j for j in range(ny)]

        i_if = nx if ON_RIGHT else 0
        self.y_if = self.ys.copy()
        self.iface_pair = [(nid(i_if, j, 0), nid(i_if, j, 1))
                           for j in range(ny + 1)]
        self.iface_all = [n for pair in self.iface_pair for n in pair]

        self.interior_j = [j for j in range(ny + 1)
                           if abs(self.ys[j] - Y0) > TOL
                           and abs(self.ys[j] - Y1) > TOL]
        self.iface_free = [n for j in self.interior_j
                           for n in self.iface_pair[j]]

        outer = set()
        i_out = 0 if ON_RIGHT else nx
        for k in range(2):
            for j in range(ny + 1):
                outer.add(nid(i_out, j, k))
            for i in range(nx + 1):
                outer.add(nid(i, 0, k))
                outer.add(nid(i, ny, k))
        self.outer = sorted(outer)

        self.faces = [[nid(i_if, j, 0), nid(i_if, j + 1, 0),
                       nid(i_if, j + 1, 1), nid(i_if, j, 1)]
                      for j in range(ny)]

    def _face_gauss(self, face):
        p = self.xyz[face]
        g = 1.0 / np.sqrt(3.0)
        out = []
        for xi in (-g, g):
            for eta in (-g, g):
                N = 0.25 * np.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                                     (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
                dNx = 0.25 * np.array([-(1 - eta), (1 - eta),
                                       (1 + eta), -(1 - eta)])
                dNe = 0.25 * np.array([-(1 - xi), -(1 + xi),
                                       (1 + xi), (1 - xi)])
                jac = np.cross(dNx @ p, dNe @ p)
                out.append((N, float(np.linalg.norm(jac))))
        return out

    def iface_weights(self):
        w = np.zeros(len(self.nodes) + 1)
        for f in self.faces:
            for N, dj in self._face_gauss(f):
                w[f] += N * dj
        return w


def write_deck(mesh, u_if):
    L = ['<?xml version="1.0" encoding="ISO-8859-1"?>',
         '<febio_spec version="4.0">',
         '  <Module type="solid"/>',
         '  <Control><analysis>STATIC</analysis><time_steps>1</time_steps>'
         '<step_size>1</step_size>'
         '<solver type="solid"><symmetric_stiffness>symmetric'
         '</symmetric_stiffness><dtol>1e-12</dtol><etol>1e-12</etol>'
         '<rtol>0</rtol>'
         f'<linear_solver type="{LINSOLVE}"/></solver></Control>',
         '  <Globals><Constants><T>0</T><R>0</R><Fc>0</Fc></Constants></Globals>',
         f'  <Material><material id="1" name="Mat" type="isotropic elastic">'
         f'<density>1</density><E>{_n(E_MOD)}</E><v>{_n(NU)}</v>'
         f'</material></Material>',
         '  <Mesh>',
         '    <Nodes name="Object1">']
    L += [f'      <node id="{n}">{_n(x)},{_n(y)},{_n(z)}</node>'
          for (n, x, y, z) in mesh.nodes]
    L += ['    </Nodes>',
          '    <Elements type="hex8" mat="1" name="Part1">']
    L += [f'      <elem id="{e}">{",".join(str(c) for c in conn)}</elem>'
          for (e, conn) in mesh.elems]
    L += ['    </Elements>',
          '    <NodeSet name="all_nodes">'
          + ",".join(str(n[0]) for n in mesh.nodes) + '</NodeSet>',
          '    <NodeSet name="outer">'
          + ",".join(str(n) for n in mesh.outer) + '</NodeSet>',
          '    <NodeSet name="iface_free">'
          + ",".join(str(n) for n in mesh.iface_free) + '</NodeSet>',
          '    <NodeSet name="iface_all">'
          + ",".join(str(n) for n in mesh.iface_all) + '</NodeSet>',
          '  </Mesh>',
          '  <MeshDomains><SolidDomain name="Part1" mat="Mat"/></MeshDomains>',
          '  <MeshData>']

    # Outer BC: u = 0
    for name, col in (("ox_map", [0.0]*len(mesh.outer)), ("oy_map", [0.0]*len(mesh.outer))):
        L.append(f'    <NodeData name="{name}" node_set="outer" '
                 f'data_type="scalar">')
        L += [f'      <node lid="{lid}">{_n(v)}</node>'
              for lid, v in enumerate(col, 1)]
        L.append('    </NodeData>')

    L.append(f'    <NodeData name="ix_map" node_set="iface_free" '
             f'data_type="scalar">')
    L += [f'      <node lid="{lid}">{_n(u_if[n][0])}</node>'
          for lid, n in enumerate(mesh.iface_free, 1)]
    L.append('    </NodeData>')

    L.append(f'    <NodeData name="iy_map" node_set="iface_free" '
             f'data_type="scalar">')
    L += [f'      <node lid="{lid}">{_n(u_if[n][1])}</node>'
          for lid, n in enumerate(mesh.iface_free, 1)]
    L.append('    </NodeData>')

    L.append('  </MeshData>')

    L += ['  <Boundary>',
          '    <bc name="planar" type="zero displacement" node_set="all_nodes">'
          '<x_dof>0</x_dof><y_dof>0</y_dof><z_dof>1</z_dof></bc>',
          '    <bc name="ox" type="prescribed displacement" node_set="outer">'
          '<dof=x</dof><value lc="1" type="map">ox_map</value>'
          '<relative>0</relative></bc>',
          '    <bc name="oy" type="prescribed displacement" node_set="outer">'
          '<dof=y</dof><value lc="1" type="map">oy_map</value>'
          '<relative>0</relative></bc>',
          '    <bc name="ix" type="prescribed displacement" '
          'node_set="iface_free">'
          '<dof=x</dof><value lc="1" type="map">ix_map</value>'
          '<relative>0</relative></bc>',
          '    <bc name="iy" type="prescribed displacement" '
          'node_set="iface_free">'
          '<dof=y</dof><value lc="1" type="map">iy_map</value>'
          '<relative>0</relative></bc>']
    L.append('  </Boundary>')

    L += ['  <LoadData><load_controller id="1" type="loadcurve">'
          '<interpolate>LINEAR</interpolate><extend>CONSTANT</extend>'
          '<points><pt>0,0</pt><pt>1,1</pt></points>'
          '</load_controller></LoadData>',
          '  <Output><logfile>',
          f'    <node_data data="ux;uy" delim="," file="{LOG_U}" '
          f'node_set="iface_all"/>',
          f'    <node_data data="Rx;Ry" delim="," file="{LOG_R}" '
          f'node_set="iface_all"/>',
          '  </logfile></Output>', '</febio_spec>']
    Path(DECK).write_text("\n".join(L) + "\n")


def parse_log(path, ncol):
    out = {}
    txt = Path(path).read_text()
    blocks = txt.split("*Step")
    body = blocks[-1] if len(blocks) > 1 else txt
    for line in body.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split(",")
        if len(parts) < ncol + 1:
            continue
        try:
            out[int(float(parts[0]))] = tuple(float(p) for p in parts[1:ncol + 1])
        except ValueError:
            continue
    return out


if __name__ == "__main__":
    # Get NX, NY from environment or defaults
    nx_env = int(Path("NX.txt").read_text().strip()) if Path("NX.txt").exists() else 5
    ny_env = int(Path("NY.txt").read_text().strip()) if Path("NY.txt").exists() else 8
    
    mesh = Mesh(nx_env, ny_env)
    imp = read_imports()
    y_if = mesh.y_if

    u_if = {}
    if SIDE == "dirichlet":
        u_line = sample(imp, "values", (UI_X, UI_Y), y_if)
        for j, (nb, nt) in enumerate(mesh.iface_pair):
            u_if[nb] = u_if[nt] = (float(u_line[j, 0]), float(u_line[j, 1]))

    write_deck(mesh, u_if)
    for f in (LOG_U, LOG_R):
        Path(f).unlink(missing_ok=True)

    r = subprocess.run([FEBIO, "-i", DECK], capture_output=True, text=True, timeout=3600)
    if "N O R M A L   T E R M I N A T I O N" not in (r.stdout or ""):
        sys.stderr.write(f"FEBio did not terminate normally (rc={r.returncode})\n"
                         f"{(r.stdout or '')[-2000:]}\n")
        sys.exit(1)

    ulog = parse_log(LOG_U, 2)
    rlog = parse_log(LOG_R, 2)
    
    if not ulog or not rlog:
        sys.stderr.write(f"empty FEBio logfile output\n")
        sys.exit(2)

    U = np.array([[0.5 * (ulog[nb][c] + ulog[nt][c]) for c in (0, 1)]
                  for (nb, nt) in mesh.iface_pair], float)

    w = mesh.iface_weights()
    R = np.array([[rlog[nb][c] + rlog[nt][c] for c in (0, 1)]
                  for (nb, nt) in mesh.iface_pair], float)
    W = np.array([w[nb] + w[nt] for (nb, nt) in mesh.iface_pair], float)
    Q = np.zeros_like(R)
    ok = np.abs(W) > 1e-14
    Q[ok] = -R[ok] / W[ok, None]

    good = np.array(mesh.interior_j, dtype=int)
    good = good[ok[good]]
    if len(good):
        for j in range(len(y_if)):
            if j not in good:
                Q[j] = Q[good[np.argmin(np.abs(good - j))]]

    # Write NDOF to log file
    ndof = len(mesh.nodes) * 2  # 2 DOFs per node (x, y displacement, z is constrained)
    with open("run.log", "w") as f:
        f.write(f"NDOF = {ndof}\n")

    print(f"[febio A] interface n={len(U)} "
          f"ux=[{U[:,0].min():.6g},{U[:,0].max():.6g}] "
          f"uy=[{U[:,1].min():.6g},{U[:,1].max():.6g}] "
          f"tx=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}] "
          f"ty=[{Q[:,1].min():.6g},{Q[:,1].max():.6g}]")

    Path("exports.json").write_text(json.dumps({
        "field_name": "displacement",
        "n_points": int(len(y_if)),
        "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
        "values": [[float(a_), float(b_)] for a_, b_ in U],
        "normal_fluxes": [[float(a_), float(b_)] for a_, b_ in Q],
    }, indent=2))
