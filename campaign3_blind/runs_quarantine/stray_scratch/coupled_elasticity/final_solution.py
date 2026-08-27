#!/usr/bin/env python3
"""Final coupled elasticity simulation."""
import json
import numpy as np
from pathlib import Path
import subprocess

PYTHON = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
BASE_DIR = Path("/tmp/coupled_elasticity")
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

MESH_LEVELS = [
    {"level": 1, "NX_A": 8, "NY_A": 5, "NX_B": 8, "NY_B": 7},
    {"level": 2, "NX_A": 16, "NY_A": 10, "NX_B": 16, "NY_B": 14},
    {"level": 3, "NX_A": 32, "NY_A": 20, "NX_B": 32, "NY_B": 28},
]

INTERFACE_PROBE = np.array([( 0.25+(i+0.5)/(2*44), 5/8 ) for i in range(44)])


def create_participant_A(level_config, work_dir):
    NX, NY = level_config["NX_A"], level_config["NY_A"]
    level = level_config["level"]
    
    script = f'''#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

SIDE = "neumann"
PARTNER = "B"
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.0, 0.625
IFACE_Y = 0.625
LAMBDA, MU = 480.0, 1200.0
NX, NY = {NX}, {NY}
TOL = 1e-9

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER)
    except:
        return None

def sample_vector(imp, key, fallback, x_coords):
    if not imp or not imp.get("coordinates"):
        return np.full((len(x_coords), 2), float(fallback))
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float)
    if vs.ndim == 1:
        return np.column_stack([np.interp(x_coords, xs, vs), np.zeros_like(x_coords)])
    result = np.zeros((len(x_coords), 2))
    for comp in range(2):
        result[:, comp] = np.interp(x_coords, xs, vs[:, comp])
    return result

imp = read_imports()

geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1), bcs=("outer", "outer", "interface", "outer"))
mesh = Mesh(geo.GenerateMesh(maxh=min((X1-X0)/NX, (Y1-Y0)/NY)))

fes = VectorH1(mesh, order=1, dirichlet="outer")
u, v = fes.TnT()

def Strain(u):
    return 0.5 * (grad(u) + grad(u).trans)

def Stress(u):
    eps = Strain(u)
    return 2 * MU * eps + LAMBDA * Trace(eps) * Id(2)

a = BilinearForm(fes)
a += InnerProduct(Stress(u), Strain(v)) * dx

f = LinearForm(fes)
f += CoefficientFunction((-63*x**4*y/3125 + 99*x**4/5000 + 168*x**3*y/15625 - 33*x**3/3125 + 
                          216*x**2*y**2/625 - 1557*x**2*y/3125 - 99*x**2/5000 - 288*x*y**2/3125 + 
                          1992*x*y/15625 + 33*x/3125 - 36*y**2/625 + 54*y/625)) * v[0] * dx
f += CoefficientFunction(-108*x**5/15625 + 72*x**4/15625 - 18*x**3*y**2/625 + 153*x**3*y/1250 - 
                         279*x**3/3125 + 36*x**2*y**2/3125 - 153*x**2*y/3125 + 486*x**2/15625 + 
                         9*x*y**2/625 - 153*x*y/2500 + 63*x/1250 - 12*y**2/3125 + 51*y/3125 - 
                         42/3125) * v[1] * dx

vdof = np.array([[fes.GetDofNrs(NodeId(VERTEX, i))[c] for c in range(2)] for i in range(mesh.nv)], int)
vxy = np.array([mesh.vertices[i].point for i in range(mesh.nv)], float)

iface_v = np.where(np.abs(vxy[:, 1] - IFACE_Y) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 0])]
x_if = vxy[iface_v, 0]

iface_dofs_x = vdof[iface_v, 0]
iface_dofs_y = vdof[iface_v, 1]

if SIDE == "neumann":
    tx_if = sample_vector(imp, "normal_fluxes", 0.0, x_if)
    gfun = GridFunction(fes)
    for d, t in zip(iface_dofs_x, tx_if[:, 0]):
        gfun.vec[int(d)] = float(t)
    for d, t in zip(iface_dofs_y, tx_if[:, 1]):
        gfun.vec[int(d)] = float(t)
    f += gfun[0] * v[0] * ds("interface")
    f += gfun[1] * v[1] * ds("interface")

gfu = GridFunction(fes)
gfu.vec[:] = 0.0

with TaskManager():
    a.Assemble()
    f.Assemble()
    res = f.vec.CreateVector()
    res.data = f.vec - a.mat * gfu.vec
    gfu.vec.data += a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * res

ux_if = np.array([gfu.vec[int(d)] for d in iface_dofs_x], float)
uy_if = np.array([gfu.vec[int(d)] for d in iface_dofs_y], float)
disp_if = np.column_stack([ux_if, uy_if])

fesq_x = H1(mesh, order=1)
fesq_y = H1(mesh, order=1)
px, wx = fesq_x.TnT()
py, wy = fesq_y.TnT()

mx = BilinearForm(fesq_x); mx += px * wx * dx
my = BilinearForm(fesq_y); my += py * wy * dx

ftx = LinearForm(fesq_x); ftx += MU * (grad(gfu)[0, 1] + grad(gfu)[1, 0]) * wx * dx
fty = LinearForm(fesq_y); fty += (LAMBDA * (grad(gfu)[0, 0] + grad(gfu)[1, 1]) + 2*MU*grad(gfu)[1, 1]) * wy * dx

with TaskManager():
    mx.Assemble(); my.Assemble(); ftx.Assemble(); fty.Assemble()

qh_tx = GridFunction(fesq_x)
qh_ty = GridFunction(fesq_y)
qh_tx.vec.data = mx.mat.Inverse(fesq_x.FreeDofs(), inverse="sparsecholesky") * ftx.vec
qh_ty.vec.data = my.mat.Inverse(fesq_y.FreeDofs(), inverse="sparsecholesky") * fty.vec

tx_export = np.array([qh_tx(mesh(*(mesh.vertices[vi].point))) for vi in iface_v])
ty_export = np.array([qh_ty(mesh(*(mesh.vertices[vi].point))) for vi in iface_v])
traction_if = np.column_stack([tx_export, ty_export])

Path("exports.json").write_text(json.dumps({{
    "field_name": "displacement",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(xx), float(IFACE_Y)] for xx in x_if],
    "values": disp_if.tolist(),
    "normal_fluxes": traction_if.tolist(),
}}))

# Extract at probe points
probe_points = [( (i_x+0.5)/44, (i_y+0.5)*0.625/44 ) for i_y in range(44) for i_x in range(44)]
probe_results = []
for x, y in probe_points:
    ux = gfu.components[0](mesh(x, y))
    uy = gfu.components[1](mesh(x, y))
    probe_results.append([x, y, ux, uy])

Path(f"/tmp/coupled_elasticity/output/solution_level{level}_A.csv").write_text(
    "x,y,ux,uy\\n" + "\\n".join([f"{{r[0]}},{{r[1]}},{{r[2]}},{{r[3]}}" for r in probe_results])
)

Path("run.log").write_text(f"NDOF = {{fes.ndof}}\\nElements = {{mesh.ne}}\\n")
print(f"A: {{fes.ndof}} DOFs, {{len(iface_v)}} iface pts")
'''
    work_dir.mkdir(exist_ok=True)
    (work_dir / "participant_A.py").write_text(script)
    return work_dir / "participant_A.py"


def create_participant_B(level_config, work_dir):
    NX, NY = level_config["NX_B"], level_config["NY_B"]
    level = level_config["level"]
    
    script = f'''#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from skfem import (Basis, ElementTriP1, FacetBasis, LinearForm, MeshTri, 
                   condense, solve, ElementVector)
from skfem.models.elasticity import linear_elasticity

SIDE = "dirichlet"
PARTNER = "A"
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.625, 1.5
IFACE_Y = 0.625
LAMBDA, MU = 480.0, 240.0
NX, NY = {NX}, {NY}
TOL = 1e-9

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER)
    except:
        return None

def sample_vector(imp, key, fallback, x_coords):
    if not imp or not imp.get("coordinates"):
        return np.full((len(x_coords), 2), float(fallback))
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float)
    if vs.ndim == 1:
        return np.column_stack([np.interp(x_coords, xs, vs), np.zeros_like(x_coords)])
    result = np.zeros((len(x_coords), 2))
    for comp in range(2):
        result[:, comp] = np.interp(x_coords, xs, vs[:, comp])
    return result

imp = read_imports()

mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1), np.linspace(Y0, Y1, NY + 1))
_tol = 1e-10
mesh = mesh.with_boundaries({{
    'left': lambda x: x[0] < _tol,
    'right': lambda x: x[0] > 1.0 - _tol,
    'top': lambda x: x[1] > Y1 - _tol,
    'interface': lambda x: x[1] < Y0 + _tol,
}})

elem = ElementVector(ElementTriP1())
basis = Basis(mesh, elem)
n2d = basis.nodal_dofs
px, py = mesh.p[0], mesh.p[1]

iface_n = np.where(np.abs(py - IFACE_Y) < TOL)[0]
iface_n = iface_n[np.argsort(px[iface_n])]
x_if = px[iface_n]

iface_dofs_x = n2d[0, iface_n]
iface_dofs_y = n2d[1, iface_n]

outer_mask = (np.abs(px - X0) < TOL) | (np.abs(px - X1) < TOL) | (np.abs(py - Y1) < TOL)
outer_n = np.where(outer_mask)[0]
outer_dofs = np.concatenate([n2d[0, outer_n], n2d[1, outer_n]])

K = linear_elasticity(LAMBDA, MU).assemble(basis)

@LinearForm
def body_force(v, w):
    x, y = w['x'][0], w['x'][1]
    fx = (1737*x**4*y/30625 - 9357*x**4/245000 - 4632*x**3*y/153125 + 3119*x**3/153125 + 
          396*x**2*y**2/875 - 57969*x**2*y/61250 + 86847*x**2/245000 - 528*x*y**2/4375 + 
          40962*x*y/153125 - 16034*x/153125 - 66*y**2/875 + 519*y/3500 - 369/7000)
    fy = (2316*x**5/153125 - 1544*x**4/153125 + 1158*x**3*y**2/30625 + 9201*x**3*y/61250 - 
          53561*x**3/245000 - 2316*x**2*y**2/153125 - 9201*x**2*y/153125 + 59737*x**2/612500 - 
          579*x*y**2/30625 - 9201*x*y/122500 + 9477*x/98000 + 772*y**2/153125 + 
          3067*y/153125 - 3159/122500)
    return fx * v[0] + fy * v[1]

f = body_force.assemble(basis)

sol = basis.zeros()
if SIDE == "dirichlet":
    disp_if = sample_vector(imp, "values", 0.0, x_if)
    sol[iface_dofs_x] = disp_if[:, 0]
    sol[iface_dofs_y] = disp_if[:, 1]
    D = np.concatenate([outer_dofs, iface_dofs_x, iface_dofs_y])
else:
    D = outer_dofs

sol = solve(*condense(K, f, x=sol, D=D))

r = K @ sol - f

dx_if = np.diff(x_if)
wgt_x = np.zeros(len(iface_n))
wgt_x[0] = dx_if[0] / 2 if len(dx_if) > 0 else 1.0/NX
wgt_x[-1] = dx_if[-1] / 2 if len(dx_if) > 0 else 1.0/NX
wgt_x[1:-1] = (dx_if[:-1] + dx_if[1:]) / 4 if len(dx_if) > 1 else 1.0/NX

tx_export = np.zeros(len(iface_n))
ty_export = np.zeros(len(iface_n))
for i in range(len(iface_n)):
    if abs(wgt_x[i]) > 1e-14:
        tx_export[i] = -r[iface_dofs_x[i]] / wgt_x[i]
        ty_export[i] = -r[iface_dofs_y[i]] / wgt_x[i]

corner_mask = (np.abs(x_if - X0) < TOL) | (np.abs(x_if - X1) < TOL)
interior_mask = ~corner_mask
if np.any(interior_mask) and np.any(corner_mask):
    interior_indices = np.where(interior_mask)[0]
    for i in np.where(corner_mask)[0]:
        nearest = interior_indices[np.argmin(np.abs(interior_indices - i))]
        tx_export[i] = tx_export[nearest]
        ty_export[i] = ty_export[nearest]

traction_if = np.column_stack([tx_export, ty_export])
disp_if = np.column_stack([sol[iface_dofs_x], sol[iface_dofs_y]])

Path("exports.json").write_text(json.dumps({{
    "field_name": "displacement",
    "n_points": int(len(iface_n)),
    "coordinates": [[float(xx), float(IFACE_Y)] for xx in x_if],
    "values": disp_if.tolist(),
    "normal_fluxes": traction_if.tolist(),
}}))

# Extract at probe points using projection onto test functions
# For P1 elements, we can evaluate by finding the element containing the point
probe_points = [( (i_x+0.5)/44, 0.625+(i_y+0.5)*0.875/44 ) for i_y in range(44) for i_x in range(44)]
probe_results = []

# Get nodal solution
nodal_sol = sol.reshape(2, -1)  # (2, n_nodes)

for x, y in probe_points:
    # Find nearest node for simple interpolation
    dists = (mesh.p[0] - x)**2 + (mesh.p[1] - y)**2
    nearest_node = np.argmin(dists)
    ux = nodal_sol[0, nearest_node]
    uy = nodal_sol[1, nearest_node]
    probe_results.append([x, y, ux, uy])

Path(f"/tmp/coupled_elasticity/output/solution_level{level}_B.csv").write_text(
    "x,y,ux,uy\\n" + "\\n".join([f"{{r[0]}},{{r[1]}},{{r[2]}},{{r[3]}}" for r in probe_results])
)

Path("run.log").write_text(f"NDOF = {{K.shape[0]}}\\nElements = {{mesh.nelements}}\\n")
print(f"B: {{K.shape[0]}} DOFs, {{len(iface_n)}} iface pts")
'''
    work_dir.mkdir(exist_ok=True)
    (work_dir / "participant_B.py").write_text(script)
    return work_dir / "participant_B.py"


def run_coupling(level_config):
    level = level_config["level"]
    work_dir_A = BASE_DIR / f"level{level}_A"
    work_dir_B = BASE_DIR / f"level{level}_B"
    
    script_A = create_participant_A(level_config, work_dir_A)
    script_B = create_participant_B(level_config, work_dir_B)
    
    print(f"\nLevel {level}: NX_A={level_config['NX_A']}, NY_A={level_config['NY_A']}, "
          f"NX_B={level_config['NX_B']}, NY_B={level_config['NY_B']}")
    
    max_iter = 100
    tol = 1e-6
    theta = 0.5
    
    prev_A = prev_B = None
    residual_history = []
    
    for iteration in range(max_iter):
        (work_dir_A / "imports.json").write_text(json.dumps({"B": prev_B} if prev_B else {}))
        (work_dir_B / "imports.json").write_text(json.dumps({"A": prev_A} if prev_A else {}))
        
        for script, wdir in [(script_A, work_dir_A), (script_B, work_dir_B)]:
            result = subprocess.run([PYTHON, str(script)], cwd=wdir, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Failed: {result.stderr[:300]}")
                return None
        
        export_A = json.loads((work_dir_A / "exports.json").read_text())
        export_B = json.loads((work_dir_B / "exports.json").read_text())
        
        if prev_A:
            res_A = np.linalg.norm(np.array(export_A["values"]) - np.array(prev_A["values"])) / \
                   (np.linalg.norm(np.array(prev_A["values"])) + 1e-15)
            res_B = np.linalg.norm(np.array(export_B["values"]) - np.array(prev_B["values"])) / \
                   (np.linalg.norm(np.array(prev_B["values"])) + 1e-15)
            residual = max(res_A, res_B)
        else:
            residual = float('inf')
        
        residual_history.append(residual)
        if iteration % 10 == 0 or residual < tol:
            print(f"  Iter {iteration+1}: residual = {residual:.6e}")
        
        if prev_A:
            export_A["values"] = ((1-theta)*np.array(prev_A["values"]) + 
                                  theta*np.array(export_A["values"])).tolist()
            export_B["values"] = ((1-theta)*np.array(prev_B["values"]) + 
                                  theta*np.array(export_B["values"])).tolist()
        
        prev_A, prev_B = export_A, export_B
        
        if residual < tol:
            print(f"  Converged at iteration {iteration+1}")
            break
    
    with open(OUTPUT_DIR / f"residual_level{level}.csv", 'w') as f:
        f.write("iteration,interface_residual\n")
        for i, res in enumerate(residual_history):
            f.write(f"{i+1},{res}\n")
    
    for side, wdir in [("A", work_dir_A), ("B", work_dir_B)]:
        if (wdir / "run.log").exists():
            (OUTPUT_DIR / f"run_level{level}_{side}.log").write_text(
                (wdir / "run.log").read_text())
    
    for side, export in [("A", export_A), ("B", export_B)]:
        coords = np.array(export["coordinates"])
        values = np.array(export["values"])
        fluxes = np.array(export["normal_fluxes"])
        
        with open(OUTPUT_DIR / f"interface_level{level}_{side}.csv", 'w') as f:
            f.write("x,y,ux,uy,tx,ty\n")
            for x, y in INTERFACE_PROBE:
                idx = np.argmin(np.abs(coords[:, 0] - x))
                ux, uy = values[idx]
                tx, ty = fluxes[idx]
                f.write(f"{x},{y},{ux},{uy},{tx},{ty}\n")
    
    return {
        "level": level,
        "converged": residual < tol,
        "iterations": len(residual_history),
        "final_residual": residual_history[-1] if residual_history else float('inf'),
        "export_A": export_A,
        "export_B": export_B,
    }


def main():
    print("="*60)
    print("Coupled Elasticity Simulation")
    print("="*60)
    
    results = []
    for level_config in MESH_LEVELS:
        result = run_coupling(level_config)
        if result:
            results.append(result)
    
    if results:
        finest = results[-1]
        files = []
        for lc in MESH_LEVELS:
            l = lc["level"]
            files.extend([
                f"solution_level{l}_A.csv", f"solution_level{l}_B.csv",
                f"interface_level{l}_A.csv", f"interface_level{l}_B.csv",
                f"residual_level{l}.csv", f"run_level{l}_A.log", f"run_level{l}_B.log",
            ])
        
        if len(results) >= 2:
            prev_coords = np.array(results[-2]["export_A"]["coordinates"])
            curr_coords = np.array(results[-1]["export_A"]["coordinates"])
            prev_vals = np.array(results[-2]["export_A"]["values"])
            curr_vals = np.array(results[-1]["export_A"]["values"])
            
            common_x = np.intersect1d(prev_coords[:, 0], curr_coords[:, 0])
            if len(common_x) > 0:
                prev_interp = np.array([np.interp(common_x, prev_coords[:, 0], prev_vals[:, i]) 
                                       for i in range(2)]).T
                curr_interp = np.array([np.interp(common_x, curr_coords[:, 0], curr_vals[:, i]) 
                                       for i in range(2)]).T
                max_rel = np.max(np.abs(curr_interp - prev_interp) / 
                                (np.linalg.norm(prev_interp) + 1e-15))
            else:
                max_rel = 0.01
            mesh_indep = "CONVERGED" if max_rel < 0.01 else "NOT_CONVERGED"
        else:
            mesh_indep = "NOT_CONVERGED"
            max_rel = 0.01
        
        result_txt = f"""LEVELS = {len(results)}
FILES = {",".join(files)}
INTERFACE_RESIDUAL = {finest["final_residual"]:.6e}
COUPLING_ITERATIONS = {finest["iterations"]}
MESH_INDEPENDENCE = {mesh_indep}
MAX_REL_CHANGE = {max_rel:.6e}
"""
        (OUTPUT_DIR / "RESULT.txt").write_text(result_txt)
        print(f"\nResults written to {OUTPUT_DIR / 'RESULT.txt'}")
        print(result_txt)


if __name__ == "__main__":
    main()
