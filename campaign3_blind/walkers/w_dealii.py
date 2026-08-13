"""deal.II path-walk participant: a wrapper around the compiled C++ solver.

deal.II has no Python API, so unlike every other backend there is a BUILD STEP
before anything can run at all. It happens on the FIRST coupling iteration only
-- the work_dir persists across iterations, so the wrapper rebuilds only when the
source is newer than the binary. A build per iteration would dominate the run.

This install is deal.II 9.8.0-pre, Release, SERIAL: no MPI, no PETSc, no
Trilinos, no p4est. Anything needing those does not configure, so the solver uses
deal.II's own SparseMatrix and UMFPACK.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wcommon as W                                              # noqa: E402

cfg = W.load_cfg()
dim, axis, xi = cfg["dim"], cfg["axis"], cfg["xi"]
ext, n = cfg["extent"], cfg["n"]
vec = cfg["physics"] == "vector"
ncomp = dim if vec else 1
free_axis = 1 - axis
if dim != 2:
    sys.exit("the deal.II walk participant serves 2-D only")

# The C++ source is staged into the work_dir alongside this wrapper, because
# the wrapper itself is copied there and cannot reach back to the walkers tree.
SRC = Path("iface_dealii.cc").resolve()
DEAL = cfg.get("dealii_dir", "/home/alexander/dealii/build")
EXE = Path("iface_dealii")


def build():
    Path("CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.13)\n"
        f"find_package(deal.II 9.0 REQUIRED HINTS {DEAL})\n"
        "deal_ii_initialize_cached_variables()\n"
        "project(iface_dealii CXX)\n"
        "add_executable(iface_dealii iface_dealii.cc)\n"
        "deal_ii_setup_target(iface_dealii)\n")
    b = Path("build")
    b.mkdir(exist_ok=True)
    for cmd in (["cmake", "-S", ".", "-B", "build",
                 f"-DDEAL_II_DIR={DEAL}", "-DCMAKE_BUILD_TYPE=Release"],
                ["cmake", "--build", "build", "-j", "4"]):
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        Path("dealii_build.log").write_text(
            (Path("dealii_build.log").read_text()
             if Path("dealii_build.log").is_file() else "")
            + "\n$ " + " ".join(cmd) + "\n" + r.stdout[-40000:] + r.stderr[-40000:])
        if r.returncode != 0:
            sys.exit(f"deal.II build failed:\n{r.stdout[-3000:]}\n"
                     f"{r.stderr[-3000:]}")
    shutil.copy("build/iface_dealii", EXE)


if not EXE.is_file() or EXE.stat().st_mtime < SRC.stat().st_mtime:
    build()

# ── the interface points this subdomain owns, and the partner's samples ──
h = [(ext[a][1] - ext[a][0]) / n[a] for a in range(dim)]
sfree = np.linspace(ext[free_axis][0], ext[free_axis][1], n[free_axis] + 1)
ipts = np.zeros((len(sfree), dim))
ipts[:, axis] = xi
ipts[:, free_axis] = sfree

TR = cfg.get("transient")
NSTEPS = int(round(TR["t_end"] / TR["dt"])) if TR else 0

imp = W.read_imports(cfg["partner"])
key = "values" if cfg["side"] == "dirichlet" else "normal_fluxes"
# WAVEFORM exchange for the transient instance: the imported object is the
# whole space-time trace, nsteps components per interface point
g = W.sample(imp, key, ipts, 0.0, NSTEPS if TR else ncomp, [free_axis])


def _mu(e):
    """sympy prints ** for powers; muparser wants ^."""
    return str(e).replace("**", "^")


if vec:
    m = [float(cfg["lam"]), float(cfg["mu"]), 0.0, 0.0]
    s0, s1 = (_mu(e) for e in cfg["source"])
    mode = 1
elif TR:
    K = np.asarray(cfg["K"], float)
    # mode 2 packs (k, dt, nsteps) into the material line
    m = [K[0, 0], float(TR["dt"]), float(NSTEPS), 0.0]
    s0, s1 = _mu(cfg["source"]), "0"
    mode = 2
else:
    K = np.asarray(cfg["K"], float)
    m = [K[0, 0], K[0, 1], K[1, 0], K[1, 1]]
    s0, s1 = _mu(cfg["source"]), "0"
    mode = 0

lines = [f"{mode} {0 if cfg['side'] == 'dirichlet' else 1} "
         f"{axis} {xi!r}",
         f"{ext[0][0]!r} {ext[0][1]!r} {ext[1][0]!r} {ext[1][1]!r} "
         f"{n[0]} {n[1]}",
         " ".join(repr(float(v)) for v in m),
         s0, s1, str(len(sfree))]
for p, row in zip(sfree, g):
    lines.append(" ".join([repr(float(p))]
                          + [repr(float(v)) for v in np.atleast_1d(row)]))
Path("in.txt").write_text("\n".join(lines) + "\n")

env = dict(os.environ)
env["LD_LIBRARY_PATH"] = ("/opt/4C-dependencies/lib" + os.pathsep
                          + env.get("LD_LIBRARY_PATH", ""))
r = subprocess.run([str(EXE.resolve()), "in.txt", "out.txt"],
                   capture_output=True, text=True, env=env, timeout=3600)
Path("dealii_run.log").write_text(r.stdout + "\n" + r.stderr)
if r.returncode != 0 or not Path("out.txt").is_file():
    sys.exit(f"deal.II solver failed (rc={r.returncode}):\n{r.stderr[-3000:]}")

txt = Path("out.txt").read_text().splitlines()
iface, nodes, ndof, sect = [], [], 0, 0
for ln in txt:
    if ln.startswith("#NODES"):
        sect = 1
        continue
    if ln.startswith("#NDOF"):
        ndof = int(ln.split()[1])
        continue
    v = [float(a) for a in ln.split()]
    (nodes if sect else iface).append(v)
iface = np.array(iface, float)
nodes = np.array(nodes, float)
nslot = NSTEPS if TR else ncomp
U = iface[:, 1:1 + nslot]
Q = iface[:, 1 + nslot:1 + 2 * nslot]

if cfg["side"] == "neumann":
    # The Dirichlet partner reads this side's VALUES, not its flux; the C++ side
    # leaves the reaction slot zero here on purpose, because on free dofs the
    # discrete equations hold and the reaction formula would export zero anyway.
    Q = np.zeros_like(U)

W.write_nodes("nodes.csv", nodes[:, :dim], nodes[:, dim:dim + ncomp])
W.write_log(cfg, ndof, r.stdout.strip().replace("\n", " | "))
print(f"[dealii {cfg['sidename']} {cfg['side']}] NDOF={ndof} "
      f"iface_n={len(iface)} u=[{U.min():.6g},{U.max():.6g}] "
      f"q=[{Q.min():.6g},{Q.max():.6g}]")
W.write_exports(ipts, U, Q, "displacement" if vec else "temperature")
