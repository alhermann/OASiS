"""The REAL Kratos ConvectionDiffusion route, verified by execution.

Why this module exists: the `heat` and `poisson` templates in this backend used
to emit a numpy/scipy assembly with no `import KratosMultiphysics` anywhere in
them — heat.py's own first line read "Heat conduction — Kratos (manual
assembly)". An agent asked to solve a problem WITH KRATOS was handed a script
that cannot run Kratos, and the resulting submission cannot be attributed to
the code the task named. That is not a documentation defect, it is the wrong
artefact.

Every API fact below was established on this install (Kratos 10.3.0) by running
it, and each one had a wrong first guess:

  * `LaplacianElement2D3N` is registered; `LaplacianElement2D4N` is NOT
    ("The Element ... is not registered!"). 2D is P1 TRIANGLES here.
  * the reaction-carrying DOF overload is `AddDof(var, reaction, mp)`. There is
    no `AddDofWithReaction`. Registering no reaction and then letting the
    strategy compute reactions aborts in EVERY worker thread with "This
    container only can store the variables specified in its variables list".
  * CONDUCTIVITY is read NODALLY, through ConvectionDiffusionSettings, not from
    Properties (see the poisson pitfall: nodal 999 + Properties 1 -> 999).
  * the volumetric source HEAT_FLUX is the NODAL value interpolated at ONE
    centroid point, i.e. mean(f_nodes) * A / 3. Matched to 4.3e-16 against an
    independent assembly using that rule; the exact P1 mass matrix is 50% off,
    so this is not a detail.
  * FACE_HEAT_FLUX on a `ThermalFace2D2N` is the INWARD normal flux. Measured
    against a closed form chosen so the two readings differ by a sign:
    u(interface) came out +0.875 where inward predicts +0.875.
  * Kratos prints from C++ streams. An in-process `os.dup2` redirect of fd 1
    captured ZERO bytes. If a log has to show which code ran, run the solve in
    a SUBPROCESS and capture that.
"""
from __future__ import annotations

_REAL = '''\
"""{title}

Solved by Kratos Multiphysics ConvectionDiffusionApplication -- the solver is
Kratos, not an assembly written here. Run it with `run_simulation`.
"""
import json

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401
# ^ the import is what REGISTERS LaplacianElement2D3N; without it
#   CreateNewElement raises 'The Element "LaplacianElement2D3N" is not
#   registered!'

NX, NY = {nx}, {ny}
X0, X1, Y0, Y1 = {x0}, {x1}, {y0}, {y1}
K_VAL = {k}


def source(x, y):
    """The volumetric source f in -div(k grad u) = f. EDIT THIS."""
    return {f_expr}


def boundary_value(x, y):
    """The prescribed value on the Dirichlet boundary. EDIT THIS."""
    return {g_expr}


# ---- mesh: P1 triangles, two per cell (2D3N is the only 2D Laplacian element)
xs = np.linspace(X0, X1, NX + 1)
ys = np.linspace(Y0, Y1, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731  0-based
tris = []
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
        tris += [[a, b, c], [a, c, d]]
tris = np.array(tris)

# ---- model part
model = KM.Model()
mp = model.CreateModelPart("domain")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
for v in (KM.TEMPERATURE, KM.HEAT_FLUX, KM.CONDUCTIVITY, KM.FACE_HEAT_FLUX,
          KM.REACTION_FLUX, KM.SPECIFIC_HEAT, KM.DENSITY, KM.VELOCITY,
          KM.MESH_VELOCITY):
    mp.AddNodalSolutionStepVariable(v)

# ConvectionDiffusionSettings tells the element WHICH variable is which. Every
# entry must be set: an unset one leaves the element reading variable #0.
s = KM.ConvectionDiffusionSettings()
s.SetUnknownVariable(KM.TEMPERATURE)
s.SetDiffusionVariable(KM.CONDUCTIVITY)
s.SetVolumeSourceVariable(KM.HEAT_FLUX)
s.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
s.SetDensityVariable(KM.DENSITY)
s.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
s.SetVelocityVariable(KM.VELOCITY)
s.SetMeshVelocityVariable(KM.MESH_VELOCITY)
s.SetReactionVariable(KM.REACTION_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, s)

prop = mp.CreateNewProperties(1)
for i, (px, py) in enumerate(nodes):
    n = mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
    # CONDUCTIVITY IS READ NODALLY, not from Properties. Setting it only on
    # the Properties object gives a zero-diffusivity, singular system.
    n.SetSolutionStepValue(KM.CONDUCTIVITY, K_VAL)
    # HEAT_FLUX is the volumetric source, interpolated at the centroid:
    # the element assembles mean(f_nodes) * A / 3.
    n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(px, py)))
    n.SetSolutionStepValue(KM.DENSITY, 1.0)
    n.SetSolutionStepValue(KM.SPECIFIC_HEAT, 1.0)
for e, el in enumerate(tris):
    mp.CreateNewElement("LaplacianElement2D3N", e + 1,
                        [int(v) + 1 for v in el], prop)

# DOFs must carry the reaction, or CalculateReactions aborts in every thread.
KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)

tol = 1e-12
on_bnd = ((np.abs(nodes[:, 0] - X0) < tol) | (np.abs(nodes[:, 0] - X1) < tol)
          | (np.abs(nodes[:, 1] - Y0) < tol) | (np.abs(nodes[:, 1] - Y1) < tol))
for i in np.where(on_bnd)[0]:
    nd = mp.GetNode(int(i) + 1)
    nd.SetSolutionStepValue(KM.TEMPERATURE, float(boundary_value(*nodes[i])))
    nd.Fix(KM.TEMPERATURE)

lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
bas = KM.ResidualBasedBlockBuilderAndSolver(lin)
strat = KM.ResidualBasedLinearStrategy(mp, scheme, bas, True, False, False,
                                       False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = 1.0
mp.CloneTimeStep(1.0)
strat.Initialize()
strat.Solve()

u = np.array([mp.GetNode(i + 1).GetSolutionStepValue(KM.TEMPERATURE)
              for i in range(len(nodes))])
print(f"Kratos LaplacianElement2D3N: nodes={{len(nodes)}} "
      f"elements={{len(tris)}} max|u|={{np.abs(u).max():.10e}}")
# AN ALL-ZERO FIELD WITH A NONZERO SOURCE IS NOT A CONVERGED SOLVE. It is a
# source or a settings entry that never reached the element -- check it here
# rather than downstream.
if np.abs(u).max() == 0.0 and any(abs(source(*p)) > 0 for p in nodes[:50]):
    raise SystemExit("Kratos returned an identically zero field against a "
                     "nonzero source: the volume source did not reach the "
                     "element. Check ConvectionDiffusionSettings and that "
                     "HEAT_FLUX is set on NODES.")

np.savetxt("solution.csv",
           np.column_stack([nodes[:, 0], nodes[:, 1], u]),
           delimiter=", ", header="x, y, u", comments="",
           fmt="%.15e")
json.dump({{"max_abs": float(np.abs(u).max()), "n_nodes": int(len(nodes)),
           "n_elements": int(len(tris)), "element": "LaplacianElement2D3N",
           "solver": "Kratos ConvectionDiffusionApplication"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos solve complete.")
'''


def real_convdiff_script(title: str, nx: int, ny: int, k: float,
                         f_expr: str, g_expr: str,
                         x0: float = 0.0, x1: float = 1.0,
                         y0: float = 0.0, y1: float = 1.0) -> str:
    """A runnable Kratos ConvectionDiffusion script for -div(k grad u) = f."""
    return _REAL.format(title=title, nx=nx, ny=ny, k=k, f_expr=f_expr,
                        g_expr=g_expr, x0=x0, x1=x1, y0=y0, y1=y1)


CROSS_CHECK_NOTE = (
    "To VERIFY a Kratos answer, assemble the same P1 system independently "
    "(numpy is enough) with the SAME source rule -- mean(f_nodes) * A / 3, "
    "the one-point centroid rule Kratos uses -- and compare. That comparison "
    "is what established the rule: it matched to 4.3e-16, while the exact P1 "
    "mass matrix was 50% off. An independent assembly is a CHECK on the "
    "solver's answer, never a replacement for running it: a submission "
    "produced by the assembly alone cannot be attributed to Kratos, and a "
    "coupled task that names two codes is failed by it."
)


_TRANSIENT = '''\
"""{title}

Transient conduction solved by Kratos Multiphysics ConvectionDiffusionApplication
using EulerianDiffusion2D3N, which carries the capacity term. Run with
`run_simulation`.
"""
import json

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

NX, NY = {nx}, {ny}
X0, X1, Y0, Y1 = {x0}, {x1}, {y0}, {y1}
K_VAL, RHO, CP = {k}, {rho}, {cp}
DT, T_END = {dt}, {t_end}
T_INIT = {t_init}


def source(x, y):
    """Volumetric source. EDIT THIS."""
    return {f_expr}


def dirichlet(x, y):
    """Return the prescribed temperature, or None where the node is free.
    EDIT THIS."""
    {dirichlet_body}


xs = np.linspace(X0, X1, NX + 1)
ys = np.linspace(Y0, Y1, NY + 1)
nodes = np.array([(xs[i], ys[j]) for j in range(NY + 1) for i in range(NX + 1)])
nid = lambda i, j: j * (NX + 1) + i          # noqa: E731
tris = []
for j in range(NY):
    for i in range(NX):
        a, b, c, d = nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)
        tris += [[a, b, c], [a, c, d]]
tris = np.array(tris)

model = KM.Model()
mp = model.CreateModelPart("domain")
mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
for v in (KM.TEMPERATURE, KM.HEAT_FLUX, KM.CONDUCTIVITY, KM.FACE_HEAT_FLUX,
          KM.REACTION_FLUX, KM.SPECIFIC_HEAT, KM.DENSITY, KM.VELOCITY,
          KM.MESH_VELOCITY):
    mp.AddNodalSolutionStepVariable(v)
s = KM.ConvectionDiffusionSettings()
s.SetUnknownVariable(KM.TEMPERATURE)
s.SetDiffusionVariable(KM.CONDUCTIVITY)
s.SetVolumeSourceVariable(KM.HEAT_FLUX)
s.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
s.SetDensityVariable(KM.DENSITY)
s.SetSpecificHeatVariable(KM.SPECIFIC_HEAT)
s.SetVelocityVariable(KM.VELOCITY)
s.SetMeshVelocityVariable(KM.MESH_VELOCITY)
s.SetReactionVariable(KM.REACTION_FLUX)
mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, s)

prop = mp.CreateNewProperties(1)
for i, (px, py) in enumerate(nodes):
    n = mp.CreateNewNode(i + 1, float(px), float(py), 0.0)
    # ALL of these are read NODALLY through the settings object, not from
    # Properties. RHO * CP is the volumetric capacity: it is what makes this
    # transient rather than steady, and setting it to 0 silently gives the
    # steady answer at the first step.
    n.SetSolutionStepValue(KM.CONDUCTIVITY, K_VAL)
    n.SetSolutionStepValue(KM.DENSITY, RHO)
    n.SetSolutionStepValue(KM.SPECIFIC_HEAT, CP)
    n.SetSolutionStepValue(KM.HEAT_FLUX, float(source(px, py)))
    n.SetSolutionStepValue(KM.TEMPERATURE, T_INIT)
for e, el in enumerate(tris):
    # EulerianDiffusion2D3N, NOT LaplacianElement2D3N: the Laplacian element
    # has no capacity term, so a "transient" run built on it returns the steady
    # solution at every step and the history is flat.
    mp.CreateNewElement("EulerianDiffusion2D3N", e + 1,
                        [int(v) + 1 for v in el], prop)

KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
for i, (px, py) in enumerate(nodes):
    val = dirichlet(px, py)
    if val is not None:
        nd = mp.GetNode(i + 1)
        nd.SetSolutionStepValue(KM.TEMPERATURE, float(val))
        nd.Fix(KM.TEMPERATURE)

mp.SetBufferSize(2)      # the scheme reads the previous step
lin = KM.LinearSolverFactory().Create(
    KM.Parameters('{{"solver_type":"skyline_lu_factorization"}}'))
strat = KM.ResidualBasedLinearStrategy(
    mp, KM.ResidualBasedIncrementalUpdateStaticScheme(),
    KM.ResidualBasedBlockBuilderAndSolver(lin), True, False, False, False)
strat.SetEchoLevel(1)
mp.ProcessInfo[KM.DELTA_TIME] = DT
strat.Initialize()

probe = int(nid(NX // 2, NY // 2))
t, hist = 0.0, []
while t < T_END - 1e-12:
    t += DT
    mp.CloneTimeStep(t)
    strat.Solve()
    hist.append((t, float(mp.GetNode(probe + 1)
                          .GetSolutionStepValue(KM.TEMPERATURE))))
print(f"Kratos EulerianDiffusion2D3N: {{len(hist)}} steps, probe T "
      f"{{hist[0][1]:.6f}} -> {{hist[-1][1]:.6f}}")
if len(hist) > 2 and max(abs(v - hist[0][1]) for _, v in hist) == 0.0:
    raise SystemExit("the probe temperature is identical at every step: the "
                     "capacity term is not in play. Check DENSITY and "
                     "SPECIFIC_HEAT are nonzero on the NODES and that the "
                     "element is EulerianDiffusion2D3N.")

u = np.array([mp.GetNode(i + 1).GetSolutionStepValue(KM.TEMPERATURE)
              for i in range(len(nodes))])
np.savetxt("history.csv", np.array(hist), delimiter=", ",
           header="t, probe_T", comments="", fmt="%.15e")
np.savetxt("solution.csv", np.column_stack([nodes[:, 0], nodes[:, 1], u]),
           delimiter=", ", header="x, y, u", comments="", fmt="%.15e")
json.dump({{"n_steps": len(hist), "dt": DT, "t_end": T_END,
           "probe_T_final": hist[-1][1], "max_abs": float(np.abs(u).max()),
           "element": "EulerianDiffusion2D3N",
           "solver": "Kratos ConvectionDiffusionApplication"}},
          open("results_summary.json", "w"), indent=2)
print("Kratos transient conduction complete.")
'''


def real_transient_script(title: str, nx: int, ny: int, k: float, rho: float,
                          cp: float, dt: float, t_end: float,
                          f_expr: str, dirichlet_body: str,
                          t_init: float = 0.0,
                          x0: float = 0.0, x1: float = 1.0,
                          y0: float = 0.0, y1: float = 1.0) -> str:
    """A runnable Kratos transient conduction script.

    `dirichlet_body` is the body of dirichlet(x, y): it must `return` a value
    on constrained nodes and `return None` elsewhere.
    """
    return _TRANSIENT.format(title=title, nx=nx, ny=ny, k=k, rho=rho, cp=cp,
                             dt=dt, t_end=t_end, f_expr=f_expr,
                             dirichlet_body=dirichlet_body, t_init=t_init,
                             x0=x0, x1=x1, y0=y0, y1=y1)
