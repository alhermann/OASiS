"""Kratos Multiphysics participant for the OASiS `couple` driver (DIRICHLET side).

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST. Needs KratosMultiphysics + ConvectionDiffusionApplication
importable in the interpreter named in `command`.

This is the DIRICHLET side: it imports the partner's `values` (interface temperature), applies them as fixed nodal
TEMPERATURE on the interface, exports temperature + its own outward normal flux.

Slab [X0, X1] x [0, H], conductivity K.
  x = X0 : Dirichlet T = T_OUTER (hot side)
  x = X1 : INTERFACE, Dirichlet T = imported partner values (interpolated over y)
  top/bot: insulated (natural)

The exported outward normal flux density is the CONSISTENT (reaction) flux, not
a difference quotient of the solution — see the block above the export below.
"""
import json
from pathlib import Path

import numpy as np
import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication  # noqa: F401

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
PARTNER = "right"
X0, X1 = 0.0, 0.5
H = 1.0
K = 1.0
T_OUTER = 100.0
T_INIT = 50.0            # iteration-1 fallback interface temperature
nx, ny = 32, 32
# ─────────────────────────────────────────────────────────────────────────


def imported_T(y_coords: np.ndarray) -> np.ndarray:
    p = Path("imports.json")
    imp = json.loads(p.read_text() or "{}") if p.is_file() else {}
    if PARTNER not in imp or "values" not in imp[PARTNER]:
        return np.full_like(y_coords, float(T_INIT))
    d = imp[PARTNER]
    src_y = np.asarray(d["coordinates"], float)[:, 1]
    src_v = np.asarray(d["values"], float).ravel()
    o = np.argsort(src_y)
    return np.interp(y_coords, src_y[o], src_v[o])


def solve(T_if_in: np.ndarray):
    model = KM.Model()
    mp = model.CreateModelPart("thermal")
    mp.ProcessInfo[KM.DOMAIN_SIZE] = 2
    settings = KM.ConvectionDiffusionSettings()
    settings.SetUnknownVariable(KM.TEMPERATURE)
    settings.SetDiffusionVariable(KM.CONDUCTIVITY)
    settings.SetVolumeSourceVariable(KM.HEAT_FLUX)
    settings.SetSurfaceSourceVariable(KM.FACE_HEAT_FLUX)
    mp.ProcessInfo.SetValue(KM.CONVECTION_DIFFUSION_SETTINGS, settings)
    for v in (KM.TEMPERATURE, KM.CONDUCTIVITY, KM.HEAT_FLUX, KM.FACE_HEAT_FLUX,
              KM.REACTION_FLUX):   # REACTION_FLUX carries the interface reaction
        mp.AddNodalSolutionStepVariable(v)
    mp.SetBufferSize(1)

    props = mp.CreateNewProperties(1)
    nid, cnt = {}, 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            mp.CreateNewNode(cnt, X0 + (X1 - X0) * i / nx, H * j / ny, 0.0)
            nid[(i, j)] = cnt
            cnt += 1
    eid = 1
    for j in range(ny):
        for i in range(nx):
            a, b, c, d = nid[(i, j)], nid[(i+1, j)], nid[(i+1, j+1)], nid[(i, j+1)]
            mp.CreateNewElement("LaplacianElement2D3N", eid, [a, b, d], props); eid += 1
            mp.CreateNewElement("LaplacianElement2D3N", eid, [b, c, d], props); eid += 1

    for node in mp.Nodes:
        node.SetSolutionStepValue(KM.CONDUCTIVITY, K)
        node.SetSolutionStepValue(KM.HEAT_FLUX, 0.0)
        node.SetSolutionStepValue(KM.FACE_HEAT_FLUX, 0.0)
    for j in range(ny + 1):                       # outer hot side
        n = mp.Nodes[nid[(0, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, T_OUTER)
        n.Fix(KM.TEMPERATURE)
    for j in range(ny + 1):                       # INTERFACE Dirichlet = imported
        n = mp.Nodes[nid[(nx, j)]]
        n.SetSolutionStepValue(KM.TEMPERATURE, float(T_if_in[j]))
        n.Fix(KM.TEMPERATURE)

    # AddDof with a REACTION variable: without the second argument the fixed
    # dofs have nowhere to store their reaction and it is silently discarded.
    KM.VariableUtils().AddDof(KM.TEMPERATURE, KM.REACTION_FLUX, mp)
    scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
    builder = KM.ResidualBasedBlockBuilderAndSolver(KM.SkylineLUFactorizationSolver())
    # arg 4 is CalculateReactionsFlag: it must be True, the interface flux below
    # is read out of the reactions.
    strategy = KM.ResidualBasedLinearStrategy(mp, scheme, builder,
                                              True, False, False, False)
    strategy.Initialize()
    strategy.Solve()
    return mp, nid


def main() -> None:
    y_if = np.array([H * j / ny for j in range(ny + 1)])
    T_in = imported_T(y_if)
    mp, nid = solve(T_in)

    T_if = np.array([mp.Nodes[nid[(nx, j)]].GetSolutionStepValue(KM.TEMPERATURE)
                     for j in range(ny + 1)])

    # Outward normal flux density q = -(K grad T).n on the interface.
    #
    # WHY NOT A DIFFERENCE QUOTIENT. That is what this file used to do:
    # q = -K * (T_if - T_near) / dx, a one-sided backward difference. It is only
    # O(h) accurate, and so is the L2 projection of grad T that the other
    # participants used, for the same reason: the gradient of a P1 solution is
    # only O(h) accurate ON the boundary — the superconvergence points are
    # interior — and the boundary trace is exactly what the coupling reads.
    # Measured against a manufactured solution with a known exact interface
    # flux, that recovery converges at order ~1 while the consistent flux below
    # converges at ~2, so the recovery, not the physics and not the partner, was
    # setting the answer.
    #
    # THE CONSISTENT (REACTION) FLUX. From
    #     a(u,v) - (f,v) = int_dOmega (K grad u . n) v ds = -int_Gamma qn v ds
    # it follows that for every basis function phi_i on the interface
    #     int_Gamma qn phi_i ds = -r_i,   r = A u_h - b
    # with r the UNCONSTRAINED residual on the constrained rows — which is
    # precisely what Kratos calls the REACTION: its builder recomputes the RHS
    # with NO Dirichlet condition applied and stores -b_i = (A u_h - b)_i at
    # every fixed dof. Dividing by w_i = int_Gamma phi_i ds turns the functional
    # into a density the partner can interpolate pointwise.
    #
    # w_i for the P1 trace on this interface: the interface is the straight
    # segment x = X1 cut into ny equal pieces by THIS script's own node layout,
    # so int_Gamma phi_i ds is hy for an interior node and hy/2 at the two ends,
    # exactly. Regenerate this if the mesh above is ever made non-uniform.
    hy = H / ny
    w = np.full(ny + 1, hy)
    w[0] = w[-1] = 0.5 * hy

    r = np.array([mp.Nodes[nid[(nx, j)]].GetSolutionStepValue(KM.REACTION_FLUX)
                  for j in range(ny + 1)])
    q_out = -r / w

    # An interface node that ALSO lies on the outer Dirichlet boundary carries
    # the OUTER reaction as well, so its residual is not this interface's flux.
    # With nx >= 1 the two x-faces share no node, but the guard is kept because
    # an edited geometry can make them coincide.
    outer_ids = {nid[(0, j)] for j in range(ny + 1)}
    suspect = np.array([nid[(nx, j)] in outer_ids for j in range(ny + 1)])
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            q_out[i] = q_out[good[np.argmin(np.abs(good - i))]]

    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": len(T_if),
        "coordinates": [[X1, float(y)] for y in y_if],
        "values": [float(v) for v in T_if],
        "normal_fluxes": [float(v) for v in q_out],
    }, indent=2))
    print("kratos DIRICHLET side: T_if(applied)=%.6f q_out(computed)=%.6f"
          % (float(T_if.mean()), float(q_out.mean())))


if __name__ == "__main__":
    main()
