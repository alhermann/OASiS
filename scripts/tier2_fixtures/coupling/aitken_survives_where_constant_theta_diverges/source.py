"""The default accelerator survives the case the constant one blows up on —
"survives" meaning it LANDS ON THE ANSWER, not that it converges.

THE CLAIM UNDER TEST is the corrected sentence attached to the knowledge's own
divergence example, and it is the one that decides what an agent does when a
run at rho = 4 ends "did not converge":

  "at rho = 4, theta = 0.5 with a CONSTANT accelerator DIVERGES — the interface
   values run away by many orders of magnitude (measured: 4.1e+13 relative at
   300 iterations) and the conservation check fires — while theta = 0.2
   converges in 83. ... The default 'aitken' does NOT diverge on this case, and
   it does not solve it either: it lands on the closed-form interface value to
   1.2e-05 relative and its residual is still 1.4e-04 against tol = 1e-4 after
   300 iterations. ... its own adaptation drove theta onto the 0.05 clamp for 60
   of 299 adaptations, and once the raw output has settled the residual can only
   fall like (1-theta) per iteration, measured here at 0.968. So the answer was
   found and the convergence test could not certify it inside the budget."

WHAT THIS FIXTURE USED TO TEST, AND WHY IT HAD TO CHANGE. It tested the previous
sentence, "Note the default 'aitken' survives this particular case", read as
`converged=True`, and it failed. The sentence was written from a 40-cell sweep
run on the PER-PARTICIPANT Aitken theta that feature/coupling-robustness
replaced with ONE global theta, on an analytic stand-in for the participants,
at tol=1e-8 over 400 iterations, with a harness that was never committed. On the
driver that ships, re-measured with `scripts/sweep_accelerators.py`, the default
at rho = 4, theta = 0.5 does not diverge and does not converge: it sits on the
right answer with a residual that will not come down. The knowledge now says
that, and this fixture now tests that.

THIS IS THE DISTINCTION THE FIXTURE EXISTS FOR. `converged` and "found the
answer" are different questions on a partitioned driver, and here they give
opposite verdicts on the same run. An agent that reads `converged=False` as "the
physics is wrong" will go and change the physics, which is the expensive mistake
this whole section is written to prevent. So the fixture asserts BOTH: that the
interface state really is the analytic one, from both sides, and that the
residual really did not reach tol. Dropping either half would let a genuinely
broken driver pass — the first half alone would accept a run that never settles,
the second alone would accept a run that settled on the wrong fixed point.

BUDGET: 300 ITERATIONS, AND THE CHANGE IS DELIBERATE. This fixture ran at 200.
The claim it now tests is a 300-iteration statement, because the sweep behind it
gave every cell of a 70-setting grid the same max_iter=300. At 200 the Aitken
arm has landed on the interface VALUE (deviation 3.1e-04) but its interface flux
is still 0.30 W/m^2 from the closed form against the 0.2 tolerance used below,
so the physics half could not be checked at all and "landed on the answer" would
have been asserted on the temperature alone. Both arms still get the identical
budget and the identical tol, which is the property that matters: without it the
comparison is between an accelerator and a budget.

THE EXPERIMENT is one problem, run twice, with ONE string different. Same rho,
same theta0, same max_iter, same tol, same meshes, same initial guess — only
`accelerator` changes.

WHAT IS MEASURED, and why not the obvious thing. The residual is normalised by
the raw export magnitude, so a diverging run SATURATES it near a constant of
order one instead of sending it to infinity — the constant arm here ends with a
residual of order 1, which a merely-slow run could also produce. The raw
interface value is no better: it sits near 317 K, so an error amplified
fiftyfold still leaves it within a factor of two of the right answer. What
separates a runaway from a settling iteration is the DEVIATION from the exact
interface temperature relative to it, which grows as the amplification factor to
the power of the iteration count with nothing to hide behind.

WHY THE COPY OF probe_theta. couplinglib's probe_theta hard-codes
accelerator="constant" and this fixture's whole point is the other value, so the
probe is reproduced here with the accelerator as a parameter. It is otherwise
the same construction — same problem_with_rho, same heat_edits, same
SWEEP_T_INIT, same meshes, same `pair` through the registered `couple` tool —
so the two arms of this fixture differ in exactly one argument.
"""
from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

# The shared library lives in a sibling `_lib/` DIRECTORY, not as a bare
# file, because scripts/mutate_tier2_fixtures.py stages a fixture into a
# scratch tree and copies only sibling directories whose name starts with
# `_`. As a bare file it would not be copied, the staged fixture could not
# import it, and every mutation verdict would be VACUOUS_BASELINE.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_lib"))
import couplinglib as L                                     # noqa: E402

RHO = 4.0               # the ratio the knowledge's own divergence example uses
THETA0 = 0.5            # and its theta
MAX_ITER = 300          # the SAME budget for both arms — see the docstring
TOL = 1e-4              # and the SAME tolerance
# The two accelerator strings, named so the difference between the arms is one
# identifier and the mutation can move it.
AITKEN = "aitken"
CONSTANT = "constant"

# Deviation from the exact interface temperature, relative to it.
RUNAWAY = 1.0           # above this the interface value is not even the right
                        # order — the run has run away
SETTLED = 1e-3          # below this it has landed on the answer

# Physics, against the closed form. A run converged to tol=1e-4 sits a few
# 1e-3 K and a few 1e-2 W/m^2 from the exact interface values — these
# thresholds sit an order of magnitude above that and orders of magnitude below
# any pathology worth catching: a wrong fixed point is tens of K away and a sign
# error is O(1) in the flux.
T_ATOL = 0.05
Q_ATOL = 0.2
BALANCE_RTOL = 1e-2


def _quiet_stage(root, name, backend, edits):
    """`L.stage` prints the interpreter it resolved, which is right for a pair
    fixture and noise for a two-arm comparison that stages four participants."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return L.stage(root, name, backend, edits)


def probe(tag: str, accelerator: str, rho: float = RHO, theta: float = THETA0,
          max_iter: int = MAX_ITER, tol: float = TOL, dirichlet: str = "left",
          backend: str = "skfem", mesh_l=(16, 16), mesh_r=(14, 12)) -> dict:
    """couplinglib.probe_theta with the ACCELERATOR made a parameter.

    Everything else is identical to it, deliberately: same problem, same
    starting guess away from every ratio's answer, same non-matching interface
    meshes, and the same registered `couple` tool through `pair`.
    """
    p = L.problem_with_rho(rho)
    roles = {"left": "dirichlet" if dirichlet == "left" else "neumann",
             "right": "dirichlet" if dirichlet == "right" else "neumann"}
    root = L.workroot(tag)
    specs = []
    for pos, mesh in (("left", mesh_l), ("right", mesh_r)):
        partner = "right" if pos == "left" else "left"
        e = L.heat_edits(p, pos, roles[pos], partner, mesh)
        e["T_INIT"] = f"{L.SWEEP_T_INIT}"
        specs.append(_quiet_stage(root, pos, backend, e))
    res = L.pair(specs, max_iter=max_iter, tol=tol, accelerator=accelerator,
                 theta=theta)
    worst = 0.0
    for ex in (res.get("exports") or {}).values():
        for v in ex.get("values", []) or []:
            d = abs(float(v) - p.t_iface)
            worst = max(worst, d) if math.isfinite(d) else float("inf")
    deviation = worst / abs(p.t_iface) if math.isfinite(worst) else float("inf")
    return {"converged": bool(res.get("converged")),
            "iterations": int(res.get("iterations", 0)),
            "residual": float(res.get("residual", float("nan"))),
            "deviation": deviation, "result": res, "problem": p,
            "accelerator": accelerator, "budget": (rho, theta, max_iter, tol)}


def shown(dev: float) -> str:
    return "inf" if not math.isfinite(dev) else f"{dev:.3e}"


def report(tag: str, r: dict) -> None:
    print(f"{tag}_accelerator={r['accelerator']}")
    print(f"{tag}_converged={r['converged']}")
    print(f"{tag}_iterations={r['iterations']}")
    print(f"{tag}_residual={r['residual']:.3e}")
    print(f"{tag}_deviation_from_exact={shown(r['deviation'])}")


def landed_on_the_closed_form(tag: str, r: dict) -> bool:
    """Not `converged`: the interface VALUES, against the analytic answer.

    A partitioned fixed-point scheme converges to a fixed point, which is the
    solution only if the two participants exchange the right quantity with the
    right sign in the right units. So this checks the temperature from both
    sides, the flux from both sides with their opposite outward normals, and
    the conservation balance.

    `converged` is deliberately NOT part of this. On this driver at this ratio
    the two answers differ, and separating them is the fixture's subject.
    """
    p, res = r["problem"], r["result"]
    ex = res.get("exports") or {}
    if not ex:
        return bool(L.check(False, f"{tag}_no_exports",
                            "the run returned no interface exports at all"))
    nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
    print(f"{tag}_n_points={nl}/{nr}")
    ok = L.check(nl != nr, f"{tag}_matching_meshes",
                 f"both sides used {nl} interface points, so the "
                 f"non-matching-interface claim was not exercised")
    for side in ("left", "right"):
        lo, hi = L.span(ex[side]["values"])
        print(f"{tag}_{side}_T_span=[{lo:.10g},{hi:.10g}]")
        ok = L.close(0.5 * (lo + hi), p.t_iface, T_ATOL,
                     f"{tag}_{side}_T_err") and ok
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = L.span(ex[side]["normal_fluxes"])
        print(f"{tag}_{side}_q_span=[{lo:.10g},{hi:.10g}]")
        ok = L.close(0.5 * (lo + hi), sign * p.q, Q_ATOL,
                     f"{tag}_{side}_q_err") and ok
    net_l, net_r = L.net_flux(ex["left"]), L.net_flux(ex["right"])
    rel = abs(net_l + net_r) / max(abs(net_l), abs(net_r), 1e-30)
    print(f"{tag}_flux_balance_rel={rel:.3e}")
    ok = L.check(rel < BALANCE_RTOL, f"{tag}_flux_not_balanced",
                 f"net(left)={net_l:.6e} net(right)={net_r:.6e}") and ok
    return bool(ok)


def body() -> None:
    L.require_available("skfem")
    p = L.problem_with_rho(RHO)
    print(f"rho={RHO:g} theta0={THETA0} max_iter={MAX_ITER} tol={TOL:g}")
    print(f"k_left={p.kl:.4f} T_exact={p.t_iface:.6f} q_exact={p.q:.6f}")
    print(f"constant_amplification={p.amplification('left', THETA0):.4f}")
    print(f"stability_limit_for_this_rho={2.0 / (1.0 + RHO):.6f}")

    # ── arm 1: the constant accelerator, which the knowledge says diverges ──
    c = probe("cst", CONSTANT)
    report("constant", c)

    # ── arm 2: the DEFAULT accelerator, same everything else ───────────────
    a = probe("ait", AITKEN)
    report("aitken", a)

    diverged = (not math.isfinite(c["deviation"])) or c["deviation"] > RUNAWAY
    print(f"constant_theta_diverged={bool(diverged)}")
    L.check(diverged, "constant_theta_did_not_diverge",
            f"deviation from the exact interface temperature ended at "
            f"{shown(c['deviation'])}, which is not a runaway; the knowledge's "
            f"whole divergence example rests on it being one")
    L.check(not c["converged"], "constant_theta_converged",
            "the constant arm reached the tolerance, so there is nothing for "
            "the default to survive")

    # ── what "survives" means here: the default did NOT run away ───────────
    a_away = (not math.isfinite(a["deviation"])) or a["deviation"] > RUNAWAY
    print(f"aitken_diverged={bool(a_away)}")
    L.check(not a_away, "aitken_diverged_too",
            f"the knowledge says the default does not diverge on this case; "
            f"its deviation ended at {shown(a['deviation'])}")

    settled = L.check(a["deviation"] < SETTLED, "aitken_deviation_too_large",
                      f"deviation ended at {shown(a['deviation'])}, above "
                      f"{SETTLED:.0e}")
    physics = landed_on_the_closed_form("aitken", a)
    landed = bool(settled and physics)
    print(f"aitken_landed_on_the_closed_form={landed}")
    L.check(landed, "aitken_did_not_land_on_the_closed_form",
            "the knowledge says the default lands on the closed-form interface "
            "state at this ratio; on this driver it did not")

    # ── and the other half: it did NOT certify that answer ─────────────────
    #
    # Asserted, not merely printed. The served sentence is that the answer was
    # found and the convergence test could not confirm it inside the budget; a
    # driver on which the default now DOES reach tol here is a better driver
    # and a stale claim, and this fixture is the thing that has to say so.
    print(f"aitken_met_tol={a['converged']}")
    L.check(not a["converged"], "aitken_now_meets_tol_here",
            f"the served claim is that at rho={RHO:g}, theta={THETA0} the "
            f"default lands on the answer WITHOUT its residual reaching "
            f"tol={TOL:g} inside {MAX_ITER} iterations. It converged in "
            f"{a['iterations']} with residual {a['residual']:.3e}. Re-run "
            f"scripts/sweep_accelerators.py and rewrite the claim from it")
    print(f"aitken_residual_above_tol="
          f"{bool(a['residual'] > TOL)}")
    print(f"aitken_residual_over_tol_ratio={a['residual'] / TOL:.2f}")

    # The gap between the two arms, so the run reports its own numbers rather
    # than the fixture pinning one.
    better = (not math.isfinite(c["deviation"])) or (
        a["deviation"] < c["deviation"])
    if math.isfinite(c["deviation"]) and a["deviation"] > 0.0:
        print(f"orders_of_magnitude_between_the_arms="
              f"{math.log10(c['deviation'] / a['deviation']):.1f}")
    print(f"aitken_deviation_below_constant={bool(better)}")
    L.check(better, "aitken_was_not_better_than_constant",
            f"aitken {shown(a['deviation'])} vs constant "
            f"{shown(c['deviation'])}")

    # A two-arm comparison in which both arms ran the same accelerator would
    # be vacuous however green it looked, and one in which the arms got
    # different budgets would prove nothing. Both are read back from the runs.
    distinct = bool(AITKEN != CONSTANT and a["accelerator"] != c["accelerator"])
    print(f"the_two_arms_ran_different_accelerators={distinct}")
    L.check(distinct, "both_arms_ran_the_same_accelerator",
            f"arm 1 ran {c['accelerator']!r} and arm 2 ran "
            f"{a['accelerator']!r}; there is no comparison here")
    same_budget = bool(a["budget"] == c["budget"])
    print(f"arms_differed_only_in_the_accelerator={same_budget}")
    L.check(same_budget, "the_two_arms_did_not_share_a_budget",
            f"aitken ran (rho, theta, max_iter, tol)={a['budget']} and "
            f"constant ran {c['budget']}")
    print("arms=2")


L.main(body)
