"""At a strongly unbalanced ratio the default accelerator is tens of orders of
magnitude closer to the answer than the naive constant theta — and NEITHER of
them reaches it. Both halves are the claim.

THE CLAIM UNDER TEST is the comparative half of the accelerator advice:

  "over the whole grid Aitken matched or beat the same constant theta in 65 of
   the 70 settings"
  "at rho = 6 and rho = 9 it solved NOTHING and landed on NOTHING. At every
   theta on the grid it diverged, by 4.1e+03 to 2.0e+17 relative"

Note what the claim says and does not say. It says MATCHED OR BEAT. It does not
say Aitken converges wherever a constant theta fails, and on this driver it very
often does not: over the 70-setting grid behind that sentence, Aitken solved
where the same constant theta ran away in only 4 settings, all of them at
rho = 2.

WHAT THIS FIXTURE USED TO ASSERT, AND WHY THAT WAS WRONG. It required the
default to REACH the closed-form interface state here, at rho = 6, on the
strength of a preamble that said rho = 6 was "inside the region the default
still recovers". That was never measured on this driver — it was carried over
from a 40-cell sweep run on the PER-PARTICIPANT Aitken theta that
feature/coupling-robustness replaced with one global theta, on an analytic
stand-in for the participants rather than the participants themselves, and
whose harness was never committed. Re-measured with
`scripts/sweep_accelerators.py` on the driver and participants that ship:

    at theta = 0.5 the default SOLVES up to rho = 2 and no further.
    At rho = 4 it lands on the closed-form interface value (1.2e-05 relative)
    without its residual reaching tol in 300 iterations.
    At rho = 6 and rho = 9 it solves nothing and lands on nothing, at any theta
    from 0.1 to 1.0.

So rho = 6 is OUTSIDE the recovery region by two swept ratios, and the fixture
was demanding something the knowledge does not claim, at a ratio the knowledge
now says the opposite about. The assertion is corrected to the claim; the ratio
is NOT moved to one where the default wins, because moving it would be choosing
the setting that makes a sentence true instead of measuring it.

WHY rho = 6 IS STILL THE RIGHT RATIO. It is the setting where the two halves of
the claim come apart, which is exactly what a fixture is for. theta = 0.5 sits
above the constant stability limit 2/(1+rho) = 0.286, so the naive theta is
genuinely bad for the constant scheme and there is a real contest; and the
default is measured to lose the absolute contest here while winning the
comparative one by about thirty orders of magnitude. A fixture at a ratio where
the default simply wins would not notice if "matched or beat" quietly became
"always solves".

BOTH ARMS USE THE SAME max_iter AND THE SAME tol, read back from each run rather
than assumed. Without that the comparison is between an accelerator and a
budget.

WHAT IS AND IS NOT PINNED. The GAP between the two arms is asserted as a floor
in orders of magnitude, never as a value. The constant arm at this setting is
bit-reproducible — its residual and deviation come back identical run after run
— but the AITKEN arm is not reproducible across environments: its theta is a
nonlinear function of the residual history and it is clamped, so on a diverging
run a last-bit difference in a BLAS reduction changes which adaptation hits the
clamp, and from there the trajectories differ. Measured here the gap was about
31 orders; the same cell measured elsewhere put the Aitken arm at 2.3e+04 rather
than 2.9e+04. The VERDICT was identical in every environment tried. So this
fixture asserts verdicts and a floor, and prints the digits for the record.
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

RHO = 6.0               # strongly unbalanced: theta=0.5 is far above 2/(1+rho)
THETA = 0.5             # the knowledge's own "cannot estimate rho" default
MAX_ITER = 300          # the SAME budget for both arms
TOL = 1e-4              # and the SAME tolerance

RUNAWAY = 1.0           # deviation above this: the value is not even the right
                        # order of magnitude
SETTLED = 1e-3          # below this: the run landed on the answer

# The floor on the gap between the arms, in orders of magnitude. Measured at
# about 31 here; the floor sits far below that because the Aitken arm's digits
# move between environments (see the docstring) while the verdict does not.
MIN_ORDERS = 10.0

# Physics against the closed form, used to establish that NEITHER arm reached
# it. A run converged to tol=1e-4 sits a few 1e-3 K and a few 1e-2 W/m^2 away;
# these thresholds sit an order of magnitude above that and orders of magnitude
# below a wrong fixed point (tens of K) or a flux sign error (O(1) relative).
T_ATOL = 0.05
Q_ATOL = 0.2
BALANCE_RTOL = 1e-2


def _quiet_stage(root, name, backend, edits):
    """`L.stage` prints the interpreter it resolved, which is right for a pair
    fixture and noise for a two-arm comparison that stages four participants."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return L.stage(root, name, backend, edits)


def probe(tag: str, accelerator: str, rho: float = RHO, theta: float = THETA,
          max_iter: int = MAX_ITER, tol: float = TOL, dirichlet: str = "left",
          backend: str = "skfem", mesh_l=(16, 16), mesh_r=(14, 12)) -> dict:
    """couplinglib.probe_theta with the ACCELERATOR made a parameter.

    Same problem, same starting guess away from every ratio's answer, same
    non-matching interface meshes, same registered `couple` tool through
    `pair`. The deviation is probe_theta's: how far the interface value ended
    from the exact answer, relative to it — the residual is normalised by the
    export magnitude and SATURATES near a constant of order one on a diverging
    run, so it cannot tell a runaway from a slow iteration.
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


def reached_closed_form(tag: str, r: dict) -> bool:
    """Did this arm land on the analytic interface state?

    Reported, never asserted True: at this ratio the measured answer is that
    NEITHER arm does, and that is the fixture's evidence rather than its
    failure. Printed for both arms so the boundary claim has numbers behind it.
    """
    p, res = r["problem"], r["result"]
    ok = bool(r["converged"]) and r["deviation"] < SETTLED
    ex = res.get("exports") or {}
    if not ex:
        print(f"{tag}_no_exports=True")
        return False
    nl, nr = len(ex["left"]["coordinates"]), len(ex["right"]["coordinates"])
    print(f"{tag}_n_points={nl}/{nr}")
    if nl == nr:
        # The non-matching-interface claim is part of what this setting
        # exercises, so say so out loud rather than letting it pass unnoticed.
        ok = False
        L.check(False, f"{tag}_matching_meshes",
                f"both sides used {nl} interface points, so the "
                f"non-matching-interface claim was not exercised")
    for side in ("left", "right"):
        lo, hi = L.span(ex[side]["values"])
        print(f"{tag}_{side}_T_span=[{lo:.10g},{hi:.10g}]")
        err = abs(0.5 * (lo + hi) - p.t_iface)
        print(f"{tag}_{side}_T_err={err:.3e}")
        ok = ok and err <= T_ATOL
    for side, sign in (("left", +1.0), ("right", -1.0)):
        lo, hi = L.span(ex[side]["normal_fluxes"])
        print(f"{tag}_{side}_q_span=[{lo:.10g},{hi:.10g}]")
        err = abs(0.5 * (lo + hi) - sign * p.q)
        print(f"{tag}_{side}_q_err={err:.3e}")
        ok = ok and err <= Q_ATOL
    net_l, net_r = L.net_flux(ex["left"]), L.net_flux(ex["right"])
    rel = abs(net_l + net_r) / max(abs(net_l), abs(net_r), 1e-30)
    print(f"{tag}_flux_balance_rel={rel:.3e}")
    ok = ok and math.isfinite(rel) and rel < BALANCE_RTOL
    return bool(ok)


def body() -> None:
    L.require_available("skfem")
    p = L.problem_with_rho(RHO)
    limit = 2.0 / (1.0 + RHO)
    print(f"rho={RHO:g} theta={THETA} max_iter={MAX_ITER} tol={TOL:g}")
    print(f"k_left={p.kl:.4f} T_exact={p.t_iface:.6f} q_exact={p.q:.6f}")
    print(f"theta_opt_for_this_rho={p.theta_opt('left'):.6f}")
    print(f"stability_limit_for_this_rho={limit:.6f}")
    print(f"constant_amplification_at_this_theta="
          f"{p.amplification('left', THETA):.4f}")

    # The premise: the naive theta must be a BAD constant choice here, or the
    # contest is against nothing. This is arithmetic on the knowledge's own
    # formula, evaluated before anything runs.
    naive_is_bad = THETA > limit
    print(f"naive_theta_above_the_constant_stability_limit={bool(naive_is_bad)}")
    L.check(naive_is_bad, "naive_theta_is_already_stable_here",
            f"theta={THETA} is below the limit {limit:.6f} at rho={RHO:g}, so "
            f"a constant theta converges and there is nothing for the "
            f"accelerator to beat")

    a = probe("ait", "aitken")
    c = probe("cst", "constant")
    for tag, r in (("aitken", a), ("constant", c)):
        print(f"{tag}_converged={r['converged']}")
        print(f"{tag}_iterations={r['iterations']}")
        print(f"{tag}_residual={r['residual']:.3e}")
        print(f"{tag}_deviation_from_exact={shown(r['deviation'])}")

    # ── half one: the CONSTANT arm must be the disaster the claim says it is ──
    c_away = (not math.isfinite(c["deviation"])) or c["deviation"] > RUNAWAY
    print(f"constant_theta_diverged={bool(c_away)}")
    L.check(c_away, "constant_theta_did_not_diverge",
            f"deviation ended at {shown(c['deviation'])}, which is not a "
            f"runaway; the comparative claim rests on it being one")

    # ── half two: the MEASURED BOUNDARY. rho = 6 is outside the region the
    # default recovers, and this fixture is the place that says so with runs.
    a_ok = reached_closed_form("aitken", a)
    c_ok = reached_closed_form("constant", c)
    print(f"aitken_reached_closed_form={a_ok}")
    print(f"constant_reached_closed_form={c_ok}")
    print(f"neither_arm_reached_the_closed_form={bool(not a_ok and not c_ok)}")
    L.check(not c_ok, "the_same_constant_theta_also_worked",
            f"the constant arm reached the closed form in {c['iterations']} "
            f"iterations with deviation {shown(c['deviation'])}, so this ratio "
            f"shows no advantage for the default and the claim is not being "
            f"tested here")
    L.check(not a_ok, "aitken_recovered_at_a_ratio_the_knowledge_says_it_cannot",
            f"the served accelerator advice states that at rho = 6 the default "
            f"'solved NOTHING and landed on NOTHING ... at every theta on the "
            f"grid'. Here it reached the closed form at rho={RHO:g}, "
            f"theta={THETA} with deviation {shown(a['deviation'])}. That is a "
            f"better driver and a stale claim: re-run "
            f"scripts/sweep_accelerators.py and rewrite the boundary from it")
    a_away = (not math.isfinite(a["deviation"])) or a["deviation"] > RUNAWAY
    print(f"aitken_also_diverged={bool(a_away)}")
    L.check(a_away, "aitken_did_not_diverge_here",
            f"the boundary claim says the default diverges at rho = 6 for every "
            f"theta measured; here its deviation ended at "
            f"{shown(a['deviation'])}, inside the runaway threshold {RUNAWAY:g}")

    # ── half three: MATCHED OR BEAT, which is what the claim actually says ──
    beat = bool(a["deviation"] < c["deviation"])
    orders = (math.log10(c["deviation"] / a["deviation"])
              if (math.isfinite(c["deviation"]) and math.isfinite(a["deviation"])
                  and a["deviation"] > 0.0) else float("inf"))
    print(f"orders_of_magnitude_between_the_arms="
          f"{'inf' if not math.isfinite(orders) else f'{orders:.1f}'}")
    print(f"aitken_beat_constant={beat}")
    L.check(beat, "aitken_did_not_beat_constant",
            f"at rho={RHO:g}, theta={THETA}, max_iter={MAX_ITER}, tol={TOL:g}: "
            f"aitken deviation {shown(a['deviation'])}, constant deviation "
            f"{shown(c['deviation'])}")
    wide = bool(orders >= MIN_ORDERS)
    print(f"gap_is_at_least_{MIN_ORDERS:g}_orders={wide}")
    L.check(wide, "the_gap_between_the_arms_is_not_wide",
            f"the claim is that the default is tens of orders of magnitude "
            f"closer to the answer here; measured gap {orders:.1f} orders, "
            f"floor {MIN_ORDERS:g}")

    # The two arms differ in ONE argument. Read back what each run was
    # actually given rather than asserting it from the constants: a comparison
    # in which the accelerator arm also got a different budget would prove
    # nothing, however green it looked.
    same_budget = bool(a["budget"] == c["budget"]
                       and a["accelerator"] != c["accelerator"])
    print(f"arms_differed_only_in_the_accelerator={same_budget}")
    L.check(same_budget, "the_two_arms_did_not_share_a_budget",
            f"aitken ran (rho, theta, max_iter, tol)={a['budget']} and "
            f"constant ran {c['budget']}")
    print("ratios_tested=1")


L.main(body)
