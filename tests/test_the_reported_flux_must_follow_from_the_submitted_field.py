"""One-sided coupled evidence: does each side's flux follow from its own field?

THE HOLE. Every other coupled check compares the two sides to each other, so a
submission whose two sides come from a single computation passes all of them.
The bit-exact-zero rule catches the crudest form. `flux_ratio_consistency`
catches the general form: for a scalar conduction flux the reported ``q_n`` must
equal ``-k du/dn`` of the SAME side's submitted field, so the ratio it implies
must be the same constant at every interface point.

WHY IT DID NOT ALREADY WORK. `recover_flux_from_field` existed for exactly this
and keyed the field columns on each interface probe's exact tangential
coordinate. The interface probes are not on the field grid's tangential lines,
by construction: measured on `C8_27b_BARE_seed4`, side A's field is the 44x44
midpoint rule on (0, 0.625) x (0, 1) — y = (j+0.5)/44 — while the interface
probes are y = 1/4 + (i+0.5)/88, and **0 of 44 share a y**. So the recovery
returned None everywhere and the check could only answer NOT_ASSESSED. Sixth
instance of a mechanism instrumented and unable to reach its own case, and the
real reason issue #78 sat behind "needs a key-schema field": the blocker was
geometric.

These tests are built on manufactured submissions with a KNOWN conductivity, so
the assertions are about numbers rather than about code shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "campaign3_blind"))

from blind_eval import interface as IF          # noqa: E402

M = 44                    # interface probes, as the tasks prescribe
N = 44                    # field probes per axis
XI = 0.625                # the interface


def _field(k, n=N, x0=0.0, x1=XI, slope=3.0):
    """u = slope * x on a midpoint grid, so du/dx is exactly `slope`."""
    pts, vals = [], []
    hx, hy = (x1 - x0) / n, 1.0 / n
    for i in range(n):
        for j in range(n):
            x = x0 + (i + 0.5) * hx
            y = (j + 0.5) * hy
            pts.append((x, y))
            vals.append((slope * x,))
    return pts, vals


def _iface_pts():
    return [(XI, 0.25 + (i + 0.5) * 0.5 / M) for i in range(M)]


def test_the_interface_probes_are_not_on_the_field_grid():
    """The premise of the whole repair, asserted rather than assumed."""
    fpts, _ = _field(1.0)
    field_y = {round(p[1], 9) for p in fpts}
    iface_y = {round(p[1], 9) for p in _iface_pts()}
    assert not (field_y & iface_y), (
        "if the probes ever coincide, the simpler column lookup would do and "
        "this test is measuring the wrong geometry")


def test_an_honest_flux_recovers_its_own_conductivity():
    """q_n = -k du/dn reported honestly must imply exactly k."""
    for k in (1.0, 3.0, 1000.0):
        fpts, fvals = _field(k)
        ipts = _iface_pts()
        # side A occupies x < XI, so its outward normal is +x and q_n = -k du/dx
        reported = [(-k * 3.0,) for _ in ipts]
        dudn = IF.recover_normal_derivative(fpts, fvals, ipts, 0, XI, +1.0)
        assert all(d is not None for d in dudn), (
            "the field could not be interpolated to the interface probes")
        res = IF.flux_ratio_consistency(reported, dudn, components=(0,))
        assert res["verdict"] == "CONSISTENT", res
        got = res["per_component"][0]["implied_coefficient"]
        assert abs(got - k) < 1e-6 * k, (
            f"implied coefficient {got} should recover k = {k}")


def test_a_flux_invented_independently_of_the_field_is_caught():
    """A plausible-looking profile that the field does not generate."""
    fpts, fvals = _field(1.0)
    ipts = _iface_pts()
    # a smooth, entirely reasonable-looking flux profile — but not this field's
    reported = [(-3.0 * (1.0 + 4.0 * p[1] ** 2),) for p in ipts]
    dudn = IF.recover_normal_derivative(fpts, fvals, ipts, 0, XI, +1.0)
    res = IF.flux_ratio_consistency(reported, dudn, components=(0,))
    assert res["verdict"] == "INCONSISTENT", res


def test_the_detection_FLOOR_is_stated_rather_than_assumed():
    """What this check does NOT catch, asserted so nobody overclaims it.

    The 30% threshold was chosen from measurement: across the 98 applicable
    coupled runs in the tree, consistent submissions' implied coefficient
    varies by 0.9% at the median and 17.7% at p90, while inconsistent ones
    start at 39.3%. A flux whose implied coefficient wanders by ~23% therefore
    passes — and must, because firing there would start condemning honest runs.

    So the claim this check supports is bounded: it catches a reported flux
    that does not track the submitted field's normal derivative to within 30%
    along the interface. Evading it requires inventing a profile proportional
    to the true normal derivative, which is most of the work of computing it.
    A paper sentence stronger than that is not supported by this code.
    """
    fpts, fvals = _field(1.0)
    ipts = _iface_pts()
    mild = [(-3.0 * (1.0 + 0.8 * p[1]),) for p in ipts]
    dudn = IF.recover_normal_derivative(fpts, fvals, ipts, 0, XI, +1.0)
    res = IF.flux_ratio_consistency(mild, dudn, components=(0,))
    assert res["verdict"] == "CONSISTENT", (
        "a ~23% wander sits below the measured false-alarm floor and is not "
        "flagged; if this ever fails, the threshold moved and the calibration "
        "in flux_from_field_phase's docstring has to be redone")
    assert 0.15 < res["per_component"][0]["spread"] < 0.30


def test_an_inward_normal_is_named_as_a_sign_convention_not_a_forgery():
    fpts, fvals = _field(2.0)
    ipts = _iface_pts()
    reported = [(+2.0 * 3.0,) for _ in ipts]        # correct size, wrong sign
    dudn = IF.recover_normal_derivative(fpts, fvals, ipts, 0, XI, +1.0)
    res = IF.flux_ratio_consistency(reported, dudn, components=(0,))
    assert res["verdict"] == "SIGN_CONVENTION", res
    assert "OPPOSITE sign" in res["detail"]


def test_a_flat_field_says_so_instead_of_abstaining_silently():
    """A zero field is a statement about the submission, not missing geometry."""
    fpts, fvals = _field(1.0, slope=0.0)           # u == 0 everywhere
    ipts = _iface_pts()
    res = IF.flux_ratio_consistency([(1.0,)] * M,
                                    IF.recover_normal_derivative(
                                        fpts, fvals, ipts, 0, XI, +1.0),
                                    components=(0,))
    assert res["verdict"] == "NOT_ASSESSED"
    assert "flat at the interface" in res["per_component"][0]["detail"]


def test_the_check_refuses_the_families_where_the_ratio_need_not_be_constant():
    """Elasticity and anisotropy would otherwise produce invented defects."""
    cases = {
        "elasticity": (),
        "fluid_structure_interaction": (),
        "fem_dsmc": (),
        "diffusion": (0,),
        "reaction_diffusion": (0,),
        "transient_diffusion": (0,),
    }
    for fam, want in cases.items():
        got, why = IF.scalar_flux_components({"physics_family": fam})
        assert got == want, f"{fam}: expected {want}, got {got} ({why})"
    aniso = {"physics_family": "diffusion",
             "coefficients": "anisotropic conductivity K = [[1, 1/2], [1/2, 2]]"}
    got, why = IF.scalar_flux_components(aniso)
    assert got == (), f"a full tensor K must be refused, got {got}"
    assert "tangential" in why
    unknown, why = IF.scalar_flux_components({"physics_family": "magneto_hd"})
    assert unknown == () and "abstains" in why


def test_a_flipped_component_is_not_absorbed_by_a_healthy_one():
    """Worst across components, never best.

    C1 transmits a temperature and two traction components. The first
    aggregation reported CONSISTENT whenever ANY component was consistent, so a
    reversed sign could hide behind a healthy neighbour — measurable in the
    calibration as 262 triples booked CONSISTENT whose spreads reached 2.54.
    """
    fpts, fvals = _field(1.0)
    ipts = _iface_pts()
    dudn = IF.recover_normal_derivative(fpts, fvals, ipts, 0, XI, +1.0)
    two = [(d[0], d[0]) for d in dudn]             # a second, identical channel
    reported = [(-3.0, +3.0) for _ in ipts]        # component 1 sign-flipped
    res = IF.flux_ratio_consistency(reported, two, components=(0, 1))
    assert res["verdict"] == "SIGN_CONVENTION", res


def test_the_grader_records_it_for_every_coupled_run_and_gates_on_nothing():
    """What the GRADER ends up with, at the real entry point.

    The reasons this check produces must not reach `out["reasons"]`: the caller
    does `all_reasons = list(iface_reasons)` and feeds them to the outcome, so a
    reason here would silently change grades before its false-positive rate has
    been measured.
    """
    iface = pytest.importorskip("grading.iface")
    src = (ROOT / "campaign3_blind" / "grading" / "iface.py").read_text()
    assert "out[\"flux_from_field\"] = flux_from_field_phase(" in src
    assert "out[\"findings\"].extend(out[\"flux_from_field\"][\"findings\"])" in src
    assert "out[\"reasons\"].extend(out[\"flux_from_field\"]" not in src, (
        "the flux check's reasons must not join the phase reasons — those gate "
        "the outcome in grade_blind_v2")
    assert hasattr(iface, "flux_from_field_phase")
    assert hasattr(iface, "outward_signs")


def test_the_outward_normal_is_derived_from_extents_or_refused():
    iface = pytest.importorskip("grading.iface")
    leg = iface.Leg(axis=0, value=0.625, band=None)
    ok = iface.outward_signs({"extent_a": [[0.0, 0.625], [0.0, 1.0]],
                              "extent_b": [[0.625, 1.5], [0.0, 1.0]]}, leg)
    assert ok == {"A": 1.0, "B": -1.0}
    flipped = iface.outward_signs({"extent_a": [[0.625, 1.5], [0.0, 1.0]],
                                   "extent_b": [[0.0, 0.625], [0.0, 1.0]]}, leg)
    assert flipped == {"A": -1.0, "B": 1.0}
    assert iface.outward_signs({}, leg) is None, (
        "without extents the sign must be refused, not guessed — a guessed "
        "sign turns a correct submission into a reversed-convention finding")
    assert iface.outward_signs(
        {"extent_a": [[0.0, 0.4], [0.0, 1.0]],
         "extent_b": [[0.8, 1.5], [0.0, 1.0]]}, leg) is None
