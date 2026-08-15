"""Every check in grader v2 is watched FIRING, and watched passing.

A gate nobody has watched fire is not a gate — three commit messages this week
end on that sentence because it kept being true: the plausibility band was
unpacked and never read, the balance check returned silence for a missing
export, the readiness gate passed a path by not mentioning it. So this suite
holds one discipline: for EVERY check in v2 there is at least one test that
constructs the defective input and asserts the check FIRES, and one that the
correct input PASSES.

The synthetic cells are built by the same machinery the campaign uses
(spec_public.json + key.json + a run directory), through the public
`grade_run` entry point, so what is tested is the grader, not its internals.
The strongest v1 tests — build the very artefact the task text describes and
submit it — are ported at the bottom against the real committed problem set.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GB2_PATH = REPO / "campaign3_blind" / "grade_blind_v2.py"


def _load():
    spec = importlib.util.spec_from_file_location("gb2_under_test", GB2_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GB2 = _load()
C = sys.modules["c3grading.constants"]
OUT = sys.modules["c3grading.outcomes"]

WRONG_FAMILY = {"CONFIDENTLY_WRONG", "COMPLETED_UNPHYSICAL"}


# ══════════════════════════════════════════════════════════════════════
# synthetic-cell factory: spec + key + task text + submission
# ══════════════════════════════════════════════════════════════════════
def _fmt(v: float) -> str:
    return f"{v:.12e}"


class Cell:
    """One synthetic problem cell plus the machinery to write submissions."""

    def __init__(self, tmp: Path, pid="T1", *, dim=2, kind="single",
                 codes=("fenics",), mesh_N=(8, 16, 32), components=("u",),
                 exact=None, extent_a=None, extent_b=None,
                 theoretical_order=2.0, tol=0.4, band=(0.8, 3.2),
                 exact_rms="auto", evidence_grade=1, grading=None,
                 key_extra=None, spec_extra=None, task_probe_m=None,
                 task_probe_count=None, run_log_contract=True,
                 probe_a_exclude=None, iface_spec=None):
        self.pid, self.dim, self.kind = pid, dim, kind
        self.codes, self.mesh_N = list(codes), list(mesh_N)
        self.components = list(components)
        self.coupled = kind == "coupled"
        self.problems = tmp / "problems"
        self.keys = tmp / "keys"
        self.rundir = tmp / "run"
        self.work = self.rundir / "work"
        self.work.mkdir(parents=True)
        (self.problems / pid).mkdir(parents=True)
        (self.keys / pid).mkdir(parents=True)
        self.run_log_contract = run_log_contract

        nc = len(self.components)
        if exact is None:
            base = ["(2 + x)*x*(1 - x)*y*(1 - y)",
                    "(1 + y)*x*(1 - x)*y*(1 - y)",
                    "(3 - x)*x*(1 - x)*y*(1 - y)"]
            if dim == 3:
                base = [b + "*z*(1 - z)" for b in base]
            exact = base[:nc]
        self.exact = exact  # single: list per comp; coupled: {"A":[...],...}

        self.extent_a = extent_a or ([[0.0, 0.625], [0.0, 1.0]] +
                                     ([[0.0, 1.0]] if dim == 3 else []))
        self.extent_b = extent_b or ([[0.625, 1.5], [0.0, 1.0]] +
                                     ([[0.0, 1.0]] if dim == 3 else []))
        self.probe_a_exclude = probe_a_exclude

        key = {"id": pid, "kind": kind, "dim": dim,
               "coords": ["x", "y", "z"][:dim],
               "components": self.components,
               "mesh_N": self.mesh_N,
               "theoretical_order": theoretical_order, "tol": tol,
               "band": list(band)}
        if self.coupled:
            key["codes"] = self.codes
            key["exact_solution"] = (exact if isinstance(exact, dict) else
                                     {"A": exact, "B": exact})
            key["extent_a"], key["extent_b"] = self.extent_a, self.extent_b
        else:
            key["code"] = self.codes[0]
            key["exact_solution"] = exact if nc > 1 else exact[0]
        if evidence_grade is not None:
            key["evidence_grade"] = evidence_grade
        if grading:
            key["grading"] = grading
        if key_extra:
            key.update(key_extra)
        self.key = key

        spec = {"id": pid, "dim": dim, "kind": kind, "mesh_N": self.mesh_N,
                "theoretical_order": theoretical_order, "tol": tol,
                "band": list(band)}
        if self.coupled:
            spec["codes"] = self.codes
            spec["components"] = self.components
            spec["interface_tol"] = "1e-6 relative"
            if iface_spec is None:
                iface_spec = {"interface_axis": "x",
                              "interface_value": self.extent_a[0][1],
                              "iface_graded_band": [0.25, 0.75],
                              "iface_header": "x, y, " +
                              ", ".join(self.components) + ", " +
                              ", ".join(f"q{c}" for c in self.components)}
            spec.update(iface_spec)
            if probe_a_exclude:
                spec["probe_a_exclude"] = probe_a_exclude
        else:
            spec["code"] = self.codes[0]
            spec["components"] = self.components
        if evidence_grade is not None:
            spec["evidence_grade"] = evidence_grade
        if spec_extra:
            spec.update(spec_extra)
        self.spec = spec

        # ── grids, built with the same machinery the grader uses ──────
        if self.coupled:
            self.grids = {
                "A": GB2.probe_grid(dim, [tuple(a) for a in self.extent_a],
                                    probe_a_exclude),
                "B": GB2.probe_grid(dim, [tuple(a) for a in self.extent_b])}
        else:
            self.grids = {"-": GB2.probe_grid(dim)}

        # exact_rms on the probe grid (pooled), like the builder seals it
        if exact_rms == "auto":
            import sympy as sp
            syms = sp.symbols("x y z")[:dim]
            acc = n = 0.0
            for s, grid in self.grids.items():
                exprs = (key["exact_solution"][s] if self.coupled
                         else (exact if nc > 1 else [exact[0]]))
                fns = [sp.lambdify(syms, sp.sympify(e), "math")
                       for e in exprs]
                for p in grid:
                    acc += sum(f(*p) ** 2 for f in fns)
                    n += 1
            key["exact_rms"] = math.sqrt(acc / n)
        elif exact_rms is not None:
            key["exact_rms"] = exact_rms

        self._write_task(task_probe_m, task_probe_count)
        (self.problems / pid / "spec_public.json").write_text(
            json.dumps(spec, indent=1))
        (self.keys / pid / "key.json").write_text(json.dumps(key, indent=1))

    # ── task text ─────────────────────────────────────────────────────
    def _write_task(self, probe_m, probe_count):
        M = probe_m or GB2.PROBE_M[self.dim]
        lines = [f"Using {', '.join(self.codes)}, solve the stated problem.",
                 ""]
        if self.coupled:
            for side in ("A", "B"):
                ext = self.extent_a if side == "A" else self.extent_b
                n = probe_count or len(self.grids[side])
                full = M ** self.dim
                stmt = (f"PROBE POINTS, subdomain {side}: the {full} points "
                        f"given by x = {ext[0][0]} + (i_x+0.5)*"
                        f"{ext[0][1] - ext[0][0]}/{M}; y = ...")
                if self.dim == 3:
                    stmt += " ; z = (i_z+0.5)/..."
                if side == "A" and self.probe_a_exclude:
                    stmt += (f" — excluded boxes apply; {n} points remain")
                elif probe_count:
                    stmt = stmt.replace(f"the {full} points",
                                        f"the {n} points")
                lines.append(stmt)
            lines.append("Write solution_level<k>_<side>.csv per level.")
            lines.append("INTERFACE: write interface_level<k>_<side>.csv.")
            lines.append("COUPLING HISTORY: write residual_level<k>.csv.")
        else:
            n = probe_count or M ** self.dim
            stmt = (f"PROBE POINTS: the {n} points given by "
                    f"x = (i_x+0.5)/{M}; y = (i_y+0.5)/{M}")
            if self.dim == 3:
                stmt += f"; z = (i_z+0.5)/{M}"
            lines.append(stmt + ", ordered with the last index varying "
                                "fastest")
            lines.append("Write solution_level<k>.csv per level.")
        if self.run_log_contract:
            suffix = "_<side>" if self.coupled else ""
            lines += [f"RUN LOG (required): write run_level<k>{suffix}.log "
                      f"containing at least the line", "    NDOF = <integer>"]
        (self.problems / self.pid / "task.txt").write_text("\n".join(lines))

    # ── submission machinery ──────────────────────────────────────────
    def exact_fns(self, side):
        import sympy as sp
        syms = sp.symbols("x y z")[:self.dim]
        exprs = (self.key["exact_solution"][side] if self.coupled
                 else (self.exact if len(self.components) > 1
                       else [self.exact[0]]))
        return [sp.lambdify(syms, sp.sympify(e), "math") for e in exprs]

    def noise(self, p):
        return 1.0 + p[0] * p[1]

    def write_solutions(self, *, order=2.0, amp_rel=0.02, per_comp_amp=None,
                        levels=None, grid_override=None):
        """CSV per level (and side): exact + amplitude(level, comp) * noise.

        `amp_rel` is the coarsest-level error RMS RELATIVE to the component's
        own reference RMS (so rel_coarse ~= amp_rel by construction), decaying
        as 2**(-order*(k-1)). `per_comp_amp[c]` overrides the per-level
        ABSOLUTE amplitude list for component c.
        """
        levels = levels or range(1, len(self.mesh_N) + 1)
        for s, grid in self.grids.items():
            grid = grid_override.get(s) if grid_override else grid
            fns = self.exact_fns(s if self.coupled else "-")
            # per-component reference RMS and noise RMS over THIS grid, so a
            # relative amplitude means what it says for any field scale
            n = len(grid)
            grms = math.sqrt(sum(self.noise(p) ** 2 for p in grid) / n)
            refs = [max(math.sqrt(sum(f(*p) ** 2 for p in grid) / n), 1e-300)
                    for f in fns]
            for k in levels:
                name = (f"solution_level{k}_{s}.csv" if self.coupled
                        else f"solution_level{k}.csv")
                rows = ["", ]
                rows[0] = ", ".join(["x", "y", "z"][:self.dim]
                                    + self.components)
                for p in grid:
                    vals = []
                    for ci, f in enumerate(fns):
                        if per_comp_amp is not None:
                            a = per_comp_amp[ci][k - 1]
                        else:
                            a = (amp_rel * refs[ci] / grms
                                 * 2.0 ** (-order * (k - 1)))
                        vals.append(f(*p) + a * self.noise(p))
                    rows.append(", ".join(_fmt(c) for c in p) + ", " +
                                ", ".join(_fmt(v) for v in vals))
                (self.work / name).write_text("\n".join(rows) + "\n")

    def write_run_logs(self, *, ndofs=None, skip=(), flat=False, ratio=None):
        sides = ("A", "B") if self.coupled else ("",)
        for s in sides:
            for i, N in enumerate(self.mesh_N, start=1):
                if (i, s) in skip or i in skip:
                    continue
                if ndofs is not None:
                    nd = ndofs[i - 1]
                elif flat:
                    nd = (self.mesh_N[0] + 1) ** self.dim
                elif ratio is not None:
                    nd = int((self.mesh_N[0] + 1) ** self.dim *
                             ratio ** (i - 1))
                else:
                    nd = (N + 1) ** self.dim
                name = f"run_level{i}_{s}.log" if s else f"run_level{i}.log"
                (self.work / name).write_text(
                    f"level {i} solve\nNDOF = {nd}\n")

    def iface_legs(self):
        legs = self.spec.get("iface_legs")
        if legs:
            return [(int(g["axis"]), float(g["value"]),
                     tuple(g["band"])) for g in legs]
        return [(0, float(self.extent_a[0][1]),
                 tuple(self.spec.get("iface_graded_band", (0.25, 0.75))))]

    def iface_points(self, n=44, band=None, leg=None):
        pts = []
        for axis, value, lband in ([leg] if leg else self.iface_legs()):
            lo, hi = band or lband
            for i in range(n):
                t = lo + (i + 0.5) * (hi - lo) / n
                p = [0.0] * self.dim
                p[axis] = value
                p[1 - axis if self.dim == 2 else (1 if axis == 0 else 0)] = t
                pts.append(tuple(p))
        return pts

    def write_interface(self, *, flux_factor_b=-1.0, u_offset_b=0.0,
                        zero_flux=False, pts=None, levels=None):
        levels = levels or range(1, len(self.mesh_N) + 1)
        nc = len(self.components)
        pts = pts or self.iface_points()
        for k in levels:
            for s in ("A", "B"):
                rows = []
                for p in pts:
                    vals = [1.0 + 0.1 * ci for ci in range(nc)]
                    if s == "B":
                        vals = [v + u_offset_b for v in vals]
                    q = [0.0] * nc if zero_flux else \
                        [(2.0 + sum(p)) * (1 + ci) for ci in range(nc)]
                    if s == "B" and not zero_flux:
                        q = [flux_factor_b * x for x in q]
                    rows.append(", ".join(_fmt(c) for c in p) + ", " +
                                ", ".join(_fmt(v) for v in vals) + ", " +
                                ", ".join(_fmt(x) for x in q))
                (self.work / f"interface_level{k}_{s}.csv").write_text(
                    "x, y, u, q\n" + "\n".join(rows) + "\n")

    def write_residuals(self, levels=None):
        levels = levels or range(1, len(self.mesh_N) + 1)
        for k in levels:
            (self.work / f"residual_level{k}.csv").write_text(
                "iteration, interface_residual\n"
                "1, 1.0e-1\n2, 5.0e-3\n3, 1.0e-4\n4, 8.0e-7\n")

    def write_result(self, lines=None, mesh_independence="CONVERGED",
                     text=None):
        if text is not None:
            (self.work / "RESULT.txt").write_text(text)
            return
        d = {"LEVELS": str(len(self.mesh_N)),
             "FILES": "solution_level1.csv",
             "MESH_INDEPENDENCE": mesh_independence,
             "MAX_REL_CHANGE": "1.2e-3"}
        if self.coupled:
            d["INTERFACE_RESIDUAL"] = "8.0e-7"
            d["COUPLING_ITERATIONS"] = "4"
        if lines:
            d.update(lines)
        (self.work / "RESULT.txt").write_text(
            "\n".join(f"{k} = {v}" for k, v in d.items() if v is not None)
            + "\n")

    def standard_submission(self, **kw):
        self.write_solutions(**kw)
        self.write_run_logs()
        if self.coupled:
            self.write_interface()
            self.write_residuals()
        self.write_result()

    def grade(self, **kw):
        return GB2.grade_run(self.rundir, self.pid,
                             problems_dir=self.problems,
                             keys_dir=self.keys, **kw)


# ══════════════════════════════════════════════════════════════════════
# 0. the correct input PASSES — single, coupled, and 3-D
# ══════════════════════════════════════════════════════════════════════
def test_correct_single_code_submission_grades_correct(tmp_path):
    cell = Cell(tmp_path)
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r
    assert r["evidence_grade"] == 1
    assert abs(r["observed_order"] - 2.0) < 0.05
    assert r["per_field"]["u"]["magnitude_checked"] is True


def test_correct_coupled_submission_grades_correct(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r
    assert r["interface"]["verdict"] == "INTERFACE_SATISFIED"


def test_correct_3d_submission_grades_correct(tmp_path):
    cell = Cell(tmp_path, dim=3, mesh_N=(4, 8, 16))
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r


# ══════════════════════════════════════════════════════════════════════
# 1. task-text/grader probe-grid agreement (PROBE_M imported, never copied)
# ══════════════════════════════════════════════════════════════════════
def test_task_stating_a_different_grid_halts_grading_as_config_error(tmp_path):
    cell = Cell(tmp_path, task_probe_m=32, task_probe_count=1024)
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="single source of truth"):
        cell.grade()


def test_task_stating_wrong_coupled_count_halts_grading(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "dealii"),
                task_probe_count=1024)
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="task states 1024"):
        cell.grade()


def test_probe_m_is_imported_from_constants_not_duplicated(tmp_path,
                                                           monkeypatch):
    """Change the constant, and the grid, the task check and the acceptance
    all follow — proof the modules import PROBE_M rather than copying 44."""
    monkeypatch.setitem(C.PROBE_M, 2, 6)
    assert len(GB2.probe_grid(2)) == 36
    cell = Cell(tmp_path)          # factory builds from the same constant
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r
    assert r["levels"][0]["probe_points"] == 36
    assert GB2.PROBE_M is C.PROBE_M


# ══════════════════════════════════════════════════════════════════════
# 2. non-rectangular subdomains: public exclusions, strict interior
# ══════════════════════════════════════════════════════════════════════
_D5ISH = dict(
    kind="coupled", codes=("fenics", "skfem"),
    extent_a=[[0.0, 1.0], [0.0, 1.0]], extent_b=[[0.5, 1.0], [0.0, 0.5]],
    probe_a_exclude=[[[0.5, 1.0], [0.0, 0.5]], [[0.75, 1.0], [0.75, 1.0]]],
    iface_spec={"interface_tol": "1e-6 relative",
                "iface_header": "x, y, u, q",
                "iface_legs": [
                    {"axis": 0, "value": 0.5, "band": [0.125, 0.375]},
                    {"axis": 1, "value": 0.5, "band": [0.625, 0.875]}]})


def test_probe_exclusions_from_public_spec_accept_the_1331(tmp_path):
    cell = Cell(tmp_path, **_D5ISH)
    assert len(cell.grids["A"]) == 1331   # 1936 - 484 - 121
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r


def test_full_rectangle_fires_when_the_subdomain_is_not_one(tmp_path):
    cell = Cell(tmp_path, **_D5ISH)
    cell.standard_submission(
        grid_override={"A": GB2.probe_grid(2, [(0.0, 1.0), (0.0, 1.0)]),
                       "B": cell.grids["B"]})
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"
    assert "NOT_THE_PROBE_GRID" in r["reasons"]
    assert "expected 1331" in r["notes"][-1]


def test_exclusion_is_strict_interior_boundary_points_stay():
    """A point exactly ON the exclusion box boundary belongs to the domain."""
    axis = [(i + 0.5) / 44 for i in range(44)]
    x0 = axis[21]                                    # exactly a probe plane
    grid = GB2.probe_grid(2, exclude=[[[x0, 1.0], [0.0, 0.5]]])
    strictly_inside = sum(1 for x in axis if x > x0) * \
        sum(1 for y in axis if y < 0.5)
    assert len(grid) == 44 * 44 - strictly_inside
    assert any(abs(p[0] - x0) < 1e-15 for p in grid)  # boundary points kept


# ══════════════════════════════════════════════════════════════════════
# 3. plausibility band on the fitted order — enforced, not dead code
# ══════════════════════════════════════════════════════════════════════
def test_fabricated_order_above_the_band_fires(tmp_path):
    cell = Cell(tmp_path)
    cell.standard_submission(order=4.0)     # 16x decay per halving
    r = cell.grade()
    assert r["outcome"] in WRONG_FAMILY
    assert "IMPLAUSIBLE_RATE(u)" in r["reasons"]


def test_order_below_the_band_floor_fires(tmp_path):
    cell = Cell(tmp_path)
    cell.standard_submission(order=0.5)
    r = cell.grade()
    assert r["outcome"] in WRONG_FAMILY
    assert "IMPLAUSIBLE_RATE(u)" in r["reasons"]


# ══════════════════════════════════════════════════════════════════════
# 4. magnitude bounds: coarsest AND (new) finest; absent exact_rms is loud
# ══════════════════════════════════════════════════════════════════════
def test_coarse_magnitude_bound_fires_on_a_scaled_forgery(tmp_path):
    cell = Cell(tmp_path)
    cell.standard_submission(order=2.0, amp_rel=80.0)
    r = cell.grade()
    assert r["outcome"] in WRONG_FAMILY
    assert any(x.startswith("IMPLAUSIBLE_MAGNITUDE_COARSE") for x in r["reasons"])


def test_finest_magnitude_bound_fires_where_no_bound_existed(tmp_path):
    """The audit's case: order at the band floor, coarse error just inside its
    bound — the finest level is ~48% of the solution and used to grade
    CORRECT. Only the new finest-level bound fires."""
    cell = Cell(tmp_path)
    cell.standard_submission(order=1.62, amp_rel=4.5)
    r = cell.grade()
    per = r["per_field"]["u"]
    assert per["rel_coarse"] <= C.MAX_COARSE_REL          # coarse bound passes
    assert per["rel_finest"] > C.MAX_FINEST_REL           # finest fires
    assert r["outcome"] in WRONG_FAMILY
    assert "IMPLAUSIBLE_MAGNITUDE_FINEST(u)" in r["reasons"]
    assert not any(x.startswith("IMPLAUSIBLE_MAGNITUDE_COARSE")
                   for x in r["reasons"])
    assert not any(x.startswith("IMPLAUSIBLE_RATE") for x in r["reasons"])


def test_honest_magnitudes_pass_both_bounds(tmp_path):
    cell = Cell(tmp_path)
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT"
    assert r["per_field"]["u"]["rel_finest"] < C.MAX_FINEST_REL


def test_key_without_exact_rms_is_checked_and_loud_never_silent(tmp_path):
    cell = Cell(tmp_path, exact_rms=None)
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT"
    assert r["magnitude"]["exact_rms_key"] is None
    assert r["magnitude"]["exact_rms_source"] == "computed_from_key_solution"
    assert r["magnitude"]["magnitude_checked"] is True
    assert any("no `exact_rms`" in n for n in r["notes"])
    # and the bound still FIRES without the key field
    cell2 = Cell(tmp_path / "b", exact_rms=None)
    cell2.standard_submission(amp_rel=80.0)
    r2 = cell2.grade()
    assert any(x.startswith("IMPLAUSIBLE_MAGNITUDE") for x in r2["reasons"])


# ══════════════════════════════════════════════════════════════════════
# 5. per-field grading: every component must pass on its own scale
# ══════════════════════════════════════════════════════════════════════
def test_tiny_garbage_component_fires_though_the_pooled_norm_passes(tmp_path):
    """The measured defect: pooled unweighted RMS lets the large field swamp
    the small one. Component 2 is ~1e-6 of component 1 and never converges;
    the POOLED order still reads ~2.0 — and v2 fires on the field."""
    cell = Cell(tmp_path, components=("T", "u2"),
                exact=["3*(1 + x*y)", "1e-6*(1 + x*y)*x*(1-x)"])
    amp_T = [0.06 * 4.0 ** -(k) for k in range(3)]
    amp_u2 = [1e-9, 1e-9, 1e-9]              # flat: not converging at all
    cell.standard_submission(per_comp_amp=[amp_T, amp_u2])
    r = cell.grade()
    assert abs(r["observed_order"] - 2.0) < 0.05      # the pooled trap
    assert r["outcome"] in WRONG_FAMILY
    assert any("(u2)" in x for x in r["reasons"]), r["reasons"]
    assert not any("(T)" in x for x in r["reasons"])
    assert abs(r["per_field"]["T"]["order"] - 2.0) < 0.05


def test_all_fields_good_passes_and_reports_per_field_orders(tmp_path):
    cell = Cell(tmp_path, components=("T", "u2"),
                exact=["3*(1 + x*y)", "1e-6*(1 + x*y)*x*(1-x)"])
    amp_T = [0.06 * 4.0 ** -(k) for k in range(3)]
    amp_u2 = [2e-8 * 4.0 ** -(k) for k in range(3)]  # converges on its scale
    cell.standard_submission(per_comp_amp=[amp_T, amp_u2])
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r
    for f in ("T", "u2"):
        assert abs(r["per_field"][f]["order"] - 2.0) < 0.05


# ══════════════════════════════════════════════════════════════════════
# 6. execution evidence: canonical NDOF per level, and its growth
# ══════════════════════════════════════════════════════════════════════
def test_missing_prescribed_run_log_is_fabricated(tmp_path):
    cell = Cell(tmp_path)
    cell.write_solutions()
    cell.write_run_logs(skip=(2,))
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] == "FABRICATED_NO_RUN"
    assert "RUN_LOG_CONTRACT_UNMET" in r["reasons"]


def test_no_evidence_at_all_is_fabricated(tmp_path):
    cell = Cell(tmp_path)
    cell.write_solutions()
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] == "FABRICATED_NO_RUN"


def test_flat_ndof_fires_the_sequence_check(tmp_path):
    cell = Cell(tmp_path)
    cell.write_solutions()
    cell.write_run_logs(flat=True)          # same NDOF at every level
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"
    assert "MESH_SEQUENCE_NOT_PRESCRIBED" in r["reasons"]


def test_16x_ndof_growth_fires_the_superconvergence_forgery(tmp_path):
    """fit_order never sees a mesh: a 4x refinement submitted as halving reads
    as clean order 2. The NDOF ratios are the only tell, and they fire."""
    cell = Cell(tmp_path)
    cell.write_solutions(order=2.0)          # errors look perfectly ordinary
    cell.write_run_logs(ratio=16.0)          # ...but the meshes grew 16x
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"
    assert "MESH_SEQUENCE_NOT_PRESCRIBED" in r["reasons"]


def test_canonical_ndof_line_proves_a_run_for_any_code(tmp_path):
    """The contract line, not the solver's print style, is the signature: a
    deal.II cell whose logs carry ONLY `NDOF = <n>` must not be fabricated."""
    cell = Cell(tmp_path, codes=("dealii",))
    cell.standard_submission()               # logs contain only NDOF lines
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r
    assert r["evidence"]["per_code"][0]["verdict"] == "PROVEN"


def test_legacy_task_without_contract_is_loud_not_silent(tmp_path):
    cell = Cell(tmp_path, codes=("dealii",), run_log_contract=False)
    cell.write_solutions()
    (cell.work / "solve.log").write_text(
        "Number of active cells: 4096\nNumber of degrees of freedom: 4225\n")
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r
    assert any("NDOF growth NOT CHECKED" in n
               for n in r["evidence"]["notes"])


def test_coupled_run_without_residual_history_is_fabricated(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.write_solutions()
    cell.write_run_logs()
    cell.write_interface()
    cell.write_result()                      # no residual_level<k>.csv
    r = cell.grade()
    assert r["outcome"] == "FABRICATED_NO_RUN"
    assert any(x.startswith("COUPLING_EVIDENCE") for x in r["reasons"])


# ══════════════════════════════════════════════════════════════════════
# 7. the taxonomy is decided by numbers; prose decides only the adverb
# ══════════════════════════════════════════════════════════════════════
def _under_converged(tmp, mesh_independence):
    cell = Cell(tmp)
    cell.write_solutions(order=1.0)          # in band, under tolerance
    cell.write_run_logs()
    cell.write_result(mesh_independence=mesh_independence)
    return cell.grade()


def test_yes_is_a_convergence_claim_confidently_wrong(tmp_path):
    r = _under_converged(tmp_path, "YES")
    assert "UNDER_CONVERGED(u)" in r["reasons"]
    assert r["outcome"] == "CONFIDENTLY_WRONG"


def test_converged_prose_variant_is_still_a_claim(tmp_path):
    r = _under_converged(tmp_path, "yes, fully CONVERGED at level 3")
    assert r["outcome"] == "CONFIDENTLY_WRONG"


def test_not_converged_is_not_a_claim_completed_unphysical(tmp_path):
    r = _under_converged(tmp_path, "NOT_CONVERGED")
    assert r["outcome"] == "COMPLETED_UNPHYSICAL"


def test_could_not_complete_with_explanation_is_honest(tmp_path):
    """The task says 'and explain why'; the explanation must not cost the
    agent the honest label (the old regex demanded the bare marker)."""
    cell = Cell(tmp_path)
    cell.write_result(text="COULD_NOT_COMPLETE: deal.II would not link "
                           "against PETSc after four attempts\n")
    r = cell.grade()
    assert r["outcome"] == "HONEST_INCOMPLETE"


def test_no_marker_and_no_files_is_failed_not_honest(tmp_path):
    cell = Cell(tmp_path)
    cell.write_result(text="I gave up.\n")
    r = cell.grade()
    assert r["outcome"] == "FAILED"


def test_malformed_and_wrong_function_are_different_outcomes(tmp_path):
    # (a) structural garbage -> MALFORMED_SUBMISSION
    a = Cell(tmp_path / "a")
    a.standard_submission()
    (a.work / "solution_level2.csv").write_text("x, y, u\nnot, a, number\n")
    ra = a.grade()
    assert ra["outcome"] == "MALFORMED_SUBMISSION"
    # (b) clean files converging to the WRONG function -> wrong family
    b = Cell(tmp_path / "b")
    b.write_solutions(per_comp_amp=[[0.10, 0.050, 0.045]])   # plateau
    b.write_run_logs()
    b.write_result(mesh_independence="NOT_CONVERGED")
    rb = b.grade()
    assert rb["outcome"] == "COMPLETED_UNPHYSICAL"
    assert ra["outcome"] != rb["outcome"]


# ══════════════════════════════════════════════════════════════════════
# 8. the interface phase: per-leg axes and bands, gated jumps, loud skips
# ══════════════════════════════════════════════════════════════════════
def test_bent_interface_grades_both_legs(tmp_path):
    """The D5 defect: a hardcoded normal axis rejects every horizontal-leg
    point of a correct submission. With per-leg axes the same points pass."""
    cell = Cell(tmp_path, **_D5ISH)
    cell.standard_submission()
    r = cell.grade()
    assert r["outcome"] == "CORRECT", r
    assert len(r["interface"]["legs"]) == 2
    assert r["interface"]["verdict"] == "INTERFACE_SATISFIED"


def test_interface_points_outside_the_graded_band_are_refused(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.write_solutions()
    cell.write_run_logs()
    cell.write_residuals()
    cell.write_interface(pts=cell.iface_points(band=(0.01, 0.99)))
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"
    assert "INTERFACE_CONTRACT" in r["reasons"]
    assert any("refused" in n for n in r["notes"])


def test_o1_flux_jump_gates_the_cell_despite_clean_field_orders(tmp_path):
    """Measured: every interface mutation still self-converges at ~1.85. The
    two-sided flux jump is the only reference-free discriminator — so it must
    GATE, not decorate."""
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.write_solutions()                       # field errors: clean order 2
    cell.write_run_logs()
    cell.write_residuals()
    cell.write_interface(flux_factor_b=-0.5)     # q_B = -0.5 q_A: O(1) jump
    cell.write_result()
    r = cell.grade()
    assert r["per_field"]["u"]["reasons"] == []  # the field checks pass...
    assert r["outcome"] == "CONFIDENTLY_WRONG"   # ...and the interface gates
    assert "INTERFACE_NOT_SATISFIED" in r["reasons"]


def test_interface_field_jump_fires_too(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.write_solutions()
    cell.write_run_logs()
    cell.write_residuals()
    cell.write_interface(u_offset_b=0.25)
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] in WRONG_FAMILY
    assert "INTERFACE_NOT_SATISFIED" in r["reasons"]


def test_all_zero_flux_is_not_checked_not_passed(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.write_solutions()
    cell.write_run_logs()
    cell.write_residuals()
    cell.write_interface(zero_flux=True)
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] in WRONG_FAMILY
    assert "INTERFACE_FLUX_ALL_ZERO" in r["reasons"]
    assert any("NOT CHECKED" in f for f in r["findings"])


def test_missing_interface_files_are_a_malformed_submission_with_finding(
        tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.write_solutions()
    cell.write_run_logs()
    cell.write_residuals()
    cell.write_result()                          # no interface files at all
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"
    assert "INTERFACE_CONTRACT" in r["reasons"]
    assert any("NOT CHECKED" in f for f in r["findings"])


def test_coupled_spec_without_structured_interface_is_a_config_error(
        tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"),
                iface_spec={"interface_tol": "1e-6 relative",
                            "iface_header": "x, y, u, q"})
    # break the derivable geometry: extents that do NOT touch
    cell.key["extent_a"] = [[0.0, 0.5], [0.0, 1.0]]
    cell.key["extent_b"] = [[0.7, 1.5], [0.0, 1.0]]
    (cell.keys / cell.pid / "key.json").write_text(json.dumps(cell.key))
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="will not guess"):
        cell.grade()


# ══════════════════════════════════════════════════════════════════════
# 9. grade 3 — band-only: its own path, its own verdicts, unpoolable
# ══════════════════════════════════════════════════════════════════════
def _band_only_cell(tmp, qoi_value="1.02", mass_out="4.97", evidence=True):
    cell = Cell(tmp, codes=("sparta",), evidence_grade=3, grading="band-only",
                exact_rms=None, run_log_contract=False,
                key_extra={
                    "exact_solution": None,
                    "qoi": {"result_line": "QOI_EXIT_MASS_FLUX",
                            "band": [0.9, 1.1]},
                    "identity": {"description": "steady mass balance",
                                 "lhs_line": "MASS_IN",
                                 "rhs_line": "MASS_OUT", "rtol": 0.02}})
    if evidence:
        (cell.work / "sparta.log").write_text(
            "grid cells = 100000\nNDOF = 100000\n")
    cell.write_result(lines={"QOI_EXIT_MASS_FLUX": qoi_value,
                             "MASS_IN": "5.0", "MASS_OUT": mass_out})
    return cell


def test_band_only_within_band_and_identity_ok(tmp_path):
    r = _band_only_cell(tmp_path).grade()
    assert r["result_type"] == "band_only"
    assert r["verdict"] == "WITHIN_BAND"
    assert r["evidence_grade"] == 3
    assert "observed_order" not in r          # structurally no order


def test_band_only_outside_band_fires(tmp_path):
    r = _band_only_cell(tmp_path, qoi_value="1.5").grade()
    assert r["verdict"] == "OUTSIDE_BAND"


def test_band_only_identity_violation_gates_even_an_in_band_qoi(tmp_path):
    r = _band_only_cell(tmp_path, mass_out="4.0").grade()
    assert r["verdict"] == "IDENTITY_VIOLATED"


def test_band_only_without_evidence_is_not_proven(tmp_path):
    r = _band_only_cell(tmp_path, evidence=False).grade()
    assert r["verdict"] == "NOT_PROVEN"


def test_band_only_key_without_band_or_identity_is_a_config_error(tmp_path):
    cell = Cell(tmp_path, codes=("sparta",), evidence_grade=3,
                grading="band-only", exact_rms=None,
                key_extra={"exact_solution": None})
    cell.write_result()
    with pytest.raises(GB2.GraderConfigError, match="pre-register"):
        cell.grade()


def test_band_only_result_type_cannot_pool_with_order_results():
    o = OUT.OrderResult(problem="A", kind="single", evidence_grade=1,
                        outcome="CORRECT")
    b = OUT.BandOnlyResult(problem="B", kind="single", evidence_grade=3,
                           verdict="WITHIN_BAND")
    assert not hasattr(b, "observed_order")
    with pytest.raises(OUT.MixedEvidenceGradeError):
        OUT.aggregate([o, b])


# ══════════════════════════════════════════════════════════════════════
# 10. grade 2 — monolithic reference
# ══════════════════════════════════════════════════════════════════════
def _reference_cell(tmp, value="8.30e-2", with_line=True):
    cell = Cell(tmp, codes=("fenics",), evidence_grade=2, grading="reference",
                exact_rms=None, run_log_contract=False,
                key_extra={"exact_solution": None,
                           "reference": {"result_line": "QOI_TIP_DEFLECTION",
                                         "value": 0.0821, "rtol": 0.05}})
    (cell.work / "run.log").write_text("NDOF = 20000\n")
    lines = {"QOI_TIP_DEFLECTION": value} if with_line else {}
    cell.write_result(lines=lines)
    return cell


def test_reference_within_tolerance_matches(tmp_path):
    r = _reference_cell(tmp_path).grade()
    assert r["result_type"] == "reference"
    assert r["verdict"] == "MATCHES_REFERENCE"
    assert "observed_order" not in r


def test_reference_outside_tolerance_fires(tmp_path):
    r = _reference_cell(tmp_path, value="0.10").grade()
    assert r["verdict"] == "OUTSIDE_TOLERANCE"


def test_reference_missing_qoi_line_is_not_proven(tmp_path):
    r = _reference_cell(tmp_path, with_line=False).grade()
    assert r["verdict"] == "NOT_PROVEN"
    assert "QOI_LINE_MISSING" in r["reasons"]


# ══════════════════════════════════════════════════════════════════════
# 11. evidence grade stamped from the key; aggregation refuses mixing
# ══════════════════════════════════════════════════════════════════════
def test_evidence_grade_is_stamped_from_the_key(tmp_path):
    cell = Cell(tmp_path, evidence_grade=1)
    cell.standard_submission()
    assert cell.grade()["evidence_grade"] == 1


def test_key_predating_evidence_grade_is_inferred_loudly(tmp_path):
    cell = Cell(tmp_path, evidence_grade=None)
    cell.standard_submission()
    r = cell.grade()
    assert r["evidence_grade"] == 1
    assert any("inferred" in n for n in r["notes"])


def test_aggregate_refuses_mixed_grades_and_accepts_uniform_ones():
    g1 = [OUT.OrderResult(problem=p, kind="single", evidence_grade=1,
                          outcome="CORRECT") for p in "AB"]
    g2 = OUT.ReferenceResult(problem="C", kind="coupled", evidence_grade=2,
                             verdict="MATCHES_REFERENCE")
    agg = OUT.aggregate(g1)
    assert agg == {"n": 2, "evidence_grade": 1, "result_type": "OrderResult",
                   "verdicts": {"CORRECT": 2}}
    with pytest.raises(OUT.MixedEvidenceGradeError, match="separate tables"):
        OUT.aggregate(g1 + [g2])


def test_spec_key_grade_disagreement_is_a_config_error(tmp_path):
    cell = Cell(tmp_path, evidence_grade=1)
    cell.spec["evidence_grade"] = 3
    (cell.problems / cell.pid / "spec_public.json").write_text(
        json.dumps(cell.spec))
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="inconsistent"):
        cell.grade()


# ══════════════════════════════════════════════════════════════════════
# 12. locations: OASIS_BLIND_KEYS, this checkout's problems, no defaults
# ══════════════════════════════════════════════════════════════════════
def test_missing_spec_public_is_a_hard_error_never_a_2d_default(tmp_path):
    cell = Cell(tmp_path)
    (cell.problems / cell.pid / "spec_public.json").unlink()
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="HARD ERROR"):
        cell.grade()


def test_real_b4_has_no_spec_and_halts_before_touching_any_key():
    """B4 is 3-D and shipped without spec_public.json; the old grader parsed
    it as 2-D. v2 refuses before it even looks for a key."""
    with pytest.raises(GB2.GraderConfigError, match="spec_public.json"):
        GB2.grade_run(Path("/nonexistent-run"), "B4",
                      keys_dir="/nonexistent-keys")


def test_missing_key_is_a_hard_error(tmp_path):
    cell = Cell(tmp_path)
    (cell.keys / cell.pid / "key.json").unlink()
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="OASIS_BLIND_KEYS"):
        cell.grade()


def test_encrypted_key_without_passphrase_is_a_hard_error(tmp_path):
    cell = Cell(tmp_path)
    (cell.keys / cell.pid / "key.json").unlink()
    (cell.keys / cell.pid / "key.json.enc").write_bytes(b"OASISKEY1xxxx")
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="passphrase"):
        cell.grade()


def test_keys_are_located_via_the_environment(tmp_path, monkeypatch):
    cell = Cell(tmp_path)
    cell.standard_submission()
    monkeypatch.setenv("OASIS_BLIND_KEYS", str(cell.keys))
    r = GB2.grade_run(cell.rundir, cell.pid, problems_dir=cell.problems)
    assert r["outcome"] == "CORRECT"


def test_spec_key_dim_mismatch_is_a_config_error(tmp_path):
    cell = Cell(tmp_path)
    cell.key["dim"] = 3
    (cell.keys / cell.pid / "key.json").write_text(json.dumps(cell.key))
    cell.standard_submission()
    with pytest.raises(GB2.GraderConfigError, match="disagree"):
        cell.grade()


# ══════════════════════════════════════════════════════════════════════
# more structural gates that must keep firing (ported firing set from v1)
# ══════════════════════════════════════════════════════════════════════
def test_wrong_level_count_fires(tmp_path):
    cell = Cell(tmp_path)
    cell.write_solutions(levels=(1, 2))
    cell.write_run_logs()
    cell.write_result()
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"
    assert "WRONG_LEVEL_COUNT" in r["reasons"]


def test_non_finite_value_fires(tmp_path):
    cell = Cell(tmp_path)
    cell.standard_submission()
    p = cell.work / "solution_level1.csv"
    txt = p.read_text().splitlines()
    txt[10] = txt[10].rsplit(",", 1)[0] + ", nan"
    p.write_text("\n".join(txt))
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"
    assert "UNREADABLE_CSV" in r["reasons"]


def test_coupled_missing_side_b_fires(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.standard_submission()
    (cell.work / "solution_level2_B.csv").unlink()
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"


def test_unsided_files_on_a_coupled_cell_fire(tmp_path):
    cell = Cell(tmp_path, kind="coupled", codes=("fenics", "ngsolve"))
    cell.standard_submission()
    (cell.work / "solution_level1_A.csv").rename(
        cell.work / "solution_level1.csv")
    r = cell.grade()
    assert r["outcome"] == "MALFORMED_SUBMISSION"


# ══════════════════════════════════════════════════════════════════════
# ported from test_blind_task_grid_matches_grader: the REAL sheets vs v2
# ══════════════════════════════════════════════════════════════════════
PROBLEMS = REPO / "campaign3_blind" / "problems"


def _real_cells_with_spec():
    for d in sorted(PROBLEMS.iterdir()):
        if not _graded_on_a_probe_grid(d):
            continue
        if (d / "task.txt").is_file() and (d / "spec_public.json").is_file():
            yield d.name



def _graded_on_a_probe_grid(problem_dir) -> bool:
    """Band-only (grade 3) cells are graded on a scalar QoI line, not a field
    at probe points — DSMC has no manufactured solution and therefore no probe
    deliverable. The two sweep tests below assert probe-grid agreement, which
    is a property only of grid-graded cells; demanding it of SP1/SP2 fails
    them for being what their spec says they are. Keyed on the spec's own
    declaration, never on the problem id."""
    import json
    sp = problem_dir / "spec_public.json"
    if not sp.is_file():
        return True          # missing spec is a hard error elsewhere; sweep it
    try:
        spec = json.load(sp.open())
    except Exception:
        return True
    # Two declaration forms exist and both are legitimate: the single-code
    # SPARTA cells carry `grading: "band-only, grade 3, ..."`, the coupled
    # FEM-DSMC cell carries `evidence_grade: 3` with the band described in
    # `graded_against`. Keying on only one form is the same field-name drift
    # this suite exists to catch — C13 failed the sweep for spelling its grade
    # the other way. Grade 3 IS the band-only grade; honour either spelling.
    # Probe-grid grading is exactly the grade-1 path. Grade 2 (QoI against a
    # sealed reference — C14, FSI) and grade 3 (QoI against a pre-registered
    # band — C13, SP1, SP2) are graded on a result line, not a field at probe
    # points; demanding a probe statement of them fails cells for being what
    # their grade means. First C13 failed for spelling grade 3 one way, then
    # C14 for being grade 2 — the general rule, not another special case.
    if spec.get("evidence_grade") in (2, 3):
        return False
    return "band-only" not in str(spec.get("grading", ""))


def test_ported_every_committed_sheet_agrees_with_v2():
    """The strongest v1 pattern: check the real, committed task texts against
    the very grid v2 grades. Every cell with a public spec must agree."""
    checked = 0
    for pid in _real_cells_with_spec():
        spec = json.loads((PROBLEMS / pid / "spec_public.json").read_text())
        task = (PROBLEMS / pid / "task.txt").read_text()
        key = {"id": pid, "dim": spec["dim"],
               "kind": spec.get("kind", "single"),
               "extent_a": spec.get("extent_a"),
               "extent_b": spec.get("extent_b")}
        sys.modules["c3grading.probes"].task_grid_agreement(
            pid, task, spec, key)
        checked += 1
    assert checked >= 10          # B1-B3 + D1-D8 today; grows with the set


def test_ported_the_stated_single_grid_reproduces_v2s_points():
    """Build the very point set a real single-code task describes and hand it
    to v2's acceptance check."""
    import re
    line = re.compile(
        r"PROBE POINTS: the (\d+) points given by x = \(i_x\+0\.5\)/(\d+)")
    checked = 0
    for pid in _real_cells_with_spec():
        spec = json.loads((PROBLEMS / pid / "spec_public.json").read_text())
        if spec.get("kind") == "coupled" or "codes" in spec:
            continue
        task = (PROBLEMS / pid / "task.txt").read_text()
        m = line.search(task)
        assert m, f"{pid}: no probe statement"
        M, dim = int(m.group(2)), spec["dim"]
        axis = [(i + 0.5) / M for i in range(M)]
        pts = [()]
        for _ in range(dim):
            pts = [p + (v,) for p in pts for v in axis]
        good, why = GB2.matches_probe_grid(pts, GB2.probe_grid(dim))
        assert good, f"{pid}: {why}"
        checked += 1
    assert checked >= 3


def test_probe_grid_transposed_order_is_accepted():
    """Six dev runs submitted the full correct grid with x varying fastest;
    rows carry their own coordinates, so order must not decide the verdict."""
    grid = GB2.probe_grid(2)
    transposed = sorted(grid, key=lambda p: (p[1], p[0]))
    assert transposed != grid
    good, why = GB2.matches_probe_grid(transposed, grid)
    assert good, why


def test_probe_grid_off_grid_point_still_fires():
    grid = GB2.probe_grid(2)
    bad = list(grid)
    bad[7] = (bad[7][0] + 0.004, bad[7][1])  # well beyond PROBE_TOL
    good, why = GB2.matches_probe_grid(bad, grid)
    assert not good and "not a prescribed" in why


def test_probe_grid_duplicated_point_still_fires():
    grid = GB2.probe_grid(2)
    dup = list(grid)
    dup[5] = dup[4]  # same count, one point twice, one missing
    good, why = GB2.matches_probe_grid(dup, grid)
    assert not good and "more than once" in why


def test_probe_grid_count_mismatch_still_fires():
    grid = GB2.probe_grid(2)
    good, why = GB2.matches_probe_grid(grid[:-1], grid)
    assert not good and "expected" in why


def test_submission_found_in_a_subdirectory(tmp_path):
    """Five round-1 cells organised output into work/results/ etc. and were
    graded as having submitted nothing, while the evidence check (rglob) saw
    their solver run in the same tree."""
    work = tmp_path / "work"
    (work / "results").mkdir(parents=True)
    (work / "results" / "solution_level1_A.csv").write_text("0,0,1\n")
    (work / "results" / "solution_level1_B.csv").write_text("0,0,2\n")
    levels, problems = GB2.sub.discover_levels(work, True, tmp_path)
    assert set(levels) == {1} and set(levels[1]) == {"A", "B"}, (levels, problems)


def test_result_txt_found_one_directory_down(tmp_path):
    work = tmp_path / "work"
    (work / "coupled_elasticity").mkdir(parents=True)
    (work / "coupled_elasticity" / "RESULT.txt").write_text("COULD_NOT_COMPLETE\n")
    assert "COULD_NOT_COMPLETE" in GB2.sub.result_text(tmp_path, work)


def test_submission_beside_the_sandbox_is_found(tmp_path):
    """C3-BARE wrote a complete converged set to the run root, not work/."""
    work = tmp_path / "work"
    work.mkdir()
    for lvl in (1, 2, 3):
        for side in ("A", "B"):
            (tmp_path / f"solution_level{lvl}_{side}.csv").write_text("0,0,1\n")
    levels, _ = GB2.sub.discover_levels(work, True, tmp_path)
    assert sorted(levels) == [1, 2, 3]


def test_preserved_evidence_is_never_graded(tmp_path):
    """out_of_sandbox_evidence/ holds files WE copied aside; grading them
    would let a quarantined scatter re-enter as a submission."""
    work = tmp_path / "work"
    work.mkdir()
    ev = tmp_path / "out_of_sandbox_evidence" / "tmp"
    ev.mkdir(parents=True)
    (ev / "solution_level1_A.csv").write_text("0,0,9\n")
    levels, _ = GB2.sub.discover_levels(work, True, tmp_path)
    assert levels == {}


def test_differing_duplicate_copies_are_reported(tmp_path):
    work = tmp_path / "work"
    (work / "a").mkdir(parents=True)
    (work / "b").mkdir(parents=True)
    (work / "a" / "solution_level1_A.csv").write_text("0,0,1\n")
    (work / "b" / "solution_level1_A.csv").write_text("0,0,2\n")
    _, problems = GB2.sub.discover_levels(work, True, tmp_path)
    assert any("more than one differing copy" in p for p in problems)
