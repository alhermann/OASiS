"""Calibrate the self-audit against every graded run, not four hand-picks.

The audit is only shippable with two measured numbers:
  * false-alarm rate on runs the grader scored CORRECT (must be ~0: an audit
    that cries wolf on good work teaches agents to ignore it), and
  * catch rate on runs that submitted and scored wrong (the population the
    audit exists for).
Case-by-case tuning against three hand-picked failures is how the prototype
got overfitted twice today; this harness is the antidote.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tools.result_audit import audit   # noqa: E402

D = Path("/home/alexander/Schreibtisch/ofa-v2/campaign3_blind")
RUNS = D / "runs"
FEM = ["FE1", "FE2", "DL1", "DL2", "NG1", "NG2", "SK1", "SK2",
       "KR1", "KR2", "DU1", "DU2", "FB1", "FB2", "FC1", "FC2"]
GOOD = {"CORRECT", "CORRECT_SUPERCONVERGENT"}
WRONG_SUBMITTED = {"COMPLETED_UNPHYSICAL", "CONFIDENTLY_WRONG"}

fa, fa_n = [], 0          # false alarms among CORRECT
hit, miss = [], []        # among wrong-submitted
for seed in (4, 5, 6, 7):
    gf = D / f"dev_grades_27b_seed{seed}.json"
    if not gf.is_file():
        continue
    g = json.loads(gf.read_text())
    for cell in FEM:
        r = g.get(f"{cell}_MCP")
        if not r:
            continue
        out = r.get("outcome") or ""
        wd = RUNS / f"{cell}_27b_MCP_seed{seed}" / "work"
        if not wd.is_dir():
            continue
        theo = r.get("theoretical_order")
        a = audit(str(wd), claimed_order=theo)
        name = f"{cell}_s{seed}"
        if out in GOOD:
            fa_n += 1
            if not a["clean"]:
                fa.append((name, [f["sequence"] + ": " +
                                  f["finding"][:42] for f in a["findings"]]))
        elif out in WRONG_SUBMITTED:
            (hit if not a["clean"] else miss).append(name)

print(f"  CORRECT runs audited      : {fa_n}")
print(f"  false alarms              : {len(fa)}  "
      f"({100*len(fa)/max(1,fa_n):.0f}%)")
for n, whys in fa:
    print(f"    FALSE ALARM {n}: {whys}")
print(f"  wrong-submitted audited   : {len(hit)+len(miss)}")
print(f"  caught before submission  : {len(hit)}  "
      f"({100*len(hit)/max(1,len(hit)+len(miss)):.0f}%)")
print(f"    caught: {' '.join(hit)}")
print(f"    missed: {' '.join(miss)}")
