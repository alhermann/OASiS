"""Every grade must trace to a ledger; every ledger to a grade — rerunnably.

Pre-freeze requirement (#63): the paper's numbers come from grades files, and
each graded cell must be backed by exactly one run ledger, live or archived.
A grade without a ledger is a number with no run behind it; a ledger without a
grade is a run silently dropped from the round. Quarantined casualties are
accounted separately — they are EXPECTED to have no grade.

Seed-1 runs live under rounds/round1_20260815/runs (archived when the round
closed); the first version of this scan looked only at runs/ and reported all
64 seed-1 grades as unbacked.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent


def ledger_cells(seed: int) -> set:
    out = set()
    roots = [D / "runs"]
    roots += [p / "runs" for p in (D / "rounds").iterdir()
              if (p / "runs").is_dir()] if (D / "rounds").is_dir() else []
    for root in roots:
        for led in root.glob(f"*_seed{seed}/ledger.json"):
            name = led.parent.name              # CELL_27b_ARM_seedN
            cell_arm = name.rsplit("_seed", 1)[0].replace("_27b", "")
            out.add(cell_arm)
    return out


def grades_file(seed: int) -> Path:
    if seed == 1:
        # seed 1 was graded twice (round 1, then round 2 on the same seed);
        # the round-2 file is the one every pooling uses.
        return D / "dev_grades_27b_seed1_round2.json"
    return D / f"dev_grades_27b_seed{seed}.json"


def main() -> int:
    bad = 0
    print(f"{'seed':>4} {'ledgers':>8} {'graded':>7} "
          f"{'unbacked-grades':>16} {'ungraded-ledgers':>17} {'quarantined':>12}")
    for seed in range(1, 20):
        gf = grades_file(seed)
        if not gf.is_file():
            continue
        graded = set(json.loads(gf.read_text()))
        led = ledger_cells(seed)
        q = len(list((D / "runs_quarantine").glob(f"*_seed{seed}_*"))) \
            if (D / "runs_quarantine").is_dir() else 0
        unbacked = sorted(graded - led)
        ungraded = sorted(led - graded)
        print(f"{seed:>4} {len(led):>8} {len(graded):>7} "
              f"{len(unbacked):>16} {len(ungraded):>17} {q:>12}")
        for x in unbacked[:3]:
            print(f"       grade without ledger: {x}")
        for x in ungraded[:3]:
            print(f"       ledger without grade: {x}")
        bad += len(unbacked) + len(ungraded)
    print(f"\n{'CONSISTENT' if bad == 0 else f'{bad} MISMATCHES'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
