#!/bin/bash
# ROUND 7 — the audit round: does looking at your own output convert wrong
# answers into right ones?
#
# Round 5/6 diagnosis, read cell by cell: 69% of OASiS runs submit a complete
# answer and most are wrong, in two shapes visible in the agent's own files —
# error levels FLAT across refinement (14/18: a source built but never wired
# in, fields of literal zeros, tolerance floors) and a rate exactly one order
# low (4/18: element degree, locking). Nothing in the loop ever made the agent
# LOOK at its own numbers: the critic reviews the plan before the run, the
# grader the answer after, and 51/53 runs execute via run_bash where no check
# lives.
#
# What lands with this round, none of it seen by rounds <= 6:
#   * audit_results — a tool reading ONLY the agent's own files: near-zero
#     field, floor, order-vs-claim, non-monotone. Calibrated on 40 graded
#     runs: 0 false alarms on CORRECT, 15/18 known-wrong caught.
#   * the instruction to run it before submitting, in the universal block
#     every knowledge payload carries (the one channel read 53/53);
#   * the FEBio door-in-the-wall: separable f(x,y)*g(t) via nodal map + load
#     curve, after an agent read my warning as "impossible" and quit.
#
# THE METRIC: the submitted-and-wrong share of OASiS single-code runs
# (rounds 4-6 baseline: 18 of ~48). If audit findings get fixed rather than
# ignored, that share falls and the solve rate rises without one new physics
# fact. Secondary: audit_results call rate (did they even run it), and
# coupled cells writing field files.
#
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
export PATH="/home/alexander/FEBio/bin:$PATH"
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen

exec 9>/tmp/oasis_round7_driver.lock
flock -n 9 || { echo "REFUSING: another round7 driver holds the lock" >&2; exit 1; }
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >&2
  exit 1
fi

# CREDIT PREFLIGHT. Round 5 lost 18 coupled runs mid-round to HTTP 402 and they
# were briefly booked as model results. A round is worth nothing if it dies of
# an unpaid invoice halfway, so refuse to start without enough headroom for the
# whole matrix. MEASURED cost per 128-run round: ~60 credits (round 5) and 59
# (round 6, 186.37 -> 127.25). The floor is 100 — a full round plus two thirds
# of one in reserve. The first version guessed 150 before any round had been
# costed and refused a launch that 127.25 covers with 2x margin; a floor may
# be lowered by a MEASUREMENT, never by impatience, and this one is.
BAL=$(curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        https://openrouter.ai/api/v1/credits 2>/dev/null \
      | "$OASIS_PYTHON" -c "
import sys, json
try:
    d = json.load(sys.stdin).get('data', {})
    print(f\"{d.get('total_credits', 0) - d.get('total_usage', 0):.2f}\")
except Exception:
    print('ERR')")
case "$BAL" in
  ERR|"") echo "REFUSING: could not read the credit balance" >&2; exit 1 ;;
esac
if [ "$(awk -v b="$BAL" 'BEGIN{print (b < 100)}')" = "1" ]; then
  echo "REFUSING: credit balance $BAL is below the 100 needed for a full round." >&2
  echo "Top up at https://openrouter.ai/settings/credits and re-launch." >&2
  exit 1
fi
echo "credit preflight ok: $BAL remaining"

PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b_round7.log
CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"
CONC=4

echo ">>> round 7 (seeds 10+11, just-in-time notice) start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge $CONC ]; do sleep 30; done; }

for s in 10 11; do
  for c in $CELLS; do
    throttle
    ( $PY campaign3_blind/run_blind.py --model 27b --conditions BARE \
          --problems "$c" --seed $s --phase development >> "$LOG" 2>&1
      $PY campaign3_blind/run_blind.py --model 27b --conditions MCP \
          --problems "$c" --seed $s --phase development >> "$LOG" 2>&1
      echo "<<< cell $c seed $s done $(date '+%F %T')" >> "$LOG" ) &
  done
done
wait
echo ">>> round 7 complete $(date '+%F %T')" >> "$LOG"
"$OASIS_PYTHON" campaign3_blind/sweep_orphans.py --kill >> "$LOG" 2>&1
