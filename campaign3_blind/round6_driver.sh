#!/bin/bash
# ROUND 6 — does a JUST-IN-TIME notice do what static text could not?
#
# Three rounds have measured the same failure: runs that solve correctly and end
# with nothing a reader can find. Three of the five failing single-code OASiS
# runs in round 5 seed 6 made 85-107 tool calls, produced solver output, and
# wrote no summary at all.
#
# Two static interventions failed to move it. The rule filed in SPARTA's table
# reached 1 of 9 backends. The same rule appended to every knowledge payload
# reached 9 of 9, verified before round 5 launched, and the coupled score did
# not budge — which refuted placement as the explanation and retired that whole
# class of fix.
#
# So round 6 tests something different in KIND: run_simulation now checks, at
# the moment a solve SUCCEEDS, whether output exists with no human-readable
# summary beside it, and says so in its own response. Stateful and just-in-time,
# at the one moment numbers demonstrably exist and demonstrably are not saved.
#
# THE METRIC THIS ROUND IS FOR is not the solve rate. It is the share of runs
# that end with solver output and no written summary. If that falls, the
# intervention works and the solve rate may follow; if it does not, the failure
# is not about being reminded at all and round 7 looks at capability.
#
# Seeds 8 and 9, all 32 cells, both arms, 128 runs. The 18 credit casualties
# from round 5 are deliberately NOT folded in here — see CONVERGENCE.md; they
# would mix knowledge states within round 5.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
export PATH="/home/alexander/FEBio/bin:$PATH"
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen

exec 9>/tmp/oasis_round6_driver.lock
flock -n 9 || { echo "REFUSING: another round6 driver holds the lock" >&2; exit 1; }
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >&2
  exit 1
fi

# CREDIT PREFLIGHT. Round 5 lost 18 coupled runs mid-round to HTTP 402 and they
# were briefly booked as model results. A round is worth nothing if it dies of
# an unpaid invoice halfway, so refuse to start without enough headroom for the
# whole matrix. Round 5's 128 runs cost roughly 60 credits, so 150 is a
# comfortable floor rather than a tight estimate.
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
if [ "$(awk -v b="$BAL" 'BEGIN{print (b < 150)}')" = "1" ]; then
  echo "REFUSING: credit balance $BAL is below the 150 needed for a full round." >&2
  echo "Top up at https://openrouter.ai/settings/credits and re-launch." >&2
  exit 1
fi
echo "credit preflight ok: $BAL remaining"

PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b_round6.log
CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"
CONC=4

echo ">>> round 6 (seeds 8+9, just-in-time notice) start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge $CONC ]; do sleep 30; done; }

for s in 8 9; do
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
echo ">>> round 6 complete $(date '+%F %T')" >> "$LOG"
"$OASIS_PYTHON" campaign3_blind/sweep_orphans.py --kill >> "$LOG" 2>&1
