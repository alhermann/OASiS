#!/usr/bin/env bash
# ROUND 9 — the post-fix re-measurement. Nothing we quote was measured on the
# build the agents now get.
#
# The last full matrix is seeds 14/15, run 29-30 Aug. EIGHT commits landed on
# 1 Sep, every one of them inside the channel an agent actually reads:
#
#   56c8da62  deal.II never told an agent how to read its solution off-node
#   e0cb041e  the per-side filename rule reached the physics path, not coupled
#   0cddebcf  OASiS handed the agent node data; 1 in 7 coupled runs submitted it
#   51d6ae2f  4C: SOLID is 3-D, and 2-D TSI cannot work in this build
#   fe304208  three things my own knowledge edits broke
#   ccd922e9  served knowledge quoted convergence orders, against our own rule
#   9f926294  the lean-physics switch (default OFF — this round runs FULL)
#   7e15a627  cut-served-volume REFUTED; the elaboration is load-bearing
#
# plus, earlier the same night: knowledge() now carries the universal core on
# 100% of calls instead of 11.5%, the coupling escape hatch is real, the
# pre-submission gate sees a forgery, verify_pde_consistency exists, and four
# grader defects that invented fabrications are gone.
#
# So the honest position is that 37.5%/41.7% single-code is a PRE-FIX number
# and there is no post-fix number at all beyond three cells at one seed. This
# round produces one.
#
# THE QUESTION, and it is a single one: on the build that exists today, what is
# the OASiS arm worth against bare on the same cells at the same seeds?
# Secondary, all readable from the same runs: does the coupled half still score
# zero; is verify_pde_consistency called when it would help; does the audit
# convert a wrong submission into a right one or into an honest incomplete.
#
# Full matrix, no sampling: 9 backends x both halves + all 14 coupled cells,
# two fresh seeds, both arms. 128 runs. Measured cost of this matrix: ~60
# credits (rounds 5, 6 and 8).
#
# This is a DEVELOPMENT round. Its cells are burnt and its score is not the
# paper's. The freeze criterion is a round that teaches OASiS nothing new
# (paper section 3.2), never a score being reached.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
export PATH="/home/alexander/FEBio/bin:$PATH"
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=campaign3_blind/round9.log

exec 9>/tmp/oasis_round9_driver.lock
flock -n 9 || { echo "REFUSING: another round9 driver holds the lock" >&2; exit 1; }

# The keys must be sealed before a round and after it. A round that runs beside
# a readable key directory is not a blind round, whatever the agent did.
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >> "$LOG"
  exit 1
fi

# Nothing CPU-heavy may run beside a measurement: a niced tier-2 fixture once
# ran at 1226% CPU next to a round and inflated its wall-clock and timeouts.
# `pgrep -fc` prints 0 AND exits 1 when nothing matches, so `|| echo 0` yields
# "0\n0" and this guard would refuse forever. `| wc -l` prints one number.
if [ "$(pgrep -f 'run_blind[.]py' 2>/dev/null | wc -l)" != "0" ]; then
  echo "REFUSING: agent runs already in flight; a round must have the machine" >> "$LOG"
  exit 1
fi

# CREDIT PREFLIGHT. Round 5 lost 18 coupled runs mid-round to HTTP 402 and they
# were briefly booked as model results. The floor is 100 — one full round plus
# two thirds in reserve — and it was set by measurement, not by caution.
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env 2>/dev/null; set +a
BAL=$(curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
        https://openrouter.ai/api/v1/credits 2>/dev/null \
      | "$OASIS_PYTHON" -c "
import sys, json
try:
    d = json.load(sys.stdin).get('data', {})
    print(f\"{d.get('total_credits', 0) - d.get('total_usage', 0):.2f}\")
except Exception:
    print('ERR')")
case "$BAL" in ERR|"") echo "REFUSING: could not read the credit balance" >> "$LOG"; exit 1 ;; esac
if [ "$(awk -v b="$BAL" 'BEGIN{print (b < 100)}')" = "1" ]; then
  echo "REFUSING: credit balance $BAL is below the 100 a full round needs" >> "$LOG"; exit 1
fi
echo ">>> credit preflight ok: $BAL remaining" >> "$LOG"

CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 \
C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"
SEEDS="96 97"              # never used by any round; checked against runs/
PAR=6                      # concurrent cells; the box has 32 threads

echo ">>> round 9 (post-fix re-measurement) start $(date '+%F %T')" >> "$LOG"
echo ">>> seeds $SEEDS, $(echo $CELLS | wc -w) cells, both arms, build $(git rev-parse --short HEAD)" >> "$LOG"

# mapfile + </dev/null: a backgrounded child inside `while read ... done < list`
# inherits the loop's stdin and eats the list — that cost 15 of 17 cells once.
mapfile -t CELL_ARR <<< "$(echo $CELLS | tr ' ' '\n')"

running=0
for s in $SEEDS; do
  for c in "${CELL_ARR[@]}"; do
    [ -z "$c" ] && continue
    ( for arm in BARE MCP; do
        d="campaign3_blind/runs/${c}_27b_${arm}_seed${s}"
        [ -f "$d/ledger.json" ] && continue        # resume, never overwrite
        "$PY" campaign3_blind/run_blind.py --model 27b --conditions "$arm" \
             --problems "$c" --seed "$s" --phase development >> "$LOG" 2>&1
      done
      echo "<<< $c seed $s done $(date '+%F %T')" >> "$LOG" ) < /dev/null &
    running=$((running + 1))
    if [ "$running" -ge "$PAR" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
  done
done
wait
echo ">>> round 9 complete $(date '+%F %T')" >> "$LOG"
"$OASIS_PYTHON" campaign3_blind/sweep_orphans.py --kill >> "$LOG" 2>&1
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  chmod 000 "$OASIS_BLIND_KEYS"
  echo ">>> keys re-sealed after the round" >> "$LOG"
fi
