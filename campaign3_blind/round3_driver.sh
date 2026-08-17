#!/bin/bash
# ROUND 3 — replicates 2 and 3 of the SAME configuration as round 2.
#
# Round 2 measured one run per cell. A repeatability probe then showed only
# 1 of 8 cells reproduced its own verdict on identical seed, code and
# knowledge, with three cells giving three different verdicts in three runs.
# A single run per cell therefore supports no claim in either direction.
#
# Nothing the agent sees has changed since round 2 launched (verified by diff
# over src/tools, src/backends, src/core, data/coupling_participants,
# langgraph_eval/agent.py and the problems), so round 2's 64 runs ARE
# replicate 1. This driver adds replicates 2 and 3: seeds 2 and 3, all 32
# cells, both arms, 128 runs.
#
# The seed does not pin the trajectory on a hosted endpoint — it is simply a
# label that keeps the run directories apart.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen

exec 9>/tmp/oasis_round3_driver.lock
flock -n 9 || { echo "REFUSING: another round3 driver holds the lock" >&2; exit 1; }
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >&2
  exit 1
fi

PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b_round3.log
CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"
CONC=4

echo ">>> round 3 (replicates 2+3) start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge $CONC ]; do sleep 30; done; }

for s in 2 3; do
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
echo ">>> round 3 complete $(date '+%F %T')" >> "$LOG"
