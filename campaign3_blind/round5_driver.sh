#!/bin/bash
# ROUND 5 — does PROMINENCE change behaviour, and is the primitive rate falling?
#
# Round 4 answered one question sharply and left one open. Behaviour moved
# everywhere we aimed: coupled runs reaching `couple` 8/28 -> 13/27, convergence
# 6 -> 10, SPARTA sequences 1/6 -> 3/4 with the score reversing 116 points, and
# the FEM uplift +10.4 -> +12.5. But the pooled coupled score stayed 0/24 in
# both arms, with NO_SOLUTION_FILES still the dominant grader reason — even
# though all 27 coupled OASiS runs received the core payload carrying section
# 3b, which exists to prevent exactly that.
#
# The hypothesis this round tests: being SERVED is not being READ. Section 3b
# sits inside a 53 KB core payload. Four things have changed since round 4
# closed, three of them about reaching the agent rather than telling it more:
#
#   1. the write-the-deliverable-first rule now appends to EVERY knowledge
#      payload at the single point they all pass through, instead of living in
#      one backend's table where it reached 1 of 9 backends;
#   2. the DSMC entry states that level-to-level change is not automatically a
#      mesh effect, with two ways to measure the noise floor;
#   3. the critic refusal now names which half of the contract was broken and
#      says the review text was not lost;
#   4. the claim parser no longer books "UNKNOWN" as a denial (record only, no
#      verdict moves).
#
# If the coupled score moves, prominence was the difference. If it does not,
# the gap is not knowledge placement and the next round must look elsewhere —
# and that is a finding either way, not a failure.
#
# Same 32 cells, both arms, two replicates (seeds 6 and 7) for the same power
# as rounds 3 and 4. Earlier seeds are untouched and the runner refuses to
# overwrite an existing ledger.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
# FEBio is not on the default PATH; without this a participant that shells out
# to febio4 dies with FileNotFoundError and it looks like a model failure.
export PATH="/home/alexander/FEBio/bin:$PATH"

exec 9>/tmp/oasis_round5_driver.lock
flock -n 9 || { echo "REFUSING: another round5 driver holds the lock" >&2; exit 1; }
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >&2
  exit 1
fi

PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b_round5.log
CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"
CONC=4

echo ">>> round 5 (seeds 6+7, prominence test) start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge $CONC ]; do sleep 30; done; }

for s in 6 7; do
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
echo ">>> round 5 complete $(date '+%F %T')" >> "$LOG"
# Solvers have twice outlived their run and taken a core from whatever ran
# next. Sweep before the round is declared finished.
/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python \
  campaign3_blind/sweep_orphans.py --kill >> "$LOG" 2>&1
