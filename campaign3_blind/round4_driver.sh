#!/bin/bash
# ROUND 4 — the first round after the knowledge changed, and the round that
# tests the freeze criterion rather than the model.
#
# Round 3 produced nine primitives, and the two biggest were ours, not the
# model's: six couplings converged and scored zero for want of the four steps
# between a converged coupling and a submitted answer, and the participants we
# ship could not express a source term at all — the elastic ones hard-wired a
# zero body force outside the block we tell agents to edit, and eight heat ones
# carried the source as a single number while every problem here has a
# polynomial one. Twenty of 28 coupled OASiS runs never reached `couple`; four
# sampled at random all wrote COULD_NOT_COMPLETE naming those gaps.
#
# All of that is now fixed and verified by running. So the question this round
# asks is the paper's own:
#
#     DID THIS ROUND TEACH OASiS ANYTHING NEW AND REUSABLE?
#
# If the primitive count falls sharply, we are approaching the freeze. If it
# does not, we are not, and the answer is another round — not a lowered bar.
#
# Same 32 cells, both arms, two replicates (seeds 4 and 5) so the result has
# the same power as round 3. Seeds 2 and 3 are round 3 and are not touched;
# the runner refuses to overwrite an existing ledger in any case.
#
# The seed does not pin the trajectory on a hosted endpoint — it is a label
# that keeps run directories apart.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
# FEBio is not on the default PATH; a participant that shells out to febio4
# fails with FileNotFoundError without this, which would look like a model
# failure in the ledger.
export PATH="/home/alexander/FEBio/bin:$PATH"

exec 9>/tmp/oasis_round4_driver.lock
flock -n 9 || { echo "REFUSING: another round4 driver holds the lock" >&2; exit 1; }
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >&2
  exit 1
fi

PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b_round4.log
CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"
CONC=4

echo ">>> round 4 (seeds 4+5, post-knowledge-fix) start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge $CONC ]; do sleep 30; done; }

for s in 4 5; do
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
echo ">>> round 4 complete $(date '+%F %T')" >> "$LOG"
