#!/bin/bash
# ROUND 2 — 27B, seed 1, all 32 cells, both arms.
#
# Same seed as round 1 ON PURPOSE: the model's sampling is then as close to
# identical as the provider allows, so the difference between the two rounds
# is attributable to the twelve defects closed between them rather than to a
# different draw. Round 1's runs are archived under runs_round1/ — this
# driver writes into a clean runs/ and the runner's resume-on-existing-ledger
# shortcut therefore cannot serve a round-1 result as a round-2 one.
#
# Interleaved arms (BARE then MCP per cell) so background load hits both
# alike. Three cells concurrently. Development phase: results feed
# post-mortems and the convergence decision, never the paper's table.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
# No X11: DISPLAY with a stale cookie made round-1 cells conclude a solver
# "requires a display" and treat it as fatal.
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen

P=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b_round2.log

# ONE DRIVER AT A TIME — enforced with a LOCK, not a process grep.
#
# A previous launch whose cells were all refusing at the custody gate stayed
# alive and kept cycling; launching again put two drivers on the same 32 cells
# and six runners in flight, each cell at risk of being run twice into the same
# ledger path. The first guard here used pgrep on this script's own name, which
# matched its OWN subshells and refused every launch including the legitimate
# one — the same self-match footgun that once stalled a relauncher for eleven
# hours, and that killed the shell writing this very patch. A lock has no such
# ambiguity: this process holds it or it does not.
exec 9>/tmp/oasis_round2_driver.lock
if ! flock -n 9; then
  echo "REFUSING: another round2 driver holds the lock" >&2
  exit 1
fi

# The keys must already be sealed. The runner checks too, but failing here
# costs one second instead of 32 refused cells.
if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >&2
  exit 1
fi

CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"

echo ">>> round 2 (27b, seed 1) start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 30; done; }

pair() {
  local c=$1
  $P campaign3_blind/run_blind.py --model 27b --conditions BARE --problems "$c" \
      --seed 1 --phase development >> "$LOG" 2>&1
  $P campaign3_blind/run_blind.py --model 27b --conditions MCP  --problems "$c" \
      --seed 1 --phase development >> "$LOG" 2>&1
  echo "<<< cell $c done $(date '+%F %T')" >> "$LOG"
}

for c in $CELLS; do
  throttle
  pair "$c" &
done
wait
echo ">>> round 2 complete $(date '+%F %T')" >> "$LOG"
