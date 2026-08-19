#!/usr/bin/env bash
# Top up the two cells that decide the verdict: A (production today) and C (the
# candidate: thinking ON, prompt left exactly as pi-task emits it, /no_think and
# all). E is not the shipped config -- it strips /no_think, which C proves is
# unnecessary -- so it stays as a supporting cell.
set -uo pipefail
cd "$(dirname "$0")"
S=/tmp/claude-1000/-home-edgars--pi-agent-extensions-pi-task/58efd1e7-45dc-405a-bd1f-ff5fefe30c75/scratchpad
STIM=$S/refine-real.json
run() {
    local name=$1 n=$2; shift 2
    echo "[$(date +%H:%M:%S)] TOPUP $name +$n"
    env STIMULUS="$STIM" ROOT=/home/edgars/hub/mx5-n TRIALS="$n" MAX_CALLS=120 \
        ARM="38-$name" "$@" node replay.mjs >>"$S/matrix/38-$name.jsonl" 2>>"$S/matrix/38-$name.err"
    echo "[$(date +%H:%M:%S)] TOPUP $name done ($(wc -l <"$S/matrix/38-$name.jsonl") total)"
}
run A 20 TEMP=0.7 TOP_P=0.8 TOP_K=20 MIN_P=0.0 PP=1.5
run C 22 TEMP=1.0 TOP_P=0.95 TOP_K=20 MIN_P=0.0 PP=0.0 THINK=1
echo "[$(date +%H:%M:%S)] TOPUP COMPLETE"
