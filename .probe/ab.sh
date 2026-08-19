#!/usr/bin/env bash
# One A/B arm: run the refine replay N times against whatever model llama-server
# currently has loaded, with the candidate guard OFF and then ON.
#
# Usage: ARM=qwen3.8 TRIALS=5 ./ab.sh
set -uo pipefail

S="${S:-/tmp/claude-1000/-home-edgars--pi-agent-extensions-pi-task/58efd1e7-45dc-405a-bd1f-ff5fefe30c75/scratchpad}"
ARM="${ARM:?set ARM}"
TRIALS="${TRIALS:-5}"
MAX_CALLS="${MAX_CALLS:-150}"
STIM="${STIM:-$S/refine-real.json}"
OUT="$S/ab-results.jsonl"

loaded=$(curl -s -m 5 http://127.0.0.1:8080/v1/models | grep -o 'Qwen[^"]*\.gguf' | head -1)
echo "arm=$ARM loaded=$loaded trials=$TRIALS stimulus=$(basename "$STIM")"

for g in 0 1; do
    STIMULUS="$STIM" ARM="$ARM" GUARD="$g" TRIALS="$TRIALS" MAX_CALLS="$MAX_CALLS" \
        node /home/edgars/hub/qwen/.probe/replay.mjs | tee -a "$OUT"
done
