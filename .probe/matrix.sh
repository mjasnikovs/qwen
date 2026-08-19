#!/usr/bin/env bash
# Phase 1+2 of the Qwen3.8 refine-wander investigation.
#
# 2x2 on ONE variable pair, both of which are MODEL SETTINGS, not harness caps:
#   THINK  -- the server runs `--reasoning off`, which makes the chat template
#             prefill an empty pre-closed <think></think>. Qwen3.8 ships thinking
#             ON by default at reasoning_effort=xhigh, and its agentic planning is
#             trained to live in that trace. THINK=1 re-opens it per request.
#   PP     -- presence_penalty. Server default is 1.5 (Qwen's own NON-thinking
#             preset). Penalties run FIRST in the sampler chain and cover the last
#             64 tokens, so re-emitting a path just read is actively punished.
#             That manufactures breadth. Qwen's THINKING preset is pp=0.0.
#
# Cells:
#   A think=off pp=1.5  -- current production config (the broken one)
#   B think=off pp=0.0  -- isolates the presence penalty alone
#   C think=on  pp=0.0  -- Qwen's documented thinking preset (temp 1.0/top_p 0.95)
#   D think=on  pp=1.5  -- isolates thinking alone, samplers held at baseline
set -uo pipefail
cd "$(dirname "$0")"

S=/tmp/claude-1000/-home-edgars--pi-agent-extensions-pi-task/58efd1e7-45dc-405a-bd1f-ff5fefe30c75/scratchpad
STIM=$S/refine-real.json
OUT=${OUT:-$S/matrix}
N=${N:-20}
CAP=${CAP:-120}
ROOTDIR=${ROOTDIR:-/home/edgars/hub/mx5-n}
TAG=${TAG:-38}

mkdir -p "$OUT"

run_cell() {
    local name=$1; shift
    local f="$OUT/${TAG}-${name}.jsonl"
    echo "[$(date +%H:%M:%S)] CELL $TAG-$name start n=$N cap=$CAP -> $f"
    env STIMULUS="$STIM" ROOT="$ROOTDIR" TRIALS="$N" MAX_CALLS="$CAP" \
        ARM="$TAG-$name" "$@" node replay.mjs >>"$f" 2>>"$OUT/${TAG}-${name}.err"
    echo "[$(date +%H:%M:%S)] CELL $TAG-$name done ($(wc -l <"$f") lines)"
}

# Baseline samplers = the server defaults, stated explicitly so every cell is
# byte-comparable rather than relying on whatever the container happens to hold.
BASE_S=(TEMP=0.7 TOP_P=0.8 TOP_K=20 MIN_P=0.0)
THINK_S=(TEMP=1.0 TOP_P=0.95 TOP_K=20 MIN_P=0.0)

for cell in "${CELLS:-A B C D}"; do :; done
CELLS=${CELLS:-A B C D}

for c in $CELLS; do
    case $c in
        A) run_cell A "${BASE_S[@]}"  PP=1.5 ;;
        B) run_cell B "${BASE_S[@]}"  PP=0.0 ;;
        C) run_cell C "${THINK_S[@]}" PP=0.0 THINK=1 ;;
        D) run_cell D "${BASE_S[@]}"  PP=1.5 THINK=1 ;;
        # E is the FULL candidate fix: server thinking on, Qwen's own thinking
        # sampler preset, and pi-task's `/no_think` removed from the prompt.
        E) run_cell E "${THINK_S[@]}" PP=0.0 THINK=1 STRIP_NOTHINK=1 ;;
        # F isolates the soft switch alone. With the server still pre-closing the
        # think block this MUST look like A -- if it does not, /no_think is doing
        # something beyond thinking control and the story is wrong.
        F) run_cell F "${BASE_S[@]}"  PP=1.5 STRIP_NOTHINK=1 ;;
    esac
done
echo "[$(date +%H:%M:%S)] MATRIX $TAG COMPLETE"
