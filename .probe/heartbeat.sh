#!/usr/bin/env bash
# 10-minute heartbeat for the Qwen3.8 refine-wander investigation.
#
# Prints one status block every 10 minutes so a long unattended matrix run stays
# visible: which model is loaded, how far each cell has got, and whether the
# server is actually busy (a stalled slot looks identical to a slow trial
# without this).
S=/tmp/claude-1000/-home-edgars--pi-agent-extensions-pi-task/58efd1e7-45dc-405a-bd1f-ff5fefe30c75/scratchpad
INTERVAL=${INTERVAL:-600}

while true; do
    ts=$(date +%H:%M:%S)
    model=$(curl -s -m 5 http://127.0.0.1:8080/props 2>/dev/null \
        | grep -o '"model_path":"[^"]*"' | cut -d'"' -f4 | xargs -r basename)
    busy=$(curl -s -m 5 http://127.0.0.1:8080/slots 2>/dev/null \
        | grep -o '"is_processing":[a-z]*' | head -1 | cut -d: -f2)
    ntok=$(curl -s -m 5 http://127.0.0.1:8080/slots 2>/dev/null \
        | grep -o '"n_prompt_tokens":[0-9]*' | head -1 | cut -d: -f2)
    cells=""
    for f in "$S"/matrix/*.jsonl; do
        [ -e "$f" ] || continue
        cells="$cells $(basename "$f" .jsonl)=$(wc -l <"$f" | tr -d ' ')"
    done
    last=$(tail -1 "$S"/matrix38.log 2>/dev/null | cut -c1-100)
    echo "HB $ts model=${model:-none} busy=${busy:-?} prompt_tok=${ntok:-?}"
    echo "HB $ts cells:${cells:- none}"
    echo "HB $ts log: ${last:-none}"
    sleep "$INTERVAL"
done
