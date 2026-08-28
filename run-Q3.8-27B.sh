#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Qwen3.8-27B (unsloth/Qwen3.8-27B-GGUF, UD-Q4_K_XL).
#
# Same arch family as Qwen3.6-27B: llama.cpp calls it `qwen35`. 64 hidden layers
# + 1 MTP layer = 65 blocks, which is why --tensor-split still sums to 65.
# Hidden layout is 16 x (3 x GatedDeltaNet -> 1 x GatedAttention), so only 16 of
# the 64 layers hold a real KV cache; the other 48 hold a fixed-size recurrent
# state. That is what makes long context affordable on a 28GB rig.
# Gated attention geometry: 24 Q heads, 4 KV heads, head_dim 256 (= 2x128, so
# q8_0 / f16 KV are block-compatible).
#
# UD-Q4_K_XL is 17.9GB on disk vs 15.7GB for the old NVFP4 27B, i.e. ~2.2GB less
# VRAM left for KV. Context therefore starts at 100k, not 120k. Raise it only
# after checking idle VRAM headroom -- see the Qwen3.6-27B KV tuning notes.
#
# THINKING IS ON AND STAYS ON
# ---------------------------
# Qwen3.8 ships thinking on by default and trains its agentic planning INTO the
# trace. `--reasoning off` was inherited from the Qwen3.6 scripts, where it was
# harmless because thinking there was only narration. On 3.8 it makes the chat
# template prefill an empty pre-closed <think></think>, leaving the model nowhere
# to plan -- so it plans by PROBING THE FILESYSTEM instead.
#
# Measured on the pi-task `refine` phase, captured live prompt, greenfield repo:
#   reasoning off: 8/40 runs never terminated (up to 122 tool calls, 163 failed
#                  reads); 9/40 produced no usable answer.
#   reasoning on : 0/25 ran away, 25/25 usable, median 1 tool call, 18 failed reads.
#   Fisher exact  p=0.019 non-termination, p=0.010 usable output.
#   Cost         ~48% more wall clock per usable answer (176s vs 119s).
# presence_penalty was tested as the cause and REFUTED (p=0.74). The samplers
# below are simply Qwen's documented THINKING preset, which pairs with reasoning on.
#
# Harnesses select the LEVEL per request -- it is not a server flag. The template
# reads it and injects a matching system line. Valid: xhigh (default), medium, low;
# anything else returns HTTP 500 from the template, so a typo fails loudly.
#
#   "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}
#   "chat_template_kwargs": {"enable_thinking": false}          # per-request off
#   "chat_template_kwargs": {"preserve_thinking": false}        # keep only latest
#
# Qwen's own warning, worth heeding before anyone turns the knob down to save
# time: "In multi-turn agentic tasks, lower reasoning effort does not always
# reduce overall task completion time... it can also lead to insufficient
# analysis, more failures, and repeated retries." That is exactly the failure
# this script's `--reasoning on` was set to fix, so treat `low` with suspicion.
#
# `--reasoning-preserve` below is the server-side form of Qwen's preserve_thinking,
# on by default upstream. It keeps the trace from earlier turns instead of only
# the last, which the card calls out as important for agents and for KV reuse.
#
# Three traps found the hard way:
#   - The OpenAI top-level `reasoning_effort` field does NOT reach this template.
#     Verified via /apply-template: the think block stayed pre-closed. It must go
#     inside chat_template_kwargs.
#   - The kwarg BEATS Qwen's in-prompt `/no_think` soft switch. With thinking on
#     and `/no_think` still in the prompt the model produced a median 17k-char
#     trace anyway (n=25). A harness cannot disable thinking by prompt text alone.
#   - The samplers below are SERVER-WIDE and set to Qwen's THINKING preset. A
#     harness that turns thinking off per request should also send the instruct
#     preset per request (temp 0.7, top_p 0.80, presence_penalty 1.5), otherwise
#     it runs non-thinking decoding on thinking-tuned sampling.
#
# MTP SPECULATIVE DECODING -- RE-ENABLED ON b10453, STILL ON PROBATION
# --------------------------------------------------------------------
# This was removed after the MTP draft path aborted in `common_speculative_draft`
# ("Aborted (core dumped)"), the container silently restarting under
# --restart=unless-stopped so it looked like a hung agent. The crash rate tracked
# thinking: 0 crashes across 40 thinking-off trials, then 6/20, 12/20, 20/20 once
# thinking was on. Old flags also carried ngram-mod:
#   --spec-type draft-mtp,ngram-mod --spec-draft-n-max 3
#   --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 48
#
# What changed, and why the config below is narrower than the old one:
#
#   - ngram-mod is NOT restored. Upstream #21815 (open) is a GGML_ABORT from stale
#     speculative ngram state after a context shift. Thinking-on means long traces
#     and context growth, which is exactly when that fires -- the same correlation
#     the crash data shows. MTP alone halves the abort surface. If the abort
#     returns with MTP only, ngram was never the cause and MTP is the bug.
#   - --split-mode layer must stay. Upstream #25522 is MTP crashing on multi-GPU
#     when -sm tensor cuts a 1-KV-head layer; -sm layer is the accepted
#     workaround. This script already used layer, so #25522 is NOT our crash.
#     (Separately, -sm tensor was measured a loss on this box anyway.)
#   - n-max 3 is upstream's default and the measured optimum. PR #27173 benchmarks
#     this exact model (Qwen3.8-27B, draft-mtp, temp 0) and reports depth 3 best on
#     stock upstream; depth 5 only wins with that PR's unmerged chain-draft patch
#     (LLAMA_SPEC_CHAIN). Still open, still not in LLAMA_REF (re-checked b10665,
#     2026-08-28). Matches the 35B DFlash result:
#     small n_max wins, deep drafts are a net loss.
#   - The draft KV inherits -ctkd/-ctvd q8_0 already set below. Upstream default is
#     f16, which costs VRAM this box does not have spare.
#
# Two open quality/VRAM bugs to watch, neither a crash:
#   - #25618: draft-mtp on a QUANTIZED target diverges from vanilla under greedy
#     decode. Target here is UD-Q4_K_XL, so this applies. At --temp 1.0 the
#     divergence sits under sampling noise, but it is real. Do not assume
#     spec-decode is lossless here.
#   - #27155: draft KV cache leaking ~10MB per PP+TG cycle until OOM. Reported for
#     DSpark, not MTP, but check VRAM after a long session before trusting it.
#
# To roll back, delete the two --spec-* lines. Nothing else depends on them.

# No env exports here. Verified 2026-08-24 against the pinned llama.cpp ref:
#   CUDA_MALLOC_ASYNC_SUPPORTED -- not read by llama.cpp/ggml at all.
#   GGML_CUDA_FORCE_MMQ         -- compile-time cmake option now, not an
#                                  env var (ggml/CMakeLists.txt, default OFF).
# Both were also never reaching the container: docker create only forwards
# the -e flags listed below, not the host shell's exports.

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-host}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
# 65 blocks (64 layers + 1 MTP)
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-UD-Q4_K_XL.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Qwen3.8-27B-F16.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

PARALLEL="${PARALLEL:-1}"
CONTEXT="${CONTEXT:-120000}"

NAME="${NAME:-llama-turboquant}"

if docker inspect "${NAME}" >/dev/null 2>&1; then
    docker rm -f "${NAME}" >/dev/null
fi

docker create \
    --name "${NAME}" \
    --restart=unless-stopped \
    --gpus all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUDA_VISIBLE_DEVICES=1,0 \
    --memory=30g \
    --memory-swap=46g \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --ulimit core=0 \
    --network "${NETWORK}" \
    -v "$(pwd)/models:/models:ro" \
    -v "$(pwd)/scripts:/scripts:ro" \
    --entrypoint /scripts/entrypoint.sh \
    "${IMAGE}" \
    --model "/models/${MODEL_FILE}" \
    --mmproj "/models/${MMPROJ_FILE}" \
    --mmproj-offload \
    --image-min-tokens 1024 \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split 39,27 \
    -fit off \
    --flash-attn on \
    -c "${CONTEXT}" \
    -n -1 \
    --parallel "${PARALLEL}" \
    --kv-unified \
    -ctk q8_0 \
    -ctv q8_0 \
    -ctkd q8_0 \
    -ctvd q8_0 \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --load-mode mlock \
    --jinja \
    --reasoning on \
    --reasoning-preserve \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    -b 2048 \
    -ub 512 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
