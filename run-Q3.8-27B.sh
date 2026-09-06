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
# UD-Q4_K_XL read from the GGUF header 2026-09-02: 17.55 GB of tensors, mostly
# Q5_K (7.9 GB) + IQ4_XS/Q4_K (6.2 GB) + Q6_K (2.9 GB). output.weight is Q6_K
# (1.04 GB) and token_embd Q4_K (0.72 GB) -- both far smaller than the BF16
# pair in the NVFP4 VERY-HIGH file, so the whole file is ~2.1 GB SMALLER than
# NVFP4. The backbone itself is ~2.1 GB bigger, and the backbone is what gets
# split across the two cards, which is why --tensor-split differs from the
# NVFP4 script (see DFLASH2 below).
#
# Settings are a copy of run-Q3.8-27B-NVFP4.sh (DFlash2 sidecar + ngram-mod,
# reasoning-effort medium, same KV/batch/sampler flags). Only the model files
# and --tensor-split differ. Read that script for the DFlash2 and vision notes.
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
# DFLASH2 SPECULATIVE DECODING (replaces the baked-in MTP head, 2026-09-02)
# ----------------------------------------------------------------------
# Same drafting stack as run-Q3.8-27B-NVFP4.sh: the DFlash2 sidecar
# (Qwen3.8-27B-DFlash2-Q4_K_M-mrope.gguf, locally converted, 1.14 GB) with
# ngram-mod chained on. MTP alone measured 63.7 code / 40.4 prose on this box
# vs 93.1 / 57.4 for DFlash2 n-max 4, so draft-mtp is gone from this script.
# The old `--spec-type draft-mtp --spec-draft-n-max 3` block is in git history.
#
# The MTP head (blk.64.nextn.*, IQ4/Q6_K, ~0.4 GB) is still in the GGUF and
# still loaded; llama.cpp has no flag to skip it.
#
# TRIAL LOG 2026-09-02, b10734, ctx 120000, one cold bench.py pass per config
# (fresh container each time, --restart=no, temp 0). Repeats agree to ~0.5% on
# decode and ~3% on prefill. Columns: prose / code decode tok/s, 10k prefill.
#
#   --tensor-split 37,28 (the 2026-08-21 value)  OOM: 555 MiB draft compute
#                        buffer on the 4070. This 17.55 GB file is NOT the
#                        17.9 GB one that was measured then.
#   39,26 n4 ub512       63.6 / 86.5 / 1543   ~1.0 GB free on each card
#   40,25 n4 ub512       62.7 / 86.1 / 1542   787 MiB free on the 5070 Ti
#   n-max 3              63.3 / 82.6 / 1562
#   n-max 5              66.7 / 89.5 / 1606   <- +5% prose, +3.5% code vs 4
#   n-max 6              63.8 / 87.4 / 1658
#   n-max 8              60.3 / 92.0 / 1678   code up, prose (= thinking) down
#   -ub 256              66.9 / 89.4 / 1764   <- +10% prefill, -500 MiB VRAM
#   -ub 128              67.1 / 89.5 / 1394
#   -ub 384              66.9 / 89.6 / 1682
#   -ub 1024             OOM (1208 MiB compute buffer on the 4070)
#   -b 4096 (with 256)   65.7 / 89.5 / 1823   +3% prefill, -2% prose: rejected
#   -b 8192              63.6 / 89.4 / 1845   -5% prose: rejected
#   -ctkd/-ctvd f16      byte-identical drafts, +110 MiB: no gain
#   dflash without ngram byte-identical drafts on cold prompts (ngram needs
#                        generated text to match, kept for warm sessions)
#   --spec-draft-device  CUDA0 / CUDA1 both OOM at 39,26
#   37,28 -ub 256        59.0 / 96.8 / 1718   prose -12%: 2 more layers on the
#   38,27 -ub 256        58.2 / 76.9 / 1607   slower card cost more than the
#                                            ~1.4 GB it leaves free on the 4070
#   40,25 -ub 768        58.3 / 95.0 / 1491   worse on every axis
#   41,24 / 42,23 -ub 1024  OOM (the buffer just moves to the other card)
#   fill.py 30k/60k/100k/118k on the final config: VRAM flat at 10801 / 14966
#   MiB (4070 / 5070 Ti), zero errors, 47 tok/s decode at 118k depth.
#   The ~1.4 GB free on the 4070 at idle is real and stays free at full
#   context; nothing tested turns it into speed.
#
# Why 39,26 not 40,25: decode is a wash (bandwidth-bound, 1 layer = <1%), and
# 40,25 changes the arithmetic enough that temp-0 output diverges, so its
# numbers are not comparable anyway. 39,26 leaves ~1.5 GB free on the 5070 Ti
# with -ub 256, which is the vision headroom.
#
# Why -ub 256 beats 512 here when 768 lost on NVFP4: two-GPU layer split runs
# pipeline-parallel across ubatches, and 8 chunks per 2048 batch overlap the
# cards better than 4. Smaller than 256 and the per-launch overhead wins.
#
# For scale, NVFP4 VERY-HIGH on the same build and harness does about
# 65 / 94 / 1536 (2026-08-28 record), so this file is ~5% behind on code and
# level on prose and prefill after tuning.
#
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
DRAFT_FILE="${DRAFT_FILE:-Qwen3.8-27B-DFlash2-Q4_K_M-mrope.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

PARALLEL="${PARALLEL:-1}"
CONTEXT="${CONTEXT:-140000}"

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
    --model-draft "/models/${DRAFT_FILE}" \
    --mmproj "/models/${MMPROJ_FILE}" \
    --mmproj-offload \
    --image-min-tokens 1024 \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split 39,26 \
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
    --load-mode mlock \
    --jinja \
    --reasoning on \
    --reasoning-preserve \
    --reasoning-effort medium \
    --spec-type draft-dflash,ngram-mod \
    --spec-draft-n-max 5 \
    --spec-ngram-mod-n-match 24 \
    --spec-ngram-mod-n-min 4 \
    --spec-ngram-mod-n-max 48 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    -b 2048 \
    -ub 256 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
