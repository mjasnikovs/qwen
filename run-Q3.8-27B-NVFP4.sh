#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Qwen3.8-27B NVFP4 (esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF, VERY-HIGH tier).
#
# Same model and arch as run-Q3.8-27B.sh -- llama.cpp calls it `qwen35`, 64
# hidden layers + 1 MTP layer = 65 blocks, so --tensor-split still sums to 65.
# 16 x (3 x GatedDeltaNet -> 1 x GatedAttention), so only 16 of 64 layers hold a
# real KV cache. Gated attention: 24 Q heads, 4 KV heads, head_dim 256.
#
# VERY-HIGH tier = NVFP4 backbone (all attention + MLP) + BF16 lm_head +
# Q6_K token_embd + IQ4_XS MTP head. 18.34 GiB on disk.
#
# The MTP draft head is baked into the GGUF (blk.64.nextn.*), but it is NOT used
# any more -- see DFLASH2 below.
#
# DFLASH2
# -------
# Drafting is a DFlash2 sidecar (z-lab/Qwen3.8-27B-DFlash2-GGUF, Q4_K_M, 1.14 GB)
# instead of the baked-in MTP head. Block-diffusion drafter: it predicts a whole
# block in one pass and keeps top candidates per position.
#
# DFlash2 is not in upstream llama.cpp -- it is PR #27342, STILL OPEN as of
# 2026-08-24 (head 64f765f5, merged into upstream b10605). There is no separate
# dflash2 image: the single Dockerfile merges that PR into the pinned upstream
# ref and applies patches/*.patch (the vision fix, see VISION below).
# llama-turboquant:cuda is the patched DFlash2 build.
# Roll back with IMAGE=llama-turboquant:prev-dflash2 (the last two-image build),
# or IMAGE=llama-turboquant:prev plus the old --spec-type draft-mtp,ngram-mod
# block (see git history).
#
# ngram-mod DOES chain onto draft-dflash -- it contributed 11% of accepted
# tokens (draft positions 4+, past DFlash2's 4-wide block) and cost nothing:
# dflash-only was 93.3 / 59.9 / 1530 vs 99.9 / 60.1 / 1616 chained. A short
# probe will not show it; ngram needs generated text to match against.
#
# Draft width swept: n-max 4 wins prose by 21% over 7; 10 wins code but costs
# another 15% prefill. 4 is the pick for reasoning-heavy work.
#
# --tensor-split moved 43,22 -> 40,25: the 1.14 GB sidecar plus the 888 MiB
# mmproj no longer fit on the 5070 Ti at 43. Not a DFlash/vision incompatibility,
# just VRAM -- 40,25 loads vision AND the sidecar at the full 120k.
#
# VISION
# ------
# Images WORK, but they cost all the drafting. Measured cold on this box
# (2026-08-21, temp 0), decode tok/s, this config vs the same binary running
# --spec-type draft-mtp,ngram-mod --spec-draft-n-max 6 at --tensor-split 43,22:
#
#              draft-dflash   draft-mtp
#   code            94.6         65.2
#   prose           58.7         41.6
#   image           35.0         47.3
#   text turn after
#   an image        34.8         49.7
#
# Unpatched, an image request does not just slow down -- it ABORTS with
# `process: llama_decode(ctx_dft) failed rc=-1`. The DFlash inject batch copied
# the target's positions verbatim, but a multimodal target batch is M-RoPE and an
# image span repeats one temporal position, while ctx_dft (plain qwen3,
# n_pos_per_embd == 1) demands continuous positions.
# patches/0001-dflash-dense-inject-pos-for-vision.patch (the community fix from
# PR #27342, NOT merged into the PR head) numbers the injected rows densely and
# the request completes correctly.
#
# What the patch does NOT fix: draft() still positions its block at the target's
# TOKEN count while the draft KV ends at the target's M-RoPE position, so every
# draft decode fails and the log fills with `draft: llama_decode returned -1`.
# The answer is correct, just unaccelerated, for the WHOLE chat once an image is
# in it. If images are the main workload, switch to draft-mtp.
# RE-MEASURED 2026-08-24 on b10605 + PR head 64f765f5: unchanged. Image turn and
# the text turn after it both run 34.8 tok/s with draft_n = 0, and the follow-up
# turn added 86 more `llama_decode returned -1` lines. Still open.
#
# Alternatives measured and rejected 2026-08-21: draft-dspark (slower -- 79/35
# vs 95/59 -- and does not fit at -c 120000), swapping the ngram type (no effect;
# ngram-cache is worse), draft-eagle3 and draft-simple (no Qwen3.8 checkpoint).
#
# Caveat: the BF16 lm_head makes every verification pass more expensive.
# Upstream numbers put the BF16-head tiers at ~15.5 tok/s decode vs ~18.5 for the
# LOW/MEDIUM tiers (Q5_0/Q8_0 heads). BF16 buys output quality, not speed.
#
# NVFP4 is native only on the 5070 Ti (Blackwell). The 4070 SUPER (Ada) runs it
# through a dequant path, same as the Qwen3.6-27B-NVFP4 model already does.
#
# REASONING
# ---------
# `--reasoning on`, same as run-Q3.8-27B.sh. Same model, same chat template.
# With it off, the template prefills an empty pre-closed <think></think> and the
# model has nowhere to plan.
#
# Harnesses select the LEVEL per request -- it is not a server flag:
#   "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}
#   "chat_template_kwargs": {"enable_thinking": false}          # per-request off
#   "chat_template_kwargs": {"preserve_thinking": false}        # keep only latest
# Valid levels: xhigh (default), medium, low. The OpenAI top-level
# `reasoning_effort` field does NOT reach this template.
#
# The samplers below are still the INSTRUCT preset, not Qwen's thinking preset.

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
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-NVFP4-MTP-VERY-HIGH.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Qwen3.8-27B-NVFP4-BF16.gguf}"
DRAFT_FILE="${DRAFT_FILE:-Qwen3.8-27B-DFlash2-Q4_K_M.gguf}"
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
    --model-draft "/models/${DRAFT_FILE}" \
    --mmproj "/models/${MMPROJ_FILE}" \
    --mmproj-offload \
    --image-min-tokens 1024 \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split 40,25 \
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
    --spec-draft-n-max 4 \
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
    -ub 512 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
