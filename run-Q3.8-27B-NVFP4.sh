#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Qwen3.8-27B NVFP4 (esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF, HIGH tier).
#
# Same model and arch as run-Q3.8-27B.sh -- llama.cpp calls it `qwen35`, 64
# hidden layers + 1 MTP layer = 65 blocks, so --tensor-split still sums to 65.
# 16 x (3 x GatedDeltaNet -> 1 x GatedAttention), so only 16 of 64 layers hold a
# real KV cache. Gated attention: 24 Q heads, 4 KV heads, head_dim 256.
#
# HIGH tier = 448-tensor NVFP4 backbone (all attention + MLP) + BF16 lm_head +
# Q6_K token_embd + IQ4_XS MTP head. 16.36 GiB on disk, ~0.33 GiB SMALLER than
# the UD-Q4_K_XL file, so 120k context has slightly more headroom than the
# Q4_K_XL script, not less.
#
# The MTP draft head is baked into the GGUF (blk.64.nextn.*) -- no drafter file.
#
# Caveat: the BF16 lm_head makes every MTP verification pass more expensive.
# Upstream numbers put HIGH at ~15.5 tok/s decode vs ~18.5 for the LOW/MEDIUM
# tiers (Q5_0/Q8_0 heads). BF16 buys output quality, not speed. If decode
# matters more than quality, download the LOW file and point MODEL_FILE at it.
#
# NVFP4 is native only on the 5070 Ti (Blackwell). The 4070 SUPER (Ada) runs it
# through a dequant path, same as the Qwen3.6-27B-NVFP4 model already does.
#
# NOT YET RUN -- context and tensor-split are inherited, not measured. Check
# idle VRAM before trusting them.
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

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-host}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
# 65 blocks (64 layers + 1 MTP)
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-NVFP4-MTP-VERY-HIGH.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Qwen3.8-27B-NVFP4-BF16.gguf}"
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
    --tensor-split 43,22 \
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
    --spec-type draft-mtp,ngram-mod \
    --spec-draft-n-max 6 \
    --spec-draft-p-min 0.75 \
    --spec-ngram-mod-n-match 24 \
    --spec-ngram-mod-n-min 4 \
    --spec-ngram-mod-n-max 48 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
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
