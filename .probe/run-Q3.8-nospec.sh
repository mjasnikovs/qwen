#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

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

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

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
    --reasoning off \
    --reasoning-preserve \
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
