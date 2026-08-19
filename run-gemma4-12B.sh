#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Gemma 4 12B QAT (UD-Q4_K_XL) + MTP drafter, single GPU: RTX 5070 Ti (16 GB).
#
# CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=1 pins this to the
# 5070 Ti (PCI 26:00.0), the faster card. The 4070 SUPER (PCI 25:00.0) is
# deliberately NOT exposed, so nothing can spill onto it. Inside the container
# the 5070 Ti is the only visible device, so --main-gpu 0 still refers to it.
# See run-Q3.6-27B.sh for the two-GPU layout.
#
# Weights 6.7 GB + drafter 0.25 GB leaves ~8.5 GB for KV + compute buffers.
# Gemma 4 is 48 layers with 1024-token sliding-window attention on most of them
# (only the global layers hold a full-context KV), so the cache is cheap; the
# 262k vocab makes the logits buffer the other big consumer -- that scales with
# -ub, not with context.
#
# Measured on the 4070 SUPER (12282 MiB total, ~400 MiB taken by the desktop),
# q8_0 KV, -ub 512:
#   -c  65536 ->  8154 MiB used  (default here)
#   -c 131072 ->  9082 MiB used
#   -c 262144 -> 10948 MiB peak under a 234k-token prefill (no OOM, ~1.3 GB spare)
# The 5070 Ti has 16303 MiB, so 262144 now has ~5 GB of headroom instead of 1.3.
# VRAM figures above are 4070 SUPER measurements and have not been re-measured
# on the 5070 Ti; they are an upper bound, not a prediction.
#
# Measured throughput on the 4070 SUPER at short context: prefill ~930 t/s,
# decode ~140 tok/s with MTP draft acceptance ~0.75. At a 234k-token prompt:
# prefill 926 t/s, decode ~33 tok/s (acceptance drops to ~0.5). The 5070 Ti
# should be faster; not yet measured.

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-host}"

MODEL_FILE="${MODEL_FILE:-gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}"
# Smart Q4_0 drafter (repo default). MTP/mtp-gemma-4-12B-it-Q8_0.gguf trades
# +220 MB VRAM for a bit more draft acceptance.
DRAFT_FILE="${DRAFT_FILE:-mtp-gemma-4-12B-it-Q4_0.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

PARALLEL="${PARALLEL:-1}"
CONTEXT="${CONTEXT:-65536}"

NAME="${NAME:-llama-gemma4}"

if docker inspect "${NAME}" >/dev/null 2>&1; then
    docker rm -f "${NAME}" >/dev/null
fi

docker create \
    --name "${NAME}" \
    --restart=unless-stopped \
    --gpus all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUDA_VISIBLE_DEVICES=1 \
    --memory=30g \
    --memory-swap=46g \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --ulimit core=0 \
    --network "${NETWORK}" \
    -e LLAMA_ARG_PORT="${HOST_PORT}" \
    -v "$(pwd)/models:/models:ro" \
    -v "$(pwd)/scripts:/scripts:ro" \
    --entrypoint /scripts/entrypoint.sh \
    "${IMAGE}" \
    --model "/models/${MODEL_FILE}" \
    --model-draft "/models/${DRAFT_FILE}" \
    --spec-type draft-mtp \
    --spec-draft-n-max 4 \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --split-mode none \
    -fit off \
    --flash-attn on \
    -c "${CONTEXT}" \
    -n -1 \
    --parallel "${PARALLEL}" \
    --kv-unified \
    -ctk q8_0 \
    -ctv q8_0 \
    --jinja \
    --reasoning off \
    -b 2048 \
    -ub 512 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
