#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Parallel variant of run-Q3.6-27B.sh.
#
# Difference vs the base script: --parallel 2 with a NON-unified KV cache, so
# the server exposes TWO independent sequence slots, each with a FULL 120k
# context. --parallel splits total -c evenly across slots, so -c is set to
# 240000 (2 x 120000) to give every slot an independent 120k window.
#
# WARNING: this DOUBLES the KV-cache VRAM vs the base config (which fits ~tight
# on the 16GB card at 120k q8_0/turbo3). Two independent 120k contexts may OOM
# on this rig -- launch and check VRAM before relying on it. Levers if it does
# not fit: lower CONTEXT, or drop -ctk/-ctv to a smaller quant (turbo2).
#
# Why parallel at all: with --split-mode layer a single request pipelines
# across both GPUs and each card averages ~50% util. Two concurrent requests
# overlap in the pipeline (GPU1 runs one slot's late layers while GPU0 runs the
# other slot's early layers), raising aggregate GPU utilization and total
# throughput. Per-request latency is unchanged.

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-runner-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
# 65 layers
MODEL_FILE="${MODEL_FILE:-Qwen3.6-27B-NVFP4-MTP.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Qwen3.6-27B-F16.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

# Number of independent context slots and total context (split across slots).
# CONTEXT is total; each slot gets CONTEXT/PARALLEL. 240000/2 => 120k per slot.
PARALLEL="${PARALLEL:-2}"
CONTEXT="${CONTEXT:-240000}"

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
    -e TURBO_AUTO_ASYMMETRIC=0 \
    -p "${HOST_PORT}:8080" \
    --network "${NETWORK}" \
    --ip "${STATIC_IP}" \
    -v "$(pwd)/models:/models:ro" \
    -v "$(pwd)/scripts:/scripts:ro" \
    --entrypoint /scripts/entrypoint.sh \
    "${IMAGE}" \
    --model "/models/${MODEL_FILE}" \
    --mmproj "/models/${MMPROJ_FILE}" \
    --mmproj-offload \
    --host 0.0.0.0 \
    --port 8080 \
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
    -ctk q8_0 \
    -ctv turbo3 \
    -ctkd q8_0 \
    -ctvd turbo3 \
    --no-mmap \
    --mlock \
    --jinja \
    --reasoning off \
    --spec-type draft-mtp,ngram-mod \
    --spec-draft-n-max 3 \
    --spec-ngram-mod-n-match 24 \
    --spec-ngram-mod-n-min 4 \
    --spec-ngram-mod-n-max 48 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --repeat-penalty 1.0 \
    -b 512 \
    -ub 256 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --cache-reuse 256 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
