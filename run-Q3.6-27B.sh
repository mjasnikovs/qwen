#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-runner-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
# 65 layers
MODEL_FILE="${MODEL_FILE:-Qwen3.6-27B-NVFP4-MTP.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

if [[ ! -f "models/${MODEL_FILE}" ]]; then
    echo "Model not found: models/${MODEL_FILE}" >&2
    exit 1
fi

NAME="${NAME:-llama-turboquant}"

if docker inspect "${NAME}" >/dev/null 2>&1; then
    docker rm -f "${NAME}" >/dev/null
fi

docker create \
    --name "${NAME}" \
    --restart=unless-stopped \
    --gpus all \
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
    --host 0.0.0.0 \
    --port 8080 \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split 49,16 \
    -fit off \
    --flash-attn on \
    -c 120000 \
    -n -1 \
    --parallel 1 \
    -ctk q8_0 \
    -ctv turbo3 \
    -ctkd q8_0 \
    -ctvd turbo3 \
    --kv-unified \
    --no-mmap \
    --mlock \
    --jinja \
    --reasoning off \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --temp 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --repeat-penalty 1.0 \
    -b 256 \
    -ub 128 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --cache-reuse 256 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
