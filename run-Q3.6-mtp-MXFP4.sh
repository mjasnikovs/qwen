#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-runner-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-MXFP4_MOE.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
# qwen35moe: 41 layers (blk 0-40), 256 experts / 8 used, head 16, kv 2, ctx 262144.
# MXFP4 (~22.2GB): expert tensors are mxfp4 (144 B/256) -> identical per-block size
# to Q4_K_XL: ~486.5 MB/block (blk 34,38 ~522, blk 39 ~589). Total experts ~20.1GB,
# non-expert ~2GB (smaller than Q4_K_XL -> a touch more GPU0 headroom).
# tensor-split 1,0 puts all attn/shared tensors on CUDA0 (RTX 5070 Ti, native FP4).
# Placement matches the validated Q4_K_XL layout (identical block sizes):
OT_CUDA0='blk\.(4[0-9]|3[0-9]|2[0-9]|19|18|17)\.ffn_(gate|up|down)_exps\.=CUDA0'   # blk 19-40 -> 16 GB GPU
OT_CUDA1='blk\.(1[0-9]|[1-9])\.ffn_(gate|up|down)_exps\.=CUDA1'    # blk 3-18  -> 8 GB GPU
OT_CPU='blk\..*\.ffn_(gate|up|down)_exps\.=CPU'                     # blk 0-2   -> RAM (catch-all, keep last)
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-${OT_CUDA0},${OT_CUDA1},${OT_CPU}}"

if [[ ! -f "models/${MODEL_FILE}" ]]; then
    echo "Model not found: models/${MODEL_FILE}" >&2
    exit 1
fi

NAME="${NAME:-llama-turboquant}"

# Remove any prior container with the same name so re-running this script
# always produces a fresh, unstarted container.
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
    --tensor-split 1,0 \
    --override-tensor "${OVERRIDE_TENSOR}" \
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
    --spec-draft-n-max 2 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --repeat-penalty 1.0 \
    -b 1024 \
    -ub 256 \
    --cache-idle-slots \
    --cache-ram 16384 \
    --cache-reuse 256 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
