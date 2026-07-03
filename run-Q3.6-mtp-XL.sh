#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-runner-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
# Q6_K_XL (~31GB). tensor-split 1,0: all attention/shared tensors on CUDA0.
# Measured ~500MB per expert block. Filling free VRAM headroom from top blocks down.
# CUDA0 RTX 5070 Ti (16GB): ~3GB attn + 23 expert blocks (~12GB) ≈ 14GB used
# CUDA1 RTX 3070 Ti  ( 8GB): 9 expert blocks (~6.5GB) + compute buffers ≈ 7.5GB used
# CPU: blk 0-14 + 50+ -> RAM
OT_CUDA0='blk\.(4[0-9]|3[0-9]|20|19|18)\.ffn_(gate|up|down)_exps\.=CUDA0'  # blk 18-20,30-49 -> RTX 5070 Ti
OT_CUDA1='blk\.(2[0-9]|10)\.ffn_(gate|up|down)_exps\.=CUDA1'                     # blk 21-29 -> RTX 3070 Ti
OT_CPU='blk\..*\.ffn_(gate|up|down)_exps\.=CPU'                               # blk 0-12 + 50+ -> RAM
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-${OT_CUDA0},${OT_CUDA1},${OT_CPU}}"

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
    --memory=28g \
    --memory-swap=32g \
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
    -ctk turbo4 \
    -ctv turbo4 \
    -ctkd turbo3 \
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
    -b 1536 \
    -ub 768 \
    --cache-idle-slots \
    --cache-ram 6144 \
    --cache-reuse 256 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
