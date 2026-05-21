#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-aiz-docker_aiz-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-blk\.(38|37|36|35)\.ffn_(gate|up|down)_exps\.=CUDA0,blk\.(34|33|31|30|29|28|27|26|25|24)\.ffn_(gate|up|down)_exps\.=CUDA1,blk\..*\.ffn_(gate|up|down)_exps\.=CPU}"

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
    --restart=no \
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
    -c 128000 \
    -n -1 \
    --parallel 2 \
    -ctk turbo4 \
    -ctv turbo4 \
    -ctkd turbo3 \
    -ctvd turbo3 \
    --kv-unified \
    --no-mmap \
    --mlock \
    --jinja \
    --chat-template-kwargs '{"enable_thinking": true, "preserve_thinking": true}' \
    --reasoning on \
    --spec-type draft-mtp \
    --spec-draft-n-max 3 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    -b 4096 \
    -ub 696 \
    --cache-idle-slots \
    --cache-ram 2048 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
