#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-aiz-docker_aiz-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
MODEL_FILE="${MODEL_FILE:-qwen3.6-35b-a3b-UD-Q4_K_XL.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-blk\.(38|37|36|35|34)\.ffn_(gate|up|down)_exps\.=CUDA0,blk\.(33|31|30|29|28|27|26|25|24|23)\.ffn_(gate|up|down)_exps\.=CUDA1,blk\..*\.ffn_(gate|up|down)_exps\.=CPU}"

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
    --restart=no \
    --gpus all \
    --memory=28g \
    --memory-swap=28g \
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
    --jinja \
    --chat-template-kwargs '{"enable_thinking": false}' \
    --reasoning off \
    -fit off \
    --no-mmap \
    --mlock \
    --flash-attn on \
    -ctk turbo4 \
    -ctv turbo4 \
    --cache-ram 2048 \
    --cache-reuse 256 \
    --no-kv-unified \
    -c 128000 \
    -n -1 \
    -np 1 \
    -b 1536 \
    -ub 1536 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --min_p 0.0 \
    --presence_penalty 1.5 \
    --repeat-penalty 1.0 \
    --threads 8 \
    --cpu-range 0-7 \
    "$@" >/dev/null

docker start -a "${NAME}"
