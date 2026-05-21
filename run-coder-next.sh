#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-aiz-docker_aiz-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
MODEL_FILE="${MODEL_FILE:-Qwen3-Coder-Next-Q3_K_S.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
# qwen3next: 48 blocks (blk.0–blk.47), 512 experts/10 active, ~768MB expert weights/block
# CUDA0 (7322 MiB free): blocks 47–40 (8 blocks ~6.1GB); CUDA1 (4986 MiB free): blocks 39–35 (5 blocks ~3.8GB)
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-blk\.(47|46|45|44|43|42)\.ffn_(gate|up|down)_exps\.=CUDA0,blk\.(41|40|39|38|37|36|35)\.ffn_(gate|up|down)_exps\.=CUDA1,blk\..*\.ffn_(gate|up|down)_exps\.=CPU}"

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
    --cache-ram 4096 \
    --cache-reuse 256 \
    --no-kv-unified \
    -c 128000 \
    -n -1 \
    -np 1 \
    -b 1028 \
    -ub 1028 \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 40 \
    --min_p 0.1 \
    --presence_penalty 1.5 \
    --repeat-penalty 1.0 \
    --threads 8 \
    --cpu-range 0-7 \
    "$@" >/dev/null

docker start -a "${NAME}"
