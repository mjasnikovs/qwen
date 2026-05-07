#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
N_CPU_MOE="${N_CPU_MOE:-41}"
CTX_SIZE="${CTX_SIZE:-128000}" #200000
CACHE_TYPE_K="${CACHE_TYPE_K:-turbo4}"
CACHE_TYPE_V="${CACHE_TYPE_V:-turbo3}"
NETWORK="${NETWORK:-aiz-docker_aiz-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"

# Pick model: env var wins, otherwise prompt from the available .gguf files.
if [[ -z "${MODEL_FILE:-}" ]]; then
    mapfile -t available < <(cd models 2>/dev/null && ls -1 *.gguf 2>/dev/null || true)
    if ((${#available[@]} == 0)); then
        echo "No .gguf files in models/. Run ./download-model.sh or place a GGUF there." >&2
        exit 1
    elif ((${#available[@]} == 1)); then
        MODEL_FILE="${available[0]}"
        echo "Using only available model: ${MODEL_FILE}"
    else
        if [[ ! -t 0 ]]; then
            echo "Multiple models in models/ but stdin is not a TTY. Set MODEL_FILE=<name>." >&2
            printf '  %s\n' "${available[@]}" >&2
            exit 1
        fi
        echo "Available models:"
        for i in "${!available[@]}"; do
            printf "  %d) %s\n" $((i + 1)) "${available[$i]}"
        done
        read -rp "Select model [1-${#available[@]}]: " choice
        if ! [[ "$choice" =~ ^[0-9]+$ ]] || ((choice < 1 || choice > ${#available[@]})); then
            echo "Invalid choice: $choice" >&2
            exit 1
        fi
        MODEL_FILE="${available[$((choice - 1))]}"
    fi
fi

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
    --memory=28g \
    --memory-swap=28g \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --ulimit core=0 \
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
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --n-cpu-moe "${N_CPU_MOE}" \
    --no-mmap \
    --mlock \
    --cache-type-k "${CACHE_TYPE_K}" \
    --cache-type-v "${CACHE_TYPE_V}" \
    --cache-ram 2048 \
    -c "${CTX_SIZE}" \
    --parallel 1 \
    -b 1024 \
    -ub 256 \
    "$@" >/dev/null

cat <<EOF
Container '${NAME}' created (not started).

Start it when you want to serve:
    docker start -a ${NAME}      # foreground, follows logs
    docker start ${NAME}         # detached
    docker logs -f ${NAME}       # tail logs
    docker stop ${NAME}          # stop

It will NOT auto-start on Docker daemon boot (--restart=no).
EOF
