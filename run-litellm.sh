#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="${IMAGE:-ghcr.io/berriai/litellm:main-stable}"
HOST_PORT="${HOST_PORT:-4000}"
NETWORK="${NETWORK:-aiz-docker_aiz-network}"
STATIC_IP="${STATIC_IP:-172.18.0.11}"
NAME="${NAME:-llama-litellm}"
CONFIG_FILE="${CONFIG_FILE:-litellm/config.yaml}"
LITELLM_LOG="${LITELLM_LOG:-INFO}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "LiteLLM config not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# Remove any prior container with the same name so re-running this script
# always produces a fresh, unstarted container.
if docker inspect "${NAME}" >/dev/null 2>&1; then
    docker rm -f "${NAME}" >/dev/null
fi

docker create \
    --name "${NAME}" \
    --restart=no \
    --memory=1g \
    --memory-swap=1g \
    -p "${HOST_PORT}:4000" \
    --network "${NETWORK}" \
    --ip "${STATIC_IP}" \
    -e "LITELLM_LOG=${LITELLM_LOG}" \
    -v "$(pwd)/${CONFIG_FILE}:/app/config.yaml:ro" \
    "${IMAGE}" \
    --config /app/config.yaml \
    --port 4000 \
    --host 0.0.0.0 \
    "$@" >/dev/null

cat <<EOF
Container '${NAME}' created (not started).

Translator in front of llama-turboquant. Listens on host port ${HOST_PORT}.
Exposes BOTH:
    Anthropic Messages   POST http://localhost:${HOST_PORT}/v1/messages
    OpenAI Chat          POST http://localhost:${HOST_PORT}/v1/chat/completions
Upstream:                http://172.18.0.10:8080/v1 (llama-turboquant)

Master key (set in litellm/config.yaml): sk-litellm-local

For Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:${HOST_PORT}
    export ANTHROPIC_AUTH_TOKEN=sk-litellm-local

Start it when you want to serve:
    docker start -a ${NAME}      # foreground, follows logs
    docker start ${NAME}         # detached
    docker logs -f ${NAME}       # tail logs
    docker stop ${NAME}          # stop

It will NOT auto-start on Docker daemon boot (--restart=no).
EOF
