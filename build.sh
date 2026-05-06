#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Override CUDA arch for faster build, e.g.:
#   CUDA_DOCKER_ARCH=89-real ./build.sh   # RTX 4090
#   CUDA_DOCKER_ARCH=86-real ./build.sh   # RTX 3090
#   CUDA_DOCKER_ARCH=120-real ./build.sh  # RTX 5090
CUDA_DOCKER_ARCH="${CUDA_DOCKER_ARCH:-default}"
IMAGE="${IMAGE:-llama-turboquant:cuda}"

docker build \
    --build-arg CUDA_DOCKER_ARCH="${CUDA_DOCKER_ARCH}" \
    -t "${IMAGE}" \
    .
