#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Override CUDA arch for faster build, e.g.:
#   CUDA_DOCKER_ARCH=89-real ./build.sh   # RTX 4090
#   CUDA_DOCKER_ARCH=86-real ./build.sh   # RTX 3090
#   CUDA_DOCKER_ARCH=120-real ./build.sh  # RTX 5090
# Empty -> use the Dockerfile's own default (120-real;89-real, i.e. this box).
CUDA_DOCKER_ARCH="${CUDA_DOCKER_ARCH:-}"
# Empty -> Dockerfile default (8). Lower it if a model is resident while building.
BUILD_JOBS="${BUILD_JOBS:-}"
IMAGE="${IMAGE:-llama-turboquant:cuda}"

# Plain `if`, not `[[ ]] && ...`: under `set -e` a false one-liner would abort.
args=()
if [[ -n "${CUDA_DOCKER_ARCH}" ]]; then
    args+=(--build-arg "CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}")
fi
if [[ -n "${BUILD_JOBS}" ]]; then
    args+=(--build-arg "BUILD_JOBS=${BUILD_JOBS}")
fi

# ccache lives in a BuildKit cache mount, so BuildKit is required (not legacy builder).
DOCKER_BUILDKIT=1 docker build \
    "${args[@]}" \
    -t "${IMAGE}" \
    .
