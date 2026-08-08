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
# Image tag kept as-is so the existing run-*.sh defaults keep resolving; the build
# is upstream ggml-org/llama.cpp now, not the turboquant fork.
IMAGE="${IMAGE:-llama-turboquant:cuda}"
# Empty -> Dockerfile's pinned upstream commit. Set to a tag/sha to move the pin, e.g.:
#   LLAMA_REF=b10241 ./build.sh
LLAMA_REF="${LLAMA_REF:-}"

# Plain `if`, not `[[ ]] && ...`: under `set -e` a false one-liner would abort.
args=()
if [[ -n "${CUDA_DOCKER_ARCH}" ]]; then
    args+=(--build-arg "CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}")
fi
if [[ -n "${BUILD_JOBS}" ]]; then
    args+=(--build-arg "BUILD_JOBS=${BUILD_JOBS}")
fi
if [[ -n "${LLAMA_REF}" ]]; then
    args+=(--build-arg "LLAMA_REF=${LLAMA_REF}")
fi

# ccache lives in a BuildKit cache mount, so BuildKit is required (not legacy builder).
DOCKER_BUILDKIT=1 docker build \
    "${args[@]}" \
    -t "${IMAGE}" \
    .
