# syntax=docker/dockerfile:1.7

# Builds llama-server (CUDA) from TheTom/llama-cpp-turboquant
# which adds turbo3/turbo4 KV cache quant types on top of upstream llama.cpp.
#
# Uses the fork's default feature/turboquant-kv-cache branch: turboquant (turbo2/3/4)
# KV cache types with Qwen3 MTP speculative decoding, continuously synced from
# upstream llama.cpp master. No manual PR merge or arg.cpp patching needed.

ARG UBUNTU_VERSION=24.04
# CUDA 13.3 matches the host driver (610.x advertises CUDA 13.3) and has native
# Blackwell (sm_120) support. Container carries the CUDA runtime only; the GPU
# kernel driver comes from the host via the NVIDIA container runtime.
ARG CUDA_VERSION=13.3.0
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Pin to a feature/turboquant-kv-cache tip; override at build time if needed.
# patches/ is applied on top of this ref -- see patches/README.md.
ARG TURBOQUANT_REPO=https://github.com/TheTom/llama-cpp-turboquant.git
ARG TURBOQUANT_BRANCH=feature/turboquant-kv-cache
ARG TURBOQUANT_REF=30d6881eb97be0844b77ff7bc93175e15972d689

# CUDA archs to build for. Override e.g. with --build-arg CUDA_DOCKER_ARCH=89-real
# (4070/4090 Ada=89, 3090/A100=86/80, H100=90, RTX 50xx Blackwell=120).
# Default targets this box: RTX 5070 Ti (120) + RTX 4070 SUPER (89).
ARG CUDA_DOCKER_ARCH=120-real;89-real

# Parallel compile jobs. nvcc uses 2-4 GB RAM each, so on a desktop you'll
# want to cap this — building with -j$(nproc) on CUDA easily OOMs the host.
ARG BUILD_JOBS=4

############################
# Build stage
############################
FROM ${BASE_CUDA_DEV_CONTAINER} AS build
ARG TURBOQUANT_REPO
ARG TURBOQUANT_BRANCH
ARG TURBOQUANT_REF
ARG CUDA_DOCKER_ARCH
ARG BUILD_JOBS

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake ninja-build git ca-certificates \
        libssl-dev libgomp1 libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /src
COPY patches/ /patches/
RUN git clone --filter=blob:none --branch "${TURBOQUANT_BRANCH}" "${TURBOQUANT_REPO}" . \
    && git checkout "${TURBOQUANT_REF}" \
    && git log -1 --format='build commit: %H %s' \
    # tqp-v0.3.0 embed.cpp still requires loading.html, but the current llama-ui
    # bundle dropped it -> UI provisioning fails. Drop the stale required-check;
    # all present assets (index.html, bundle, sw.js, ...) still embed normally.
    && sed -i '/{ "loading.html",/d' tools/ui/embed.cpp \
    && ! grep -q '"loading.html"' tools/ui/embed.cpp \
    # Fork-local fixes not yet upstream. These are pinned against TURBOQUANT_REF --
    # `git apply` fails the build on a REF bump so a stale patch is never silently
    # skipped; re-check against the new ref before bumping.
    && git apply --verbose /patches/*.patch

RUN if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        EXTRA_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_CURL=ON \
        ${EXTRA_CMAKE_ARGS} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j"${BUILD_JOBS}" --target llama-server

RUN mkdir -p /out/lib /out/bin && \
    find build -name "*.so*" -exec cp -P {} /out/lib/ \; && \
    cp build/bin/llama-server /out/bin/llama-server

############################
# Runtime stage
############################
FROM ${BASE_CUDA_RUN_CONTAINER} AS server

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 curl ca-certificates \
    && apt-get autoremove -y && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /app
COPY --from=build /out/lib/ /app/
COPY --from=build /out/bin/llama-server /app/llama-server

ENV LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH}
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/llama-server"]
