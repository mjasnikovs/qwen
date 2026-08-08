# syntax=docker/dockerfile:1.7

# Builds llama-server (CUDA) from upstream ggml-org/llama.cpp.
#
# Previously built TheTom/llama-cpp-turboquant for its turbo2/3/4 KV cache types.
# Upstream now carries the speculative-decoding stack this box relies on
# (--spec-type draft-mtp/draft-dflash/ngram-mod, -ctkd/-ctvd), so the fork is gone.
# What upstream does NOT have: the turbo* KV cache types. Allowed -ctk/-ctv values
# are f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 -- run scripts passing
# turbo3 will be rejected at startup.

ARG UBUNTU_VERSION=24.04
# CUDA 13.3 matches the host driver (610.x advertises CUDA 13.3) and has native
# Blackwell (sm_120) support. Container carries the CUDA runtime only; the GPU
# kernel driver comes from the host via the NVIDIA container runtime.
ARG CUDA_VERSION=13.3.0
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Pin to an upstream master commit; override at build time if needed.
# ee0445c9 == release tag b10241 (2026-08-03).
ARG LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_BRANCH=master
ARG LLAMA_REF=ee0445c99cffbe8d920b05cad28cb055d7049c0a

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
ARG LLAMA_REPO
ARG LLAMA_BRANCH
ARG LLAMA_REF
ARG CUDA_DOCKER_ARCH
ARG BUILD_JOBS

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake ninja-build git ca-certificates \
        libssl-dev libgomp1 libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /src
RUN git clone --filter=blob:none --branch "${LLAMA_BRANCH}" "${LLAMA_REPO}" . \
    && git checkout "${LLAMA_REF}" \
    && git log -1 --format='build commit: %H %s'

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
