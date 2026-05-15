# syntax=docker/dockerfile:1.7

# Builds llama-server (CUDA) from TheTom/llama-cpp-turboquant
# which adds turbo3/turbo4 KV cache quant types on top of upstream llama.cpp.

ARG UBUNTU_VERSION=24.04
ARG CUDA_VERSION=12.8.1
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

# Pin to the turboquant branch tip; override at build time if needed.
ARG TURBOQUANT_REF=69d8e4be47243e83b3d0d71e932bc7aa61c644dc

# llama.cpp PR #22673 — MTP (Multi-Token Prediction) speculative decoding support
ARG MTP_PR=22673

# CUDA archs to build for. Override e.g. with --build-arg CUDA_DOCKER_ARCH=89-real
# (4090=89, 3090/A100=86/80, H100=90, RTX 50xx=120). Default builds all archs.
ARG CUDA_DOCKER_ARCH=86-real;61-real

# Parallel compile jobs. nvcc uses 2-4 GB RAM each, so on a desktop you'll
# want to cap this — building with -j$(nproc) on CUDA easily OOMs the host.
ARG BUILD_JOBS=4

############################
# Build stage
############################
FROM ${BASE_CUDA_DEV_CONTAINER} AS build
ARG TURBOQUANT_REF
ARG CUDA_DOCKER_ARCH
ARG MTP_PR

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake ninja-build git ca-certificates \
        libssl-dev libgomp1 libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14

WORKDIR /src
RUN git config --global user.email "build@local" && git config --global user.name "Docker Build"
RUN git clone --filter=blob:none https://github.com/TheTom/llama-cpp-turboquant.git . \
    && git checkout ${TURBOQUANT_REF} \
    && git remote add upstream https://github.com/ggml-org/llama.cpp.git \
    && git fetch upstream pull/${MTP_PR}/head:pr-mtp \
    && git merge --no-ff -X ours pr-mtp -m "Merge llama.cpp PR #${MTP_PR}: MTP speculative decoding support" \
    && git checkout pr-mtp -- common/ \
    && sed -i 's/GGML_TYPE_Q5_1,/GGML_TYPE_Q5_1,\n        GGML_TYPE_TURBO2_0,\n        GGML_TYPE_TURBO3_0,\n        GGML_TYPE_TURBO4_0,/' common/arg.cpp

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
    cmake --build build --config Release -j"$(nproc)" --target llama-server

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
