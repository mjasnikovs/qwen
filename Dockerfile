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
# a130532a == release tag b10605 (2026-08-24), 17 commits past b10588.
# Reviewed the whole range; nothing lands on this box's hot paths.
# Worth watching on the first run:
#  - #27594 mtmd: pillow-accurate resize algo, corrected for all models. This
#    CHANGES image preprocessing, so vision output can differ from b10588.
#  - #27574 tensor-parallel meta split-state fix. We run -sm layer, so this is
#    only relevant if we ever try -sm tensor again (we won't, it lost).
#  - #27573 CUDA POOL_1D support.
# The rest is webui tabs, CI, tests, DeepSeek/GLM/mamba2 model work.
ARG LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_BRANCH=master
ARG LLAMA_REF=a130532ae1c4c54daaae5527795f5b19c184f269

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
ARG DFLASH_PR
ARG DFLASH_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc-14 g++-14 build-essential cmake ninja-build git ca-certificates \
        ccache \
        libssl-dev libgomp1 libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-14 CXX=g++-14 CUDAHOSTCXX=g++-14
# ccache is what makes a LLAMA_REF bump cheap. Bumping the ref busts the clone
# layer and therefore the build layer, so cmake always re-runs -- but most .cu
# files are byte-identical between two nearby upstream commits, so ccache serves
# them from the BuildKit cache mount instead of paying nvcc twice (sm_120+sm_89).
# base_dir + the sloppiness flags are required for hits: sources live under a
# path that changes between builds, and nvcc emits __TIME__-style noise that
# would otherwise defeat the hash.
ENV CCACHE_DIR=/ccache \
    CCACHE_BASEDIR=/src \
    CCACHE_COMPILERCHECK=content \
    CCACHE_SLOPPINESS=time_macros,include_file_mtime,include_file_ctime,locale,random_seed \
    CCACHE_MAXSIZE=15G

WORKDIR /src
# ONE image, not two. The old Dockerfile.dflash2 existed only because DFlash2
# lives in an unmerged PR. Instead of a second image we merge that PR into the
# pinned upstream ref, so this single build serves every run script:
#   - DFlash2 is opt-in at runtime (--spec-type draft-dflash). Nothing else in
#     the binary changes, so the non-DFlash runs are unaffected.
#   - PR #27342 is still OPEN. Its head moved to 64f765f5 (12 commits, last
#     2026-08-24): a refactor, p_min, cost optimisation, and a top_k selector
#     move. The merge into b10605 is clean (verified 2026-08-24, no conflicts).
# patches/0001-dflash-dense-inject-pos-for-vision.patch is @Shamish's community
# fix from PR #27342, never merged into the PR head. Without it, any request
# carrying an image dies with `llama_decode(ctx_dft) failed rc=-1`: the DFlash
# inject batch copies the target's M-RoPE positions verbatim, and ctx_dft
# (plain qwen3, n_pos_per_embd == 1) demands continuous positions.
# The PR's own refactor moved that loop, so the patch was RE-AUTHORED against
# head 64f765f5 on 2026-08-24; the old 1deefcca version no longer applies. The
# fix is still NOT in the PR head, so we still carry it.
# On every LLAMA_REF or PR bump: re-check that the merge is clean and the patch
# still applies. Drop DFLASH_PR and the patch once the PR lands upstream.
ARG DFLASH_PR=27342
ARG DFLASH_REF=64f765f5adefa4620dddda436ce56f1430435536
COPY patches/ /patches/
RUN git clone --filter=blob:none --branch "${LLAMA_BRANCH}" "${LLAMA_REPO}" . \
    && git checkout "${LLAMA_REF}" \
    && git fetch origin "pull/${DFLASH_PR}/head" \
    && git -c user.email=build@local -c user.name=build \
           merge --no-edit "${DFLASH_REF}" \
    && git apply --verbose /patches/*.patch \
    && git log -1 --format='build commit: %H %s'

RUN --mount=type=cache,target=/ccache,sharing=locked \
    if [ "${CUDA_DOCKER_ARCH}" != "default" ]; then \
        EXTRA_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${CUDA_DOCKER_ARCH}"; \
    fi && \
    ccache --zero-stats && \
    cmake -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=OFF \
        -DGGML_CUDA=ON \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_CURL=ON \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        ${EXTRA_CMAKE_ARGS} \
        -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined && \
    cmake --build build --config Release -j"${BUILD_JOBS}" --target llama-server && \
    ccache --show-stats

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
