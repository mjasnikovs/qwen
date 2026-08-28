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
# ca3d5a3e == release tag b10665 (2026-08-28), 60 commits past b10605.
# Reviewed the whole range; nothing lands on this box's hot paths.
# The headline of this bump: DFlash2 IS NOW UPSTREAM -- see the clone RUN below.
# Worth watching on the first run:
#  - #27342 landed as b10f9ca5 (squashed via #27816). Two commits that were not
#    in the old DFLASH_REF f7aadef0 are now in: 11f45ed3 "Fix graph number
#    calculation" and 2f3923bc "rename hid and unary". Re-measure DFlash2
#    acceptance on text and on images before trusting the old numbers.
#  - #27762 llama: token ID tracking in KV cells. Adds a token field to the
#    per-cell `ext` struct, and now fills `ext` for plain token batches too --
#    previously it was M-RoPE-only. Host RAM, not VRAM, but it touches the same
#    2D-position bookkeeping the DFlash draft cache rides on.
#  - #27711 spec: synthetic speculative-acceptance options. Benchmark-only, but
#    it edits common/speculative.cpp, which is where our patch lands.
#  - #24124 server --kv-unified-per-slot (ctx-per-slot). Relevant to
#    run-Q3.6-27B-parallel.sh if we ever want per-slot context isolation.
#  - #27659 gguf-py now maps generation_config.json `repetition_penalty` into
#    GGUF metadata. Conversion-time only, so it affects any model re-converted
#    from here on -- including the local DFlash2 draft. Penalties wreck spec
#    decode acceptance, so check what a fresh convert bakes in.
# The rest is webui, CI, Vulkan/Metal/HIP/hexagon backends, and model work.
ARG LLAMA_REPO=https://github.com/ggml-org/llama.cpp.git
ARG LLAMA_BRANCH=master
ARG LLAMA_REF=ca3d5a3e10d53f7ea672cb9b6178faca3e2807bc

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
# lived in an unmerged PR, and this file used to merge that PR into the pinned
# upstream ref at clone time.
#
# NO LONGER. PR #27342 LANDED UPSTREAM on 2026-08-27 as b10f9ca5 (squashed via
# #27816), which is an ancestor of LLAMA_REF. The DFLASH_PR/DFLASH_REF args and
# the `git fetch pull/.../head && git merge` step are therefore GONE -- the
# clone is plain upstream at LLAMA_REF plus patches/. Do not add the merge back.
# DFlash2 stays opt-in at runtime (--spec-type draft-dflash), so nothing about
# the non-DFlash runs changes.
#
# STILL PATCHED, though: patches/0001-dflash-mtmd-zero-fill-draft-cache.patch is
# the COMPLETE vision fix from upstream issue #27408 (@fishlikeX, fork commit
# 3e008b22, never turned into a PR). Issue #27408 is STILL OPEN and none of the
# zero-fill code is in upstream master -- what landed with #27342 is only
# f5a7ec15, which is half of it. Verified against b10665 on 2026-08-28: the
# patch applies clean, 4 hunks, offset +18 lines. See VISION in
# run-Q3.8-27B-NVFP4.sh and patches/README.md.
# History: patches/0001-dflash-dense-inject-pos-for-vision.patch was @Shamish's
# community workaround. It renumbered the DFlash inject batch densely from the
# draft cache's own max position. That stopped the rc=-1 rejection, but only
# for the inject batch: common_speculative_impl_draft_dflash::draft() still
# positions its noise block at dp.n_past, which is the TARGET's position. So
# every image silently desynced the two caches by its token span (1032 here:
# --image-min-tokens 1024 + 8 markers) and drafting died with
# "inconsistent sequence positions" for the rest of the conversation.
# Upstream commit f5a7ec15 fixes it properly: it sets is_mrope from the DRAFT
# model's rope type and feeds 4 position rows per token to both the encoder and
# the inject batch, so the draft keeps the target's real positions and the two
# caches stay in lockstep.
#
# THAT FIX IS GATED ON THE DRAFT GGUF. llama_model_rope_type() only returns
# MROPE for LLM_ARCH_DFLASH when hparams.rope_sections is non-zero, and only a
# converter at >= f5a7ec15 writes dflash.rope.dimension_sections (degenerate
# [head_dim/2, 0, 0, 0]). The published z-lab/incoai DFlash2 GGUFs do NOT carry
# it, so they silently fall back to the old broken path. The draft GGUF this
# box runs is converted locally from z-lab/Qwen3.8-27B-DFlash2 safetensors --
# see models/hf/ and the note in patches/README.md. If you ever swap in a
# downloaded DFlash2 GGUF, check for dflash.rope.dimension_sections first.
#
# On every LLAMA_REF bump: re-check that patches/ still applies. It is inside
# the `&&` chain on purpose, so a stale patch fails the build loudly.
# Drop patches/ and the `git apply` line once #27408 lands upstream.
COPY patches/ /patches/
RUN git clone --filter=blob:none --branch "${LLAMA_BRANCH}" "${LLAMA_REPO}" . \
    && git checkout "${LLAMA_REF}" \
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
    cmake --build build --config Release -j"${BUILD_JOBS}" --target llama-server llama-quantize && \
    ccache --show-stats

RUN mkdir -p /out/lib /out/bin && \
    find build -name "*.so*" -exec cp -P {} /out/lib/ \; && \
    cp build/bin/llama-server /out/bin/llama-server && \
    cp build/bin/llama-quantize /out/bin/llama-quantize

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
# llama-quantize ships too: the DFlash2 draft has to be re-quantized locally
# (see patches/README.md -- published DFlash2 GGUFs lack rope sections).
COPY --from=build /out/bin/llama-quantize /app/llama-quantize

ENV LD_LIBRARY_PATH=/app:${LD_LIBRARY_PATH}
ENV LLAMA_ARG_HOST=0.0.0.0
ENV LLAMA_ARG_PORT=8080

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/llama-server"]
