#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Gemma 4 26B-A4B NVFP4 (catlilface/Gemma-4-26B-A4B-NVFP4-GGUF).
#
# MoE: 25.2B total / 3.8B active, 30 layers, 128 experts (8 active + 1 shared).
# llama.cpp arch is `gemma4` -- the same code path as the dense 12B, so no
# llama.cpp bump is needed. NVFP4 backbone, 17.68 GB on disk.
#
# NVFP4 is native only on the 5070 Ti (Blackwell). The 4070 SUPER (Ada) runs it
# through a dequant path, same as the Qwen NVFP4 models already do.
#
# MTP
# ---
# Drafting is google/gemma-4-26B-A4B-it-assistant, a 0.4B MTP head shipped as a
# SEPARATE GGUF (arch `gemma4-assistant`, in upstream since PR #23398, merged
# 2026-06-07). Q8_0 quant, 462 MB. This is NOT the baked-in-MTP layout the Qwen
# NVFP4 model uses -- the drafter is a real second model here.
#
# ngram-simple is chained after it, and it is the single biggest win in this
# file. On work where the answer copies spans of the prompt -- rewrite this file,
# apply this edit, the bread and butter of a coding agent -- it takes decode from
# 85 tok/s to 245. It does nothing on from-scratch prose, and costs nothing there
# either.
#
# It is INVISIBLE to a normal benchmark. bench.py has no prompt an ngram drafter
# can match against, so draft-mtp and draft-mtp,ngram-* score byte-identical on
# it. The numbers below come from a separate copy-heavy probe (echobench.py).
#
# ngram variant swept, echo / edit decode tok/s:
#   ngram-simple  244 / 176   <-- picked
#   ngram-mod     214 / 158
#   ngram-map-k4v 161 /  82
#   ngram-map-k   157 /  84
#   ngram-cache    83 /  80   (also drags plain coding decode down to 82)
#   none (mtp)     85 /  84
#
# --spec-ngram-simple-size-n is the lookup length. 8 beats the default 12:
# n=8 -> 89.1 coding / 243 echo / 195 edit; n=12 -> 86.8 / 244 / 177;
# n=16 -> 87.0 / 229 / 167; n=4 collapses MTP acceptance to 0.40 and coding
# decode to 80.5. size-m (draft length, default 48) measured flat from 48 to 96.
#
# --spec-draft-n-max 2 is the pick. Draft width is a real cliff here:
#   n-max   coding decode   prose decode   acceptance
#     off       70.8            70.9          --
#      2        86.8            72.1         0.72
#      3        84.9            65.8         0.63
#      4        82.0            61.3         0.46
# Wider drafts raise total accepted tokens but the acceptance RATE collapses and
# every rejected token costs a full verification pass.
#
# Drafter quant swept: Q4_K_M (325 MB) 87.6, Q8_0 (462 MB) 87.3, F16 (855 MB)
# 84.1. Q4_K_M wins on both speed and VRAM.
#
# VISION
# ------
# mmproj-Gemma4-26b-F16.gguf, 1.19 GB. Gemma 4 supports a VARIABLE image token
# budget; the supported values are 70, 140, 280, 560 and 1120. Left unset,
# llama.cpp picks a low budget and screenshot detail suffers. Both min and max
# are pinned to 1120 (the ceiling) because the workload is reading game
# screenshots.
#
# -ub MUST be >= the image budget. The vision encoder runs non-causal attention
# and asserts `n_ubatch >= n_tokens`; at -ub 512 a 1120-token image kills the
# server mid-request (llama.cpp issue #21461). MEASURED here: the request dies
# with a bare connection reset, no error in the log. -ub 1280 is the smallest
# working value -- 1152 OOMs at load, 2048 does not fit at all.
#
# Raising -ub is NOT free: prefill drops ~20% (6750 -> 5400 t/s). That is the
# price of the max image budget, and it is paid on every request, image or not.
# Decode is unaffected. If images stop mattering, -ub 512 + IMAGE_TOKENS=280
# buys the prefill back.
#
# REASONING
# ---------
# Off. The chat template defaults `enable_thinking` to false, and `--reasoning
# off` keeps it there. A harness can still turn it on per request:
#   "chat_template_kwargs": {"enable_thinking": true}
#
# SAMPLING
# --------
# Google's published defaults: temp 1.0, top-p 0.95, top-k 64. Sampler ORDER
# matters for Gemma 4 -- temp before top-p before top-k, which is
# --sampler-seq tpk, not llama.cpp's default (edskypmxt, temp last).

# No env exports here. Verified 2026-08-24 against the pinned llama.cpp ref:
#   CUDA_MALLOC_ASYNC_SUPPORTED -- not read by llama.cpp/ggml at all.
#   GGML_CUDA_FORCE_MMQ         -- compile-time cmake option now, not an
#                                  env var (ggml/CMakeLists.txt, default OFF).
# Both were also never reaching the container: docker create only forwards
# the -e flags listed below, not the host shell's exports.

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-host}"

# 30 layers
MODEL_FILE="${MODEL_FILE:-Gemma4-26b-NVFP4.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Gemma4-26B-F16.gguf}"
DRAFT_FILE="${DRAFT_FILE-gemma-4-26B-A4B-it-assistant-Q4_K_M.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
TENSOR_SPLIT="${TENSOR_SPLIT:-18,12}"

PARALLEL="${PARALLEL:-1}"
CONTEXT="${CONTEXT:-98304}"
IMAGE_TOKENS="${IMAGE_TOKENS:-1120}"

NAME="${NAME:-llama-gemma4-26b}"

# DRAFT_FILE="" disables speculative decoding entirely (baseline measurements).
DRAFT_ARGS=()
if [[ -n "${DRAFT_FILE}" ]]; then
    DRAFT_ARGS=(
        --model-draft "/models/${DRAFT_FILE}"
        --spec-type "${SPEC_TYPE:-draft-mtp,ngram-simple}"
        --spec-draft-n-max "${SPEC_N_MAX:-2}"
        --spec-ngram-simple-size-n 8
    )
fi

if docker inspect "${NAME}" >/dev/null 2>&1; then
    docker rm -f "${NAME}" >/dev/null
fi

docker create \
    --name "${NAME}" \
    --restart=unless-stopped \
    --gpus all \
    -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e CUDA_VISIBLE_DEVICES=1,0 \
    --memory=30g \
    --memory-swap=46g \
    --cap-add=IPC_LOCK \
    --ulimit memlock=-1:-1 \
    --ulimit core=0 \
    --network "${NETWORK}" \
    -e LLAMA_ARG_PORT="${HOST_PORT}" \
    -v "$(pwd)/models:/models:ro" \
    -v "$(pwd)/scripts:/scripts:ro" \
    --entrypoint /scripts/entrypoint.sh \
    "${IMAGE}" \
    --model "/models/${MODEL_FILE}" \
    "${DRAFT_ARGS[@]}" \
    --mmproj "/models/${MMPROJ_FILE}" \
    --mmproj-offload \
    --image-min-tokens "${IMAGE_TOKENS}" \
    --image-max-tokens "${IMAGE_TOKENS}" \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split "${TENSOR_SPLIT}" \
    -fit off \
    --flash-attn on \
    -c "${CONTEXT}" \
    -n -1 \
    --parallel "${PARALLEL}" \
    --kv-unified \
    -ctk q8_0 \
    -ctv q8_0 \
    --load-mode mmap \
    --jinja \
    --reasoning off \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 64 \
    --min-p 0.0 \
    --sampler-seq tpk \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    -b 2048 \
    -ub 1280 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"

# BENCHMARKS
# ----------
# All cold: container removed, server restarted, `./bench.py --runs 1`, temp 0.
# Measured 2026-08-21. decode = tok/s, prefill = pp_xl peak t/s.
#
# `coding` and `prose` are bench.py. `echo` and `edit` are echobench.py, a
# copy-heavy probe (reproduce this file / apply one edit and return the whole
# file) -- the only shape an ngram drafter can win on.
#
#   config                                 coding  prose  echo  edit  prefill
#   -------------------------------------  ------  -----  ----  ----  -------
#   no spec at all                           70.1   70.9    70    70     8319
#   MTP n-max 3                              84.9   65.8    --    --     6742
#   MTP n-max 2                              87.4   72.1    85    84     6764
#   MTP n-max 2 + ngram-simple n=8  <-THIS   89.1     --   243   195     5402
#   ... but -ub 1024                         78.8     --    --    --     5937
#   ... but KV f16                           77.0     --    --    --     5129
#   ... but -b 4096                          83.8     --    --    --     5959
#   ... but -c 131072                        83.4     --    --    --     5656
#
# Read the echo/edit column first. Chaining ngram-simple is worth ~2.5x on the
# work a coding agent actually does, and it is free everywhere else.
#
# Notes on the losers:
#   -c 131072  the compute buffer no longer fits and llama.cpp silently retries
#              "without pipeline parallelism". 98304 is the largest context that
#              keeps it on. Watch the log for that line after any change.
#   KV f16     costs 10 tok/s of decode and 270 MB for +0.02 acceptance. q8_0
#              wins despite what PR #23398 says about quantized KV and MTP.
#   -b 4096    buys 10% prefill, costs 4% decode. Not worth it for coding.
#   19,11      does not load: the mmproj no longer fits on the 5070 Ti.
#
# VRAM at rest: 4070 SUPER 11116 / 12282 MiB, 5070 Ti 15256 / 16303 MiB after an
# image request. There is no headroom left -- any added context or -ub OOMs.
