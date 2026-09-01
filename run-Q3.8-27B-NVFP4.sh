#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Qwen3.8-27B NVFP4 (esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF, VERY-HIGH tier).
#
# Same model and arch as run-Q3.8-27B.sh -- llama.cpp calls it `qwen35`, 64
# hidden layers + 1 MTP layer = 65 blocks, so --tensor-split still sums to 65.
# 16 x (3 x GatedDeltaNet -> 1 x GatedAttention), so only 16 of 64 layers hold a
# real KV cache. Gated attention: 24 Q heads, 4 KV heads, head_dim 256.
#
# VERY-HIGH tier = NVFP4 backbone (all attention + MLP) + BF16 lm_head +
# Q6_K token_embd + IQ4_XS MTP head. 18.34 GiB on disk.
#
# The MTP draft head is baked into the GGUF (blk.64.nextn.*), but it is NOT used
# any more -- see DFLASH2 below.
#
# DFLASH2
# -------
# Drafting is a DFlash2 sidecar (z-lab/Qwen3.8-27B-DFlash2-GGUF, Q4_K_M, 1.14 GB)
# instead of the baked-in MTP head. Block-diffusion drafter: it predicts a whole
# block in one pass and keeps top candidates per position.
#
# DFlash2 IS UPSTREAM NOW. PR #27342 landed 2026-08-27 as b10f9ca5 (squashed
# via #27816) and is an ancestor of LLAMA_REF (b10734). The Dockerfile no longer
# fetches or merges that PR -- it is plain upstream plus patches/. The image
# still carries ONE local patch, the #27408 vision zero-fill; see VISION.
# llama-turboquant:cuda is that build.
# Roll back with IMAGE=llama-turboquant:prev (b10665, same flags -- A/B'd
# 2026-09-01 and identical, see below), IMAGE=llama-turboquant:prev-dflash2
# (the last two-image build), or the old --spec-type draft-mtp,ngram-mod block
# (see git history).
#
# b10665 -> b10734 A/B, 2026-09-01, same flags, same prompts, temp 0:
#   text 55.2 vs 51.5 tok/s, image 54.4 vs 55.7, text-after-image 46.9 vs 44.5,
#   and the draft counts are BYTE-IDENTICAL on both builds (258/560, 201/389,
#   176/488). #27621 (CUDA MOE fusion extended to spec decode) looked like a
#   win on paper and delivered nothing here. Do not re-walk it.
#   These are cold single runs on a synthetic prose prompt, NOT comparable to
#   the vision table below (different prompts and harness) -- compare columns
#   within a table, never across.
#
# ngram-mod DOES chain onto draft-dflash -- it contributed 11% of accepted
# tokens (draft positions 4+, past DFlash2's 4-wide block) and cost nothing:
# dflash-only was 93.3 / 59.9 / 1530 vs 99.9 / 60.1 / 1616 chained. A short
# probe will not show it; ngram needs generated text to match against.
#
# Draft width swept: n-max 4 wins prose by 21% over 7; 10 wins code but costs
# another 15% prefill. 4 is the pick for reasoning-heavy work.
#
# 39,26 trialled 2026-08-25 and REJECTED: it buys ~220 MiB more headroom on the
# main GPU but decode is a wash (92.1/72.0/75.3/70.7 vs 91.8/74.7/75.7/70.8 for
# text/image/text-after-image/3-image), accept rates identical, and the image
# turn is ~4% slower. 40,25 already holds VRAM steady through a 3-image 9k-token
# request, so there is nothing to buy. Do not re-walk this.
# --tensor-split moved 43,22 -> 40,25: the 1.14 GB sidecar plus the 888 MiB
# mmproj no longer fit on the 5070 Ti at 43. Not a DFlash/vision incompatibility,
# just VRAM -- 40,25 loads vision AND the sidecar at the full 120k.
#
# VISION
# ------
# FIXED 2026-08-25, still working at b10734 (re-verified 2026-09-01: the log
# line `zero-filled 91 draft-cache hole rows` appears once per image, 51.7%
# accept on the image turn, correct description, zero decode errors).
# Measured on this box, temp 0, 2898x1068 png (~2.9k image tokens,
# 91-position span), ctx 120000:
#
#                     f5a7ec15 only  + #27408 patch
#   text only         83.6% / 96.3   83.6% / 91.8
#   image turn         0.2% / 22.4   59.1% / 74.7
#   text after image   7.4% / 27.4   52.5% / 75.7
#   3 images, 9k tok        --       56.2% / 70.8
#
# (draft accept % / decode tok/s). Zero llama_decode errors, vision output
# verified correct. For scale: the old broken setup did 35.0 on images and
# draft-mtp 47.3, so dflash now wins images too. The old "switch to draft-mtp
# for image workloads" advice is DEAD.
#
# THE HISTORY, because the trap here is subtle.
# Raw upstream: an image request ABORTED with `llama_decode(ctx_dft) rc=-1`.
# A multimodal target batch is M-RoPE and an image span repeats one temporal
# position, which the draft context rejected as non-consecutive.
# Upstream f5a7ec15 (came in with #27342) makes the draft M-RoPE and passes 4
# position rows.
# That stops the abort but is only HALF a fix: draft() still bases its noise
# block on dp.n_past, the TOKEN count, while the draft cache runs on the
# POSITION scale. An N-row image is N tokens but only ~grid_height positions,
# so the two diverge forever. Result: no crash, no errors, and 0.2% acceptance.
# patches/0001-dflash-mtmd-zero-fill-draft-cache.patch is the complete fix from
# upstream issue #27408 (@fishlikeX, fork commit 3e008b22, never opened as a
# PR): process() SKIPS embedding batches, zero-fills the hole they leave with
# zero-feature encoder rows, and draft() bases the noise block on the draft
# cache's own pos_max + 1. Look for one `zero-filled N draft-cache hole rows`
# per image in the log -- that is the fix working, not a problem.
# Their diff does NOT apply as-is: their base predates the process()
# refactor that #27342 brought in. All three hunks were re-authored, and
# f5a7ec15's M-RoPE handling was kept inside the zero-fill path so the patch
# works with either draft gguf.
# RE-AUTHORED AGAIN 2026-09-01 for b10734: upstream #27310 fused the DFlash
# encoder into the KV injection, deleting the separate llama_encode call and
# the features_buf staging vector, so the gap-fill hunk is now a memset of
# batch_inject.embd plus one llama_decode. Any future bump touching
# common_speculative_impl_draft_dflash::process() will break it again.
#
# DRAFT GGUF: either file works now. The -mrope one below is converted locally
# and carries dflash.rope.dimension_sections; the published GGUFs do not (see
# patches/README.md). With the #27408 patch that key no longer decides whether
# vision works, so treat it as belt-and-braces.
#
# Two dead ends, recorded so nobody re-walks them:
#   - `--swa-full` was REQUIRED with the f5a7ec15-only build. The draft has
#     sliding_window 2048 on all 5 layers, an image span holds the position
#     constant, the window never slides, and the cache never frees cells:
#     `failed to find a memory slot for batch of size 512`. The #27408 patch
#     skips embedding batches entirely, so the flag is NOT needed any more.
#     Verified 2026-08-25 at ctx 120000 without it.
#   - ctx had to drop to 65536 with the f5a7ec15-only build: the draft's CUDA
#     pool growth starved the vision encoder and clip_encode OOM'd at 120000.
#     Also gone. 120000 holds, ~590 MiB free on the main GPU after a 3-image
#     9k-token request, and VRAM stops growing there.
#
# Alternatives measured and rejected 2026-08-21: draft-dspark (slower -- 79/35
# vs 95/59 -- and does not fit at -c 120000), swapping the ngram type (no effect;
# ngram-cache is worse), draft-eagle3 and draft-simple (no Qwen3.8 checkpoint).
#
# Caveat: the BF16 lm_head makes every verification pass more expensive.
# Upstream numbers put the BF16-head tiers at ~15.5 tok/s decode vs ~18.5 for the
# LOW/MEDIUM tiers (Q5_0/Q8_0 heads). BF16 buys output quality, not speed.
#
# NVFP4 is native only on the 5070 Ti (Blackwell). The 4070 SUPER (Ada) runs it
# through a dequant path, same as the Qwen3.6-27B-NVFP4 model already does.
#
# REASONING
# ---------
# `--reasoning on`, same as run-Q3.8-27B.sh. Same model, same chat template.
# With it off, the template prefills an empty pre-closed <think></think> and the
# model has nowhere to plan.
#
# Harnesses select the LEVEL per request -- it is not a server flag:
#   "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "low"}
#   "chat_template_kwargs": {"enable_thinking": false}          # per-request off
#   "chat_template_kwargs": {"preserve_thinking": false}        # keep only latest
# Valid levels: xhigh (default), medium, low. The OpenAI top-level
# `reasoning_effort` field does NOT reach this template.
#
# The samplers below are still the INSTRUCT preset, not Qwen's thinking preset.

# No env exports here. Verified 2026-08-24 against the pinned llama.cpp ref:
#   CUDA_MALLOC_ASYNC_SUPPORTED -- not read by llama.cpp/ggml at all.
#   GGML_CUDA_FORCE_MMQ         -- compile-time cmake option now, not an
#                                  env var (ggml/CMakeLists.txt, default OFF).
# Both were also never reaching the container: docker create only forwards
# the -e flags listed below, not the host shell's exports.

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-host}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
# 65 blocks (64 layers + 1 MTP)
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-NVFP4-MTP-VERY-HIGH.gguf}"
MMPROJ_FILE="${MMPROJ_FILE:-mmproj-Qwen3.8-27B-NVFP4-BF16.gguf}"
DRAFT_FILE="${DRAFT_FILE:-Qwen3.8-27B-DFlash2-Q4_K_M-mrope.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

PARALLEL="${PARALLEL:-1}"
CONTEXT="${CONTEXT:-120000}"

NAME="${NAME:-llama-turboquant}"

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
    -v "$(pwd)/models:/models:ro" \
    -v "$(pwd)/scripts:/scripts:ro" \
    --entrypoint /scripts/entrypoint.sh \
    "${IMAGE}" \
    --model "/models/${MODEL_FILE}" \
    --model-draft "/models/${DRAFT_FILE}" \
    --mmproj "/models/${MMPROJ_FILE}" \
    --mmproj-offload \
    --image-min-tokens 1024 \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --split-mode layer \
    --tensor-split 40,25 \
    -fit off \
    --flash-attn on \
    -c "${CONTEXT}" \
    -n -1 \
    --parallel "${PARALLEL}" \
    --kv-unified \
    -ctk q8_0 \
    -ctv q8_0 \
    -ctkd q8_0 \
    -ctvd q8_0 \
    --load-mode mlock \
    --jinja \
    --reasoning on \
    --reasoning-preserve \
    --reasoning-effort medium \
    --spec-type draft-dflash,ngram-mod \
    --spec-draft-n-max 4 \
    --spec-ngram-mod-n-match 24 \
    --spec-ngram-mod-n-min 4 \
    --spec-ngram-mod-n-max 48 \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 0.0 \
    --repeat-penalty 1.0 \
    -b 2048 \
    -ub 512 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
