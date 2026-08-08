#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export CUDA_MALLOC_ASYNC_SUPPORTED=1
export GGML_CUDA_FORCE_MMQ=1

IMAGE="${IMAGE:-llama-turboquant:cuda}"
HOST_PORT="${HOST_PORT:-8080}"
NETWORK="${NETWORK:-runner-network}"
STATIC_IP="${STATIC_IP:-172.18.0.10}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-Fast-NVFP4.gguf}"
DRAFT_FILE="${DRAFT_FILE:-Qwen3.6-35B-A3B-DFlash-Q8_0.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"

# Env-overridable so these can be swept without editing the file. Note that
# appending `--spec-type ...` to this script's args does NOT work: the flag is
# already passed below and the parser keeps the first occurrence, so the append
# is silently ignored. Set the env var instead.
SPEC_TYPE="${SPEC_TYPE:-draft-dflash,ngram-mod}"
# 3 is MEASURED, not upstream's recommendation. llama.cpp PR #22105 suggests 15
# and z-lab's card suggests block_size 16 -- both are wrong for this box. Swept
# 2/3/5/7/10/15 on one 400-tok prompt (baseline no-spec = 83.6 t/s):
#   n_max   2      3      5      7      10     15
#   t/s     109.6  113.6  109.9  102.2  76.4   69.7
#   acc     .608   .527   .392   .296   .186   .122
#   accepted tokens: ~244-268 at EVERY depth
# Accepted count is flat while drafted grows linearly, so past ~3 the extra draft
# tokens are pure wasted MoE bandwidth: 8-of-256 routing means each speculated
# token drags a fresh expert union through memory. Deeper drafts lose outright --
# n_max>=10 is slower than no speculation at all. Re-sweep if the model changes.
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"

# VISION IS DISABLED -- --mmproj is deliberately not passed. It is not a config
# preference: draft-model speculation and images cannot coexist in this build.
# server-context.cpp:3324 feeds image chunks into the draft context whenever the
# draft is a *separate* model (ctx_other != ctx_tgt, i.e. DFlash or EAGLE3, but
# not MTP which shares ctx_tgt). DFlash has no vision encoder, so its KV falls
# behind the target's positions -> "non-consecutive token position" -> empty
# batches -> the n_empty_consecutive > 3 GGML_ABORT. The fork tags this as known
# unfinished work: [TAG_MTMD_DRAFT_PROCESSING] in server-context.cpp:20/259/3328.
# To get vision back you must give up DFlash: re-add --mmproj + --mmproj-offload,
# switch --spec-type back to draft-mtp, drop --model-draft, and move blk 40 off
# CPU in OT_CPU_MTP below (draft-mtp actually executes it). models/mmproj-Qwen3.6-
# 35B-A3B-F16.gguf is still on disk for that. Measured: DFlash text decode ran
# 78-83 tok/s here.

# qwen35moe: 41 blocks (blk 0-40), 256 experts / 8 used, head 16, kv 2, ctx 262144.
# Hybrid attention: full_attention_interval=4 -> only blk 3,7,11,...,39 (10 layers)
# carry a KV cache; the other 30 are SSM/linear and hold a small recurrent state.
# That is why 120k ctx costs ~0.7 GB of KV here, not the ~3 GB the dense 27B needs.
#
# blk 40 is the MTP/nextn head: n_layer() = 41 - n_layer_nextn = 40, so the main
# graph only runs blk 0-39. It is executed *only* by the draft-mtp speculator --
# which this script no longer uses (see DFlash below), making its 1.5 GB of BF16
# experts dead weight. Pinned to CPU to reclaim that VRAM; it is never evaluated.
#
# Sizes measured from the GGUF tensor table (nvfp4 = 64 elems / 36 B):
#   experts blk 0-39 : 432 MiB each, uniform     (nvfp4)
#   experts blk 40   : 1536 MiB                  (bf16, MTP head -> CPU)
#   non-expert       : 4508 MiB total            (incl. bf16 token_embd + output)
#   whole file       : 22.9 GiB
# tensor-split 1,0 puts all attn/shared/embed tensors *and* the KV cache on CUDA0.
# CUDA0 is the RTX 5070 Ti (16 GB, native FP4) and also drives the desktop, so it
# carries the fixed ~6.6 GB (non-expert + KV + draft + compute bufs) and gets the
# smaller expert share; CUDA1 (RTX 4070 SUPER, 12 GB, headless) gets more.
# Dropping --mmproj freed ~858 MiB on CUDA0, so blk 24 moved CUDA1 -> CUDA0 to
# relieve CUDA1, which was the tighter card at ~600 MiB slack.
# Budget: CUDA0 ~13.5/15.9 GB free, CUDA1 ~10.9/11.9 GB free -> nothing on CPU.
OT_CPU_MTP='blk\.40\.ffn_(gate|up|down)_exps\.=CPU'                      # blk 40    -> RAM (dead MTP head)
OT_CUDA0='blk\.(2[6-9]|3[0-9])\.ffn_(gate|up|down)_exps\.=CUDA0'         # blk 24-39 -> 16 GB GPU (6912 MiB)
OT_CUDA1='blk\.(2[0-5]|1[0-9]|[0-9])\.ffn_(gate|up|down)_exps\.=CUDA1'   # blk 0-23  -> 12 GB GPU (10368 MiB)
OVERRIDE_TENSOR="${OVERRIDE_TENSOR:-${OT_CPU_MTP},${OT_CUDA0},${OT_CUDA1}}"

# Speculative decoding: DFlash *replaces* draft-mtp rather than stacking with it.
# Both draft-mtp and draft-dflash bind to the one ctx_dft that --model-draft
# creates, so listing both would hand the DFlash weights to the MTP speculator.
# DFlash (arch "dflash"): 6 blocks / 386M, reads the target's hidden states at
# layers 2,7,12,17,23,28,33,38 and emits a whole block per step instead of one
# token. dflash.block_size is 16, so --spec-draft-n-max is clamped to 15 by the
# server. 7 is a starting point, NOT a measured optimum -- sweep it.
# -ctkd/-ctvd now quantize the DFlash cache (head_dim 128, SWA 4096 on 5 of 6
# blocks): q8_0/turbo3 keeps it near 170 MB vs ~575 MB at f16.

for f in "${MODEL_FILE}" "${DRAFT_FILE}"; do
    if [[ ! -f "models/${f}" ]]; then
        echo "Not found: models/${f}" >&2
        exit 1
    fi
done

NAME="${NAME:-llama-turboquant}"

# Remove any prior container with the same name so re-running this script
# always produces a fresh, unstarted container.
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
    -e TURBO_AUTO_ASYMMETRIC=0 \
    -p "${HOST_PORT}:8080" \
    --network "${NETWORK}" \
    --ip "${STATIC_IP}" \
    -v "$(pwd)/models:/models:ro" \
    -v "$(pwd)/scripts:/scripts:ro" \
    --entrypoint /scripts/entrypoint.sh \
    "${IMAGE}" \
    --model "/models/${MODEL_FILE}" \
    --model-draft "/models/${DRAFT_FILE}" \
    --host 0.0.0.0 \
    --port 8080 \
    --metrics \
    --n-gpu-layers "${N_GPU_LAYERS}" \
    --main-gpu 0 \
    --tensor-split 1,0 \
    --override-tensor "${OVERRIDE_TENSOR}" \
    -fit off \
    --flash-attn on \
    -c 120000 \
    -n -1 \
    --parallel 1 \
    -ctk q8_0 \
    -ctv q8_0 \
    -ctkd q8_0 \
    -ctvd q8_0 \
    --kv-unified \
    --no-mmap \
    --mlock \
    --jinja \
    --reasoning off \
    --temp 1.0 \
    --top-p 0.95 \
    --top-k 20 \
    --min-p 0.0 \
    --presence-penalty 1.5 \
    --repeat-penalty 1.0 \
    --spec-type "${SPEC_TYPE}" \
    --spec-draft-n-max "${SPEC_DRAFT_N_MAX}" \
    --spec-ngram-mod-n-match 24 \
    --spec-ngram-mod-n-min 4 \
    --spec-ngram-mod-n-max 48 \
    -b 2048 \
    -ub 1024 \
    --cache-idle-slots \
    --cache-ram 8192 \
    --cache-reuse 256 \
    --threads 8 \
    --cpu-range 0-7 \
    --timeout 360 \
    "$@" >/dev/null

docker start -a "${NAME}"
