# Patches

The build is plain upstream `ggml-org/llama.cpp` at `LLAMA_REF`, plus every
`*.patch` in this directory. `git apply` runs inside the clone `RUN` chain's
`&&` list on purpose — a stale patch then fails the build instead of being
silently skipped.

**No PR is merged in any more.** PR #27342 (DFlash2) landed upstream on
2026-08-27 as `b10f9ca5` (squashed via #27816), so the old
`git fetch pull/27342/head && git merge` step and the `DFLASH_PR`/`DFLASH_REF`
build args are gone from the `Dockerfile`. Do not add them back.

## Active: 0001-dflash-mtmd-zero-fill-draft-cache.patch

The complete vision fix from upstream issue #27408 (@fishlikeX, fork commit
`3e008b22`, never opened as a PR). It zero-fills the holes that skipped mtmd
image chunks leave in the DFlash draft KV cache, via the same encode+inject path
with all-zero features, so the draft and target caches stay position-aligned.

**Still needed at `LLAMA_REF`.** Issue #27408 is still open and none of the
zero-fill code is in upstream master — what came in with #27342 is only
`f5a7ec15`, which is half of it (see below). Re-verified against `b10665` on
2026-08-28: applies clean, 4 hunks, offset +18 lines.

Drop this patch, the `COPY patches/` line and the `git apply` line once #27408
lands upstream.

## Dropped: 0001-dflash-dense-inject-pos-for-vision.patch (2026-08-25)

@Shamish's community workaround from PR #27342, carried 2026-08-19 .. 2026-08-25.
It renumbered the DFlash inject batch densely, seeding from
`llama_memory_seq_pos_max()`, so `llama_decode(ctx_dft)` stopped returning
`rc=-1` on image requests.

It was only half a fix. `common_speculative_impl_draft_dflash::draft()` still
positions its noise block at `dp.n_past`, which is the **target's** position. So
the draft cache advanced densely while the target advanced with the image span
included. Every image opened a permanent 1032-position gap here
(`--image-min-tokens 1024` + 8 markers), and from the first image onward every
draft decode failed with `inconsistent sequence positions`. Silent: llama-server
just falls back to plain decode, so vision chats lost all speculative decoding
and only the log showed it (40k errors in one 5-hour run).

Superseded by upstream `f5a7ec15` ("Apply patch to fix the mrope bug"), which
is now in master (it came in with #27342 on 2026-08-27) and no longer needs to
be merged in. It sets `is_mrope` from the **draft** model's rope type
and writes 4 position rows per token into both the DFlash encoder batch and the
inject batch, so the draft carries the target's real M-RoPE positions and the two
caches stay in lockstep.

### The fix is gated on the draft GGUF — read this before swapping models

`llama_model_rope_type()` returns `LLAMA_ROPE_TYPE_MROPE` for `LLM_ARCH_DFLASH`
only when `hparams.rope_sections` is non-zero. That key
(`dflash.rope.dimension_sections`, degenerate `[head_dim/2, 0, 0, 0]`) is written
only by a converter at >= `f5a7ec15`. **The published DFlash2 GGUFs do not have
it** — checked z-lab/Qwen3.8-27B-DFlash2-GGUF (re-uploaded 2026-08-24 23:07) and
incoai/Qwen3.8-27B-DFlash2-GGUF. With those, `is_mrope` is false, the fix is dead
code, and vision requests go back to `rc=-1`.

So the draft GGUF on this box is converted locally from the z-lab safetensors,
with a converter at >= `f5a7ec15` (now just upstream `LLAMA_REF`). Sources live
in `models/hf/`.

Before trusting any DFlash2 draft GGUF:

    python3 -c "…"   # or: strings -n 8 <file> | grep dimension_sections

No `dflash.rope.dimension_sections` means no vision drafting.

## Dropped earlier, when the turboquant fork was

- `0001-fix-spec-i-batch-guard.patch` — guarded an empty `spec_i_batch` in the
  server's speculative verify loop. It only existed because the fork's
  `has_checkpoint_restored_prompt` skip could hold a GENERATING slot out of the
  batch. That symbol does not exist upstream, so the patch neither applies nor is
  needed.
- `0002-qwen35moe-expose-layer-inp.patch` — `res->t_layer_inp[il] = inpL;` in
  `src/models/qwen35moe.cpp`, needed for draft-dflash/draft-eagle3 hidden-state
  extraction. Now present upstream verbatim (`qwen35moe.cpp:183`).
