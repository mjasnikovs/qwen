# Patches

One patch, and it belongs to `Dockerfile.dflash2` only. The plain `Dockerfile`
build is unpatched upstream `ggml-org/llama.cpp` (see `LLAMA_REF` there) and does
not copy this directory.

- `0001-dflash-dense-inject-pos-for-vision.patch` — makes image requests work with
  `--spec-type draft-dflash`. The DFlash inject batch copied the target's
  positions verbatim; a multimodal target batch is M-RoPE, so an image span
  repeats the temporal component. `ctx_dft` is a plain `qwen3` with
  `n_pos_per_embd == 1`, `llama_batch_allocr` demands continuous positions, and
  `llama_decode(ctx_dft)` returned `rc=-1` — the request died. The patch numbers
  the injected rows densely per sequence, seeding from
  `llama_memory_seq_pos_max()`. No-op for text-only batches, where the target
  positions are already dense.

  Source: community fix posted by @Shamish in PR #27342 on 2026-08-19, not merged
  into the PR head. Applies to `5ecbe1ac` only — re-check when the PR head moves.
  Drop it once the PR lands with the fix included.

Two older patches were dropped when the turboquant fork was:

- `0001-fix-spec-i-batch-guard.patch` — guarded an empty `spec_i_batch` in the
  server's speculative verify loop. It only existed because the fork's
  `has_checkpoint_restored_prompt` skip could hold a GENERATING slot out of the
  batch. That symbol does not exist upstream, so the patch neither applies nor is
  needed.
- `0002-qwen35moe-expose-layer-inp.patch` — `res->t_layer_inp[il] = inpL;` in
  `src/models/qwen35moe.cpp`, needed for draft-dflash/draft-eagle3 hidden-state
  extraction. Now present upstream verbatim (`qwen35moe.cpp:183`).

Patching is wired in as `COPY patches/ /patches/` plus
`&& git apply --verbose /patches/*.patch` inside the clone `RUN` chain. Keeping it
in the `&&` chain is deliberate — a stale patch then fails the build instead of
being silently skipped.
