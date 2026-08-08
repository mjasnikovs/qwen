# Patches

Empty. The build is plain upstream `ggml-org/llama.cpp` (see `LLAMA_REF` in the
Dockerfile) and applies no patches.

Both former patches were dropped when the turboquant fork was:

- `0001-fix-spec-i-batch-guard.patch` — guarded an empty `spec_i_batch` in the
  server's speculative verify loop. It only existed because the fork's
  `has_checkpoint_restored_prompt` skip could hold a GENERATING slot out of the
  batch. That symbol does not exist upstream, so the patch neither applies nor is
  needed.
- `0002-qwen35moe-expose-layer-inp.patch` — `res->t_layer_inp[il] = inpL;` in
  `src/models/qwen35moe.cpp`, needed for draft-dflash/draft-eagle3 hidden-state
  extraction. Now present upstream verbatim (`qwen35moe.cpp:183`).

To reintroduce patching: `COPY patches/ /patches/` in the build stage and add
`&& git apply --verbose /patches/*.patch` to the clone RUN chain. Keeping it in
the `&&` chain is deliberate — a stale patch then fails the build instead of
being silently skipped.
