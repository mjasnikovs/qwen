# Fork-local patches

Applied by the Dockerfile build stage on top of `TURBOQUANT_REF`, after the
clone/checkout. `git apply` runs with `&&` in the RUN chain, so a patch that no
longer applies **fails the build** rather than being silently skipped. On a
`TURBOQUANT_REF` bump, re-check each patch against the new ref and drop any that
landed upstream.

## 0001-fix-spec-i-batch-guard.patch

Fixes a crash with `--parallel 2` + `--spec-type draft-mtp` + context checkpoints:

```
/src/tools/server/server-context.cpp:3777: GGML_ASSERT(slot.spec_i_batch.size() == n_draft + 1) failed
```

Two fork commits don't compose:

- `455d8e4b server : speculative checkpointing (#19493)` (upstream) added the
  spec verify loop, which assumes every GENERATING slot holding a `spec_draft`
  was batched this iteration and therefore has a populated `spec_i_batch`.
- `d6ae83f6 fix server checkpoints with parallel restore` (fork-local, on
  `feature/turboquant-kv-cache` only) added `has_checkpoint_restored_prompt`,
  which skips *every* generating slot when another slot restored a context
  checkpoint — a workaround for CUDA instability when a restored prompt suffix
  is evaluated in a mixed batch.

When the skip fires, `update_batch()` never runs, so `spec_i_batch` stays empty
while `spec_draft` still holds a draft carried over from a partial acceptance.
The verify loop filters only on state/`can_speculate()`/`spec_draft`, so the slot
reaches the assert with `0 != n_draft + 1` and aborts.

Needs two slots (one GENERATING, one PROCESSING_PROMPT), so `--parallel 1` is
unaffected. Still present on the fork's HEAD as of 2026-07-16 — bumping the ref
does not fix it.

The patch adds `|| slot.spec_i_batch.empty()` to the verify loop's skip
condition: if the slot wasn't batched there are no logits to verify against, so
the draft is left for the existing reuse path (`server-context.cpp:2831`) to pick
up next iteration. This preserves d6ae83f6's intent and restores #19493's
invariant; an abort becomes a one-iteration defer.

Note: this fixes the assert only. The CUDA instability that d6ae83f6 works
around is unaddressed and still underlies the parallel config.

Upstream status: not reported. Belongs in `TheTom/llama-cpp-turboquant`
(`feature/turboquant-kv-cache`), not upstream llama.cpp, which has no
`has_checkpoint_restored_prompt`.
