# 80B autoretry verification + blendcorpus cold-cache race fix (end-to-end)

- **Date:** 2026-06-28
- **Machine:** Aurora
- **Goal:** verify `scripts/submit_agpt_80b_autoretry.sh` works on debug-scaling
  from the up-to-date `ezpz` main repo, and (as it turned out) fix the
  blendcorpus index-cache race that blocked a cold-`CKPT_DIR` 80B run.
- **Status:** RESOLVED. Autoretry script validated; blendcorpus race fixed
  at the source and confirmed end-to-end (80B trains from a cold cache).

## Jobs

| Job | N | Queue | venv blendcorpus | Result |
|-----|---|-------|------------------|--------|
| 8574063 | 32 | debug-scaling | 74b09fd | mass race (EOFError ~75% ranks) at `_build_index_mappings`; stuck_pre_training bail |
| 8574172 | 4 | debug-scaling | 74b09fd | same mass race |
| 8574084 | 2 | debug | 74b09fd | all-rank prewarm also raced (TP=2) |
| 8574194 | 1 | debug | 74b09fd | single-rank prewarm built cache cleanly (no concurrent readers) |
| 8574237 | 4 | debug-scaling | 041d015f | mass race GONE (0 EOFError); 2 ranks hit a TOCTOU FileNotFoundError in the new poll guard |
| **8574261** | **4** | **debug-scaling** | **1f7e9c0** | **PASS: 0 EOFError, 0 FileNotFoundError, cold build OK, trains (step 1 loss 12.91)** |

Config (all 80B smokes): TP=4, LBS=1, GAS=1, dp_degree<=96 (safe corner),
AdamW LR=1e-6, compile OFF, AC=full, GBS=12 (4N) / 90 (32N),
`TRAINING_STEPS=20 CKPT_INTERVAL=10 WARMUP_STEPS=5 VALIDATOR_ENABLE=0`.

## The autoretry script: validated (job 8574063)

Every mechanical phase worked first try: venv yeet (90s @ 32N), device-mesh
build (`pp=1 dp_shard=90 tp=4`), launch-command assembly (incl.
`activation-checkpoint:full` passed positional-last), auto-retry active/spare
split, and the `stuck_pre_training` guard bailing cleanly (rc=143) after two
zero-progress attempts instead of burning the whole machine. The script is not
the blocker.

## The blocker: blendcorpus `_build_index_mappings` race (TP>1, cold cache)

`gpt_dataset.py::_build_index_mappings` built each per-corpus
`*_{doc,sample,shuffle}_idx.npy` on rank 0 (writing to the FINAL name via
`np.save`), then ALL ranks `np.load`'d them with no synchronization. At TP>1,
~(1 - 1/TP) of ranks (TP-coord != 0, not gated by the sibling subgroup
barriers) race ahead and mmap a file mid-write -> EOFError / "mmap length >
file size" / "invalid load key '\x00'". ~75% of ranks crash at TP=4.

A `dist.barrier()` cannot fix it: `_build_index_mappings` is called a
data-dependent number of times (per-corpus x per-split, cache-hit vs miss), so
a barrier there fires a mismatched count across ranks -> oneCCL participation
hang (that attempt was `c7eb628e`, reverted in `74b09fd`). `74b09fd` added
safe global barriers in the SIBLING paths (`blendable` / `_cache_indices` /
`build_corpus_datasets`) but left `_build_index_mappings` racy, with prewarm as
the documented mitigation.

## The fix (two commits, saforem2/blendcorpus@feat/remove-deepspeed)

1. **`041d015f`** (atomic writes + poll) -- the real runtime fix. Writer:
   `_atomic_save` writes to `f"{final}.tmp.{pid}.{rank}"` then
   `os.replace(saved, final)` (atomic on Lustre). Readers:
   `_wait_for_index_files` polls each path until it exists, is non-empty, and
   `np.load(mmap)` validates the header. File-polling, not a collective ->
   immune to the participation hang. Result (job 8574237): mass race GONE
   (0 EOFError).

2. **`1f7e9c0`** (TOCTOU fix) -- the poll guard was
   `if os.path.isfile(target) and os.path.getsize(target) > 0:` with only
   `np.load` inside the `try`. When the builder's `os.replace` swapped `target`
   between `isfile()` and `getsize()`, `getsize` raised `FileNotFoundError`
   OUTSIDE the try -> escaped uncaught, killing 2/48 ranks (job 8574237).
   Fix: move the size probe inside the `try` (FileNotFoundError is-a OSError,
   already caught); drop the separate `isfile` to remove the two-syscall
   window. Result (job 8574261): 0 EOFError AND 0 FileNotFoundError, cold
   index "finished saving index map files in 0.03s", 80B trains.

## Validation evidence (job 8574261, cold cache)

```
17:28:16  Rank 0: building blendcorpus datasets (cache=.../n4-gbs12/...)
17:28:53  building indices for blendable datasets ...
          finished saving index map files in 0.03s
17:29:54  step:  1  loss: 12.91013  grad_norm: 5.1032  (correct fresh-init 80B loss)
17:30:27  step:  2  loss: 12.84005  grad_norm: 5.0389  mfu: 11.48%
```
0 EOFError, 0 FileNotFoundError. Prior runs crashed within ~5s of the
"building blendcorpus" line; this one sailed through to training.

## Operational takeaways

- 80B can now run from a fresh `CKPT_DIR` with NO prewarm (venv >= `1f7e9c0`).
- `scripts/prewarm_blendcorpus_singlerank.sh` (single-rank builder) remains as
  a belt-and-suspenders option / for older venvs.
- Production runs from the up-to-date main repo dir (this dir), which is in
  sync with `origin/ezpz`. The `agpt-80b-v2` clone is stale (Jun 12) and holds
  no checkpoints.

## Cross-refs
- [`docs/guides/known-bugs/blendcorpus-eoferror-race.md`](../../../guides/known-bugs/blendcorpus-eoferror-race.md)
- [`docs/upstream-issues/blendcorpus-atomic-rename-index-fix.md`](../../../upstream-issues/blendcorpus-atomic-rename-index-fix.md)
