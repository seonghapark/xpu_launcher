---
author: Sam Foreman
date: 2026-05-24
status: open
---

# Blendcorpus EOFError Race in `_build_index_mappings`

> **Symptom:** at init time, 20+ ranks crash with
> `EOFError: No data left in file` from `np.load(idx_path["shuffle"],
> mmap_mode="r")`. The wrapper classifies the failure as a bad-node
> issue (because of crash-pattern matching), swaps a spare in, retries
> — and **the same failure recurs every attempt**. Wrapper exhausts
> retries.

## TL;DR

When multiple ranks race to read `*_shuffle_idx.npy`, `*_doc_idx.npy`,
or `*_sample_idx.npy` files from a fresh dataset cache, ranks observing
the file **mid-write** (size = 0 at that moment) crash with
`EOFError: No data left in file`. The cascade takes down 20+ ranks
within a few seconds. Reproduced deterministically at **80B 8N** on
2026-05-24 (smoke job `8505326`); previously masked as a bad-node
failure at **80B 522N** (job `8485515`).

This is **NOT** a node-health issue. Wrapper swap-in + retry does not
help because the new active set still races on the same fresh cache
dir.

## Reproducer

Production smoke job `8505326` (2026-05-24): 80B failover script on
8 train + 2 spare nodes (96 GPUs total), with a brand-new `CKPT_DIR`
(no pre-existing index cache). All 3 wrapper attempts hit the same
pattern.

Each attempt's stack trace is identical:

```
File "/tmp/.venv/lib/python3.14/site-packages/blendcorpus/data/gpt_dataset.py", line 1132, in _build_index_mappings
    shuffle_idx = np.load(idx_path["shuffle"], allow_pickle=True, mmap_mode="r")
File "/tmp/.venv/lib/python3.14/site-packages/numpy/lib/_npyio_impl.py", line 463, in load
    raise EOFError("No data left in file")
EOFError: No data left in file
```

Inspecting the cache dir AFTER the crash shows the smoking gun:

```bash
$ find /flare/.../checkpoints/smoke-N8-20260523-211437/.cache -size 0 -type f
.../index-cache/e4aef10bca4b77fe2b13669e623dc82e_shuffle_idx.npy
.../index-cache/3895a3c2cecfa5dd26cd5a5a238cc65d_shuffle_idx.npy
.../index-cache/a3d5e7c224ede6e86d55dffa6a17d504_doc_idx.npy
.../index-cache/413201691da80c5744896c21c3e8c0fc.dsc
.../index-cache/8c0511cb00845ae7786fe0d61b089175_doc_idx.npy
.../index-cache/2886966e81c1e5579e982897621959e5_sample_idx.npy
.../index-cache/d67b0cbc95ab2cb68087a8629036be43_shuffle_idx.npy
```

**Seven zero-byte `.npy` files** — the rank-0-as-builder cache write
hadn't flushed when the other ranks tried to mmap. Once those zero-byte
files exist on disk, every subsequent attempt sees them and re-races.

## Why the wrapper can't recover

The failover wrapper's swap-and-retry logic assumes failures are
node-correlated. For this bug:

1. Attempt 1: rank-0 starts writing the cache, ranks 1-95 race ahead,
   ~20 ranks mmap the half-written file → EOFError → wrapper sees
   crash patterns → marks node bad → swaps a spare.
2. Attempt 2: **same code path on the new active set**, hits the SAME
   partially-written files left on disk by attempt 1 (the .cache dir
   persists across attempts via the same `CKPT_DIR`). Re-races again.
3. Attempt 3: still using the same cache dir. Still racing. Wrapper
   exhausts retries, exits.

I verified this by `rm -f`'ing the partial files between attempts 2
and 3 of `8505326` — attempt 3 STILL hit EOFError on freshly-built
partial files. The race is on every fresh build, not just on
leftover artifacts.

## Where the bug lives

`blendcorpus/data/gpt_dataset.py` line ~1132 (`_build_index_mappings`).
The function tries to coordinate across ranks via filesystem presence
checks ("does this file exist yet?") but doesn't account for the
fact that **file presence != file completeness**. A non-builder rank
that gets to `os.path.exists(idx_path["shuffle"])` between the
builder's `open(...)` and the builder's final `flush+close` sees
the file but mmaps zero bytes.

## Workarounds

1. **Pre-build the cache once on rank 0 (offline)**, then point production
   at the existing cache. The 2B/20B/80B production trajectories that
   work today all do this implicitly — they resume from prior dispatches
   where the cache was already built.
2. **Don't change CKPT_DIR between dispatches** of the same model/data
   config. Reusing the same `CKPT_DIR` means the cache stays built. The
   chain-continuation pattern (`afterany:`) naturally satisfies this.
3. For the rare **fresh-CKPT_DIR case** (new model size, new dataset,
   new GBS), do a 1-rank "warmup" run that pre-builds the cache before
   launching the multi-rank training. Could be wired into `failover_run`
   as a pre-step.

## Fix (upstream) -- LANDED 2026-06-28

**Root cause (refined):** the build path already had DP-group and
PP-group barriers, but at **TP>1** those subgroup barriers do NOT gate
ranks whose TP coordinate != 0 against the global rank-0 writer -- those
ranks live in DP/PP subgroups that exclude rank 0, so their subgroup
barriers self-satisfy and ~(1 - 1/TP) of ranks race ahead to `np.load`
the index before rank 0 finished writing it. That is exactly why an 80B
TP=4 run saw ~75% of ranks crash (EOFError / "mmap length > file size" /
"invalid load key '\x00'").

**The fix** is `saforem2/blendcorpus@74b09fd` (branch
`feat/remove-deepspeed`): add a plain **global** `torch.distributed.barrier()`
right after the existing DP/PP-subgroup barriers in three places --
`blendable_dataset.py` (blendable index) and `gpt_dataset.py`
`_cache_indices` + `build_corpus_datasets` (per-corpus index). Every rank
now waits for the rank-0 writer before reading.

IMPORTANT: the same commit **deliberately does NOT** add a barrier in
`_build_index_mappings` (an earlier attempt to, commits `debfff5` /
`c7eb628e`, was reverted in `74b09fd`): that function is called per-corpus
x per-split and its build-vs-read branch is data-dependent (cache hit vs
miss), so a barrier there fires a mismatched number of times across ranks
-> oneCCL `allreduce_scaleout` participation mismatch / hang
(`atl_comm->wait fails with status: 1`), seen on 80B TP=4 / 744 ranks.

To get the fix into a venv (torch-free package, safe with `--no-deps`):
```bash
.venv/bin/python3 -m pip install --no-deps --no-cache --force-reinstall \
  'git+https://github.com/saforem2/blendcorpus@74b09fd44f8a9974568c977b23c05b11b1148668'
# then rebuild the broadcast tarball: tar -czf .venv.tar.gz --directory <parent> .venv
```
The repo `.venv` was on the pre-fix `3926c542` (Jun 2) which had the
subgroup barriers but not the global one.

### IMPORTANT: `74b09fd` does NOT fix the `_build_index_mappings` race

Verified on hardware 2026-06-28 (jobs 8574063 32N, 8574172 4N, both TP=4,
cold cache): even with `74b09fd` installed, the 80B smoke STILL crashes
with the identical EOFError/mmap pattern. The crash traceback is:
```
gpt_dataset.py:835  __init__ -> _build_index_mappings(
gpt_dataset.py:1158 shuffle_idx = np.load(idx_path["shuffle"], ...)  <- EOFError
```
i.e. the race is in **`_build_index_mappings`** -- the ONE path where
`74b09fd` *deliberately reverted* the barrier (a barrier there fires a
data-dependent (cache-hit vs miss) number of times across ranks ->
oneCCL `allreduce_scaleout` participation mismatch / hang, strictly worse
than the race). The 3 global barriers `74b09fd` adds protect the
*blendable* index + `_cache_indices` + `build_corpus_datasets` paths --
real races, but NOT the per-corpus `_build_index_mappings` one that an 80B
TP=4 cold-cache run actually hits first.

The in-code NOTE at `gpt_dataset.py:1135` is explicit: the
`_build_index_mappings` race "is mitigated operationally by pre-warming
the index cache before large runs
(`scripts/prewarm_blendcorpus_cache.sh`); a deadlock is strictly worse
than that rare race." So **prewarm is the intended mitigation for this
path**, not a barrier. (And empirically at TP=4 cold-cache the race is
not "rare" -- it hit immediately on both runs.)

### Mitigation: single-rank prewarm (the real workaround)

The race is between rank-0 (writer) and the other ranks (`np.load`
readers) with no barrier between in `_build_index_mappings`. The ONLY
race-free build is **a single rank** (no concurrent readers at all).
Caveats for `prewarm_blendcorpus_cache.sh`:
- It launches with ALL ranks of its allocation, so at TP>1 it can itself
  hit this same race (observed: the 2N prewarm 8574084 raced). Run it so
  the dataset build sees effectively one reader -- e.g. a 1-node / TP=1
  build, or extend the script to run the dataloader build on a single
  rank.
- The cache hash keys on `num_samples = GBS x train_iters` (per-corpus
  `desc = prefix + num_samples + seq_length + seed`, `gpt_dataset.py:121`),
  so the prewarm MUST pass the SAME `GBS` and `TRAINING_STEPS` as the
  target run or the hash won't match and the target rebuilds cold. The
  script does NOT currently forward `--training.global-batch-size`; pass
  `GBS=` explicitly or fix the script.
- Reusing the same `CKPT_DIR` across dispatches keeps the cache warm
  (chain-continuation `afterany:` naturally satisfies this).

## Related

- **80B 522N production `8485515` (2026-05-22)**: same EOFError
  pattern, was misattributed at the time to bad-node prevalence
  (the failover wrapper's regex matches "EOFError" → swap → retry,
  same trap). Re-analyzing the log shows zero-byte cache files
  identical to those in 8505326.
- [`blendcorpus-megatron-aliasing.md`](blendcorpus-megatron-aliasing.md):
  sibling guide covering separate blendcorpus issue (7 leftover
  Megatron-style aliases in the dataloader).
- [`bad-node-failover.md`](../bad-node-failover.md): the wrapper
  guide — note that the wrapper's swap-and-retry is a hammer that
  doesn't work on every nail. For this bug, the right escape hatch
  is the wrapper bailing after retries exhausted, then a human
  pre-building the cache.

## Status

**FIXED at the source 2026-06-28** by `saforem2/blendcorpus@041d015f`
("data: fix build-then-load index race at TP>1 (atomic writes +
poll-for-complete)") -- the atomic-rename approach. rank 0 writes each
`*_idx.npy` to a per-pid/per-rank temp then `os.replace()`s it onto the
final path (atomic on Lustre); every rank polls via `_wait_for_index_files`
until all three files are complete before `np.load`. File-polling instead
of a barrier => immune to the oneCCL participation hang that killed the
barrier approach (`c7eb628e`, reverted in `74b09fd`). Works at any TP,
needs no prewarm. The repo `.venv` was reinstalled `74b09fd` -> `041d015f`
and the broadcast tarball rebuilt.

Pending on-hardware confirmation: a cold-`CKPT_DIR` 80B TP=4 smoke (the
exact config that raced on `74b09fd`) -- see
[`docs/experiments/agpt/aurora`](../../experiments/agpt/aurora/). (Do NOT
mark this fully closed until that cold run passes; an earlier in-session
"FIXED" claim for `74b09fd` was premature -- the hardware re-run then
showed `74b09fd` only fixed the sibling paths, not `_build_index_mappings`.
`041d015f` is the commit that actually patches `_build_index_mappings`.)

### Timeline of the fix
- `3926c542` (Jun 2): DP/PP-subgroup barriers only -> races at TP>1.
- `debfff5` / `c7eb628e`: global barrier in `_build_index_mappings` ->
  oneCCL participation hang (data-dependent call count).
- `74b09fd`: reverts that barrier; adds safe global barriers in the
  sibling `blendable` / `_cache_indices` / `build_corpus_datasets` paths.
  Does NOT fix `_build_index_mappings` (relies on prewarm). 80B TP=4
  cold-cache STILL raced (jobs 8574063 32N, 8574172 4N).
- `041d015f`: atomic writes + poll in `_build_index_mappings`. THE fix.

### Legacy workaround (pre-`041d015f` venvs only)
Single-rank prewarm (`scripts/prewarm_blendcorpus_singlerank.sh`): build
the index with `--nproc 1` (no concurrent readers) at matching
`GBS`/`TRAINING_STEPS`/`SEQ_LEN`/`CKPT_DIR`. No longer needed with
`041d015f`, but kept for reproducing older runs.
