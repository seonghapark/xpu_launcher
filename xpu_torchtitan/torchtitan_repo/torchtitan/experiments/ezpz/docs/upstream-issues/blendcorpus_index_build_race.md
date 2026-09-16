# blendcorpus per-corpus index build/load race at small dataset sizes

## TL;DR

`deps/blendcorpus/blendcorpus/data/gpt_dataset.py:_build_index_mappings`
(per-corpus path, lines ~1050-1135) does **rank-0-write then
all-ranks-load via mmap with no `torch.distributed.barrier()` in
between**. At small dataset sizes — where rank 0 finishes writing
the indices in milliseconds — the next distributed op's implicit
barrier isn't tight enough to close the race, and a sibling rank
hits:

```
EOFError: No data left in file
  shuffle_idx = np.load(idx_path["shuffle"], allow_pickle=True, mmap_mode="r")
```

The fix is one line: `torch.distributed.barrier()` between the
rank-0-write block and the all-ranks-load block, matching the
pattern the **blendable dataset** path uses one function up
(lines 215-265, three barriers around the build/load handoff).

## Symptom

Reproduced on Sunspot 4N TP=2 (jobs 12468195 + 12468196,
2026-06-07) with the `books` data-list (3 shards, 11 GB, 4826
total samples for a 200-step run at `GBS=24`):

- **12468195**: rank 0 built per-corpus shuffle/doc/sample idx in
  **8 ms**. Rank 2 tried to load `shuffle_idx.npy` mid-write, hit
  `EOFError: No data left in file`. Whole job killed at 2:34.
- **12468196**: per-corpus indices now cached from 12468195's
  rank-0 write — but the **next** index layer (the *blendable
  dataset* index built by combining the per-corpus indices)
  hadn't been written yet. Rank 0 built it in **11 ms**. Same
  race, next layer. Whole job killed at 2:34 again.
- **12468197**: with both index layers now durably on disk, every
  rank hit the cache at trainer init, skipped both build paths,
  and the run proceeded cleanly to step 164 + a 904 GB checkpoint
  save (the load-bearing milestone in the parent session).

The bug only fires at small dataset sizes because the
implicit-barrier window for the rank-0 write is the wall-clock
duration of the *next* dist op chain — typically tens of
milliseconds. Once the index build itself takes longer than that
window (canonical olmo-mix-1124 indices take seconds), the race
closes naturally and the bug is invisible.

## Why the per-corpus path is racy but the blendable path isn't

Compare the two index-build sites in
`deps/blendcorpus/blendcorpus/data/gpt_dataset.py`:

**Blendable dataset path** (lines 215-265, NOT racy):

```python
if torch.distributed.get_rank() == 0 and not cache_hit:
    dataset_index, dataset_sample_index = _build_indices()
    ...
    np.save(index_path, dataset_index, allow_pickle=True)
    np.save(sample_index_path, dataset_sample_index, allow_pickle=True)
    ...

torch.distributed.barrier(group=mpu.get_data_parallel_group())       # <-- this
torch.distributed.barrier(group=mpu.get_pipeline_model_parallel_group())
torch.distributed.barrier(group=mpu.get_data_parallel_group())

self.dataset_index = np.load(index_path, ...)
```

Three explicit barriers between the rank-0 write block and the
all-ranks `np.load` — race window closed.

**Per-corpus path** (lines ~1050-1135, RACY):

```python
try:
    os.makedirs(data_cache_dir, exist_ok=True)
    with open(idx_path["desc"], "wt") as fd:
        fd.write(desc)
    doc_idx = _build_doc_idx(...)
    np.save(idx_path["doc"], doc_idx, ...)
    sample_idx = helpers.build_sample_idx(..., torch.distributed.get_rank() == 0, ...)
    np.save(idx_path["sample"], sample_idx, ...)
    shuffle_idx = _build_shuffle_idx(...)
    np.save(idx_path["shuffle"], shuffle_idx, ...)
except OSError:
    ...

# <-- no barrier here, just falls through
doc_idx = np.load(idx_path["doc"], allow_pickle=True, mmap_mode="r")
sample_idx = np.load(idx_path["sample"], allow_pickle=True, mmap_mode="r")
shuffle_idx = np.load(idx_path["shuffle"], allow_pickle=True, mmap_mode="r")
```

No barrier between the rank-0-only writes and the all-ranks
mmap-loads.

## Operational workaround

If you hit the EOFError on first launch with a small dataset:
just **resubmit the job**. The first attempt died but left the
indices durably on disk; the second attempt hits the cache at
both layers and skips the build paths entirely. This is what
unblocked the 80B path in 12468197 — three submissions for what
should have been one. Not a permanent fix, but cheap.

If you can pre-build the indices on a single rank ahead of time
(via a one-node smoke run that targets the same `CKPT_DIR`), do
that — once the cache is populated nothing in the production
launch will hit the racy path.

## Permanent fix

Add a barrier between the `_build_index_mappings` write block and
the all-ranks load block in
`deps/blendcorpus/blendcorpus/data/gpt_dataset.py:_build_index_mappings`,
matching the blendable-dataset pattern. Minimal diff:

```python
    except OSError:
        print(...)
        data_cache_success = False

+   if torch.distributed.is_initialized():
+       torch.distributed.barrier()
+
    # Load mappings.
    start_time = time.time()
    logger.debug(f" > loading doc-idx mapping from {idx_path['doc']}")
    doc_idx = np.load(idx_path["doc"], allow_pickle=True, mmap_mode="r")
```

A `torch.distributed.is_initialized()` guard keeps the
single-process / non-distributed code path working. A
`group=mpu.get_data_parallel_group()` scope would also work and
matches the blendable path more exactly, but plain global barrier
is sufficient and avoids requiring `mpu` to be set up at the
point this function runs.

## Filing status

Not yet filed upstream. The fix is small and the workaround is
cheap (resubmit until cached), so this is low-priority — but
worth a one-line PR to `deps/blendcorpus` when convenient. Open
question: which is the canonical upstream? `deps/blendcorpus/` is
vendored here; if there's a parent repo we should patch that
instead of our copy.

## Related

- [`checkpoint_async_gloo_on_xpu.md`](checkpoint_async_gloo_on_xpu.md)
  — sibling "upstream code path silently assumes barriers /
  bundled backends that XPU doesn't provide" pattern from the
  same session.
