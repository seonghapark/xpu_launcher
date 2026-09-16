# blendcorpus index-cache race: atomic-rename fix landed (`041d015f`) + one follow-up TOCTOU bug

> **Target repo:** `saforem2/blendcorpus` (branch `feat/remove-deepspeed`).
> The atomic-rename fix `041d015f` is the right approach and eliminates the
> mass race. Hardware testing surfaced ONE small remaining bug in the poll
> guard (a TOCTOU on `os.path.getsize`) -- details + one-line fix below.

## Background: the original race (fixed)

`gpt_dataset.py::_build_index_mappings` built the per-corpus
`*_{doc,sample,shuffle}_idx.npy` on rank 0, writing each to its **final**
filename via `np.save`, then ALL ranks `np.load`'d them with no
synchronization. At TP>1, ~(1 - 1/TP) of ranks (TP-coord != 0, not gated by
any sibling subgroup barrier) raced ahead and mmap'd a file mid-write:
`EOFError` / `mmap length > file size` / `invalid load key '\x00'`. Verified
on 80B TP=4 cold cache: jobs 8574063 (32N) and 8574172 (4N) crashed with
~75% of ranks hitting it. A `dist.barrier()` could not fix it (the
data-dependent per-corpus/per-split call count -> oneCCL participation hang;
that attempt was `c7eb628e`, reverted in `74b09fd`).

## The fix that landed: `041d015f` (atomic writes + poll)

`saforem2/blendcorpus@041d015f` ("data: fix build-then-load index race at
TP>1 (atomic writes + poll-for-complete)"):
- **Writer (rank 0):** `_atomic_save` writes to
  `f"{final}.tmp.{pid}.{rank}"` then `os.replace(saved, final)` -- atomic on
  Lustre, so a reader never sees a torn file.
- **Readers (all ranks):** `_wait_for_index_files(paths, timeout=1800, poll=0.5)`
  polls each path until it exists, is non-empty, and `np.load(mmap)` validates
  the header. File-polling, NOT a collective -> immune to the participation
  hang the barrier had.

**Hardware result (job 8574237, 4N TP=4, COLD cache, 2026-06-28):** the mass
race is GONE -- **0 EOFError / mmap / invalid-load** across all ranks (vs ~75%
before). The atomic-write side works as intended.

## Remaining bug: TOCTOU on `os.path.getsize` in `_wait_for_index_files`

Job 8574237 still failed -- but with a *different*, much narrower error: **2
ranks** (rank1, rank2) raised an UNCAUGHT
`FileNotFoundError: ... _shuffle_idx.npy` at `gpt_dataset.py:66`. That line is
the poll guard:

```python
while True:
    ok = False
    if os.path.isfile(target) and os.path.getsize(target) > 0:   # <-- line 66
        try:
            np.load(target, allow_pickle=True, mmap_mode="r")
            ok = True
        except (ValueError, EOFError, OSError):
            ok = False
    if ok:
        break
    ...
```

**Root cause (TOCTOU):** `os.path.isfile(target)` returns True, but between
that check and `os.path.getsize(target)`, the builder's `os.replace(saved,
final)` swaps the inode for `target` (the builder writes per-corpus indices
sequentially; a reader polling corpus N can observe corpus N's file being
(re)published). `getsize` on the briefly-absent path raises
`FileNotFoundError` -- which is OUTSIDE the `try`, so it escapes uncaught and
kills the rank. The `np.load` inside the `try` is protected (it catches
`OSError`, and `FileNotFoundError` is an `OSError` subclass), but the
`isfile`/`getsize` GUARD is not.

This is rare (narrow window), which is why only 2 of 48 ranks hit it. The
cascade: those 2 ranks die -> launch tears down -> rank-0 (builder) gets
SIGTERM (signal 15) -> autoretry sees rc=143 and (mis)labels it "walltime".

### Fix -- LANDED `saforem2/blendcorpus@1f7e9c0` (2026-06-28)

Moved the existence/size probe inside the `try` (so every filesystem access on
`target` is covered by the same `except OSError`):

```python
while True:
    ok = False
    try:
        if os.path.getsize(target) > 0:          # raises FileNotFoundError if absent -> caught below
            np.load(target, allow_pickle=True, mmap_mode="r")
            ok = True
    except (ValueError, EOFError, OSError):       # FileNotFoundError is-a OSError
        ok = False
    if ok:
        break
    if time.time() - start > timeout:
        raise TimeoutError(f"index file not complete after {timeout:.0f}s: {target}")
    time.sleep(poll)
```

(`os.path.getsize` alone is enough -- it raises if the file is absent, which
the `except` now treats as "keep polling". Drop the separate `os.path.isfile`
to remove the two-syscall window entirely.)

### Validation once patched
Re-run the cold-cache 4N TP=4 80B smoke (the config in 8574237) -- it should
now build the index and proceed to training + checkpoint with zero
FileNotFoundError and zero EOFError. Then 32N to confirm at scale.

## Operational status
- `041d015f` = atomic writes + poll (kills the mass race).
- `1f7e9c0` = TOCTOU fix (getsize inside the try). **Both now on
  `feat/remove-deepspeed`.**
- Venv reinstalled to `1f7e9c0`, tarball rebuilt. Cold-cache 80B TP=4
  re-validation pending (re-run of the 8574237 config) -- mark this fully
  closed once that passes with 0 EOFError AND 0 FileNotFoundError through to
  training + checkpoint.
