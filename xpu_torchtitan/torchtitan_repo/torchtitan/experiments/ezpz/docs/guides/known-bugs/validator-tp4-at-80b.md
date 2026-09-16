# The "validator CCL deadlock at 80B TP=4" was a phantom -- two unrelated bugs

**Status (2026-06-28): root-caused AND fixed -- CONFIRMED.** There is no
validator collective deadlock at 80B TP=4. The label conflated two *separate*,
non-collective failures plus one genuine latent crash in the validator; all
three are fixed. **Confirming run: job 12469784 (80B, TP=4, 4N, warm cache,
`--validator.freq=1`) ran `validate()` to completion at step 1 --
`log_validation validate step: 1 loss: 47.60` -- no mmap error, no
AttributeError, no CCL hang, clean exit.** (Loss is high because it is an
untrained step-1 80B at GBS=24 -- this was a plumbing confirmation, not a
numerics run.) Confirmed at dp_degree=12 (4N); the original reports were at
dp=186 (62N) -- same code paths exercised, but a 62N validator pass would be
belt-and-suspenders.

## What the docs claimed

Prior notes said the `EzpzValidator` has a "CCL collective deadlock at 80B
TP=4/62N (works at TP=1 debug, deadlocks at TP=4)", sidestepped via
`VALIDATOR_ENABLE=0`. Taken at face value this implies a tensor-parallel
collective in `validate()` hangs.

## What actually happened (evidence)

Of the four validator-enabled 80B TP=4 jobs (12469584/590/592/597), the
failure modes are **disjoint** -- no single job shows the validator causing a
CCL hang:

1. **Job 12469584 -- a CRASH, not a hang, and the only job that reached
   `validate()`.** Step 1 ran clean, then the first validation pass crashed in
   `np.load(..., mmap_mode="r")` with `ValueError: mmap length is greater than
   file size`. Cause: the validator's blendcorpus dataloader had **no**
   `--validator.dataloader.data-cache-path`, so it kept the bare default
   `.cache/blendcorpus` and **cold-built the validation-split index at full
   TP=4 scale on first validate()**, racing the half-written mmap. A dataloader
   index-path race, not a collective.

2. **Job 12469597 -- the only real oneCCL stall, and it is NOT the
   validator.** It hung in the **training** dataloader's
   `_build_index_mappings` `torch.distributed.barrier()` (the self-inflicted
   blendcorpus barrier from `c7eb628`, since reverted in `74b09fd`), in trainer
   `__init__` **before step 1**. `serve_validation=False`. Nothing to do with
   validation.

The clean 100-step TP=4 run (12469551) and the clean 46-step run (12469609)
both had the validator **off**. So the validator's compute/collective path at
TP=4 has never run to completion -- "it deadlocks on a collective" was an
**unverified hypothesis** with no matching log.

## The genuine validator bugs (now fixed)

### 1. `loss_fn` tuple-unpack crash (`validator.py`)

`BaseLoss.__call__` returns `(loss, metrics_dict)` (`components/loss.py:328`),
and upstream `validate.py` unpacks `loss_sum, _ = self.loss_fn(...)`. The ezpz
override did `loss_sum = self.loss_fn(...)`, so `loss_sum` was a tuple and
`loss_sum.detach()` raised `AttributeError` on the **first validation batch**
-- a rank-symmetric crash, independent of TP degree. (This is distinct from
`117ac69ce`, which fixed the *other* unpack -- `post_dataloading_process`.)
**Fix:** unpack the tuple (`loss_sum, _ = self.loss_fn(...)`).

### 2. validator cold-builds its index cache (config + scripts)

The validation split is a **separate index file** from the train split
(different `serve_validation`/`eval_iters` -> different cache hash). If the
validator's loader uses the default cache dir while the trainer uses a warm
prewarmed one, the validator cold-builds at scale -> the job 12469584 race.

**Fixes (layered):**
- `submit_agpt_80b_autoretry.sh` passes
  `--validator.dataloader.data-cache-path=$DATA_CACHE_PATH` (the operative
  production fix -- CLI, applied after the config builder).
- `prewarm_blendcorpus_cache.sh` enables the validator at `--validator.freq=1`
  so the **validation** index is prewarmed into that dir too (`391a08f1d`).
- `agpt/config_registry.py` `_base_config` defaults the validator's
  `data_cache_path` to the trainer's (defense-in-depth for non-script /
  interactive callers).

## What is NOT the cause (ruled out)

- **Ragged validation batch counts across dp ranks.** The blendcorpus
  validation sampler (`MegatronPretrainingSampler`) uses `drop_last=True` and
  sizes `eval_samples = global_batch_size * eval_iters` from rank-invariant
  config, so every dp rank yields an identical batch count. And
  `validator.steps=10` is a fixed scalar, so the per-batch
  `dist_sum(local_valid_tokens, batch_mesh)` collective fires a symmetric
  number of times. (If a real cross-dp-group collective deadlock ever does
  appear, this sampler evenness is the first thing to re-check.)
- **A TP-axis collective in validate().** The reduction meshes (`batch`,
  `loss`) are orthogonal to the TP axis; training at TP=4 exercises the same
  loss-parallel CE all-reduces every step and works fine for 100+ steps.

## How it was confirmed (2026-06-28, job 12469784)

The 4N prewarm job runs the validator at `--validator.freq=1` (that is how it
builds the validation index), so it doubled as the confirming run:

1. `prewarm_blendcorpus_cache.sh` (MODEL=80b NHOSTS_TRAIN=4 GAS=2
   OPTIMIZER=adamw) built both indices into
   `checkpoints/agpt-80b-adamw-books-n4-gbs24/.cache/books/index-cache`
   (train `serve_validation=False`, then validation `serve_validation=True`).
2. With the warm validation index, `validate()` ran at step 1:
   `log_validation validate step: 1 loss: 47.5966 tps: 433`, then the job
   exited clean (`Took 100.90 seconds. Exiting.`).
3. No `mmap length > file size`, no `AttributeError`, no CCL watchdog/hang.

This is the **first time `validate()` has completed at 80B TP=4.** (Job
12469785, a 4N training run with the validator on at step 1, is the second
independent confirmation -- validator firing inside a real train loop.)

Confirmed at dp_degree=12 (4N); the original reports were at dp=186 (62N).
The same code paths are exercised (TP=4 loss-parallel collectives, the
per-batch `dist_sum`, warm-vs-cold cache), so this closes the bug; a 62N
validator pass would be belt-and-suspenders for production scale.
`VALIDATOR_ENABLE=0` remains available as an escape hatch.
