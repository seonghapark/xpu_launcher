# Training `agpt_80b` on Aurora

End-to-end guide for training the 80B AuroraGPT dense model on Aurora.
Validated end-to-end on 2026-06-08 (4N smoke, step-10 sync DCP save
landed cleanly — 904 GB, 48 `.distcp` shards).

For the underlying venv setup (torch 2.13 + uv + `ezpz yeet`) see
[`running-with-newer-pytorch.md`](../running-with-newer-pytorch.md) —
this guide assumes you've done that and have a working `.venv/` +
`.venv.tar.gz`.

> [!IMPORTANT]
> Read these caveats first:
>
> - **Optimizer**: AdamW only. SophiaG/Muon overflow Hessian/Newton-Schulz
>   in bf16 at `dim=9216`.
> - **LR**: `1e-6`. `LR=1.1e-5` NaN'd at production GBS (1536–3072).
>   The 256N attempt 8530891 at `LR=1e-6` still NaN'd at step 2 — open
>   issue, see [Known issues](#known-issues) below.
> - **`compile=OFF`**. `compile=ON` triggers a
>   `DeviceMesh`-in-saved-tensors AOT autograd crash on torch 2.13 for
>   every 80B-family config (smallest reproducer: `agpt_50b_wide`,
>   ~48B params, 2N, ~30s to crash). Workaround until upstream fix:
>   keep compile off.
> - **TP=2**. Smaller TP exhausts memory at 80B.
> - **Sync ckpt mode**. Already the default (`checkpoint.async-mode=disabled`).
>   Don't enable async — it cascades to wrapper-unrecoverable failures
>   at scale.

## Working config (proven)

| Field             | Value |
|-------------------|-------|
| Model flavor      | `agpt_80b` |
| Optimizer         | AdamW |
| LR                | `1e-6` |
| Master dtype      | `float32` (default since 2026-04-30 v2 restart) |
| TP                | 2 |
| AC                | full |
| `compile`         | **OFF** |
| Seq len           | 8192 |
| Local batch size  | 1 |
| Grad accumulation | 1 |
| Global batch size | `12 * NGPUS * LBS * GAS / TP` |
| Checkpoint mode   | sync (default) |
| Checkpoint interval | 100 |
| Dataset           | blendcorpus / olmo-mix-1124 |
| Tokenizer         | google/gemma-7b |

Memory: ~88.94% peak at 4N (TP=2, ~7 GiB per-tile headroom). At 256N+
the per-tile memory is the same — total GBS scales with the FSDP
shard count, not the per-rank memory footprint.

MFU baseline (4N validation): ~17.8%, identical to Sunspot reference.

## Prerequisites

1. **Repo + venv set up** per
   [`running-with-newer-pytorch.md`](../running-with-newer-pytorch.md).
   You should have:
   - `~/torchtitan/` or `/flare/.../<your-clone>/` with `.venv/`
     activated
   - `.venv.tar.gz` built via `ezpz tar-env` (broadcast is sub-linear
     to 4096N — see the scaling table at the bottom of the venv guide)
   - `assets/hf/gemma-7b/tokenizer.json` (from
     `scripts/download_hf_assets.py`)

2. **Extra deps installed** beyond the venv-guide baseline:

   ```bash
   uvi --no-deps spmd_types
   ```

   `spmd_types` is imported by upstream `torchtitan/components/loss.py`
   since fec0c175d. Skipping it gives `ModuleNotFoundError` at
   model-init time.

3. **xccl `split_group` workaround installed**. Already lives in
   `torchtitan/experiments/ezpz/xccl_split_group_workaround.py` and is
   auto-installed at module import time. Verifies via this log line at
   startup:

   ```
   Successfully created meshes with active dimensions:
       ['batch', 'loss', 'tp', 'efsdp', 'fsdp']
   ```

   If you see `RuntimeError: No backend ... does not support
   splitting`, the workaround is not installed — see
   [`docs/upstream-issues/xccl_split_group_unsupported.md`](../../upstream-issues/xccl_split_group_unsupported.md).

## Interactive launch (4N smoke validation)

Use this to verify your stack end-to-end before submitting at scale.
This is the path that worked on 2026-06-08:

```bash
# Get an allocation
qsub -A AuroraGPT -q debug -l select=4 -l walltime=01:00:00 \
    -l filesystems=home:flare -I

# Once interactive shell lands on the head compute node:
cd /flare/AuroraGPT/<your-user>/<your-clone>/

# Load env (per running-with-newer-pytorch.md)
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="localhost,127.0.0.1,*.alcf.anl.gov,*.anl.gov"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job && ezpz_load_modules
source .venv/bin/activate

# Broadcast tarball to all 4 nodes (~70s for 8 nodes, ~91s at 64)
ezpz yeet-env --src .venv.tar.gz
deactivate
source /tmp/.venv/bin/activate

# Run 10 steps
CKPT_DIR=outputs/checkpoints/agpt-80b-adamw-olmo-mix-1124-n4-gbs24
DATA_CACHE_PATH="${CKPT_DIR}/.cache/olmo-mix-1124/index-cache"
DFL=torchtitan/experiments/ezpz/data-lists/aurora/olmo-mix-1124.txt

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config=agpt_80b \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval=10 \
    --checkpoint.keep-latest-k=0 \
    --checkpoint.no-last-save-model-only \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --optimizer=adamw \
    --optimizer.lr=1e-6 \
    --parallelism.tensor-parallel-degree=2 \
    --training.local-batch-size=1 \
    --training.global-batch-size=24 \
    --training.seq-len=8192 \
    --training.steps=10 \
    --compile.no-enable
```

Expected outcome (per [4N validation](../../production/agpt/80b/n4/README.md)):

- Step 1: loss 12.93, mem 53 GiB (82.9%), MFU ~13% (warm-up)
- Step 2: loss 12.92, mem 56.9 GiB (88.97%), MFU ~16.6%
- Step 3+: loss descending, MFU climbs to ~17.9%
- Step 10: sync DCP save lands cleanly — `outputs/checkpoints/.../step-10/`
  with `.metadata` + 48 `.distcp` shards (~904 GB total) at 4N TP=2

If anything diverges from the above by step 2, **stop and triage** —
don't burn 256N walltime chasing a stack issue.

## Submitting at scale via PBS

The production submit script is
[`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../../scripts/submit_agpt_80b_aurora_venv_failover.sh).
It wraps the interactive launcher with the bad-node failover wrapper
(`failover_lib.sh`): if a node dies mid-init or mid-step, the wrapper
swaps in a spare from the over-allocated set and retries.

```bash
# 256N production, 12h, with 10-node spare pool (266 total) + 2 retries
qsub -A AuroraGPT -q small -l select=266 -l walltime=12:00:00 \
    -l filesystems=home:flare \
    -N agpt-80b-n256 \
    -v NHOSTS_TRAIN=256,FAILOVER_MAX_RETRIES=2 \
    -- scripts/submit_agpt_80b_aurora_venv_failover.sh
```

### Spare-pool sizing

| `NHOSTS_TRAIN` | Recommended `select=` | Reasoning |
|---------------:|---------------------:|-----------|
|              4 | 4 (no spares)        | smoke; failover not worth the queue padding |
|             64 | 68 (+4)              | low bad-node hit-rate at this scale |
|            128 | 134 (+6)             | start padding more |
|            256 | 266 (+10) — fragile  | hit-rate climbs; consider 296 (+40) if cascading retries |
|            512 | 522 (+10) — fragile  | as 256N |
|           1024 | 1038 (+14)           | first-attempt init crashes (12,288-rank `set_determinism`) — bracket with 768/896 first |

The 256N+ spare-pool sizing is an open question. 2026-05-24's 8505222
exhausted 5 retries before giving up — every attempt hit SIGSEGV on a
different bad node, 3 of them from the x4101c5/c6 rack. If you see
that pattern recurring, file an ALCF support ticket on the rack and
bump the spare count.

### Chain continuations

Production training is long: 80B target is 4.67T tokens at GBS=1536,
~3,041,667 steps (years of walltime). Always queue one continuation
behind every head job:

```bash
qsub -W depend=afterany:<head-jobid> -A AuroraGPT -q small \
    -l select=266 -l walltime=12:00:00 \
    -l filesystems=home:flare \
    -N agpt-80b-n256-cont1 \
    -v NHOSTS_TRAIN=256,FAILOVER_MAX_RETRIES=2 \
    -- scripts/submit_agpt_80b_aurora_venv_failover.sh
```

`afterany` triggers regardless of head exit status — important
because PBS reports walltime exit as failure, but walltime-exit is
the *expected* end-of-shift behavior for long runs.

See [`feedback_always_have_chain_continuation`] (memory) and
[`docs/production/agpt/80b/README.md`](../../production/agpt/80b/README.md)
for the current chain status.

## Verification at each scale

Before treating any new scale as "production-ready", do this in order:

1. **4N smoke**: 10 steps, sync ckpt save, in interactive shell.
   This is the canonical sanity check. The [n4 README](../../production/agpt/80b/n4/README.md)
   is the reference.
2. **8N smoke** in `debug-scaling` queue (`-q debug-scaling -l select=8
   -l walltime=01:00:00`): same config, 20 steps.
3. **64N validation in `debug-scaling`**: same config, 50 steps. If
   step-50 ckpt save (~900 GB) lands cleanly, the path is good.
4. **256N production**: queue the full 12h with failover.

Step 3 is the critical gate — anything that's going to break at 256N
will almost always also break at 64N (init memory, blendcorpus init,
xccl process-group setup), and a 1h debug-scaling slot is cheaper than
a 12h production slot.

## Production checkpointing

The 80B production ckpt is **904 GB at 4N (48 shards)** and scales
linearly with FSDP shard count. At 256N (TP=2 → 1,536 FSDP shards),
expect ~29 TB per ckpt and proportionally-longer sync save times. Keep
this in mind for:

- **Disk quota**. `CKPT_KEEP_LATEST_K=0` (default, **mandatory** —
  setting it >0 destroyed 334 2B chain ckpts on 2026-05-25; see
  `feedback_never_use_keep_latest_k` memory).
- **Save interval**. Sync save dominates step time at production
  scale. Default `--checkpoint.interval=100` is OK at 256N (save
  ~every 30 min); reduce to 200 if save is taking >10% of wall.

## Known issues

### 256N NaN at step 2 (open, 2026-06-09)

[8530891](../../production/agpt/80b/n4/README.md) (256N, LR=1e-6,
GBS=1536, same config as the validated 4N smoke) trained step 1 cleanly
(loss 12.94) but **NaN'd at step 2** and didn't recover. Even with a
scheduler-clamped LR ≈ 1.8e-8 the NaN persists. Open hypotheses:

- bf16 overflow somewhere in the activation path that doesn't trigger
  at 4N because GBS is 64× smaller
- TP=2 loss-reduction bug (`_dist_reduce` short-circuits DTensor inputs;
  full repro in [`loss-reporting-tp-dist-reduce.md`](../loss-reporting-tp-dist-reduce.md))
- fp32 second-moment overflow under accumulated grad-norm spikes

LR=1e-7 retries (8531345, 8531721) failed for unrelated reasons
(`std::bad_alloc` at model construction on contiguous-block rank
failures). Diagnostic plan:

1. Try TP=4 to halve effective per-replica GBS (currently TP=2, GBS=1536)
2. Try with `--validator.no-enable` to rule out validator-loss-path issues
3. Dump per-tensor stats at step 1 to localize which weight first goes NaN
4. Bisect bf16-vs-fp32 master at 256N (4N is fp32-master, working)

### Other known issues

- **`compile=ON` AOT autograd crash** on torch 2.13 for the entire
  80B family. See
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../../upstream-issues/repro_devicemesh_in_saved_tensors.py).
  Workaround: `--compile.no-enable`.
- **`compile` on torch 2.10** hangs at step 1 for 80B TP=2 since
  upstream changes April 16-23. The torch-2.13 stack works
  (`compile=OFF`); don't fall back to torch 2.10.
- **1024N init OOM/SIGSEGV** at 12,288 ranks in `set_determinism`
  `torch.distributed.broadcast`. 256N/512N unaffected. Bracket
  768N/896N before trying 1024N. See
  [`memory/project_1024n_init_crash.md`](.).
- **Aurora pals-RPC launcher infra failures** (exit 127 with
  `Couldn't forward RPC launch`) are transient infra issues, not bad
  nodes; the failover wrapper can't recover from them (the wrong-node
  swap doesn't help). Just resubmit.
- **`set_determinism std::bad_alloc`** at 6,144+ ranks during init is
  a documented intermittent. Doesn't repro on retry; just bounce the
  job.
- **`signal 9` Aurora NODE_FAIL** on long-walltime jobs (4 jobs killed
  across 4 different nodes in May 2026). Cause unclear; resume from
  most recent ckpt is the operational workaround.

## References

- [`running-with-newer-pytorch.md`](../running-with-newer-pytorch.md) —
  underlying torch 2.13 venv setup
- [`training-dtype-bf16-norm-freeze.md`](../training-dtype-bf16-norm-freeze.md) —
  why v2 is fp32-master, not bf16-master
- [`loss-reporting-tp-dist-reduce.md`](../loss-reporting-tp-dist-reduce.md) —
  TP > 1 loss reporting bug + `EzpzValidator` workaround
- [`bad-node-failover.md`](../bad-node-failover.md) — failover wrapper
  details
- [`docs/production/agpt/80b/README.md`](../../production/agpt/80b/README.md) —
  live status, dispatch log, eval (when production starts persisting)
- [`docs/production/agpt/80b/n4/README.md`](../../production/agpt/80b/n4/README.md) —
  the validated 4N reference run
- [`docs/experiments/agpt/sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md`](../../experiments/agpt/sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md) —
  Sunspot 4N validation with xccl_split_group workaround
- [`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../../scripts/submit_agpt_80b_aurora_venv_failover.sh) —
  PBS submit script
- [`scripts/train_agpt_80b_venv.sh`](../../../scripts/train_agpt_80b_venv.sh) —
  interactive launcher (called from a compute node)
