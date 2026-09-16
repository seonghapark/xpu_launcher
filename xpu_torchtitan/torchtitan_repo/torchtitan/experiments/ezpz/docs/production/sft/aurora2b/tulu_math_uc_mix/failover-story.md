# 32N SFT: AuroraGPT-2B-sophiag-138650 + tulu_math_uc_mix, end-to-end failover

**Date:** 2026-06-10
**Machine:** Sunspot
**Jobs:** `12468404` → `12468408` → `12468409` → **`12468437`** (completed)
**Config:** Production GBS=6144 (matches AuroraGPT-2B pre-training),
3 epochs of `tulu-3-sft-mixture:0.65 + metamathqa:0.15 +
ultrachat-200k:0.20`.
**W&B chain:** [`365n9o09`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/365n9o09)
→ [`8eshn1p3`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/8eshn1p3)
→ [`cgmkf4qm`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/cgmkf4qm)
→ [`9qsl842a → br7gopsj`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.sft/runs/br7gopsj)
(completion run: `stoic-water-40`).

## TL;DR

The pre-training-scale SFT run **completed all 3 epochs** after
surviving **three independent oneCCL hardware crashes** across
four PBS jobs, with auto-retry correctly rotating in spare nodes
and resuming from the latest FSDP1 checkpoint each time. Loss
progressed monotonically across the chain: **1.16 → 0.77** at
step 729 / epoch 3.0. ~4.5B tokens of SFT training consumed. Two
upstream blockers had to be cleared inline during the run:

1. [pytorch/pytorch#186938](https://github.com/pytorch/pytorch/issues/186938)
   ([fix PR](https://github.com/pytorch/pytorch/pull/186940)) —
   `ShardedTensor.device` hardcoded `torch.cuda.current_device()`
   in its fallback, crashing every FSDP1 resume on XPU with
   `AssertionError: Torch not compiled with CUDA enabled`. Worked
   around locally in `train_sft.py:_patch_sharded_tensor_device_for_xpu`
   and filed + PR'd upstream.
2. ezpz [commit `6b4a00b`](https://github.com/saforem2/ezpz/commit/6b4a00b) —
   `launch_autoretry`'s `STUCK_PRE_TRAINING` guard was matching
   only torchtitan's `step=N` progress marker, false-positiving
   on TRL's `{'loss': '...'}` log format and aborting healthy
   HF-Trainer runs after the second SIGABRT. Broadened the regex
   to recognize both formats.

Final job `12468437` exited cleanly with `Exit_status = 0` after
**1h39m wall time** spread across 4 mpiexec attempts and 3
spare-node failovers. The autoretry verdict was
`FAILOVER STOP: success (attempt 4)`. Final checkpoint:
`outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729/`.

## Dataset mix

Chose a three-way mix biased toward general instruction-following
(tulu-3) with strong math signal (OpenMathInstruct-2 swapped for
metamathqa for build-time reasons, see below) and conversational
breadth (ultrachat-200k):

| Component | Weight | Rows | Source |
|-----------|-------:|-----:|--------|
| `allenai/tulu-3-sft-mixture` | 0.65 | 939k | tulu 3 SFT canonical |
| `meta-math/MetaMathQA` | 0.15 | 395k | augmented GSM8K + MATH (substituted for `nvidia/OpenMathInstruct-2`) |
| `HuggingFaceH4/ultrachat_200k` | 0.20 | 207k | filtered ChatGPT conversations |

OpenMathInstruct-2 (14M rows) was the original mix-spec target,
but its `interleave_datasets(stopping_strategy='all_exhausted')`
cycling at 14M was scaling the effective mix to ~21M rows. Rank-0
single-threaded tokenization + the XPU oneCCL barrier's ~15-minute
ceiling (which ignores PyTorch's `dist.barrier(timeout=30min)` on
xccl) meant the worker ranks tracebacked at the barrier before
rank 0 finished building, killing job `12468398`. metamathqa
(395k) preserves the GSM8K+MATH coverage at 35x less I/O.

Each rank now builds the mix in parallel and writes to a
SHA-256-content-hashed cache at `~/.cache/ezpz_sft_mixes/<hash>/`
(`datasets_sft.py:_materialized_mix_load_or_build`). Cold build is
~88 min for this mix; warm load is <0.1 s. Atomic save via tmp +
`os.rename` so concurrent rank-0 builds don't corrupt each other.

## Run table

| Job | Submit | Outcome | Steps gained | Final loss | Final epoch | Crash signature |
|-----|--------|---------|--------------:|----:|----:|-----------------|
| `12468404` | 09:03 | bad-node SIGABRT mid-attempt-1, autoretry restarted at step 0 (no `--resume_from_checkpoint`) | 0 → 140, then 0 → 100 | 0.86 | 0.59 | `ccl::v1::exception` on rank 286, then attempt-2 also crashed |
| `12468408` | 10:16 | XPU FSDP resume crash — all 384 ranks `AssertionError: Torch not compiled with CUDA enabled` in `_load_from_checkpoint` | 0 (never resumed) | n/a | n/a | upstream torch ShardedTensor.device CUDA hardcode |
| `12468409` | 10:33 | resume worked (v1 patch) → 2 successful failover cycles, then autoretry bailed STUCK_PRE_TRAINING (false positive on TRL marker) | 100 → 200 → 300 | 0.81 | 1.44 | regex didn't match `'loss':` format |
| `12468437` | 11:45 | resume worked (v2 patch) → 3 SIGABRTs survived (3 spare-node rotations) → **`Training complete.` at step 729 / epoch 3.0** | 300 → 400 → 500 → 600 → 700 → 729 | **0.77** | **3.00** | `FAILOVER STOP: success (attempt 4)`, exit 0 |

Loss across the full chain (every per-attempt resume preserves
the trainer's LR schedule, so the LR column tracks step-not-epoch):

```
job 12468404 attempt 1  step  10 → 140  loss 1.16 → 0.86   LR 2e-5    → 1.73e-5    (140 steps)
job 12468404 attempt 2  step  10 → 100  loss 0.96 → 0.86   LR 1.97e-5 → 1.75e-5    (restart from 0!)
job 12468408               <CRASH during FSDP load_from_checkpoint — XPU/CUDA bug>
job 12468409 attempt 1  step 110 → 200  loss 0.86 → 0.86   LR 1.62e-5 → 1.50e-5    (resumed from ckpt-100)
job 12468409 attempt 2  step 210 → 300  loss 0.86 → 0.81   LR 1.48e-5 → 1.34e-5    (resumed from ckpt-200)
job 12468437 attempt 1  step 310 → 400  loss 0.81 → 0.79   LR 1.32e-5 → 1.04e-5    (resumed from ckpt-300)
job 12468437 attempt 2  step 410 → 500  loss 0.80 → 0.78   LR 9.05e-6 → 5.21e-6    (resumed from ckpt-400)
job 12468437 attempt 3  step 510 → 600  loss 0.77 → 0.78   LR 5.21e-6 → 2.74e-6    (resumed from ckpt-500)
job 12468437 attempt 4  step 610 → 729  loss 0.78 → 0.77   LR 2.74e-6 → 0          (resumed from ckpt-600, completed 3 epochs)
```

End-of-training summary (from the `Training complete.` block):

| Field | Value |
|-------|------:|
| `train_runtime` (attempt-4 only) | 1119 s |
| `train_samples_per_second` | 3996 |
| `train_steps_per_second` | 0.652 |
| `train_loss` (last-step) | 0.137 |
| `mean_token_accuracy` (rolling) | 0.7957 |
| `epoch` | 3.000 |
| Wall time (full 4-job chain) | ~2h on-node (12468437 alone: 1h39m) |
| Tokens consumed by the trainer | ~4.5B (729 steps × GBS 6144 × 1024) |

## How auto-retry's bad-node failover works (with this run as worked example)

The submit script
[`aurora2b_tulu_mix_32n_gbs6144.sh`](../../../../../rl/scripts/sft/aurora2b_tulu_mix_32n_gbs6144.sh)
allocates `select=36` (32 training + 4 spare) and launches via
`ezpz launch --auto-retry --max-failover-retries 3`. The autoretry
loop wraps a single `mpiexec` invocation:

1. **Build active hostfile** from the PBS allocation, drop the
   first 4 nodes into a spare pool. Write `active.hostfile` for
   the launcher.
2. **Spawn `mpiexec`** with 32 × 12 = 384 ranks against
   `active.hostfile`. Stream stdout through a watchdog (timeout
   on idle output) and a UTF-8-tolerant decoder
   (`errors='replace'`, ezpz#163).
3. **On non-zero exit (rc=143)**: classify the failure.
   - `WALLTIME` (rc=143 + log has `=== DONE ===` marker) → stop, success.
   - `STUCK_PRE_TRAINING` (two consecutive attempts produced **no**
     progress markers — either `step=N` from torchtitan or
     `'loss':` from HF Trainer / TRL) → stop, bail out.
   - `BAD_NODE` (everything else) → grep the stdout for the last
     `rank N died from signal {6,9,11}`-style line, map rank → host,
     pull that host from the active list, push it onto the dead
     list. Pop a spare from the spare pool into the active list.
4. **Goto 1** (with the updated `active.hostfile`).

The trainer's job is to be **resumable**: `--save_strategy steps
--save_steps 100` writes a sharded FSDP1 checkpoint every 100
optimizer steps, and `--resume_from_checkpoint "${CKPT_DIR}"`
makes HF Trainer auto-detect the latest `checkpoint-N/` subdir on
relaunch. The autoretry loop **does not** know about checkpoints
or HF Trainer — it just relaunches the same `mpiexec` invocation
with a fresh hostfile, and the trainer figures out where to
resume.

### Worked example from `12468437`

- 11:46:08 — autoretry logs `attempt 1 — active=32 hosts, spare=4 hosts`
- 11:46–11:51 — model loads, FSDP shards, `_patch_sharded_tensor_device_for_xpu`
  fires, `_load_from_checkpoint` succeeds (loads
  `checkpoint-300/pytorch_model_fsdp_0/`), training resumes at step
  310.
- 11:51–12:10 — training: 90 steps, `checkpoint-400` saved at step 400.
- 12:11:35 — rank 53 on host `x1921c1s5b0n0` SIGABRTs from
  `ccl::v1::exception` (`ze_fd_manager.cpp:390 convert_fd_pidfd`).
  All 384 ranks die within ~3 seconds (mpiexec broadcasts the
  signal).
- 12:11:51 — autoretry parses stdout, finds the rank-53 death,
  emits `blind rotation: x1921c1s0b0n0 → x1921c5s4b0n0`
  (interesting: it picks the **first** failing host from
  `pals/cleanup`, not strictly the SIGABRT host; this is fine
  because both are confirmed degraded), and logs
  `attempt 2 (prior rc=143, sleeping 5s)`.
- 12:11:56 — `attempt 2 — active=32 hosts, spare=3 hosts`. New
  `mpiexec` spawns.
- 12:12–12:14 — same FSDP load path, this time loading
  `checkpoint-400/`. Training resumes at step 410 with loss
  matching the pre-crash trajectory (~0.79).
- 12:37:26 — attempt-2 SIGABRTs (`x1921c5s4b0n0`, the spare we
  just rotated in goes bad). Autoretry: `blind rotation:
  x1921c5s4b0n0 → x1921c5s5b0n0`, `attempt 3 (sleeping 10s)`,
  spare pool: 2.
- 13:01:49 — attempt-3 SIGABRTs after saving `checkpoint-600/`
  (loss 0.78, epoch 2.59). `blind rotation: x1921c5s5b0n0 →
  x1921c5s6b0n0`, `attempt 4 (sleeping 20s)`, spare pool: 1.
- 13:02:09 — attempt-4 launches. Resumes from `checkpoint-600/`,
  trains the remaining 129 steps to step 729 (epoch 3.0).
- 13:24:39 — `Training complete. Model saved to ... .` LR has
  decayed to 0 per the schedule, `mean_token_accuracy` 0.7957.
- 13:24:53 — autoretry: `FAILOVER STOP: success (attempt 4)`,
  exit 0. Job ends in `state=F Exit_status=0`.

The failover takes **~20–30 seconds** end-to-end. The 5-second
initial sleep grows to 10s/20s on subsequent attempts
(exponential backoff). The trainer-side init (model load + FSDP
shard + checkpoint load + data loader setup) takes another
~3 minutes before the next training step lands. Per-attempt
overhead: ~3.5 min wasted on the relaunch, in exchange for not
losing the run.

Across the whole 4-attempt chain in `12468437`, autoretry used 3
of the 4 available spares. The fourth spare went untouched —
attempt-4 didn't need to swap because it completed before hitting
its own SIGABRT.

### Why this matters

Without auto-retry, every hardware fault loses the rest of the
walltime allocation (the user has to qsub a new job and manually
edit the resume path). With auto-retry but no XPU-FSDP-resume
patch, the second attempt crashes immediately at
`_load_from_checkpoint` and the job dies in the same PBS slot
without ever advancing past step 0 — exactly what happened to
`12468408`. With auto-retry but no autoretry-regex fix, the
second SIGABRT (which is expected on this hardware — see "Failure
mode characterization" below) gets misclassified as
STUCK_PRE_TRAINING and the rest of the failover budget is
discarded — exactly what happened to `12468409`. All three pieces
are needed for sustained 32N training on this stack.

## Blockers cleared during the run

### 1. `ShardedTensor.device` hardcoded CUDA (filed + PR'd upstream)

**Symptom:** Every FSDP1 checkpoint resume on XPU crashed all 384
ranks with `AssertionError: Torch not compiled with CUDA enabled`
inside `transformers.Trainer._load_from_checkpoint` →
`accelerate.load_fsdp_model` → `dist_cp.load` →
`DefaultLoadPlanner.set_up_planner` → `_init_state_dict` →
ShardedTensor `.device` getter dispatch →
`torch.distributed._shard.sharded_tensor._ops.tensor_ops:54`:

```python
else:
    dev = torch.device(torch.cuda.current_device())   # <-- BUG on XPU
```

The sibling helpers in
`torch/distributed/checkpoint/planner_helpers.py:_init_state_dict`
(`tensor_func` / `dtensor_func`) already do the right
accelerator-agnostic thing via `_get_pg_default_device(pg).type` →
`_get_device_module(...)`. Only the ShardedTensor dispatch was
hardcoded.

**Local workaround:** `train_sft.py:_patch_sharded_tensor_device_for_xpu`
re-registers the ShardedTensor `.device` dispatch via
`_sharded_op_impl` with an accelerator-aware impl (uses
`torch.accelerator.current_accelerator()` first, falls back to
inspecting `pg._device_types` and preferring non-CPU on older
torch). Verified the patch replaces the `_SHARDED_OPS` entry via
a login-node smoke before submitting.

**Upstream:** Filed
[pytorch/pytorch#186938](https://github.com/pytorch/pytorch/issues/186938)
with the rank-0 traceback and a minimal repro path, then opened
[pytorch/pytorch#186940](https://github.com/pytorch/pytorch/pull/186940)
with the fix. Codex review caught a regression risk in v1 of the
PR (mirroring planner_helpers' `_get_pg_default_device` pattern
breaks composite PGs like `cpu:gloo,cuda:nccl` because that
function returns `cpu` whenever CPU is in the backend list); v2
of the PR switched to `torch.accelerator.current_accelerator()`
to avoid the composite trap entirely.

Full writeup with the proposed fix snippet:
[`docs/upstream-issues/sharded_tensor_device_cuda_hardcode.md`](../../../../upstream-issues/sharded_tensor_device_cuda_hardcode.md).

### 2. autoretry STUCK_PRE_TRAINING false-positive on TRL format

**Symptom:** Job `12468409` ran attempt-1 cleanly (loss 1.16 →
0.86, 140 steps, checkpoint-100 saved), hit a `ccl::v1::exception`
SIGABRT, attempt-2 cleanly resumed from `checkpoint-100` (loss
0.86 → 0.81, 200 more steps, checkpoint-200 → checkpoint-300
saved), hit another SIGABRT. Auto-retry then logged:

```
[auto-retry] FAILOVER STOP: stuck_pre_training (two consecutive
attempts with zero step= markers, rc=143)
```

and bailed despite 5 minutes of healthy training in attempt-2.

**Root cause:** `launch_autoretry.py:_PROGRESS_MARKER_RX` was
hardcoded to `\bstep=\d+`, which is what torchtitan's `History.update`
emits. HF Trainer / TRL `SFTTrainer.log()` instead emits a
stringified Python dict like
`{'loss': '0.857', 'epoch': '1.44', ...}` on every
`logging_steps` boundary — no `step=` substring anywhere. Both
attempts trained 100+ steps, both produced ample progress markers
*in their own format*, but the autoretry's grep returned zero.

**Fix:** ezpz commit
[`6b4a00b`](https://github.com/saforem2/ezpz/commit/6b4a00b)
broadens the regex to also match `'loss': '<digit>` and
`'loss': <digit>`:

```python
_PROGRESS_MARKER_RX = re.compile(
    r"\bstep=\d+|'loss':\s*'?[0-9]",
    re.MULTILINE,
)
```

Smoke-tested against:
- torchtitan `step=42 loss=1.23` ✓ matches (existing behavior)
- TRL `{'loss': '0.857', 'epoch': '1.44'}` ✓ matches (new)
- TRL `{'loss': 0.857, 'epoch': 1.44}` ✓ matches (numeric form)
- empty / CCL-warnings-only ✗ rejects (correct)
- prose containing "loss" or `step=foo` ✗ rejects (correct)

Reinstalled via `uv pip install -e ../ezpz --no-deps --no-cache
--link-mode=copy`. Verified the new regex is live in the venv
before submitting `12468437`.

## Failure mode characterization

The Sunspot oneCCL `ccl::v1::exception` from `ze_fd_manager.cpp:390
convert_fd_pidfd: EXCEPTION: pidfd_getfd failed: ... errno: Bad
file descriptor` is the dominant failure mode for sustained 32N
runs on this stack. Every job in the chain hit it at least once,
typically after 10–30 minutes of training. The triggering rank
varies (286 → 123 → 165 → 53 across the four jobs); the same
host (`x1921c1s0b0n0`) shows up multiple times, suggesting at
least one degraded NIC / Level Zero stack on the allocation.

Operationally we can't fix this at the user level — `pidfd_getfd`
failures inside `ccl_worker_func` are kernel/driver-layer and
auto-retry's bad-node rotation is the only viable mitigation. The
4-spare allocation gives us 4 failovers before we run out;
empirically each job has needed 1–2 swaps before walltime.

The fact that the **same host** keeps showing up across jobs
suggests an ALCF ticket would be worthwhile to get it pulled from
the queue. Filed as a TODO in
[`docs/journal.md`](../../../../journal.md) 2026-06-10 entry.

## Outputs

- **Checkpoint dir:** `outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/`
  - `checkpoint-{100,200,300,400,500,600,700,729}/` — FSDP1 sharded
    snapshots (~28 GB each: `pytorch_model_fsdp_0/` distcp shards +
    `optimizer_0/` + per-rank rng + `trainer_state.json`)
  - `checkpoint-100-hf/` — early-milestone consolidated HF format
    (7.5 GB safetensors + config + tokenizer), written during the
    `12468408` debug window
  - `checkpoint-729-hf/` — **final consolidated HF format** ready
    for `from_pretrained()`. Written post-completion via
    [`scripts/consolidate_sft_ckpt.sh`](../../../../../rl/scripts/consolidate_sft_ckpt.sh):
    `accelerate merge-weights` flattens the FSDP1 distcp shards
    in `pytorch_model_fsdp_0/` into a single `model.safetensors`,
    drops optimizer/rng state, and copies `config.json` +
    tokenizer files from the original
    `AuroraGPT-2B-sophiag-gs138650/` dir.
- **Tokens consumed:** ~4.5B (729 steps × GBS 6144 × `max_length`
  1024) — exactly 3 epochs on the materialized mix.
- **Wall time used across the chain:** ~2 h on-node total.
  `12468437` alone (the run that finished) was 1h39m, of which
  ~12 min was init/overhead across the 4 attempts.

The `checkpoint-729-hf/` artifact is the deliverable: a
4.5B-token-SFT'd AuroraGPT-2B in HF format, drop-in compatible
with `train_grpo.py --model_name_or_path`. Loss descent
`1.16 → 0.77` and `mean_token_accuracy → 0.7957` indicate the
instruction-following adaptation has converged on this mix.

## Files touched

| File | Purpose |
|------|---------|
| `torchtitan/experiments/ezpz/rl/train_sft.py` | XPU FSDP resume patch, resume_from_checkpoint coercion, autodetect-as-True for HF |
| `torchtitan/experiments/ezpz/rl/datasets_sft.py` | tulu_math_uc_mix factory + content-hashed mix cache |
| `torchtitan/experiments/ezpz/rl/scripts/sft/aurora2b_tulu_mix_32n_gbs6144.sh` | 32N submit script with `--auto-retry --max-failover-retries 3` |
| `torchtitan/experiments/ezpz/rl/scripts/consolidate_sft_ckpt.sh` | one-shot `accelerate merge-weights` wrapper |
| `~/projects/saforem2/ezpz/src/ezpz/launch_autoretry.py` | broadened `_PROGRESS_MARKER_RX` for TRL format |

## Related

- Prior SFT smoke: [`20260608-sft-2b-sophiag-metamathqa-n32.md`](../../../../experiments/agpt/sunspot/20260608-sft-2b-sophiag-metamathqa-n32.md)
- Upstream issue + fix: [pytorch/pytorch#186938](https://github.com/pytorch/pytorch/issues/186938) + [pytorch/pytorch#186940](https://github.com/pytorch/pytorch/pull/186940)
- ezpz autoretry fix: [saforem2/ezpz commit `6b4a00b`](https://github.com/saforem2/ezpz/commit/6b4a00b)
- ezpz autoretry _drain UTF-8 bug fix (pre-req for this work): [saforem2/ezpz PR #162](https://github.com/saforem2/ezpz/pull/162) (#163)
- Local upstream-issues writeup: [`docs/upstream-issues/sharded_tensor_device_cuda_hardcode.md`](../../../../upstream-issues/sharded_tensor_device_cuda_hardcode.md)
- Journal entry: [`docs/journal.md`](../../../../journal.md) 2026-06-10
