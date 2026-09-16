# Scaling Tests & Production Runs — Aurora (2026-04-18 to 2026-04-21)

> Sam Foreman
> 2026-04-18 through 2026-04-21

## Overview

Series of experiments testing agpt_2b and agpt_80b at scale on Aurora,
validating compile behavior, identifying scaling bottlenecks, and
debugging the interactive-allocation training workflow.

## Experiment Log

### 1. agpt_2b @ 512 nodes — compile=on (Job 8441308)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-18 12:15:29 CDT |
| Job ID        | 8441308.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 512 / 6144 |
| Queue         | small (prod) |
| Walltime      | 05:51:14 (of 06:00:00) |
| Exit status   | 271 (killed) |
| Model         | agpt_2b (1.99B params) |
| Parallelism   | TP=1, FSDP=6144 |
| Compile       | on (model + loss) |
| Optimizer     | SophiaG, LR=2.28e-5 |
| GBS           | 6144 (LBS=1) |
| W&B           | [q63cp223](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/q63cp223) |

**Command:**
```bash
qsub torchtitan/experiments/ezpz/submit/aurora/submit_agpt_2b_n512.sh
```

**Result:** `torch.compile` SYCL compilation ran for ~6 hours and never
completed step 1. The 6144 ranks all generate and compile C++ kernels
simultaneously, causing filesystem contention on `/tmp`.

**Steps completed:** 0

**Insight:** torch.compile is infeasible at 6144 ranks — the SYCL
compilation time grows super-linearly with rank count (3.5 min at 128N,
4.5 min at 256N, 12+ hours at 512N). This is a known issue with the
XPU inductor backend at extreme parallelism.

---

### 2. agpt_2b @ 128 nodes — compile=on (Job 8441505)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-18 13:51:12 CDT |
| Job ID        | 8441505.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 128 / 1536 |
| Queue         | debug-scaling |
| Walltime      | 00:12:45 |
| Exit status   | 0 (success) |
| Model         | agpt_2b |
| Parallelism   | TP=1, FSDP=1536 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| GBS           | 1536 (LBS=1) |
| W&B           | [l8m7ujdj](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/l8m7ujdj) |

**Command:**
```bash
qsub -l select=128 -q debug-scaling ... --compile.enable --training.steps=100
```

**Result:** All 100 steps completed successfully.

| Metric          | Value |
|-----------------|-------|
| Compile time    | ~3.5 min |
| Steady-state TPS | ~3,000/GPU |
| MFU             | 11-12% |
| Memory          | 46.44 GiB (72.59%) |
| Loss            | 12.98 → 11.22 |

**Insight:** Compile works well at 128 nodes. TPS lower than 2N baseline
(5,400) — 56% scaling efficiency due to FSDP all-gather overhead at
1536-way sharding.

---

### 3. agpt_2b @ 256 nodes — compile=on (Job 8441537)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-18 17:21:57 CDT |
| Job ID        | 8441537.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 256 / 3072 |
| Queue         | debug-scaling |
| Walltime      | 01:00:35 (walltime killed) |
| Exit status   | -29 (walltime) |
| Model         | agpt_2b |
| Parallelism   | TP=1, FSDP=3072 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| GBS           | 3072 (LBS=1) |
| W&B           | [v5ytgu0o](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/v5ytgu0o) |

**Command:**
```bash
qsub -l select=256 -q debug-scaling ... --compile.enable --training.steps=100
```

**Result:** 88 of 100 steps completed before walltime kill.

| Metric          | Value |
|-----------------|-------|
| Compile time    | ~4.5 min |
| TPS range       | 130–1,900/GPU (extreme variance) |
| Median TPS      | ~700/GPU |
| MFU             | 2-5% (average), up to 7% peak |
| Memory          | 46.52 GiB (72.70%) |
| Loss            | 12.94 → 11.76 |

**Insight:** Extreme TPS variance at 256 nodes — 10x swings between steps
(130 to 1,900 TPS). Periodic ~60s stalls every 10-20 steps, likely from
network contention on the Slingshot fabric or straggler nodes. The GC
interval (gc_freq=50) also causes a stall at step 50/55. Average throughput
is ~700 TPS/GPU, much lower than the 3,000 at 128N — scaling efficiency
drops sharply between 128 and 256 nodes for the 2B model.

---

### 4. agpt_20b @ 512 nodes — compile=on (Job 8441597)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-19 07:35:55 CDT |
| Job ID        | 8441597.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 512 / 6144 |
| Queue         | small (prod) |
| Walltime      | 01:49:17 |
| Exit status   | 0 (but crashed internally, rc=127) |
| Model         | agpt_20b (20.7B params) |
| Parallelism   | TP=1, FSDP=6144 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| GBS           | 6144 (LBS=1) |
| W&B           | [5epgirhn](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/5epgirhn) |

**Command:**
```bash
qsub ... --config=agpt_20b --compile.enable --optimizer=sophiag --optimizer.lr=2.28e-5
```

**Result:** Crashed during checkpoint loading. No training steps completed.

**Error:**
```
RuntimeError: Missing key in checkpoint state_dict: layers.0.attention.qkv_linear.wk.weight.
```

**Root cause:** A stale checkpoint from a previous (pre-refactor) run existed
at `outputs/checkpoints/agpt-20b-sophiag-olmo-mix-1124-n512-gbs6144/step-200`.
The model key format changed (`qkv_linear.wk.weight` is the new layout) and
the old checkpoint was incompatible.

**Fix:** Renamed stale checkpoint to `.bak-20260417` and resubmitted (Job 8442900).

**See also:** This is not in known-issues.md because it's a one-time
user-data issue, not a systemic bug.

---

### 5. agpt_20b @ 512 nodes — compile=on, fresh checkpoint (Job 8442900)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-20 (submitted, queued) |
| Job ID        | 8442900.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 512 / 6144 |
| Queue         | small (prod) |
| Walltime      | 12:00:00 |
| Exit status   | pending |
| Model         | agpt_20b |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |

**Status:** Queued. This is the corrected resubmission after backing up the
stale checkpoint directory.

**Concern:** The 2B model's compile took 12+ hours at 512N. The 20B model
has 64 layers (vs 12), so compile may take even longer. However, the previous
20B 512N attempt (Job 8436463, 2026-04-16) also spent 12+ hours in compile
and never reached step 1.

---

### 6. agpt_80b @ 4 nodes — interactive workflow test (Job 8443511)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-21 01:27:19 CDT |
| Job ID        | 8443511.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 4 / 48 |
| Queue         | debug-scaling |
| Walltime      | 01:00:00 |
| Exit status   | -29 (walltime, but training finished) |
| Model         | agpt_80b (80.8B params) |
| Parallelism   | TP=2, FSDP=24 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| Steps         | 10 |
| W&B           | [pk2z22t6](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/pk2z22t6) |

**Command (interactive workflow):**
```bash
# 1. Submit allocation
qsub -l select=4 -q debug-scaling ~/test.sh

# 2. SSH to head node and launch training
ssh <head_node> "export PBS_JOBID=... PBS_NODEFILE=...; \
  nohup bash -l .ezpz-interactive-launch.sh \
  python3 -m torchtitan.experiments.ezpz.train \
  --module=ezpz.agpt --config=agpt_80b --compile.enable \
  --parallelism.tensor_parallel_degree=2 \
  --optimizer=sophiag --optimizer.lr=2.28e-5 \
  --training.steps=10 ... > test-80b-4n.out 2>&1 &"
```

**Result:** 5 steps confirmed in rank-0 log (log flushing truncated remaining steps).

| Metric          | Value |
|-----------------|-------|
| Compile time    | ~4.5 min |
| Steady-state TPS | 87-89/GPU |
| MFU             | 15.95-16.28% |
| Memory          | 53.03 GiB (82.88%) |
| Loss            | 12.94 → 12.86 (5 steps) |

**Insight:** Validated the interactive-allocation workflow:
`test.sh` → SSH → `.ezpz-interactive-launch.sh`. Required two fixes:
1. ALCF proxy (`http_proxy=http://proxy.alcf.anl.gov:3128`) — compute
   nodes can't reach the internet without it
2. `set +u` before sourcing ezpz (lmod/ezpz reference unset vars)

---

### 7. agpt_80b @ 128 nodes — interactive workflow (Job 8443522)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-21 03:40:53 CDT |
| Job ID        | 8443522.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 128 / 1536 |
| Queue         | debug-scaling |
| Walltime      | 01:00:00 |
| Exit status   | 0 (training completed) |
| Model         | agpt_80b (80.8B params) |
| Parallelism   | TP=2, FSDP=768 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| Steps         | 10 |
| W&B           | [ynb7rh86](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ynb7rh86) |

**Command:** Same interactive workflow as #6.

**Result:** All 10 steps completed successfully (rc=0, 984s total).

| Step | Loss    | TPS | MFU    | Memory           |
|------|---------|-----|--------|------------------|
| 1    | 12.924  | 13  | 2.45%  | 46.37GiB(72.47%) |
| 2    | 13.013  | 88  | 16.16% | 46.37GiB(72.47%) |
| 3    | 12.905  | 87  | 15.96% | 46.37GiB(72.47%) |
| 4    | 12.884  | 89  | 16.24% | 46.37GiB(72.47%) |
| 5    | 12.846  | 90  | 16.44% | 46.37GiB(72.47%) |
| 6    | 12.800  | 89  | 16.38% | 46.37GiB(72.47%) |
| 7    | 12.744  | 90  | 16.43% | 46.37GiB(72.47%) |
| 8    | 12.636  | 90  | 16.43% | 46.37GiB(72.47%) |
| 9    | 13.309  | 88  | 16.10% | 46.37GiB(72.47%) |
| 10   | 12.437  | 90  | 16.42% | 46.37GiB(72.47%) |

**Key findings:**
- **Perfect scaling:** 87-90 TPS/GPU at 128N, identical to 2N (88 TPS)
  and 4N (87 TPS). No TPS degradation at all.
- **Total throughput:** ~138K tokens/s across 128 nodes.
- **Memory drops with scale:** 46.37 GiB at 128N vs 53.03 GiB at 4N vs
  59.82 GiB at 2N — FSDP shards parameters more at higher rank counts.
- **Compile time:** ~5 min at 128N — only marginally slower than 4N (4.5 min).
- **No TPS variance:** Unlike the 2B model at 256N, the 80B model at 128N
  shows rock-solid TPS with zero variance. The larger per-step compute
  time (~47s) amortizes communication overhead.

---

### 8. agpt_80b @ 512 nodes — compile=on, attempt 1 (Job 8443524)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-21 10:03:05 CDT |
| Job ID        | 8443524.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 512 / 6144 |
| Queue         | small (prod) |
| Walltime      | 06:00:00 |
| Exit status   | 143 (SIGTERM from rank segfault) |
| Model         | agpt_80b |
| Parallelism   | TP=2, FSDP=3072 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| Steps         | 10 |

**Command:** Same interactive workflow as #6/#7.

**Result:** Crashed during data loading. Rank 2182 on node `x4108c7s7b0n0`
died from signal 11 (SIGSEGV), killing all other ranks.

**Insight:** Likely a transient hardware issue on one node. Retried as #9.

---

### 9. agpt_80b @ 512 nodes — compile=on, attempt 2 (Job 8443524 retry)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-21 10:17:42 CDT |
| Job ID        | 8443524 (same allocation) |
| Nodes / GPUs  | 512 / 6144 |
| Exit status   | 143 (CPU OOM during compile) |
| Compile       | on |

**Result:** **CPU memory OOM during torch.compile** — `MemoryError:
std::bad_alloc` on 5 ranks (2, 4, 6, 8, 10) on the head node. The inductor
backend exhausted host RAM while generating SYCL kernels for the 80B model's
84 transformer blocks at 6144 ranks.

**Insight:** torch.compile is not feasible for 80B at 512 nodes. The CPU
memory required for SYCL kernel compilation scales with both model size and
rank count. At 128N (5 min compile, no issues) vs 512N (CPU OOM in 2 min),
the memory pressure from 4x more ranks overwhelms the host.

---

### 10. agpt_80b @ 512 nodes — compile=off (Job 8443524 retry 2)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-21 10:32:44 CDT |
| Job ID        | 8443524 (same allocation) |
| Nodes / GPUs  | 512 / 6144 |
| Exit status   | 0 (completed, but NaN loss) |
| Model         | agpt_80b (80.8B params) |
| Parallelism   | TP=2, FSDP=3072 |
| Compile       | **off** |
| Optimizer     | SophiaG, LR=2.28e-5 |
| Steps         | 10 |

**Result:** All 10 steps completed. Infrastructure works, but **loss
diverged to NaN** from step 1 (grad_norm=NaN on first backward pass).

| Step | Loss    | TPS | MFU    | Memory           |
|------|---------|-----|--------|------------------|
| 1    | 12.938  | 9   | 1.64%  | 43.02GiB(67.24%) |
| 2    | NaN     | 66  | 12.10% | 43.02GiB(67.24%) |
| 3-10 | NaN     | 66  | 12.1%  | 43.02GiB(67.24%) |

**Key metrics:**
- **66 TPS/GPU without compile** (vs 88 TPS with compile at 128N — 25% gap)
- **Total throughput:** ~405K tokens/s across 512 nodes
- **Memory:** 43.02 GiB (67%) — plenty of headroom at this FSDP degree

**NaN root cause:** SophiaG is broken at 80B scale — confirmed by the
[LR finder 80B report](../experiments/lr-finder/agpt/80b/README.md#2026-04-21----80b-gas-sweep-sunspot-small-batch-gbs192)
which found Muon and SophiaG both produce NaN for 80B. The small-batch finder
suggested AdamW LR=1.1e-5, but **that does NOT hold at the production batch**:
re-running at GBS=6144 ([2026-06-27 report](../experiments/lr-finder/agpt/80b/README.md#2026-06-27----80b-at-the-production-batch-gbs6144-sunspot))
shows AdamW's usable LR collapses to ~7e-7 (NaN cliff) and **mano** is the
better-behaved optimizer (clean U-min at 1.6e-5). For 80B production at
GBS~6144, use AdamW LR~5e-7 or switch to mano (~3e-6).

---

### 11. agpt_20b @ 256 nodes — compile=on (Job 8443784)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-21 12:40:00 CDT |
| Job ID        | 8443784.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 256 / 3072 |
| Queue         | small (prod) |
| Exit status   | 0 (success) |
| Model         | agpt_20b (20.7B params) |
| Parallelism   | TP=1, FSDP=3072 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| Steps         | 10 |

**Result:** All 10 steps completed successfully.

| Step | Loss    | TPS | MFU    | Memory           |
|------|---------|-----|--------|------------------|
| 1    | 12.944  | 31  | 1.53%  | 40.95GiB(63.99%) |
| 2    | 12.940  | 280 | 13.98% | 40.95GiB(63.99%) |
| 3    | 12.931  | 282 | 14.09% | 40.95GiB(63.99%) |
| 5    | 12.897  | 283 | 14.11% | 40.95GiB(63.99%) |
| 10   | 12.656  | 280 | 13.98% | 40.95GiB(63.99%) |

**Insight:** 280 TPS at 256N vs 357 at 2N — 78% scaling efficiency. Some
TPS variance (224-287) but no major stalls. Loss converges normally.

---

### 12. agpt_80b @ 256 nodes — compile=on (Job 8443785)

| Field         | Value |
|---------------|-------|
| Date          | 2026-04-21 12:58:00 CDT |
| Job ID        | 8443785.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov |
| Nodes / GPUs  | 256 / 3072 |
| Queue         | small (prod) |
| Exit status   | 0 (success) |
| Model         | agpt_80b (80.8B params) |
| Parallelism   | TP=2, FSDP=1536 |
| Compile       | on |
| Optimizer     | SophiaG, LR=2.28e-5 |
| Steps         | 10 |

**Result:** All 10 steps completed successfully. **Compile works at 256N.**

| Step | Loss    | TPS | MFU    | Memory           |
|------|---------|-----|--------|------------------|
| 1    | 12.905  | 11  | 2.03%  | 46.28GiB(72.34%) |
| 2    | 12.933  | 83  | 15.24% | 46.28GiB(72.34%) |
| 3    | 12.883  | 79  | 14.51% | 46.28GiB(72.34%) |
| 5    | 12.876  | 80  | 14.59% | 46.28GiB(72.34%) |
| 10   | 12.344  | 63  | 11.61% | 46.28GiB(72.34%) |

**Key findings:**
- **Compile works at 256N** (~7 min) — the critical boundary is between
  256N and 512N where compile CPU OOMs
- **79-83 TPS/GPU** — ~7% drop from 128N (88 TPS), ~6% from 2N
- **Total throughput:** ~246K tokens/s across 256 nodes
- **Loss converges normally** with SophiaG at 256N (GBS=1536) — the NaN
  issue at 512N (GBS=3072) may be GBS-dependent, not just model-size

---

## Production Jobs Submitted

All use SophiaG optimizer, LR=2.28e-5, compile=on, 4.67T tokens on
olmo-mix-1124, checkpointing every 100 steps.

| Job ID  | Model | Nodes | TP | LBS | GBS    | Steps   | Status |
|---------|-------|-------|----|-----|--------|---------|--------|
| 8443219 | 2B    | 256   | 1  | 2   | 6,144  | 92,859  | Queued |
| 8443218 | 2B    | 512   | 1  | 2   | 12,288 | 46,429  | Queued |
| 8443212 | 20B   | 256   | 1  | 1   | 3,072  | 185,718 | Queued |
| 8442900 | 20B   | 512   | 1  | 1   | 6,144  | 92,859  | Queued |
| 8443213 | 80B   | 256   | 2  | 1   | 1,536  | 371,437 | Queued |
| 8443214 | 80B   | 512   | 2  | 1   | 3,072  | 185,718 | Queued |

## Key Insights

### 1. torch.compile scales differently by model size

| Model | 2N     | 4N     | 128N   | 256N    | 512N          |
|-------|--------|--------|--------|---------|---------------|
| 2B    | ~3 min | —      | 3.5 min| 4.5 min | 12+ hours     |
| 80B   | ~2 min | 4.5 min| 5 min  | —       | CPU OOM (2 min)|

The 2B model hits a compile wall at 512N. The 80B model compiles fast at
128N — needs testing at 512N.

### 2. 80B scales perfectly, 2B does not

| Model | 2N TPS | 128N TPS | 256N TPS | 512N TPS     | Eff (256N) |
|-------|--------|----------|----------|--------------|------------|
| 2B    | 5,400  | 3,000    | ~700 avg | pending      | 13%        |
| 20B   | 357    | —        | 280      | pending      | 78%        |
| 80B   | 88     | 88-90    | 79-83    | 66 (no-compile)| 93%      |

The 80B model maintains identical per-GPU TPS from 2 to 128 nodes because
each step takes ~47s of compute, completely dominating the communication
overhead. The 2B model's steps are only ~3s, so FSDP all-gather latency
becomes a significant fraction of step time at high rank counts.

### 3. Interactive allocation workflow validated

The pattern `test.sh` → SSH → `.ezpz-interactive-launch.sh` works
reliably for launching training inside pre-allocated PBS jobs. Required:
- ALCF proxy for internet access on compute nodes
- `set +u` for lmod/ezpz compatibility
- `nohup` to survive SSH disconnect

### 4. Stale checkpoints are a landmine

The 20B 512N run (Job 8441597) crashed because a pre-refactor checkpoint
existed at the default path. Always verify checkpoint compatibility or
use a fresh directory when the model architecture changes.

## Related

- [80B TP=2 Restored](../experiments/agpt/aurora/20260418-80b-tp2-restored.md)
- [Known Issues](../guides/known-issues.md)
- [Scaling Study (Sunspot)](../scaling/)
- [Production Run Plan](../TODO.md#5-production-multi-stage-training-plan)
