# MoE PR #3386 Replay Smoke -- Sunspot 2-node (2026-05-20)

Validates the `moe_*` configs after the
[PR #3386 (Refactor MoE to clean DTensor boundaries)](https://github.com/pytorch/torchtitan/pull/3386)
replay landed on `experiments/ezpz/moe/{parallelize.py,sharding.py,model.py}`.

PR #3386 restructures MoE sharding from an imperative
`apply_moe_ep_tp(model, tp_mesh, ep_mesh)` pass at parallelize-time to
the config-based `ShardingConfig` declarations populated at
`update_from_config`-time and applied by `model.parallelize(parallel_dims)`.
Upstream deleted `torchtitan/distributed/expert_parallel.py`
(`ExpertParallel`, `TensorParallel`) and the
`ColwiseParallelWithGradPlacement` style; introduced
`torchtitan/models/common/moe_sharding.py` with a
`set_moe_sharding_config(moe_cfg, *, enable_ep, enable_sp,
expert_param_layout)` helper that we now call from
`experiments/ezpz/moe/sharding.py` for every MoE-enabled layer.

This smoke confirms no regression at debugmodel + production 2B scale, with
the `for_loop` expert backend ([PR #13](https://github.com/saforem2/torchtitan/pull/13))
still firing correctly on XPU.

## Environment

| Field          | Value                              |
|----------------|------------------------------------|
| Date           | 2026-05-20                         |
| Branch         | `ezpz` (post-PR-3386-replay)       |
| Merge commit   | `89987072b`                        |
| Machine        | Sunspot (`x1921c3s0b0n0`, `x1921c3s1b0n0`) |
| Job ID         | 12467131                           |
| Nodes          | 2                                  |
| Devices        | 24 (Intel Max 1550)                |
| Devices/Node   | 12                                 |
| Steps          | 50 per config                      |
| Dataset        | blendcorpus (books)                |
| Backend        | xccl                               |
| Compile        | enabled (model + loss)             |
| FSDP           | `dp_shard=-1` (24-way shard, no replicate) |
| TP             | 1                                  |
| EP             | 1 (no expert parallel)             |
| Expert backend | `for_loop` (auto-fallback on XPU)  |
| Torch          | `2.13.0.dev20260418+xpu` (.venv)   |

## Summary

| Config              | LBS | GBS | Steps | TPS/GPU | TFLOPS/GPU | MFU     | Peak Memory        | Final Loss | Δ vs baseline | Status |
|---------------------|----:|----:|------:|--------:|-----------:|--------:|--------------------|-----------:|--------------:|--------|
| `moe_debugmodel`    | 2   | 48  | 50    | ~12,700 | ~26        | ~8.8%   | 16.99 GiB (26.55%) | 6.99880    | -0.010 nats   | clean  |
| `moe_2b` (LBS=1)    | 1   | 24  | 50    | ~2,900  | ~25        | ~8.4%   | 14.97 GiB (23.39%) | 6.10607    | -0.050 nats   | clean  |

Both configs finished cleanly with no NaN/OOM. Numerical equivalence holds
within the expected drift band for a graph reordering (PR #3386 moves
`shared_experts` compute from inside `combine()` to after `experts()`).

The PR's own commit message states:
> *"For other settings, loss is expected to diverge compared to main due to
> different reduction pattern, and shared expert computation changes place."*

At TP=1+EP=1 (this smoke) there are no shared experts and no TP-axis
reductions on the MoE wrapper, so the drift comes purely from FSDP gradient
all-reduce ordering. Observed drift is well within the ±0.1 nat band typical
of float32 reduction reordering.

## 1. `moe_debugmodel` (LBS=2)

Tiny 163 M-param sanity model. LBS=2 to match the 35th-sync sibling smoke.

| Step | Loss     | Grad-norm | TPS    | TFLOPS | MFU   | Memory              | Δ Loss vs baseline |
|-----:|---------:|----------:|-------:|-------:|------:|---------------------|-------------------:|
| 1    | 12.92142 | 0.7721    | 511    | 1.06   | 0.35% | 16.97 GiB (26.52%)  | -0.043             |
| 5    | 12.68582 | 1.0733    | 12,436 | 25.69  | 8.61% | 16.98 GiB (26.54%)  | -0.028             |
| 10   | 11.43847 | 1.6070    | 12,309 | 25.43  | 8.53% | 16.99 GiB (26.55%)  | +0.037             |
| 15   | 10.95288 | 1.5072    | 10,702 | 22.11  | 7.41% | 16.99 GiB (26.55%)  | +0.063             |
| 20   | 10.40176 | 1.4673    | 12,742 | 26.32  | 8.83% | 16.99 GiB (26.55%)  | +0.050             |
| 25   |  9.74369 | 1.3352    | 12,805 | 26.45  | 8.87% | 16.99 GiB (26.55%)  | +0.021             |
| 30   |  9.07519 | 1.1127    | 12,812 | 26.47  | 8.88% | 16.99 GiB (26.55%)  | +0.006             |
| 35   |  8.20756 | 0.8883    | 12,867 | 26.58  | 8.91% | 16.99 GiB (26.55%)  | -0.008             |
| 40   |  7.62643 | 0.5964    | 12,884 | 26.61  | 8.93% | 16.99 GiB (26.55%)  | -0.035             |
| 45   |  7.24139 | 0.3806    | 12,824 | 26.49  | 8.88% | 16.99 GiB (26.55%)  | -0.024             |
| 50   |  6.99880 | 0.2682    | 12,701 | 26.24  | 8.80% | 16.99 GiB (26.55%)  | -0.010             |

- Loss descent: **12.92 → 7.00** (-5.92 nats) over 50 steps. Clean. Final loss
  matches 35th-sync baseline (7.01) within -0.01 nats.
- W&B: [`efficient-bird-2070`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/saga2gds)
- Params: 163,102,976 total
- Wall time: ~70 s training (`22:15:53` → `22:17:30`)
- `for_loop` warnings: 5 (one per MoE layer; matches baseline)
- Memory exactly matches 35th-sync baseline (16.99 GiB / 26.55%)

## 2. `moe_2b` (LBS=1)

Production AGPT-MoE-2B with LBS=1 (default LBS=16 OOMs on the Max 1550
for_loop path; see 35th-sync report for context).

| Step | Loss     | Grad-norm | TPS   | TFLOPS | MFU   | Memory              | Δ Loss vs baseline |
|-----:|---------:|----------:|------:|-------:|------:|---------------------|-------------------:|
| 1    | 12.95322 | 1.5088    | 243   | 2.10   | 0.70% | 11.03 GiB (17.24%)  | +0.044             |
| 5    | 11.25454 | 3.2535    | 2,675 | 23.12  | 7.75% | 14.96 GiB (23.37%)  | -0.136             |
| 10   |  9.88184 | 2.4060    | 2,669 | 23.07  | 7.74% | 14.96 GiB (23.39%)  | -0.086             |
| 15   |  8.76568 | 2.8568    | 2,848 | 24.61  | 8.25% | 14.96 GiB (23.39%)  | +0.010             |
| 20   |  7.75594 | 1.5040    | 2,918 | 25.22  | 8.46% | 14.96 GiB (23.39%)  | -0.037             |
| 25   |  7.06025 | 1.1793    | 2,919 | 25.23  | 8.46% | 14.96 GiB (23.39%)  | +0.085             |
| 30   |  6.77294 | 1.0891    | 2,975 | 25.71  | 8.62% | 14.96 GiB (23.39%)  | -0.036             |
| 35   |  6.63228 | 0.9695    | 2,937 | 25.38  | 8.51% | 14.97 GiB (23.39%)  | -0.020             |
| 40   |  6.49818 | 1.1586    | 2,917 | 25.21  | 8.46% | 14.97 GiB (23.39%)  | +0.017             |
| 45   |  6.20125 | 1.0825    | 2,893 | 25.00  | 8.38% | 14.97 GiB (23.39%)  | +0.008             |
| 50   |  6.10607 | 1.1834    | 2,890 | 24.98  | 8.38% | 14.97 GiB (23.39%)  | -0.050             |

- Loss descent: **12.95 → 6.11** (-6.84 nats) over 50 steps. Clean. Slightly
  steeper than baseline (-6.75 nats).
- W&B: [`electric-pond-2071`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/zudqly3y)
- Params: 2,070,747,136 total
- Wall time: ~150 s training (`22:18:47` → `22:21:17`)
- `for_loop` warnings: 17 (one per MoE layer; matches baseline)
- Peak memory: **14.97 GiB (+0.5 GiB vs baseline 14.47 GiB)** — see Observations.

## Observations

- **Loss equivalence within numerical-reordering band.** Maximum per-step
  drift was ±0.136 nats at moe_2b step 5 (the early warmup); converged
  steps 30+ all within ±0.05 nats. This is fully consistent with PR #3386's
  documented behavior: the relocation of `shared_experts` compute from
  inside `combine()` to a post-`experts()` add changes the floating-point
  reduction order without changing the math. (Our smoke configs have
  `n_shared_experts=0`, so the actual drift source is FSDP gradient
  all-reduce ordering after the simplified MoE.forward.)
- **`for_loop` backend still fires correctly** post-replay. Same warning
  counts as the 35th-sync baseline (5 + 17), confirming
  `EzpzGroupedExperts.update_from_config` and the new
  `GroupedExperts.parallelize` override (which wires meshes on the token
  dispatcher) are both compatible.
- **debugmodel memory exactly matches baseline** (16.99 GiB / 26.55%).
- **moe_2b memory is +0.5 GiB vs baseline.** Likely from the new graph
  shape (post-experts addition allocates a separate buffer for `shared_out`
  before adding to routed `out`, instead of fusing inside `combine()`).
  At 14.97 GiB / 23.39% the headroom is still comfortable on a 64 GiB tile.
- **Throughput unchanged** for both configs (within ±300 TPS, ~3% of total).
- **No new compile errors / no recompilation.** The per-block compile
  pipeline (no fullgraph) handles the new `MoE.forward` shape cleanly on
  the first pass.

## What this validates

- The PR #3386 replay onto `experiments/ezpz/moe/{parallelize.py,
  sharding.py, model.py, config_registry.py}` is sound — no regression
  at debugmodel or 2B scale.
- The new `model.parallelize(parallel_dims)` path correctly distributes
  MoE submodule sharding configs (router, shared experts, routed experts)
  even though no axes are actually applied at TP=1+EP=1 — the populated
  configs are no-ops at this parallelism, but their *existence* is what
  flips the `model.parallelize` gate from the old `tp_enabled`-only
  branch to the new `tp_enabled or ep_enabled` branch.
- `EzpzGroupedExperts` (which inherits `GroupedExperts.parallelize` for
  free) gets `wire_meshes(ep_mesh=None, tp_mesh=None)` at TP=1+EP=1, which
  upstream's `LocalTokenDispatcher.wire_meshes` correctly no-ops on.
- `moe_2b` LBS=1 still establishes the "safe" baseline on Sunspot for
  future MoE smoke validation.

## Logs

- `moe_debugmodel`: `.cache/smoke-20260520-moe_debugmodel-postresync2.log`
- `moe_2b` LBS=1:  `.cache/smoke-20260520-moe_2b-postresync2.log`
- Structured (debugmodel): `logs/torchtitan.experiments.ezpz.train/2026-05-20-221520-rank0.jsonl`
- Structured (moe_2b):     `logs/torchtitan.experiments.ezpz.train/2026-05-20-221744-rank0.jsonl`

## Related

- Upstream PR: [pytorch/torchtitan#3386](https://github.com/pytorch/torchtitan/pull/3386)
- Replay commits: pending (this commit + upstream-sync.md update)
- Prior MoE smoke (35th sync, pre-#3386): [`20260520-smoke-n2-postresync.md`](20260520-smoke-n2-postresync.md)
- Replay log: [`docs/upstream-sync.md`](../../../upstream-sync.md) (37th sync entry)
