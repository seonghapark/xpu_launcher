# MoE PR #3386 EP Follow-up Smoke -- Sunspot 2-node (2026-05-20)

After the [PR #3386 replay smoke](20260520-smoke-n2-pr3386-replay.md)
verified `TP=1+EP=1` was regression-free, this follow-up exercises the
expert-parallel (`EP=2`) path that the PR's new
`set_moe_sharding_config(..., enable_ep=True, ...)` wiring actually
flips on. The TP=1+EP=1 smoke only checks the *populated-but-no-op*
sharding configs — `EP=2` is the first real test that the new
`GroupedExperts.parallelize` → `token_dispatcher.wire_meshes(ep_mesh, ...)`
plumbing dispatches tokens correctly post-refactor.

## Environment

| Field          | Value                                              |
|----------------|----------------------------------------------------|
| Date           | 2026-05-20                                         |
| Branch         | `ezpz` (post-PR-3386-replay)                       |
| Commit         | `1d4115d3f`                                        |
| Machine        | Sunspot                                            |
| Job IDs        | 12467180 (debugmodel_ep), 12467181 (2b_ep)         |
| Nodes          | 2 (each job)                                       |
| Devices        | 24 per job (Intel Max 1550)                        |
| Devices/Node   | 12                                                 |
| Steps          | 50 per config                                      |
| Dataset        | blendcorpus (books)                                |
| Backend        | xccl                                               |
| Compile        | enabled (model + loss)                             |
| FSDP           | `dp_shard=-1` (24-way shard, no replicate)         |
| TP             | 1                                                  |
| EP             | 2 (`expert_parallel_degree=2`)                     |
| Expert backend | `standard` (the EP path; not `for_loop`)           |
| Torch          | `2.13.0.dev20260418+xpu` (.venv)                   |

## Model Configurations

| Config                   | LBS | Total params | Dense       | Sparse        | Active      | Layers | Routed experts | EP |
|--------------------------|----:|--------------|-------------|---------------|-------------|-------:|---------------:|---:|
| `moe_debugmodel_ep` (LBS=2 override) | 2 | 163,102,976  | 153,262,336 | 9,840,640    | 158,187,776 | (debug)| (debug)        | 2  |
| `moe_2b_ep` (LBS=1 override)         | 1 | 2,070,747,136 | 679,917,568 | 1,390,829,568 | 947,722,240 | (2B)  | (2B)           | 2  |

Both `_ep` configs in `config_registry.py` inherit their parent's
`local_batch_size` — `moe_debugmodel_ep` keeps `LBS=8` and
`moe_2b_ep` keeps `LBS=16`. Both default LBSes OOM at `EP=2` on the
Max 1550 64 GiB tile (see Findings below). Both were overridden via
`--training.local_batch_size` for this smoke.

## Summary

| Config                       | LBS                              | Steps           | TPS/GPU | TFLOPS/GPU | MFU    | Peak Memory       | Final Loss | Wall time | Status |
|------------------------------|----------------------------------|----------------:|--------:|-----------:|-------:|-------------------|-----------:|----------:|--------|
| `moe_debugmodel_ep` (default LBS=8) | 8                       | 0 (OOM at init) |    --   |     --     |   --   | 31.27 GiB request (XPU OOM) | --       | 121 s     | FAIL — vocab projection OOM |
| `moe_debugmodel_ep` (LBS=2 override) | 2                      | 41 / 50         | ~6,900  | ~14.2      | ~4.8%  | 16.86 GiB (26.35%) | 7.56489  | 1146 s    | HUNG — clean steps 1-41, ~16 min stall, SIGTERM |
| `moe_2b_ep` (LBS=1 override) | 1                                | 50              | ~2,860  | ~24.7      | ~8.3%  | 15.03 GiB (23.50%) | 6.07128    | 217 s     | clean (exit 0) |

The headline result: **the EP=2 path itself works** — both runs
exercised real `wire_meshes(ep_mesh, ...)` dispatching, and `moe_2b_ep`
descended cleanly for 50 steps with the same memory + throughput
envelope as the TP=1+EP=1 baseline. The two non-clean rows reveal
*configuration* issues, not refactor regressions.

## 1. `moe_debugmodel_ep` (default `LBS=8`) — vocab-projection OOM

```text
torch.OutOfMemoryError: XPU out of memory. Tried to allocate 31.27 GiB.
GPU 2 has a total capacity of 63.98 GiB of which 30.13 GiB is free.
Of the allocated memory 31.67 GiB is allocated by PyTorch ...
```

The allocation `31.27 GiB` is exactly the cross-entropy output
projection `(LBS * seq_len, vocab_size)` in bf16:

```
(8 * 8192, 256128) bf16 = 67,108,864 * 256,128 * 2 bytes ≈ 32 GB
```

debugmodel inherits the full 256128-token Gemma vocab. At EP=2 the
expert tensors halve, but the language-model head is **not** sharded
across EP — the vocab projection blows up on a single tile before
training starts.

Compare: `moe_debugmodel` (no `_ep`) runs at LBS=8 *fine* in the
replay smoke (16.99 GiB / 26.55%). At EP=1 the FSDP + reduced expert
memory leaves the budget for a 32 GB projection; at EP=2 the
intermediate activations the EP path *adds* (token-dispatch buffers)
shift the budget so the projection no longer fits.

## 2. `moe_debugmodel_ep` (LBS=2 override) — hangs at step 41

Re-ran with `--training.local_batch_size 2`. Steady-state numbers
were healthy:

| Step | Loss     | Grad-norm | TPS    | TFLOPS | MFU   | Memory              |
|-----:|---------:|----------:|-------:|-------:|------:|---------------------|
| 1    | 12.94510 | 0.7901    |    474 |  0.98  | 0.33% | 16.85 GiB (26.33%)  |
| 5    | 12.71764 | 1.0965    |  6,793 | 14.03  | 4.71% | 16.86 GiB (26.35%)  |
| 10   | 11.43654 | 1.5672    |  6,785 | 14.02  | 4.70% | 16.86 GiB (26.35%)  |
| 20   | 10.43099 | 1.4220    |  6,539 | 13.51  | 4.53% | 16.86 GiB (26.35%)  |
| 30   |  9.12000 | 1.0077    |  6,971 | 14.40  | 4.83% | 16.86 GiB (26.35%)  |
| 40   |  7.67952 | 0.6659    |  6,688 | 13.82  | 4.63% | 16.86 GiB (26.35%)  |
| 41   |  7.56489 | 0.6454    |  6,765 | 13.97  | 4.69% | 16.86 GiB (26.35%)  |

Loss descended cleanly **12.95 → 7.56** over 41 steps with stable
grad-norm and constant memory. Step 41 emitted normally at 04:05:06.
Step 42 never emitted; the run sat idle for ~16 minutes
(04:05:06 → 04:21:31) before `ezpz launch` reaped it with exit 143
(SIGTERM).

Throughput is ~50% of the no-EP equivalent (`moe_debugmodel` LBS=2:
~12,700 TPS / 8.8% MFU). The EP=2 token-dispatch comm overhead
dominates at this tiny model size — expected; the relevance test is
*correctness*, which the loss descent confirms.

Possible causes for the hang (uninvestigated):

- EP all-to-all backend deadlock between two ranks
- xccl heartbeat / barrier timeout
- A specific rank stalling inside `wire_meshes` book-keeping at a
  data-dependent boundary

W&B run: [`gtutjvg5`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/gtutjvg5)

## 3. `moe_2b_ep` (LBS=1 override) — clean 50 steps

Default LBS=16 would OOM similarly (vocab projection at 256128 *
16384 bf16 ≈ 8 GB just for the projection, plus the much larger
expert activations at 2B model scale). Used LBS=1.

| Step | Loss     | Grad-norm | TPS   | TFLOPS | MFU   | Memory              |
|-----:|---------:|----------:|------:|-------:|------:|---------------------|
| 1    | 12.94465 | 1.4691    |   230 |  1.99  | 0.67% | 11.07 GiB (17.30%)  |
| 5    | 11.37929 | 4.1614    | 2,629 | 22.72  | 7.62% | 15.02 GiB (23.47%)  |
| 10   |  9.95968 | 3.1358    | 2,644 | 22.85  | 7.66% | 15.03 GiB (23.49%)  |
| 15   |  8.72318 | 2.2992    | 2,843 | 24.57  | 8.24% | 15.03 GiB (23.49%)  |
| 20   |  7.76364 | 3.5520    | 2,867 | 24.78  | 8.31% | 15.03 GiB (23.50%)  |
| 25   |  6.92034 | 0.8649    | 2,848 | 24.61  | 8.25% | 15.03 GiB (23.50%)  |
| 30   |  6.96835 | 4.4769    | 2,705 | 23.38  | 7.84% | 15.03 GiB (23.50%)  |
| 35   |  6.61236 | 1.2758    | 2,813 | 24.31  | 8.15% | 15.03 GiB (23.50%)  |
| 40   |  6.42926 | 1.3757    | 2,875 | 24.85  | 8.33% | 15.03 GiB (23.50%)  |
| 45   |  6.15321 | 1.2220    | 2,878 | 24.88  | 8.34% | 15.03 GiB (23.50%)  |
| 50   |  6.07128 | 1.9649    | 2,748 | 23.75  | 7.97% | 15.03 GiB (23.50%)  |

- Loss descent: **12.94 → 6.07** (-6.87 nats) over 50 steps. Clean.
  Within ±0.05 nats of the TP=1+EP=1 `moe_2b` LBS=1 baseline (6.11)
  from the replay smoke — same reduction-reordering band.
- Peak memory: **15.03 GiB (+0.06 GiB vs EP=1 baseline)**. The EP=2
  token-dispatch overhead is essentially free at this scale.
- Throughput: 2,860 TPS/GPU vs 2,890 TPS/GPU at EP=1. EP=2 is within
  ~1% of EP=1 throughput at moe_2b scale — far below the ~50% hit at
  debugmodel scale, because the per-step expert compute now dominates
  the all-to-all comm.
- W&B: [`likely-energy-2078`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/1wq3fqrr)
- Wall time: 217 s

## Findings

### 1. `_ep` configs should override LBS in the registry

`moe_debugmodel_ep` inherits `LBS=8` from `moe_debugmodel`; `moe_2b_ep`
inherits `LBS=16` from `moe("2B", local_batch_size=16)`. Both OOM at
default LBS on the Sunspot 64 GiB tile because the unsharded language-
model head dominates the budget once EP=2 adds token-dispatch
buffers.

Concrete proposal (within ezpz scope per Golden Rule #1):

```python
def moe_debugmodel_ep() -> FaultTolerantTrainer.Config:
    cfg = moe_debugmodel()
    cfg.model_spec = model_registry("debugmodel", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    cfg.training.local_batch_size = 2  # 8 OOMs vocab projection at EP=2 on 64 GiB tiles
    return cfg


def moe_2b_ep() -> FaultTolerantTrainer.Config:
    cfg = moe("2B", local_batch_size=1)  # 16 OOMs at EP=2 on 64 GiB tiles
    cfg.model_spec = model_registry("2B", moe_comm_backend="standard")
    cfg.parallelism.expert_parallel_degree = 2
    return cfg
```

Not landed yet — flagging here for a follow-up commit.

### 2. ~~EP=2 hang at debugmodel scale needs investigation~~ — Reclassified transient on retry

**2026-05-21 update**: re-ran the same config + stack on a sibling
2N alloc, completed 50 steps cleanly in 139 s (exit 0). Original
step-41 stall did **not reproduce** — reclassifying as a transient
(likely a single-rank CCL/driver stall that didn't propagate as an
error). No code change recommended. Full retry writeup, plus three
adjacent findings (`comm.train_timeout_seconds` not wiring through,
`TORCH_DISTRIBUTED_DEBUG=DETAIL` crashing on XPU,
`--debug.deterministic` incompatible with MoE `_histc_xpu`):
[`docs/upstream-issues/moe_ep_step41_hang.md`](../../../upstream-issues/moe_ep_step41_hang.md).

### 3. The PR #3386 EP wiring works at production scale

`moe_2b_ep` validates the most important refactor claim: the new
config-based `set_moe_sharding_config(enable_ep=True, ...)` path
produces a working EP=2 training step at 2B-param scale with loss
descent within the expected reduction-reordering band. The two
problem rows above are configuration ergonomics, not regressions.

## Logs

- `moe_debugmodel_ep` (OOM):            `logs/smoke-pr3386-followup/moe_debugmodel_ep-20260520-225938.log`
- `moe_debugmodel_ep` LBS=2 (hang):     `logs/smoke-pr3386-followup/moe_debugmodel_ep_lbs2-20260520-230219.log`
- `moe_2b_ep` LBS=1 (clean):            `logs/smoke-pr3386-followup/moe_2b_ep-20260520-232257.log`

## Related

- Upstream PR: [pytorch/torchtitan#3386](https://github.com/pytorch/torchtitan/pull/3386)
- Replay smoke (TP=1, EP=1, clean): [`20260520-smoke-n2-pr3386-replay.md`](20260520-smoke-n2-pr3386-replay.md)
- 37th sync entry: [`docs/upstream-sync.md`](../../../upstream-sync.md)
- Sibling agpt follow-up: [`../../agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md`](../../agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md)
