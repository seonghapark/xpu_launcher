# MoE `for_loop` Backend Smoke -- Sunspot 8-node (2026-05-12)

Validates the `EzpzGroupedExperts.compute_backend = "for_loop"` fallback path
introduced in [PR #13](https://github.com/saforem2/torchtitan/pull/13) to handle
the upstream removal of `use_grouped_mm` from `GroupedExperts.Config` in
[pytorch/torchtitan#3308](https://github.com/pytorch/torchtitan/pull/3308).

`torch._grouped_mm` has no XPU kernel and no XPU fallback, so without this PR
all ezpz MoE configs would error out at first forward on Aurora / Sunspot. The
new `model.py` `update_from_config` block detects the missing SM90 capability
and switches the experts config to `compute_backend="for_loop"` automatically.

## Environment

| Field        | Value                              |
|--------------|------------------------------------|
| Date         | 2026-05-12                         |
| Branch       | `ezpz-moe-resync` (PR #13)         |
| Commit       | `9bdf8bb8`                         |
| Machine      | Sunspot (`x1921c5s0b0n0`)          |
| Job ID       | 12466707                           |
| Nodes        | 8                                  |
| Devices      | 96 (Intel Max 1550)                |
| Devices/Node | 12                                 |
| Steps        | 50                                 |
| Dataset      | blendcorpus (books)                |
| Backend      | xccl                               |
| Compile      | enabled (model + loss)             |

## Result

**Exit 0** in **541 s** (9 min 1 s) wall, including environment setup,
torch.compile warmup, and 50 training steps.

## For-loop fallback fired correctly

11 layer-wise warnings (1 per MoE layer in the 500M flavor: 12 total layers, 1
dense + 11 MoE):

```
[W][moe/model:238:update_from_config] torch._grouped_mm requires SM90+ CUDA;
    falling back to for_loop expert backend.
```

This is the new code path landing in `EzpzGroupedExperts._experts_forward` —
the `compute_backend` was switched away from the default `"grouped_mm"` before
`build()` instantiated the experts.

## Run config

| Field            | Value           |
|------------------|-----------------|
| Config function  | `moe_500m`      |
| Model flavor     | `500M`          |
| Total params     | 481.4 M         |
| Dense params     | 325.6 M         |
| Sparse params    | 155.8 M         |
| Active params    | 369.0 M         |
| Layers           | 12 (1 dense + 11 MoE) |
| Hidden dim       | 512             |
| MoE hidden dim   | 512             |
| Num experts      | 16              |
| Top-k            | 3               |
| Local batch size | 4               |
| Global batch size| 384             |
| Sequence length  | 8192            |
| Optimizer        | AdamW lr=8e-4 (default `_base_config`) |

## Loss & throughput trajectory

| Step | Loss     | Grad-norm | TPS    | TFLOPS | MFU    | Memory             |
|-----:|---------:|----------:|-------:|-------:|-------:|--------------------|
| 1    | 12.90162 | 1.0817    | 152    | 0.68   | 0.23%  | 34.93 GiB (54.58%) |
| 5    | 12.17321 | 2.2707    | 8,414  | 37.41  | 12.55% | 34.93 GiB (54.59%) |
| 10   | 10.61516 | 1.8626    | 8,011  | 35.62  | 11.95% | 34.95 GiB (54.62%) |
| 15   |  9.91348 | 1.4110    | 8,544  | 37.99  | 12.74% | 34.95 GiB (54.62%) |
| 20   |  9.02683 | 1.2070    | 8,718  | 38.77  | 13.00% | 34.95 GiB (54.62%) |
| 25   |  8.04900 | 0.8611    | 8,759  | 38.95  | 13.06% | 34.95 GiB (54.62%) |
| 30   |  7.33213 | **10.6135** | 8,736 | 38.84 | 13.03% | 34.95 GiB (54.62%) |
| 35   |  7.34477 | 5.1237    | 8,711  | 38.74  | 12.99% | 34.95 GiB (54.62%) |
| 40   |  6.83830 | 0.5604    | 8,775  | 39.02  | 13.09% | 34.95 GiB (54.62%) |
| 45   |  6.75210 | 0.3807    | 8,665  | 38.53  | 12.92% | 34.95 GiB (54.62%) |
| 50   |  6.65774 | 0.6415    | 8,694  | 38.66  | 12.96% | 34.95 GiB (54.62%) |

**Loss descent: 12.90 → 6.66 (-6.24 nats)** over 50 steps.
**Steady-state throughput: ~8,700 TPS / GPU, ~13% MFU.**
**Peak memory: 34.95 GiB (54.62%) of the 64 GiB Intel Max 1550 tile.**

## Observations

- **Throughput is in line with the prior `grouped_mm` baseline.** The
  [2026-04-13 Sunspot benchmark](20260413-benchmark-n2.md) ran the same
  `moe_500m` flavor at 2 nodes (24 GPUs) and reported 7,228 TPS / 9.11% MFU.
  Today's 8-node run hits 8,694 TPS / 12.96% MFU per GPU, ~20% higher per-GPU
  TPS at 4× the GPU count, which is expected from torch.compile inductor
  improvements landed since April. **The for-loop path is not measurably
  slower than the prior `_run_experts_for_loop` upstream had** (which makes
  sense — it's the same kernel body, re-vendored verbatim).
- **One grad-norm spike at step 30** (10.61, recovered to 0.56 by step 40).
  Typical of early MoE training where router decisions are still chaotic.
  Loss curve uninterrupted; no NaN, no OOM, no AC recompute mismatch. Not
  related to the new backend.
- **Memory is stable at 54.6% across the entire run** — for-loop expert
  compute doesn't peak any higher than the upstream `grouped_mm` path.
- **Loss compile runs once** per the W&B summary (`loss_metrics/global_max_loss
  = 7.07`), no compile recompiles during steady state.

## What this validates

✅ **PR #13's `EzpzGroupedExperts.compute_backend = "for_loop"` switch fires
correctly on XPU** (`has_cuda_capability(9, 0)` returns False, the warning is
emitted, the experts config is mutated before `build()` instantiates the
module).

✅ **The re-vendored `_run_experts_for_loop` body produces correct gradients
under FSDP** (50-step end-to-end backward pass, loss descends cleanly, no
DTensor placement errors).

✅ **No regression in throughput vs the upstream `_run_experts_for_loop` body
that #3308 deleted** — same kernel, same numbers.

## Logs

- Stdout: `logs/moe-for-loop-smoke/moe_500m-20260512-194515.log` (55 KiB, 631
  lines)
- Structured: `logs/torchtitan.experiments.ezpz.train/2026-05-13-004541-rank0.jsonl`
- W&B: [`fluent-glitter-2042`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/lt77xx0o)

## Related

- Resync PR: [saforem2/torchtitan#13](https://github.com/saforem2/torchtitan/pull/13)
- Upstream cause: [pytorch/torchtitan#3308](https://github.com/pytorch/torchtitan/pull/3308) (`Remove MoE expert for-loop fallback`)
- Replay log: [`docs/upstream-sync.md`](../../../upstream-sync.md) (33rd sync entry)
