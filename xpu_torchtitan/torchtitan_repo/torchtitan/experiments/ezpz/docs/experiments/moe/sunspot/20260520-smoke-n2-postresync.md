# MoE Post-Resync Smoke -- Sunspot 2-node (2026-05-20)

Validates the `moe_*` configs after the
[PR #3159 (Full DTensor for Llama3)](https://github.com/pytorch/torchtitan/pull/3159)
replay landed on `experiments/ezpz/moe/parallelize.py` in commit
[`9924fdfd6`](https://github.com/saforem2/torchtitan/commit/9924fdfd6) (replay) and
[`f65237449`](https://github.com/saforem2/torchtitan/commit/f65237449) (upstream-sync
docs entry).

PR #3159's `Module.parallelize(mesh)` → `Module.parallelize(parallel_dims)` signature
change required the same mirror in our moe parallelize layer. This smoke confirms
no regression at debugmodel + production 2B scale, with the `for_loop` expert backend
([PR #13](https://github.com/saforem2/torchtitan/pull/13)) still firing correctly
on XPU.

## Environment

| Field          | Value                              |
|----------------|------------------------------------|
| Date           | 2026-05-20                         |
| Branch         | `ezpz` (post-resync)               |
| Commits        | `9924fdfd6`, `f65237449`           |
| Machine        | Sunspot (`x1921c0s2b0n0`, `x1921c0s3b0n0`) |
| Job ID         | 12467124                           |
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

| Config              | LBS | GBS | Steps | TPS/GPU | TFLOPS/GPU | MFU     | Peak Memory        | Final Loss | Status |
|---------------------|----:|----:|------:|--------:|-----------:|--------:|--------------------|-----------:|--------|
| `moe_debugmodel`    | 2   | 48  | 50    | ~13,000 | ~27        | ~9.0%   | 16.99 GiB (26.55%) | 7.01       | clean  |
| `moe_2b` (LBS=1)    | 1   | 24  | 50    | ~2,900  | ~25        | ~8.4%   | 14.47 GiB (22.62%) | 6.16       | clean  |

Both configs finished cleanly with no NaN/OOM. The `for_loop` fallback fires once
per MoE layer on XPU (no SM90 capability → switches the experts config away from
`grouped_mm` before `build()` instantiates the module).

## 1. `moe_debugmodel` (LBS=2)

Tiny 163 M-param sanity model. Default `moe_debugmodel()` sets LBS=8 which OOMs
on Max 1550 inductor backward -- overrode to LBS=2 to match the agpt sibling
smoke and stay well under the 64 GiB tile.

| Step | Loss     | Grad-norm | TPS    | TFLOPS | MFU   | Memory              |
|-----:|---------:|----------:|-------:|-------:|------:|---------------------|
| 1    | 12.96452 | 0.8528    | 556    | 1.15   | 0.39% | 16.97 GiB (26.52%)  |
| 5    | 12.71404 | 1.0892    | 12,632 | 26.09  | 8.75% | 16.98 GiB (26.54%)  |
| 10   | 11.40110 | 1.6767    | 12,656 | 26.14  | 8.77% | 16.98 GiB (26.54%)  |
| 15   | 10.88989 | 1.3913    | 11,988 | 24.76  | 8.30% | 16.99 GiB (26.55%)  |
| 20   | 10.35191 | 1.3454    | 13,143 | 27.15  | 9.10% | 16.99 GiB (26.55%)  |
| 25   |  9.72270 | 1.1535    | 13,084 | 27.03  | 9.06% | 16.99 GiB (26.55%)  |
| 30   |  9.06926 | 0.9399    | 13,107 | 27.07  | 9.08% | 16.99 GiB (26.55%)  |
| 35   |  8.21599 | 0.7629    | 13,193 | 27.25  | 9.14% | 16.99 GiB (26.55%)  |
| 40   |  7.66102 | 0.5543    | 13,031 | 26.92  | 9.03% | 16.99 GiB (26.55%)  |
| 45   |  7.26519 | 0.3937    | 13,031 | 26.92  | 9.03% | 16.99 GiB (26.55%)  |
| 50   |  7.00865 | 0.2692    | 12,585 | 26.00  | 8.72% | 16.99 GiB (26.55%)  |

- Loss descent: **12.96 → 7.01** (-5.96 nats) over 50 steps. Clean.
- W&B: [`worldly-music-2058`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/7h9t0q4u)
- Params: 163,102,976 total
- Wall time: ~64 s training (`13:09:00` → `13:10:04`)
- `for_loop` warnings: 5 (one per MoE layer in the debugmodel)

## 2. `moe_2b` (LBS=1)

Production AGPT-MoE-2B with a smaller per-GPU batch -- default `moe_2b()` is
LBS=16, which tries to allocate **62.53 GiB in a single call** for the
inductor-fused activation tensor and OOMs immediately on a 64 GiB tile.
LBS=1 leaves the activation footprint small enough for the for_loop expert
backend to fit comfortably.

| Step | Loss     | Grad-norm | TPS   | TFLOPS | MFU   | Memory              |
|-----:|---------:|----------:|------:|-------:|------:|---------------------|
| 1    | 12.90917 | 1.5098    | 258   | 2.23   | 0.75% | 10.52 GiB (16.45%)  |
| 5    | 11.39055 | 5.0717    | 2,708 | 23.40  | 7.85% | 14.46 GiB (22.60%)  |
| 10   |  9.96791 | 3.0372    | 2,855 | 24.68  | 8.28% | 14.46 GiB (22.60%)  |
| 15   |  8.75550 | 2.8312    | 2,904 | 25.10  | 8.42% | 14.46 GiB (22.60%)  |
| 20   |  7.79281 | 1.4533    | 2,958 | 25.57  | 8.57% | 14.46 GiB (22.60%)  |
| 25   |  6.97559 | 2.5843    | 2,935 | 25.36  | 8.51% | 14.46 GiB (22.60%)  |
| 30   |  6.80864 | 1.3108    | 2,975 | 25.71  | 8.62% | 14.47 GiB (22.62%)  |
| 35   |  6.65269 | 2.0920    | 2,917 | 25.21  | 8.46% | 14.47 GiB (22.62%)  |
| 40   |  6.48103 | 0.9353    | 2,919 | 25.23  | 8.46% | 14.47 GiB (22.62%)  |
| 45   |  6.19349 | 1.1082    | 2,890 | 24.98  | 8.38% | 14.47 GiB (22.62%)  |
| 50   |  6.15666 | 0.7874    | 2,816 | 24.34  | 8.16% | 14.47 GiB (22.62%)  |

- Loss descent: **12.91 → 6.16** (-6.75 nats) over 50 steps. Clean.
- W&B: [`azure-field-2061`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/jv6bnvuq)
- Params: 2,070,747,136 total
- Wall time: ~140 s training (`13:14:25` → `13:16:47`)
- `for_loop` warnings: 17 (one per MoE layer in the 2B flavor)

## Observations

- **`for_loop` backend still fires correctly** post-resync. The
  `EzpzGroupedExperts.update_from_config` path lands the same warning as the
  2026-05-12 8-node `moe_500m` smoke, with the same kernel body producing
  correct gradients under FSDP.
- **Throughput is in the expected range:** debugmodel is kernel-launch-bound
  so 12-13k TPS / 9% MFU is normal; `moe_2b` at LBS=1 reaches ~2,900 TPS /
  8.4% MFU per GPU, consistent with prior MoE-2B numbers (April benchmarks
  reported moe_2b at 11% MFU at LBS=16; per-GPU at LBS=1 is lower because
  the for_loop path scales linearly with batch).
- **Memory headroom is comfortable** for both configs (26.5% and 22.6% of
  the 64 GiB tile), leaving room to scale LBS or enable EP in a future smoke.
- **Default `moe_2b()` LBS=16 OOMs on Max 1550 with the for_loop backend.**
  The OOM is a single 62.53 GiB allocation -- likely the fused activation
  for all experts × full-batch-tokens. This was a pre-existing constraint
  on the XPU for_loop path; not caused by the resync. Documented here so
  the next smoke is paced accordingly.

## What this validates

- The PR #3159 signature change `Module.parallelize(parallel_dims)` was
  replayed correctly onto `experiments/ezpz/moe/parallelize.py` -- no
  regression at debugmodel or 2B scale.
- The PR #13 `for_loop` expert backend still functions end-to-end under the
  new parallelize signature. 50 steps of forward+backward+optimizer with
  clean loss descent at both flavors.
- `moe_2b` at LBS=1 establishes a new "safe" baseline for future moe-2b
  smoke validation on Sunspot.

## Logs

- `moe_debugmodel`: `.cache/smoke-20260519-moe_debugmodel.log` (51 KiB)
- `moe_2b` LBS=1:  `.cache/smoke-20260519-moe_2b.log`
- Structured (debugmodel): `logs/torchtitan.experiments.ezpz.train/2026-05-20-130759-rank0.jsonl`
- Structured (moe_2b):     `logs/torchtitan.experiments.ezpz.train/2026-05-20-131323-rank0.jsonl`

## Related

- Upstream PR: [pytorch/torchtitan#3159](https://github.com/pytorch/torchtitan/pull/3159)
- Replay commits: [`9924fdfd6`](https://github.com/saforem2/torchtitan/commit/9924fdfd6),
  [`f65237449`](https://github.com/saforem2/torchtitan/commit/f65237449)
- Prior MoE smoke: [`20260512-for-loop-smoke-n8.md`](20260512-for-loop-smoke-n8.md)
- Replay log: [`docs/upstream-sync.md`](../../../upstream-sync.md) (35th sync entry)
- Sibling agpt smoke: [`../../agpt/sunspot/20260520-smoke-n2-postresync.md`](../../agpt/sunspot/20260520-smoke-n2-postresync.md)
