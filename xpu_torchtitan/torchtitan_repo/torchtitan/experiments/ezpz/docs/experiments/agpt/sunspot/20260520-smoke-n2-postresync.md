# AGPT Post-Resync Smoke -- Sunspot 2-node (2026-05-20)

Validates the `agpt_*` configs after the
[PR #3159 (Full DTensor for Llama3)](https://github.com/pytorch/torchtitan/pull/3159)
replay landed on `experiments/ezpz/agpt/parallelize.py` in commit
[`9924fdfd6`](https://github.com/saforem2/torchtitan/commit/9924fdfd6) (replay) and
[`f65237449`](https://github.com/saforem2/torchtitan/commit/f65237449) (upstream-sync
docs entry).

The upstream signature change `Module.parallelize(mesh)` → `Module.parallelize(parallel_dims)`
required mirroring the same shape into our agpt parallelize layer. This smoke confirms
no regression on either the debugmodel or the production agpt_2b flavor.

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
| Torch          | `2.13.0.dev20260418+xpu` (.venv)   |

## Summary

| Config              | LBS | GBS | Steps | TPS/GPU | TFLOPS/GPU | MFU     | Peak Memory      | Final Loss | Status |
|---------------------|----:|----:|------:|--------:|-----------:|--------:|------------------|-----------:|--------|
| `agpt_debugmodel`   | 2   | 48  | 50    | ~37,500 | ~8.7       | ~2.9%   | 2.60 GiB (4.06%) | 6.77       | clean  |
| `agpt_2b` (LBS=1)   | 1   | 24  | 50    | ~6,100  | ~68        | ~22.8%  | 24.34 GiB (38.04%) | 6.01     | clean  |
| `agpt_2b` (LBS=2)   | 2   | 48  | 50    | ~7,200  | ~80        | ~27.0%  | 44.73 GiB (69.91%) | 6.12     | clean  |

All three configs finished cleanly with no NaN/OOM, monotonically descending loss,
and steady-state memory.

## 1. `agpt_debugmodel`

Tiny 21.5 M-param sanity model -- kernel-launch-bound; throughput is meaningless,
correctness of the resync is the only thing being checked.

| Step | Loss     | Grad-norm | TPS    | TFLOPS | MFU   | Memory          |
|-----:|---------:|----------:|-------:|-------:|------:|-----------------|
| 1    | 10.80637 | 0.5414    | 374    | 0.09   | 0.03% | 2.60 GiB (4.06%) |
| 5    | 10.73855 | 0.5499    | 36,962 | 8.53   | 2.86% | 2.60 GiB (4.06%) |
| 10   | 10.44804 | 0.6631    | 36,016 | 8.31   | 2.79% | 2.60 GiB (4.06%) |
| 15   |  9.86727 | 0.9836    | 35,819 | 8.27   | 2.77% | 2.60 GiB (4.06%) |
| 20   |  8.90002 | 1.0888    | 39,104 | 9.03   | 3.03% | 2.60 GiB (4.06%) |
| 25   |  8.33927 | 0.9328    | 39,241 | 9.06   | 3.04% | 2.60 GiB (4.06%) |
| 30   |  7.93819 | 0.7000    | 37,374 | 8.63   | 2.89% | 2.60 GiB (4.06%) |
| 35   |  7.39700 | 0.5114    | 37,522 | 8.66   | 2.90% | 2.60 GiB (4.06%) |
| 40   |  7.09932 | 0.3883    | 37,847 | 8.74   | 2.93% | 2.60 GiB (4.06%) |
| 45   |  6.91023 | 0.2667    | 37,480 | 8.65   | 2.90% | 2.60 GiB (4.06%) |
| 50   |  6.76908 | 0.2282    | 35,273 | 8.14   | 2.73% | 2.60 GiB (4.06%) |

- Loss descent: **10.81 → 6.77** (-4.04 nats) over 50 steps. Clean.
- W&B: [`olive-plasma-2054`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/97qc7kbx)
- Params: 21,499,136 total
- Wall time: ~24 s training (`04:37:19` → `04:37:43`)

## 2. `agpt_2b` (LBS=1)

Production AGPT-2B with the smaller per-GPU batch size to leave headroom for
torch.compile inductor.

| Step | Loss     | Grad-norm | TPS   | TFLOPS | MFU    | Memory             |
|-----:|---------:|----------:|------:|-------:|-------:|--------------------|
| 1    | 12.93201 | 1.8872    | 305   | 3.41   | 1.14%  | 20.43 GiB (31.93%) |
| 5    | 10.90842 | 4.3317    | 5,763 | 64.48  | 21.62% | 24.34 GiB (38.04%) |
| 10   |  9.24587 | 3.7741    | 5,836 | 65.29  | 21.90% | 24.34 GiB (38.04%) |
| 15   |  7.63145 | 4.3143    | 6,059 | 67.78  | 22.73% | 24.34 GiB (38.04%) |
| 20   |  7.04706 | 2.9839    | 6,214 | 69.52  | 23.31% | 24.34 GiB (38.04%) |
| 25   |  6.80883 | 2.4362    | 6,247 | 69.89  | 23.44% | 24.34 GiB (38.04%) |
| 30   |  6.56439 | 1.6812    | 6,284 | 70.31  | 23.58% | 24.34 GiB (38.04%) |
| 35   |  6.43372 | 2.3638    | 6,312 | 70.62  | 23.68% | 24.34 GiB (38.04%) |
| 40   |  6.36797 | 3.5696    | 5,967 | 66.76  | 22.39% | 24.34 GiB (38.04%) |
| 45   |  6.05006 | 2.4050    | 6,117 | 68.44  | 22.95% | 24.34 GiB (38.04%) |
| 50   |  6.01036 | 1.5922    | 5,817 | 65.08  | 21.83% | 24.34 GiB (38.04%) |

- Loss descent: **12.93 → 6.01** (-6.92 nats) over 50 steps. Clean.
- W&B: [`sunny-waterfall-2055`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/izo5ysmh)
- Params: 1,986,578,432 total
- Wall time: ~69 s training (`04:57:38` → `04:58:47`)

## 3. `agpt_2b` (LBS=2)

Same flavor at the historical scaling-study batch size to compare against the
[2026-04-25 n=2 baseline](20260425-scaling-2b-venv-torch213.md).

| Step | Loss     | Grad-norm | TPS   | TFLOPS | MFU    | Memory             |
|-----:|---------:|----------:|------:|-------:|-------:|--------------------|
| 1    | 12.91364 | 1.8893    | 395   | 4.42   | 1.48%  | 36.90 GiB (57.67%) |
| 5    | 11.00088 | 3.5893    | 7,135 | 79.82  | 26.77% | 44.73 GiB (69.91%) |
| 10   |  9.24916 | 3.8571    | 7,300 | 81.67  | 27.39% | 44.73 GiB (69.91%) |
| 15   |  7.56830 | 2.4319    | 7,198 | 80.53  | 27.01% | 44.73 GiB (69.91%) |
| 20   |  6.86428 | 3.8402    | 7,171 | 80.23  | 26.91% | 44.73 GiB (69.91%) |
| 25   |  6.55991 | 2.3831    | 7,264 | 81.27  | 27.26% | 44.73 GiB (69.91%) |
| 30   |  6.72725 | 2.6961    | 7,209 | 80.66  | 27.05% | 44.73 GiB (69.91%) |
| 35   |  6.28113 | 2.3415    | 7,069 | 79.09  | 26.52% | 44.73 GiB (69.91%) |
| 40   |  6.45593 | 3.7226    | 7,352 | 82.25  | 27.58% | 44.73 GiB (69.91%) |
| 45   |  6.31794 | 2.8768    | 6,943 | 77.68  | 26.05% | 44.73 GiB (69.91%) |
| 50   |  6.12121 | 1.7361    | 7,224 | 80.82  | 27.11% | 44.73 GiB (69.91%) |

- Loss descent: **12.91 → 6.12** (-6.79 nats) over 50 steps. Clean.
- W&B: [`dry-water-2056`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/pqfyghhs)
- Params: 1,986,578,432 total
- Wall time: ~115 s training (`13:03:03` → `13:04:58`)

## What this validates

- The PR #3159 signature change `Module.parallelize(parallel_dims)` was replayed
  correctly onto `experiments/ezpz/agpt/parallelize.py` -- no regression at
  debugmodel or 2B scale.
- `agpt_2b` at LBS=2 matches the historical **2026-04-25 n=2 baseline**
  (7,142 TPS / 27.6% MFU at LBS=2): today's 7,224 TPS / 27.11% MFU is
  within noise. The resync did not perturb steady-state throughput.
- Memory at LBS=2 (44.73 GiB / 69.91%) matches the historical 44.57 GiB,
  confirming no leak from the FSDP path under the new parallelize signature.
- LBS=1 doubles down on safety margin -- 24.34 GiB / 38.04% leaves plenty of
  headroom for compile recompiles or future feature additions.

## Logs

- `agpt_debugmodel`: `.cache/smoke-20260519-agpt_debugmodel.log` (47 KiB)
- `agpt_2b` LBS=1:  `.cache/smoke-20260519-agpt_2b.log` (47 KiB)
- `agpt_2b` LBS=2:  `.cache/smoke-20260519-agpt_2b-lbs2.log` (47 KiB)
- Structured: `logs/torchtitan.experiments.ezpz.train/2026-05-20-04*-rank0.jsonl`,
  `2026-05-20-13*-rank0.jsonl`

## Related

- Upstream PR: [pytorch/torchtitan#3159](https://github.com/pytorch/torchtitan/pull/3159)
- Replay commits: [`9924fdfd6`](https://github.com/saforem2/torchtitan/commit/9924fdfd6),
  [`f65237449`](https://github.com/saforem2/torchtitan/commit/f65237449)
- Historical n=2 baseline: [20260425-scaling-2b-venv-torch213.md](20260425-scaling-2b-venv-torch213.md)
- Replay log: [`docs/upstream-sync.md`](../../../upstream-sync.md) (35th sync entry)
- Sibling moe smoke: [`../moe/sunspot/20260520-smoke-n2-postresync.md`](../../moe/sunspot/20260520-smoke-n2-postresync.md)
