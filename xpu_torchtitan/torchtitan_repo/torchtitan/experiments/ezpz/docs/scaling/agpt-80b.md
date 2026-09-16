# AuroraGPT-80B Scaling & Benchmarks

## Model Variants

| Variant | HIDDEN | NLAYERS | HEADS | KV_HEADS | FFN_HIDDEN |
|---------|--------|---------|-------|----------|------------|
| 80B | 9216 | 84 | 72 | 12 | 25600 |
| 80B_alt | 9216 | 84 | 72 | 12 | 25596 |
| 80B_wide | 10752 | 48 | 84 | 12 | 39936 |
| 80B_deep | 7680 | 96 | 60 | 12 | 28672 |
| 80B_deep_alt | 7680 | 96 | 60 | 12 | 28668 |

## Sunspot Weak Scaling (torch 2.10, 4–32 nodes)

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | CRASH | — | — | — | — |
| 2 | 24 | CRASH | — | — | — | — |
| 4 | 48 | 90 | 4,320 | 16.5% | 53.03 GiB (83%) | — |
| 8 | 96 | 100 | 9,600 | 18.3% | 49.55 GiB (77%) | — |
| 16 | 192 | 96 | 18,432 | 17.6% | 48.06 GiB (75%) | — |
| 32 | 384 | 96.5+/-0.7 | 37,056+/-271 | 17.7% | 46.98 GiB (73%) | — |
| 64 | 768 | CRASH | — | — | — | — |

**Config:** TP=2, compile=on, AC=full, seq_len=8192, LBS=1

## Aurora Throughput (2 nodes, compile=on)

| Model | TP | DP | GBS | Memory | TPS | TFLOPS | MFU | W&B |
|-------|----|----|-----|--------|-----|--------|-----|-----|
| 80B | 2 | 12 | 12 | 59.82 GiB (93%) | 89 | 48.44 | 16.24 | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/oo4paoya) |
| 80B_alt | 2 | 12 | 12 | 59.82 GiB (93%) | 84 | 45.75 | 15.34 | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/n8ikfo9j) |
| 80B_deep | 2 | 12 | 12 | 58.84 GiB (92%) | 86 | 47.26 | 15.85 | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/0rllk5qt) |
| 80B_deep_alt | 2 | 12 | 12 | 58.95 GiB (92%) | 82 | 44.78 | 15.02 | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/8p4jlzir) |
| 80B_wide | 2 | 12 | 12 | 60.75 GiB (95%) | 44 | 22.62 | 7.58 | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/n8tskhu0) |

**Note:** 80B TP=2 was broken on Aurora from 2026-04-12 to 2026-04-17 (OOM by
60 MiB). Resolved 2026-04-18 — works again at 88 TPS / 16% MFU after removing
`import intel_extension_for_pytorch` (IPEX allocator overhead).
See [restoration report](../experiments/agpt/aurora/20260418-80b-tp2-restored.md).

## Sunspot TP Sweep (2 nodes)

| Model | TP | DP | GBS | Memory | TPS | TFLOPS | MFU | Status |
|-------|----|----|-----|--------|-----|--------|-----|--------|
| 80B | 2 | 12 | 12 | 59.82 GiB (93%) | 85 | 46.31 | 15.53 | OK |
| 80B_alt | 2 | 12 | 12 | 59.82 GiB (93%) | 81 | 44.34 | 14.87 | OK |
| 80B_alt | 3 | 8 | 8 | 51.61 GiB (81%) | 23 | 12.40 | 4.16 | OK |
| 80B_alt | 6 | 4 | 4 | 39.54 GiB (62%) | 43 | 23.30 | 7.82 | OK |
| 80B_alt | 12 | 2 | 2 | — | — | — | — | OOM |
| 80B_wide | 2 | 12 | 12 | 60.75 GiB (95%) | 43 | 22.23 | 7.45 | Slow |
| 80B_wide | 3 | 8 | 8 | 52.75 GiB (82%) | 29 | 15.16 | 5.08 | OK |
| 80B_wide | 6 | 4 | 4 | 40.47 GiB (63%) | 51 | 26.35 | 8.84 | OK |
| 80B_wide | 12 | 2 | 2 | — | — | — | — | OOM |
| 80B_deep | 2 | 12 | 12 | 58.84 GiB (92%) | 83 | 45.58 | 15.29 | OK |
| 80B_deep_alt | 2 | 12 | 12 | 58.95 GiB (92%) | 77 | 41.87 | 14.04 | OK |
| 80B_deep_alt | 3 | 8 | 8 | 47.43 GiB (74%) | 23 | 12.56 | 4.21 | OK |
| 80B_deep_alt | 6 | 4 | 4 | 41.63 GiB (65%) | 41 | 22.43 | 7.52 | OK |
| 80B_deep_alt | 12 | 2 | 2 | 35.48 GiB (55%) | 28 | 15.40 | 5.16 | OK |

**Best config:** 80B TP=2 — 85-89 TPS, ~16% MFU

## Results Directories

- Sunspot scaling: `outputs/scaling_study/20260412_091635/`
- Sunspot TP sweep: `/flare/datascience/foremans/projects/torchtitan/outputs/benchmarks/80b_20260330_065453/`
- Aurora: `outputs/benchmarks/20260404_212401/`

## See Also

- [80B throughput leaderboard](../experiments/agpt/aurora/80b-throughput-leaderboard.md)
- [Experiment reports](../experiments/agpt/) — per-run benchmark logs
- [Production training](../production/agpt/80b/) — live training status
- [Known issues — 80B OOM](../guides/known-issues.md)
