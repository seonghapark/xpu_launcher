# AuroraGPT-20B Scaling

## Sunspot Weak Scaling (torch 2.10, 1–64 nodes)

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 406+/-2 | 4,872+/-33 | 20.3% | 48.52 GiB (76%) | 100.0% |
| 2 | 24 | 358+/-4 | 8,592+/-101 | 17.9% | 44.64 GiB (70%) | 88.2% |
| 4 | 48 | 368+/-1 | 17,664+/-67 | 18.4% | 42.62 GiB (67%) | 90.6% |
| 8 | 96 | 366+/-2 | 35,184+/-203 | 18.3% | 41.85 GiB (65%) | 90.3% |
| 16 | 192 | 360+/-4 | 69,120+/-814 | 18.0% | 41.52 GiB (65%) | 88.7% |
| 32 | 384 | 366+/-0 | 140,736+/-271 | 18.3% | 41.12 GiB (64%) | 90.3% |
| 64 | 768 | 353 | 271,104 | 17.6% | 40.93 GiB (64%) | 86.9% |

**Config:** FSDP only (TP=1), compile=on, AC=full, seq_len=8192, LBS=1

## Aurora Weak Scaling (torch 2.13, 2–4096 nodes) — In Progress

| Nodes | GPUs | GBS | TPS/GPU | TFLOPS | MFU | Memory | Loss (final) | Status | Job |
|-------|------|-----|---------|--------|-----|--------|--------------|--------|-----|
| 1 | 12 | 24 | 413 | 61.41 | 20.59% | 48.52 GiB (76%) | — | Complete | 8437369 (2026-04-15, LBS=1 GAS=2) |
| 2 | 24 | 48 | 358 | 53.32 | 17.88% | 44.64 GiB (70%) | — | Complete | 8437409 (2026-04-15) |
| 4 | 48 | 96 | 375 | 55.80 | 18.71% | 42.62 GiB (67%) | — | Complete | 8437410 (2026-04-15) |
| 8 (LBS=1) | 96 | 192 | 371 | 55.17 | 18.50% | 41.85 GiB (65%) | — | Complete | 8437411 (2026-04-15) |
| 8 (LBS=2) | 96 | 192 | 510 | 75.86 | 25.44% | 41.51 GiB (65%) | — | Complete | [8530108](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/y1ujlog3) (2026-06-07) |
| 16 | 192 | — | — | — | — | — | — | Pending | |
| 32 | 384 | 768 | 4,500 | 50.3 | 16.9% | 43.96 GiB (69%) | 12.91 | Complete | 2026-04 |
| 64 (LBS=1) | 768 | 1,536 | 453 | 67.34 | 22.58% | 28.86 GiB (45%) | — | Complete | [8521698](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/3pm3admv) + [8528805](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/i0gra8by) (2026-06-06) |
| 64 (LBS=2) | 768 | 1,536 | 511 | 75.98 | 25.48% | 39.97 GiB (62%) | — | Complete | [8529046](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/0fyu61o4) (2026-06-06) |
| 128 (LBS=1) | 1,536 | 3,072 | 417 | 62.12 | 20.83% | 28.73 GiB (45%) | — | Complete | [8528834](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/5kpt6al1) (2026-06-06) |
| 128 (LBS=2) | 1,536 | 3,072 | 480 | 71.45 | 23.96% | 39.84 GiB (62%) | — | Complete | [8529081](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/rvueahwe) (2026-06-07) |
| **256 (LBS=1)** | 3,072 | 6,144 | 304 | 45.28 | 15.18% | 40.94 GiB (64%) | — | Complete | 8437387 (2026-04-16) |
| **512 (LBS=1)** | 6,144 | 12,288 | 241 | 35.79 | 12.00% | 41.02 GiB (64%) | — | Complete | 8437389 (2026-04-16) |
| 1024–4096 | — | — | — | — | — | — | — | Blocked: `set_determinism` init crash via bare ezpz launch (see [project_1024n_init_crash](.)). Production failover path may unblock, retry pending. |

**Config:** SophiaG LR=2.28e-5, compile=on, seq_len=8192, olmo-mix-1124

**Improvement over torch 2.10:** 440 TPS/GPU (2.13) vs 358 (2.10) at 2N = **+23%**

### Queue Mapping

| Node count | Queue | Walltime |
|------------|-------|----------|
| 2 | debug | 1h |
| 4–256 | debug-scaling | 1h |
| 256–1024 | prod → small | 12h |
| 1025–1919 | prod → medium | 12h |
| 1920+ | prod → large | 12h |

## Results Directory

- Sunspot: `outputs/scaling_study/20260412_091635/`
- Aurora (torch 2.13): per-job, see job IDs above

## See Also

- [Experiment reports](../experiments/agpt/) — per-run benchmark logs
- [Production training](../production/agpt/20b/) — live training status
