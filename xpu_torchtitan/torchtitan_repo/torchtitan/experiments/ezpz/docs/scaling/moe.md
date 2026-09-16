# MoE Model Scaling

## Sunspot Weak Scaling (torch 2.10, 1–64 nodes)

### moe_2b

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 7,121+/-77 | 85,458+/-924 | 11.9% | 15.65 GiB (24%) | 100.0% |
| 2 | 24 | 6,970+/-7 | 167,280+/-169 | 11.7% | 15.34 GiB (24%) | 97.9% |
| 4 | 48 | 6,784+/-76 | 325,632+/-3665 | 11.4% | 15.92 GiB (25%) | 95.3% |
| 8 | 96 | 6,392+/-45 | 613,680+/-4412 | 10.7% | 16.84 GiB (26%) | 89.8% |
| 16 | 192 | 4,959+/-49 | 952,128+/-9503 | 8.3% | 19.09 GiB (30%) | 69.6% |
| 32 | 384 | 5,170+/-12 | 1,985,472+/-4615 | 8.7% | 26.27 GiB (41%) | 72.6% |
| 64 | 768 | 3,376 | 2,592,768 | 5.7% | 37.54 GiB (59%) | 47.4% |

### moe_7b

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 1,841+/-13 | 22,098+/-161 | 8.4% | 33.47 GiB (52%) | 100.0% |
| 2 | 24 | 1,197+/-18 | 28,728+/-441 | 5.4% | 33.31 GiB (52%) | 65.0% |
| 4 | 48 | 1,417+/-28 | 68,016+/-1357 | 6.4% | 31.09 GiB (49%) | 76.9% |
| 8 | 96 | 1,111+/-0 | 106,704+/-67 | 5.0% | 35.87 GiB (56%) | 60.4% |
| 16 | 192 | 421+/-0 | 80,928+/-135 | 1.9% | 40.78 GiB (64%) | 22.9% |
| 32 | 384 | OOM | — | — | — | — |
| 64 | 768 | OOM | — | — | — | — |

**Config:** no-compile, AC=none, FSDP only (TP=1), seq_len=4096, LBS=1

**Note:** MoE scaling efficiency degrades significantly beyond 8 nodes.
Memory usage grows with node count (all-to-all communication buffers).
moe_7b OOMs at 32+ nodes. See [TODO — MoE throughput optimization](../TODO.md#3-moe-throughput-optimization)
for planned experiments (EP, TP, float8).

## Aurora torch 2.13 — partial

### moe_2b — works at small N, blocked at 128+

| Nodes | GPUs | GBS | TPS/GPU | TFLOPS | MFU | Memory | Status | Job |
|-------|------|-----|---------|--------|-----|--------|--------|-----|
| 8 | 96 | 192 | 2,586 | 16.50 | 5.53% | 31.42 GiB (49%) | Complete | [8530108](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/2zubxd41) (2026-06-07) |
| 64 | 768 | 1,536 | — | — | — | — | NO_OUTPUT | 8528805 (2026-06-06, pre-spmd_types fix, 251s wall) + 8529046 (post-fix, 235s wall, same failure) |
| 128 | 1,536 | 3,072 | — | — | — | — | **OOM** | 8529081 (2026-06-07), 616s wall — all-to-all buffer growth past 1,536 ranks |
| 256 | 3,072 | 6,144 | — | — | — | — | **OOM** | 8529509 (2026-06-07), 1,149s wall |

64N NO_OUTPUT likely the upstream `edp_mesh=None` SIGABRT regression
noted in CLAUDE.md "MoE SIGABRT". 128N+ is a genuine memory wall in
the all-to-all expert routing path. Open diagnosis task: needs the
upstream MoE expert-routing memory fix before we can scale past
128 nodes. Logs preserved at
`outputs/scaling_study_aurora/20260606_{170634,175221}/n{64,128}/light/moe_2b.log`
and `outputs/scaling_study_aurora/20260607_{144151,203508}/n{128,256}/light/moe_2b.log`.

### Aurora large-N — blocked (same `set_determinism` wall as agpt)

| Nodes | Status | Notes |
|-------|--------|-------|
| 1024 | CRASH | 8437390 (2026-04-16, 841s wall); 8529597 (2026-06-10, 102s wall) |
| 2048 | NO_OUTPUT | 8529598 (2026-06-08), 169s wall |
| 4096 | NO_OUTPUT | 8529599 (2026-06-08), 70s wall |

Same root cause as agpt: bare `ezpz launch` hits `set_determinism
std::bad_alloc` at 12k+ ranks. The production failover path
unblocks this for agpt; once that's wired into the scaling harness
we can retry MoE here too.

### moe_10b_2b / moe_7b — Aurora not retried (already OOM on Sunspot at 32N+)

Both larger MoE configs OOM'd at 32N on Sunspot; haven't retried on
Aurora yet. The 2026-06-07 sweep at n=8 caught one OK row for moe_10b_2b:

| Model | Nodes | GPUs | TPS/GPU | MFU | Status | Job |
|-------|------|------|---------|-----|--------|-----|
| moe_10b_2b | 1 | 12 | 844 | 3.98% | Complete | 8437408 (2026-04-15) |
| moe_10b_2b | 2 | 24 | 558 | 2.63% | Complete | 8437409 (2026-04-15) |
| moe_10b_2b | 4 | 48 | 636 | 3.00% | Complete | 8437410 (2026-04-15) |
| moe_10b_2b | 8 | 96 | 484 | 2.28% | Complete | 8437411 (2026-04-15) |
| moe_10b_2b | 32+ | — | — | — | OOM | 8437750 (2026-04-15) + later |

| Model | Nodes | GPUs | TPS/GPU | MFU | Status | Job |
|-------|------|------|---------|-----|--------|-----|
| moe_7b | 1 | 12 | 1,626 | 7.39% | Complete | 8437408 (2026-04-15) |
| moe_7b | 2 | 24 | 1,185 | 5.38% | Complete | 8437409 (2026-04-15) |
| moe_7b | 4 | 48 | 1,334 | 6.06% | Complete | 8437410 (2026-04-15) |
| moe_7b | 8 | 96 | 1,067 | 4.85% | Complete | 8437411 (2026-04-15) |
| moe_7b | 32+ | — | — | — | OOM | 8437750 (2026-04-15) + later |

## Results Directory

- Sunspot: `outputs/scaling_study/20260412_091635/`

## See Also

- [MoE configs](../configs/moe.md) — model architecture details
- [Experiment reports](../experiments/moe/) — per-run benchmark logs
