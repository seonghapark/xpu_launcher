# Scaling Results

Index + dashboard for scaling data across all models, machines, and
PyTorch versions. Detailed per-model tables, raw run dirs, and
workarounds live on the per-model pages.

## Quick Reference

| Model | Best TPS/GPU | Peak MFU | Validated Max N | Latest run |
|-------|-------------|----------|-----------------|------------|
| agpt_2b | 7,344 (4N) | 27.55% | 256 (clean), 128N just refilled 2026-06-06 | [agpt-2b.md](agpt-2b.md) |
| agpt_20b | 453 (64N) | 22.58% | 128 (2026-06-06) | [agpt-20b.md](agpt-20b.md) |
| agpt_80b | 100 (8N) | 18.3% | 32 | [agpt-80b.md](agpt-80b.md) |
| moe_2b | 7,121 (1N) | 11.9% | 64 | [moe.md](moe.md) |
| moe_7b | 1,841 (1N) | 8.4% | 16 | [moe.md](moe.md) |

## Per-Model Pages

- [agpt-2b.md](agpt-2b.md) — 2B dense model scaling (Sunspot torch 2.10
  + Aurora torch 2.13, up to 256N validated)
- [agpt-20b.md](agpt-20b.md) — 20B dense model scaling (Sunspot torch
  2.10 + Aurora torch 2.13, up to 128N validated)
- [agpt-80b.md](agpt-80b.md) — 80B dense model variants and TP sweeps
- [moe.md](moe.md) — MoE model scaling (2B and 7B)

## Aurora queue mapping (current)

| N | Queue | Walltime |
|---|-------|----------|
| 1–2 | debug | 1h |
| 3–256 | debug-scaling | 1h |
| 256–1,024 | prod → small | 12h |
| 1,025–1,919 | prod → medium | 12h |
| 1,920+ | prod → large | 12h |

## Reproducing

```bash
# Submit single-N scaling sweep on Aurora (LBS defaults to production-matched values)
qsub -A AuroraGPT -q debug-scaling -l walltime=01:00:00 -l select=<N> \
    -l filesystems=home:flare -N 2b-scale-n<N> -k doe -j oe \
    -v SCALING_GROUP=light,SCALING_DATASET=eliplutchok/fineweb-small-sample \
    torchtitan/experiments/ezpz/scripts/run_scaling_study_aurora.sh

# Aggregate results across runs
python3 torchtitan/experiments/ezpz/utils/aggregate_scaling.py \
    outputs/scaling_study_aurora/<TS1> outputs/scaling_study_aurora/<TS2> ...
```

## Raw run dirs

Per-run results.json + report.md + agpt_*.log under:
`outputs/scaling_study_aurora/<TIMESTAMP>/n<N>/<group>/`

## See Also

- [Per-run experiment reports](../experiments/agpt/) and [MoE per-run](../experiments/moe/)
- [yeet-env scaling](yeet_env/) — venv broadcast wall-clock by N
- [Known issues](../guides/known-issues.md)
