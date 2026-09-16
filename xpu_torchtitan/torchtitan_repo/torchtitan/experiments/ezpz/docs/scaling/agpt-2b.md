# AuroraGPT-2B Scaling

## Aurora Weak Scaling (torch 2.13 + `ezpz yeet-env` tarball)

> Single sweep across 4 → 256 nodes with `ezpz yeet-env` tarball
> broadcast (the production env-setup pattern). All entries are 20-step
> bench runs (`BENCH_STEPS=20 SCALING_GROUP=light`).
> **Config:** FSDP-only (TP=1), `compile=on`, AC=full, seq_len=8192,
> `LBS=2` to match the production submit script (GBS = N × 12 × 2).

| Nodes | GPUs | GBS    | TPS/GPU | Total TPS  | MFU    | Efficiency vs 4N | Job |
|-------|------|--------|---------|------------|--------|------------------|-----|
| 4     | 48   | 96     | 7,344   | 352,512    | 27.55% | 100.0%           | 2026-05-29 sweep |
| 8     | 96   | 192    | 7,291   | 699,936    | 27.36% | 99.3%            | 2026-05-29 sweep |
| 16    | 192  | 384    | 6,803   | 1,306,176  | 25.53% | 92.6%            | 2026-05-29 sweep |
| 32    | 384  | 768    | 6,984   | 2,681,856  | 26.20% | 95.1%            | 2026-05-29 sweep |
| 64    | 768  | 1,536  | 6,083 (6,553) | 4,671,744 | 22.82% (24.59%) | 82.8% (89.3%) | [8529046](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ihiy4ej1), [8528940](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/_) (2026-06-06) |
| 128   | 1,536| 3,072  | 4,934   | 7,578,624  | 18.51% | 67.2%            | [8529081](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/m9i0long) (2026-06-07) |
| 256   | 3,072| 6,144  | 5,002   | 15,366,144 | 18.77% | 68.1%            | 2026-05-29 sweep |
| **512 (LBS=1)** | 6,144| 6,144 | 1,988 | 12,214,272 | 7.46% | 27.1% | 8437389 (2026-04-16, GBS=6,144 because that sweep ran with LBS=1) |
| 512 (LBS=2) | 6,144 | 12,288 | — | — | — | — | Bare-launch path blocked: `set_determinism` `std::bad_alloc` at 6,144 ranks. Production failover path works (we have a 2B 512N production chain running). Retry via failover wrapper pending. |
| 1,024 | 12,288| 24,576| —       | —          | —      | —                | Blocked: `set_determinism` init crash, see [project_1024n_init_crash](.) |
| 2,048 | 24,576| 49,152| —       | —          | —      | —                | Blocked: same as 1,024 |
| 4,096 | 49,152| 98,304| —       | —          | —      | —                | Blocked: same as 1,024 |

64N has two LBS=2 data points (8528940 ran with the prior wrapper
that still had agpt_20b at LBS=1; 8529046 ran with both at LBS=2 and
the spmd_types-fixed venv). Numbers in parens are from 8528940.

**Headline:** Small-N (4–32) lands at **~27% MFU** (matches the torch
2.13 Sunspot target and well above the torch-2.10 Sunspot baseline
below). At N=256 throughput drops to 18.77% MFU (≈68% efficiency vs
4N). 64N at LBS=2 hits 22.8–24.6% MFU (in line with the small-N
trend); 128N drops to 18.5% MFU.

### Historical: the n=64 / n=128 CRASH era (2026-05-29 → 2026-06-06)

The 64N / 128N cells used to read `CRASH` because of five stacked
bugs in the scaling wrapper. All resolved by 2026-06-06:

1. **`.venv.tar.gz` rebuild lost `.venv/bin/`** (empty in tarball
   even though present in source venv). Rebuilt manually.
2. **Wrapper's trailing `"$@"`** on the inner `ezpz launch python3 -m
   torchtitan...train` invocation leaked PBS `-v` CLI args straight
   into the training entry point, causing instant arg-parse failure
   (NO_OUTPUT wall <60s). Removed (commit `6c6235fdf`).
3. **blendcorpus segfault at ≥768 ranks** in
   `blendcorpus_builder.py:275 __init__`. Bypassed by adding a
   `SCALING_DATASET` env knob so the sweep can run against an HF
   streaming dataset (sweeps now default to
   `eliplutchok/fineweb-small-sample`).
4. **`qsub -- /bin/bash -c "..."` swallowed the `#!/bin/bash --login`
   shebang**, leaving `module` undefined on the PBS-spawned shell →
   `module load oneapi/release/2025.3.1` silently failed → oneAPI MPI
   binaries (`mpiexec`, `qstat`) not on PATH →
   `from sh import qstat` ImportError. Fix: submit the script
   directly (`qsub <script>`), not via `bash -c`.
5. **PATH-order race** propagated rank-N `python3` ahead of
   `/tmp/.venv/bin`, so the rank-N `python3 -m torchtitan...train`
   imported a system Python with no ezpz. Pinned the inner command to
   `${VIRTUAL_ENV:-/tmp/.venv}/bin/python3`.

512N+ NO_OUTPUT is a distinct failure mode: at ≥6,144 ranks the
init wall is an XCCL communicator segfault or `set_determinism
std::bad_alloc`; see [`memory/project_1024n_init_crash.md`](.). The
yeet helped at N=256 but doesn't fully clear the larger-N init
scaling wall.

**Raw run dirs:**

- `outputs/scaling_2b_aurora/20260529_081759/` (n=4/8/16/32 OK)
- `outputs/scaling_2b_aurora/20260529_075349/` (n=256 OK)
- `outputs/scaling_study_aurora/20260606_170634/n64/` (LBS=1 OK)
- `outputs/scaling_study_aurora/20260606_175221/n128/` (LBS=1 OK)

## Sunspot Weak Scaling (torch 2.10, 1–64 nodes, April 2026)

Historical reference — pre-`yeet-env` torch-2.10 baseline.

| Nodes | GPUs | TPS/GPU | Total TPS | MFU | Memory | Efficiency |
|-------|------|---------|-----------|-----|--------|------------|
| 1 | 12 | 6,457+/-38 | 77,490+/-466 | 24.2% | 47.29 GiB (74%) | 100.0% |
| 2 | 24 | 5,529+/-71 | 132,708+/-1714 | 20.8% | 46.80 GiB (73%) | 85.6% |
| 4 | 48 | 5,580+/-81 | 267,864+/-3903 | 20.9% | 46.66 GiB (73%) | 86.4% |
| 8 | 96 | 5,384+/-173 | 516,912+/-16631 | 20.2% | 46.58 GiB (73%) | 83.4% |
| 16 | 192 | 5,032+/-44 | 966,240+/-8553 | 18.9% | 46.42 GiB (73%) | 77.9% |
| 32 | 384 | 5,091+/-118 | 1,954,944+/-45616 | 19.1% | 46.41 GiB (73%) | 78.8% |
| 64 | 768 | 4,783 | 3,673,344 | 17.9% | 46.38 GiB (72%) | 74.1% |

**Config:** FSDP only (TP=1), compile=on, AC=full, seq_len=4096, LBS=1

**Results directory:** `outputs/scaling_study/20260412_091635/`

## See Also

- [Experiment reports](../experiments/agpt/) — per-run benchmark logs
- [Production training](../production/agpt/2b/) — live training status
