# `agpt_80b` TP=2 4N Sunspot smoke (2026-06-02)

Second validation of the working 80B v2 config (`AdamW LR=1e-6, TP=2,
AC=full, compile=OFF, fp32-master`), this time on **Sunspot torch 2.13
+ the new xccl split_group workaround**. The May 5 baseline (job
12466025, Aurora 4N) established the recipe; this run confirms the
recipe still works after merging the workaround that makes nested
DeviceMesh creation safe under xccl.

## TL;DR

- **Exit 0**, 20 steps, **902 s wall** total (model init + 20 steps).
- `RESULT: SUCCESS — 80B v2 trains on torch 2.13 with compile=OFF`
  (auto-detected by the submit script's log-grep).
- xccl_split_group workaround installed and exercised:
  `Successfully created meshes with active dimensions: ['batch',
   'loss', 'tp', 'efsdp', 'fsdp']` — 5 nested PGs built without the
  `RuntimeError: No backend ... does not support splitting`.
- **Δloss: -2.55 nats (12.94 → 10.39)** across 20 steps — numerically
  matches the May 5 baseline's **-2.52 nats** (12.98 → 10.46).
- **Peak memory 56.91 GiB (88.94%)** — identical to May 5.
- **MFU 17.6% mean / 18.21% max** — matches May 5's ~17.8%.
- **grad-norm peaked at 33.10 (step 16)**, recovered to 12.68 by
  step 20 — same shape as May 5 (peak ~34 around steps 15-16). Both
  runs need the same 200-step warmup the 2B/20B configs use before
  production.

## Environment

| Field          | Value                                              |
|----------------|----------------------------------------------------|
| Date           | 2026-06-02                                         |
| Branch         | `ezpz` (HEAD `03c6f8b4a` + smoke commits)          |
| Workaround     | commit `8031d1d3a` (xccl split_group)              |
| Machine        | Sunspot                                            |
| Job ID         | 12467825                                           |
| Nodes          | 4                                                  |
| Devices        | 48 (Intel Max 1550, 12/node)                       |
| Steps          | 20                                                 |
| Config         | `agpt_80b` (TP=2, dp_shard=24, AC=full, compile=OFF) |
| Optimizer      | AdamW LR=1e-6, fp32 master                         |
| Backend        | xccl                                               |
| Torch          | `2.13.0.dev20260519+xpu` (.venv, py3.14)           |
| Checkpoint     | disabled (`--checkpoint.no_enable`)                |
| Compile        | disabled (`--compile.no_enable`)                   |
| W&B            | https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/a782sf8y |
| Tee log        | `logs/agpt-80b-no-compile-t213-12467825/run.log`   |
| PBS log        | `agpt-80b-no-compile-t213.o12467825`               |

## Parallelism

```
pp=1, dp_replicate=1, dp_shard=24, cp=1, tp=2, ep=1
```

Active mesh dimensions: `['batch', 'loss', 'tp', 'efsdp', 'fsdp']` —
TP axis present (TP=2), EP axis absent (EP=1), 5 PGs total
timeout-bound by the existing xccl-timeout shim (`Applied train
timeout 0:01:40 to 5 xccl ProcessGroup(s)`).

## Loss trajectory

| step | loss     | grad_norm | mem GiB | mfu    |
|------|----------|-----------|---------|--------|
|    1 | 12.94189 |   8.0486  | 53.00   | 11.57% |
|    2 | 12.91847 |   8.3099  | 56.91   | 17.20% |
|    3 | 12.85631 |   8.2043  | 56.91   | 17.48% |
|    4 | 12.77929 |   8.1875  | 56.91   | 17.80% |
|    5 | 12.63911 |   8.3575  | 56.91   | 17.82% |
|    6 | 12.52057 |   8.1444  | 56.91   | 17.66% |
|    7 | 12.32238 |   8.5679  | 56.91   | 17.80% |
|    8 | 12.16265 |   8.5748  | 56.91   | 17.66% |
|    9 | 11.85536 |   8.9725  | 56.91   | 17.60% |
|   10 | 11.58394 |   8.3782  | 56.91   | 17.81% |
|   11 | 11.49433 |  10.8564  | 56.91   | 18.03% |
|   12 | 11.33220 |  12.7709  | 56.91   | 17.77% |
|   13 | 11.26758 |  15.9550  | 56.91   | 18.11% |
|   14 | 10.99269 |  16.2705  | 56.91   | 18.01% |
|   15 | 10.94048 |  28.0267  | 56.91   | 18.12% |
|   16 | 10.94874 |  33.0977  | 56.91   | 18.01% |
|   17 | 10.75857 |  28.9873  | 56.91   | 18.21% |
|   18 | 10.82554 |  23.3147  | 56.91   | 17.88% |
|   19 | 10.67505 |  22.3449  | 56.91   | 17.70% |
|   20 | 10.39405 |  12.6759  | 56.91   | 17.80% |

Loss is monotonically descending except for noise at steps
{16,18} — same shape as the May 5 run. grad-norm grows steadily
from step 10 (8.38) through step 16 (33.10) then recovers — typical
of a no-warmup run; production should add the 200-step linear
warmup the 2B/20B v2 configs use.

## May 5 baseline (Aurora 4N, job 12466025) comparison

|             | May 5 (Aurora) | 2026-06-02 (Sunspot) |
|-------------|---------------:|---------------------:|
| Δloss       | -2.52          | **-2.55**            |
| Peak mem    | 88.94%         | **88.94%**           |
| Steady MFU  | ~17.8%         | **~17.8%**           |
| grad-norm peak | ~34 @ step 15-16 | **33.10 @ step 16** |
| Step time   | ~45 s          | **~42 s**            |

Numerically equivalent within run-to-run noise — the xccl
workaround does NOT regress the 80B TP=2 config, and Sunspot
matches Aurora's per-node throughput for the same config.

## Workaround validation

```
[ezpz/xccl_split_group_workaround:172:maybe_install_xccl_split_group_workaround]
  Installed xccl split_group workaround on DeviceMesh._init_one_process_group
  (upstream ProcessGroupXCCL has no supportsSplitting() override;
  see experiments/ezpz/xccl_split_group_workaround.py).
```

Followed by:

```
[distributed/parallel_dims:239:build_mesh] Successfully created meshes
  with active dimensions: ['batch', 'loss', 'tp', 'efsdp', 'fsdp']
```

And the xccl-timeout shim:

```
[ezpz/trainer:81:_set_pg_timeouts_xpu_aware] Applied train timeout
  0:01:40 to 5 xccl ProcessGroup(s) (upstream _set_pg_timeout has no
  xpu branch).
```

5 PGs timeout-bound — matches the active mesh axes (`tp`, `fsdp`,
`efsdp`, `batch`/`loss` flatten down).

## Pre-existing warnings observed (unrelated)

- `torch/distributed/tensor/_redistribute.py:371`: "While
  redistributing from (_NormPartial(2.0), _NormPartial(2.0)) to
  (Replicate(), Replicate()), 2 sequential all_reduce operations
  will be performed. This is suboptimal: ... To optimize, flatten
  mesh dimensions ['fsdp', 'tp'] so DTensor..." — well-known fsdp×tp
  redistribute that could be flattened upstream; doesn't affect
  correctness.

## Verdict

The 80B TP=2 + compile=OFF v2 path is **healthy on Sunspot torch
2.13** with the new xccl workaround in place. Loss / memory /
throughput numbers all reproduce the May 5 Aurora baseline within
noise. Ready for the next production attempt — pending the
follow-ups in the 80B production page (add warmup, file ALCF
ticket for x4101c5/c6 bad nodes).

## Open follow-ups

- Add a 200-step linear warmup to the 80B production config (the
  grad-norm climb to 33.10 by step 16 is the same shape as the May 5
  run and would benefit from warmup before scaling out to 256N).
- Mirror the `SUBMIT_DIR` + `ezpz yeet` fixes (this run also fixed
  `submit_80b_no_compile_t213.sh` — commit `03c6f8b4a`) into
  `scripts/submit_agpt_80b_aurora_venv_failover.sh` for production.
- Cross-cluster validation: re-run the same smoke on Aurora 4N to
  confirm xccl workaround behaviour is identical there.
