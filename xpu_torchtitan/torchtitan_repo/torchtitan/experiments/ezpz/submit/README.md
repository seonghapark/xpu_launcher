# `submit/` — Legacy v1 Submit Scripts

> **Status: legacy / historical.** No current production training uses
> these scripts. They are kept for reference (to reproduce or inspect
> the v1 bf16-master runs) and are linked from the v1 sections of the
> per-node-count READMEs under `docs/production/agpt/`.

## Current production scripts

All current (v2) production training is driven by the **failover**
submit scripts:

- [`scripts/submit_agpt_2b_aurora_venv_failover.sh`](../scripts/submit_agpt_2b_aurora_venv_failover.sh)
- [`scripts/submit_agpt_20b_aurora_venv_failover.sh`](../scripts/submit_agpt_20b_aurora_venv_failover.sh)
- [`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../scripts/submit_agpt_80b_aurora_venv_failover.sh)

One script per model handles all node counts (256/512/1024) via env
vars (`NHOSTS_TRAIN`, `FAILOVER_MAX_RETRIES`, `LBS`, `CKPT_DIR`,
`CHECKPOINT_ASYNC_MODE`, …) and runs on the **torch 2.13 venv**
(`.venv/` + `ezpz yeet-env` tarball broadcast). The wrapper splits the
PBS nodefile into active + spare nodes, runs a preflight smoke, and
swaps in spares on a bad-node hit.

> The non-failover `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` were
> **removed 2026-06-24** — superseded by the failover versions above.
> Recover from git history if a pre-failover dispatch ever needs
> reproducing.

## Why these are legacy

The `submit/{aurora,sunspot}/*.sh` scripts use the **torch 2.10 conda
stack** (`ezpz_setup_env`, `frameworks/2025.3.1`) and were the
production driver from the migration through 2026-04-29. They are the
scripts that produced every v1 trajectory (bf16 master, frozen
RMSNorm) — see
[`../docs/guides/training-dtype-bf16-norm-freeze.md`](../docs/guides/training-dtype-bf16-norm-freeze.md)
for the bug that triggered the v2 restart.

After 2026-04-30 (v2 restart on `--training.dtype=float32`),
production moved to the torch 2.13 venv stack under `scripts/`, and
nothing live calls into this directory anymore.

## What's in here

| File | What it ran |
|------|-------------|
| `aurora/submit_agpt_2b.sh` | v1 2B at 256N |
| `aurora/submit_agpt_2b_n512.sh` | v1 2B at 512N |
| `aurora/submit_agpt_2b_n1024.sh` | v1 2B at 1024N |
| `aurora/submit_agpt_20b.sh` | v1 20B at 256N |
| `aurora/submit_agpt_20b_n512.sh` | v1 20B at 512N |
| `aurora/submit_agpt_20b_n1024.sh` | v1 20B at 1024N |
| `aurora/submit_benchmark_80b.sh` | v1 80B benchmark sweep |
| `sunspot/*` | Sunspot mirrors of the same v1 scripts |

Don't add new scripts here. New per-node-count entry points should go
under `scripts/`.
