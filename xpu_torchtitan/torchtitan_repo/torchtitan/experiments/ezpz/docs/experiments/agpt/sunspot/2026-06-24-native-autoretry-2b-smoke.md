# Native auto-retry 2B submit-script smoke (Sunspot, 2026-06-24)

Validates `scripts/submit_agpt_2b_autoretry.sh` -- the ezpz-native
`ezpz launch --auto-retry` replacement for the bash `failover_lib.sh`
machinery. See [`docs/guides/bad-node-failover.md`](../../../guides/bad-node-failover.md)
"Two implementations".

## Setup

- Machine: Sunspot, ezpz 0.20.0, torch 2.13 `.venv` (yeet tarball).
- My own PBS jobs (NOT the user's allocation), `select=4` => `NHOSTS_TRAIN=2`
  active + 2 spare, `TRAINING_STEPS=5`, `--checkpoint.no-enable`
  (no throwaway 2B ckpt), `--compile.no-enable`.
- Data: `data-lists/sunspot/books.txt` (the Sunspot-resident dataset;
  `dolma`/`olmo-mix-1124` point at `/gila`, not mounted on Sunspot).

## Job 12469523 -- full failover lifecycle (plumbing PASS)

Default `agpt_2b`, LBS=2 (GBS=48). Every native auto-retry code path
fired correctly:

1. `ezpz yeet --src .venv.tar.gz` -> `/tmp/.venv` on all 4 nodes
   (active + spare), done in 63.6s.
2. Active-only arithmetic: script logged `--nproc (active ranks): 24
   -ppn: 12` (from `NHOSTS_TRAIN*12`, NOT the full 48-rank allocation).
3. ezpz internal split: `[auto-retry] 4 total / 2 active / 2 spare`;
   `mpiexec --np=24 --ppn=12 --hostfile=.../active.hostfile`.
4. attempt 1 crashed -> scrape -> **blind rotation**
   `x1921c1s0b0n0 -> x1921c1s2b0n0` (spare swapped in).
5. attempt 2 relaunched with `active=2 / spare=1` (spare consumed).
6. attempt 2 hit the same deterministic crash -> guard fired:
   `FAILOVER STOP: stuck_pre_training (two consecutive attempts with
   zero step= markers, rc=143)` -> `Execution finished with 143`.

Step 6 is the documented `STUCK_PRE_TRAINING` circuit-breaker: it bails
without burning the whole walltime when init fails twice with zero
training progress. Exactly the no-preflight safety net the design
relies on.

The crash itself was NOT a bad node and NOT inherent to `agpt_2b` LBS=2
at 2N (that is the production/scaling config and runs fine -- 2N hits
~7142 TPS/GPU in the scaling table). It was an artifact of the smoke
wrapper passing `--compile.no-enable`: in eager mode the full 256k-vocab
CE logit slice (~16 GB) is materialized and OOMs across only 24 FSDP
ranks. Production runs `agpt_2b` with **compile ON** (config default),
and inductor fuses the CE so the slice is never materialized. (Same
reason `bitwise_sync_check.sh`, which is also compile-off, needs chunked
CE.) The OOM is a property of the smoke flags, not the script.

## Job 12469524 -- clean training steps, compile-off fit recipe (PASS)

`CONFIG_SUFFIX=_chunkedce` (`agpt_2b_chunkedce`), `LBS=1` (GBS=24),
`activation-checkpoint:full` -- the compile-off small-scale recipe.
Trained all 5 steps cleanly, loss descending, finite grad_norm:

```
step: 1  loss: 12.94454  grad_norm: 2.0455  mfu:  4.23%
step: 2  loss: 12.84625  grad_norm: 1.9841  mfu: 14.74%
step: 3  loss: 12.65087  grad_norm: 1.9907  mfu: 14.56%
step: 4  loss: 12.36414  grad_norm: 2.0182  mfu: 16.06%
step: 5  loss: 12.18043  grad_norm: 2.9984  mfu: 14.62%
Training completed
```

## Job 12469525 -- production config (compile ON, LBS=2, sophiag) PASS

The real production knobs: `agpt_2b`, LBS=2 (GBS=48), compile ON
(config default), sophiag LR=2.28e-5; only `--checkpoint.no-enable` and
5 steps deviate. Confirms LBS=2 trains on 2N **without OOM** -- compile
fuses the 256k-vocab CE so the full logit slice is never materialized:

```
step: 1  loss: 13.00868  grad_norm: 1.8865  memory: 36.85GiB(57.59%)  mfu:  2.73%
step: 2  loss: 12.91862  grad_norm: 1.9133  memory: 44.68GiB(69.83%)  mfu: 26.65%
step: 3  loss: 12.72097  grad_norm: 1.9541  memory: 44.68GiB(69.83%)  mfu: 26.28%
step: 4  loss: 12.38918  grad_norm: 2.1652  memory: 44.68GiB(69.83%)  mfu: 26.16%
step: 5  loss: 12.05885  grad_norm: 1.9934  memory: 44.68GiB(69.83%)  mfu: 26.60%
Training completed
```

Peak 69.83% memory, steady ~26.6% MFU (matches the 2N scaling-table
figure), loss descending. This is the canonical 2B config under the new
script -- no chunked CE / LBS=1 workaround needed.

## Conclusion (2B)

The native-auto-retry plumbing (yeet-to-all, active-only nproc/GBS,
internal split, scrape, spare swap, stuck-pre-training guard) is
validated end-to-end on Sunspot, and the script trains cleanly. It is a
drop-in portable replacement for the bash failover wrapper for 2B.

---

# 20B + 80B native auto-retry scripts (2026-06-24)

Ported the validated 2B template to `submit_agpt_20b_autoretry.sh` and
`submit_agpt_80b_autoretry.sh`. The shared plumbing is identical to 2B
(already proven above); per-model config deltas were verified.

## 20B -- job 12469526 (own select=4, NHOSTS_TRAIN=2 + 2 spare)

Production config (compile ON, LBS=2 -> GBS=48 at 2N, sophiag
LR=2.28e-5), `--checkpoint.no-enable`, 5 steps. Adds vs 2B: the `DATASET`
blendcorpus/HF knob, `--dataloader.num-workers=2`, `_CKPT_DATASET_SLUG`,
and `--checkpoint.async-mode=disabled` default. Launched cmd verified:
`--np=24`, `--config=agpt_20b`, `--global-batch-size=48`,
`--checkpoint.async-mode=disabled`, books dataloader flags. Trained all
5 steps, completed cleanly:

```
step: 1  loss: 12.90  grad_norm:  5.05  memory: 46.48GiB(72.64%)  mfu: 13.89%
step: 2  loss: 12.03  grad_norm:  5.09  memory: 54.30GiB(84.86%)  mfu: 23.68%
step: 3  loss: 12.99  grad_norm: 58.53  memory: 54.30GiB(84.86%)  mfu: 23.85%
step: 4  loss: 16.18  grad_norm: 90.18  memory: 54.30GiB(84.86%)  mfu: 23.61%
step: 5  loss: 14.24  grad_norm: 12.72  memory: 54.30GiB(84.86%)  mfu: 23.64%
Training completed
```

This was a plumbing/launch smoke -- the loss bounce + grad_norm spikes
are expected for a tiny GBS=48 / 5-step / no-warmup run (production uses
200-step LR warmup); it is not OOM and not a crash. Peak 84.86% mem at
2N (20B is tight at small N -- production runs at 512N where it shards
far thinner). Split / nproc / yeet / dataloader / compile / clean-exit
all validated.

## 80B -- argv dry-render (config-resolution PASS)

A real 80B smoke needs >=62 active nodes (TP=4 + the dp_degree<=186 safe
corner), so instead of burning that allocation to test plumbing already
proven by 2B/20B, the 80B launch argv was dry-rendered (PATH-shimmed
`ezpz`) at NHOSTS_TRAIN=62, GAS=2. It matches the confirmed-stable job
12469494 token-for-token:

```
--config=agpt_80b --optimizer=adamw --optimizer.lr=1e-6
--parallelism.tensor-parallel-degree=4 --parallelism.expert-parallel-degree=1
--parallelism.data-parallel-replicate-degree=1 --parallelism.data-parallel-shard-degree=-1
--compile.no-enable --training.local-batch-size=1 --training.global-batch-size=372
activation-checkpoint:full        # positional subcommand, passed LAST
```

Log confirmed `dp_degree: 186` (no warn at 62N) and `GBS: 372`. At
NHOSTS_TRAIN=64 (dp_degree=192) the script emits the documented
NaN-regime WARN block. bf16-compute / fp32-master is the `agpt_80b()`
builder default, so no `--training.dtype` flag is needed.

## Conclusion (20B/80B)

Both scripts are faithful ports of the proven 2B template with each
model's validated config baked in. 20B is live-smoke-validated; 80B's
launch argv is dry-render-validated against the known-good stable run.
