---
author: Sam Foreman
date: 2026-06-26
jobid: 8564220,8564221,8567005,8567614,8566938,8567189,8567358,8567461,8567626,8567781,8568429
status: MIXED (ckpt-resume incident recovered + verified)
---

# Breaking 512N queue starvation: sneak jobs, the multi-chain umbrella, and walltime-aware checkpointing

> **2026-06-26, Aurora.** The 512N canonical chains (2B + 20B) had sat in
> the `small` queue ~21 days (since 2026-06-04) with no slot -- a 522-node
> scatter request rarely wins against the crowd. This session attacked that
> three ways: (1) short "sneak" jobs that backfill into gaps, (2) a
> multi-chain "umbrella" job that packs all 4 canonical chains into one
> `medium`-queue allocation, and (3) walltime-aware checkpointing so any
> short job always saves before its walltime expires. Along the way it
> surfaced + fixed three production bugs.

## TL;DR

| Thread | Outcome | Evidence |
|---|---|---|
| Short 2h 512N sneak jobs land where 12h heads can't | **TACTIC WORKS** | 8564220 + 8564221 both went Q->R after 21 days stuck |
| First 20B sneak trained but saved nothing (interval too coarse) | **FAILURE MODE FOUND** | 8564221: step 4400->4489, walltime SIGTERM, 0 new ckpt |
| Walltime-aware checkpointing | **BUILT + VERIFIED** | 8567781: forced save fired at step 9 / elapsed 321s, valid ckpt, rc=0 |
| Multi-chain umbrella (1536 train nodes -> medium queue) | **BUILT + VALIDATED** | 8567461: trainer reached train->checkpoint->rc=0 under 4-way concurrency |
| 2b-v2 venv missing spmd_types | **FOUND + FIXED** | rebuilt .venv.tar.gz (2.7G, 56 spmd entries) |
| First real umbrella production submit | **QUEUED** | 8568429, 1576N, 12h, medium |
| Clone-sync broke DCP resume (optim-statedict format migration, #3623/#3269) | **CAUSED + RECOVERED + VERIFIED** | rolled 3 clones back to pre-migration 1263e5a1; load-test 8570407 loaded step-86200 in 32.6s, exit 0 |

## 1. The 512N starvation + the sneak tactic

Both 512N chains (`8521631` 2B cont10, `8521632` 20B cont11) had been
eligible since 2026-06-07 with `estimated_start = NONE`, comment "Not
enough free nodes available". `eligible_time` had maxed (~460h / 406h) --
priority accrual was no longer the lever; the 522-node scatter simply
rarely fits.

**Insight:** a *short-walltime* 512N job is far more backfill-eligible
than the 12h heads (`backfill_max=50`, `backfill_factor=84600` are active
in `small`). Submitted 2h sneaks resuming the SAME canonical ckpt dir
(CKPT_DIR is model/node/gbs-derived, so a sneak advances the chain, not a
fork):

| Job | Chain | Walltime | Result |
|---|---|---|---|
| 8564220 | 2B 512N | 2h | Q->R, but died in preflight (CCL-KVS timeout, see #2); 0 progress, chain safe at step-30400 |
| 8564221 | 20B 512N | 2h | Q->R, trained step 4400 -> 4489, but **saved nothing** (see #3) |
| 8567005 | 2B 512N | 2h | resubmit of 8564220; queued |
| 8567614 | 20B 512N | 2h | resubmit with CKPT_INTERVAL=25 + walltime backstop; queued |

The window matters: both landed during a transient drain (free nodes
swung 184 <-> 9663 within ~20 min), then the machine refilled and later
sneaks sat queued. The tactic is real but opportunistic.

## 2. Failure mode: CCL/PMI KVS-timeout during init

The first 2B sneak (8564220) died on attempt 1 with
`|CCL_ERROR| pmi_resizable_simple_internal.cpp kvs_get_value: KVS get
error: timeout` -> `comm_create error` -> `Fatal Python error:
Segmentation fault` in `libccl.so`, during collective init at 6144 ranks.
The failover wrapper burned all 3 retries swapping one spare each time,
but a KVS timeout is system-wide (PMI/pals load under the busy
post-drain machine), not a single bad node -- swaps cannot fix it. Same
class as the pals-RPC transient; the fix is resubmit. The sibling 20B
sneak on a different node slice did NOT hit it.
(See `project_ccl_kvs_timeout_init_crash` memory.)

## 3. Failure mode: short jobs save nothing -> walltime-aware checkpointing

20B sneak 8564221 trained cleanly from step 4400 but at ~48s/step, and
the default `checkpoint.interval=100` puts the next save at step 4500 =
~80 min of training. With ~45 min startup (yeet + preflight + 244GB DCP
load) eating into the 2h window, it reached only **step 4489** before the
walltime SIGTERM (exit -29) -- missing step-4500 by ~9 min, saving
nothing. ~2h of 512N compute, zero on-disk progress. (Chain ckpt
verified intact at step-4400 -- the failed attempt never corrupted it.)

**Fix: walltime-aware checkpointing** (commit `3d9044993`).
`FaultTolerantTrainer.train()` now watches a monotonic clock from loop
entry; once elapsed reaches `walltime_seconds - walltime_checkpoint_margin_seconds`
it forces a final checkpoint (reusing the `last_step` interval-bypass +
`maybe_wait_for_saving()` to flush async) and stops cleanly.
`walltime_seconds=0` (default) disables it -- exact prior behavior. The
failover submit scripts export REMAINING walltime via a new
`failover_remaining_walltime()` helper (qstat Resource_List.walltime minus
elapsed); `mpiexec --envall` propagates it to all ranks.

**Verification (8567781, 2N debug-scaling, 20b clone):** with
`WALLTIME_SECONDS=900`, default margin 600 (so wall_limit=300s), and
`CKPT_INTERVAL=99999` to isolate the walltime path as the ONLY way to
save:
```
walltime budget reached at step 9 (elapsed ~321s of 900s, margin 600s);
forcing final checkpoint and stopping
-> Training completed -> Execution finished with 0
-> VALID forced-save ckpt: step-9 (.metadata present)
```
Every assertion held: trigger fired just past the 300s limit, the save
bypassed the never-firing interval, the checkpoint landed valid, clean
rc=0. (See `feedback_short_jobs_need_short_ckpt_interval` memory for the
interim CKPT_INTERVAL=25 mitigation.)

## 4. The multi-chain "umbrella" job

The durable answer to per-chain starvation: pack all 4 canonical chains
(2B/20B x 512N/256N = **1536 train nodes**) into ONE PBS job. 1536 routes
to the `medium` queue (1025-1999 band), which is far less contended than
`small` (21-day starved) or `large` (an incumbent holds ~10k nodes + 33
queued). NB: a literal 2048N would route to that jammed `large` queue --
1536 is both the natural chain sum AND the better-scheduling size.

`scripts/submit_agpt_multi_aurora_venv_failover.sh` (commit `b66ba7235`)
slices `PBS_NODEFILE` into 4 disjoint hostfiles and launches the existing
per-chain submit scripts as 4 concurrent background subprocesses, each
fully isolated via env (PBS_NODEFILE slice, FAILOVER_LOG_DIR,
QUEUE_WAIT_CSV, MASTER_PORT, PBS_O_WORKDIR, CKPT_DIR). DRY_RUN +
MULTI_PROFILE=tiny test paths. No core torchtitan changes.

**Four tiny-scale smokes (debug-scaling) each caught a real bug** -- the
exact reason to test concurrency small before a 1576-node submit:

| Smoke | Bug found | Fix |
|---|---|---|
| 8566938 | `ezpz yeet` local-copy always runs on the shared mom node at a fixed /tmp dst -> 4 concurrent yeets corrupt each other (rc=127) | pre-stage each venv ONCE, serially, to per-model /tmp/.venv-<model>; children get SKIP_YEET=1 + VENV_DST |
| 8567189 | clones ran STALE children (edits were only in main repo) | git pull all 3 v2 clones |
| 8567358 | umbrella `wait` saw trainers as non-children -> spurious rc=127 the instant they launched | launch_trainer was called via `$(...)` (command-substitution subshell); background directly + set PIDS[i]=$! (commit `514586276`) |
| 8567461 | (validation) | trainer 1 reached preflight->mesh->ckpt-load->train->SAVE->rc=0; real per-trainer rc 0/1/143/143; failure isolation confirmed (t1 finished while siblings died on tiny-scale out-of-spares) |

8567461 wrote `multi-smoke-8567461/t1-20b-n2/step-3/.metadata` -- a valid
checkpoint from one of 4 concurrent isolated trainers in a single
allocation. The 3 "failures" were transient bad nodes at the tiny 2N /
1-spare scale (non-issue at production SPARES=10).
(See memories `project_yeet_single_job_per_alloc`,
`feedback_pull_v2_clones_after_main_changes`.)

## 5. Bug: 2b-v2 venv missing spmd_types

The walltime test's first attempt (8567626, run from the 2b-v2 clone)
failed with `ModuleNotFoundError: No module named 'spmd_types'`
(`agpt/sharding.py:17` imports it). Root cause: the task-#183 spmd_types
fix was applied ONLY to the 20b-v2 clone; the 2b-v2 `.venv` AND its
2026-05-23 `.venv.tar.gz` both lacked it. It went unnoticed because the
2B 256N chain reuses warm `/tmp/.venv` on production nodes; only a
FRESH-node debug-scaling job exposed it.

**Fix:** `uv pip install --no-deps --no-cache --link-mode=copy
spmd_types==0.2.1` into the 2b-v2 `.venv` (verified `spmd.I` imports),
then rebuilt `.venv.tar.gz` (pigz; 2.7G, 56 spmd_types entries; old one
backed up `.bak-no-spmd-20260626-104344`). Unblocks the 2B chain, 2B
sneak, and umbrella 2B trainers on fresh nodes.
(See `project_2b_v2_venv_missing_spmd_types` memory.)

## 6. Incident: clone-sync broke DCP resume; rolled back + verified

While shipping the umbrella/walltime code, the v2 clones were `git pull`'d
from their old HEAD `1263e5a1` (2026-05-27) to today's `b66ba723`. That
crossed an **optimizer state-dict FORMAT migration** -- upstream
**#3623 "[Checkpointer] Remove the dependencies on PyTorch distributed
state_dict APIs" (772dd1b6c)** plus **#3269 "support mixed optimizers"
(632f67f12)** -- which swapped `get/set_optimizer_state_dict` for
`get_flat_optim_state_dict` / `load_flat_optim_state_dict` (a different,
flattened serialization, not a key rename). Every production checkpoint was
saved with the old nested format, so resume on the new code crashed:
```
RuntimeError: Missing key in checkpoint state_dict:
optimizer.param_groups.tok_embeddings.weight.fused.
```
The 2B + 20B 512N sneaks (8567005 / 8567614) exposed it -- first jobs to
actually resume on the new code. ALL chains were affected (same clones,
pre-migration ckpts). Root cause: a clone-pull updates the TRAINING code
too, not just the infra files intended.

> **Correction (2026-06-28):** initially attributed (this session, in flight)
> to #3714 "Enable Fused qkv by default" and then to the optimizer
> `implementation="fused"` flag. Both were ruled out by direct evidence
> (`implementation` defaulted to `fused` in BOTH the pre and post commits;
> FusedQKV hooks only touch MODEL state, never `tok_embeddings` optimizer
> state). The verified cause is the #3623/#3269 flat-optim-state migration.
> Full diagnosis + the `git diff` reproduction:
> [`docs/guides/known-bugs/pre3623-optim-statedict-resume.md`](../../../guides/known-bugs/pre3623-optim-statedict-resume.md).

**Recovery (per-clone, reversible):** for each of agpt-2b-v2, agpt-20b-v2,
agpt-20b-n256: created a `rollback-safety-<ts>` branch, `git reset --hard
1263e5a1` (pre-fused, the proven-working commit), then re-applied this
session's ckpt-safe infra (walltime checkpointing, umbrella + skip-yeet
hooks, failover_remaining_walltime, umbrella wait-fix) via cherry-pick
(2b-v2) / file-copy from the fixed clone (20b clones). Verified per clone:
#3714 absent, files compile, infra markers present. **Checkpoints were
never touched -- only code moved.**

**Verification (8570407, 8N debug-scaling):** loaded the real 2B-256N
step-86200 with the rolled-back code using `TRAINING_STEPS=86200` so the
loop exits immediately after a full model+optimizer load (the exact crash
path) -- no training, no save, read-only on the chain.
```
[18:22:05] Loading the checkpoint from ./outputs/checkpoints/agpt-2b-...n256...
[18:22:38] Finished loading the checkpoint in 32.62 seconds.
[18:22:40] Training completed   [18:22:42] Execution finished with 0.
```
PASS: no Missing-key, exit 0, chain still tops out at step-86200 (nothing
written). DCP resharding 256N->8N also confirmed. The production chains
will now resume cleanly.

**Lesson** ([[feedback_clone_pull_can_break_ckpt_resume]]): to ship infra to
a production clone, do NOT blanket `git pull` -- pin the training code at
the commit matching its checkpoints and only `cp` / `checkout` the specific
infra paths. The durable fix is a checkpoint-compat shim (fused-key
migration) on the up-to-date branch (see What's open).

## What's open / next

- **First real umbrella submit `8568429`** (1576N, 12h, medium) -- queued,
  awaiting a medium-sized opening or the post-PM flood.
- **Maintenance window Mon 2026-06-29 06:00 -> Wed 07-01 03:30 UTC.** The
  post-PM flood is the umbrella's best landing window; short 2h sneaks
  stay launchable ~10h longer into the pre-PM drain than 12h heads.
- 2B + 20B 512N sneaks (8567005 / 8567614) queued, now both able to
  capture progress (2B venv fixed; 20B interval=25 + walltime backstop).
- The TP=2/LBS>1 80B grad-path bug and the >512N init crash remain open
  (separate threads, not touched here).
- **Checkpoint-compat shim (planned):** the production clones are now pinned
  pre-#3714 so they resume existing ckpts. To eventually return them to
  current code, a DCP compat shim must migrate pre-fused optimizer state
  (`optimizer.param_groups.X.weight`) to the fused layout
  (`...weight.fused`) on load, with a test asserting bit-identical
  loss/grad_norm pre/post migration. Until that exists + is validated, keep
  clones pinned pre-#3714.
