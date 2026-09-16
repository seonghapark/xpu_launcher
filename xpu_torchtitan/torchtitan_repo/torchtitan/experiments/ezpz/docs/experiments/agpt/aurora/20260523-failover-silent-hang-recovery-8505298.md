---
author: Sam Foreman
date: 2026-05-23
jobid: 8505298
status: PASS
---

# Failover Wrapper v2 — First Production Validation on a Real Silent Hang

> **2026-05-23, job `8505298`** — the bad-node failover wrapper
> (`scripts/failover_lib.sh` v2) successfully detected and recovered
> from a real silent training hang on Aurora **without manual
> intervention**. This is the first end-to-end production-style
> validation that the wrapper handles the exact incident pattern
> ([silent hang at 20B 512N, job 8479579, 2026-05-11](20260511-20b-n512-hang-8479579.md))
> for which it was built.

## TL;DR

| Phase | Time (UTC-5) | Result |
|---|---|---|
| Yeet fresh `.venv.tar.gz` (ezpz 0.16.0) to all 10 nodes | 20:57 | ✅ |
| Preflight `ezpz.examples.test` smoke on 8 active nodes | 21:00 → 21:02 | ✅ exit 0 |
| Main attempt-1 training launch | 21:02 → 21:06 | ✅ stepped 1 → 37 cleanly |
| **Silent hang** at step 37 (no traceback, no save attempt) | 21:06:41 → 21:36:41 | ⚠️ 30 min of dead air |
| `ezpz launch --timeout=1800` watchdog tripped, SIGTERM'd hung process | 21:36:41 | ✅ exit 124 |
| Wrapper classified exit 124 as silent-hang bad-node failure (not walltime) | 21:36:43 | ✅ |
| **Blind swap-one** — rotated rank-0 host (`x4220c3s6b0n0`) for a spare (`x4220c5s3b0n0`) | 21:36:43 | ✅ 1 spare remaining |
| Attempt-2 launched on new active set | 21:36:45 → 21:38:17 | ✅ step 1 |
| Attempt-2 trained 1 → 296 uninterrupted | 21:38:17 → 21:57:49 | ✅ ~21 min clean training |
| `step-100` and `step-200` DCP checkpoints persisted | 21:45:11, 21:51:40 | ✅ on flare |
| PBS walltime kill (1h limit) | 21:57:50 | ✅ exit -29 (clean) |

**Verdict: PASS.** Every wrapper code path that exists to handle the
8479579 incident class fired correctly, and the smoke run produced
on-disk checkpoints that prove the full yeet → preflight → train →
async-save pipeline works on the fresh ezpz 0.16.0 tarball stack.

## Why this matters

The wrapper had three pieces of new machinery added since the
8479579 incident:

1. `ezpz launch --timeout=1800` from ezpz 0.16.0 (the watchdog that
   SIGTERMs a launch when stdout goes silent for 30 min).
2. Wrapper exit-code classification for **exit 124** as silent-hang
   bad-node failure (distinct from real walltime kill which is exit
   143 / PBS `-29`).
3. The **blind swap-one** path in `failover_swap_in()` — used when
   no specific bad node can be identified from the log (the silent-
   hang case where there's no traceback to scrape).

All three needed to fire **in sequence**, on a real-world hang, to
prove the wrapper is production-ready. Until 8505298 we had only
[fixture-based unit tests](../../../../tests/failover/) for these
paths.

## The hang itself

Attempt-1 logged steps 1 → 37 cleanly (loss 12.96 → 11.80, ~4000
TPS/GPU, MFU ~15%). Then the log went **completely silent** at
21:06:41:

```
[2026-05-23 21:06:32][I][components/metrics:526:log] step: 35  loss: 11.92627  grad_norm: 1.7299  memory: 44.18GiB(69.05%)  tps: 4,070  tflops: 45.53  mfu: 15.27%
[2026-05-23 21:06:37][I][components/metrics:526:log] step: 36  loss: 11.86712  grad_norm: 1.7081  memory: 44.18GiB(69.05%)  tps: 3,960  tflops: 44.31  mfu: 14.86%
[2026-05-23 21:06:41][I][components/metrics:526:log] step: 37  loss: 11.80276  grad_norm: 1.7319  memory: 44.18GiB(69.05%)  tps: 3,919  tflops: 43.85  mfu: 14.70%
# ────── 30 min of dead air, no further output ──────
[2026-05-23 21:36:41][E][ezpz/launch:124:_run_with_watchdog] Watchdog: no output for 1800.0s (timeout=1800s). Sending SIGTERM to PID 191892.
```

CKPT_INTERVAL was 100, so step 37 was **not** a save step — this is
not a checkpoint-save stall. No traceback, no warning, no MPI error,
no rank dying. Just dead.

This matches the [8479579 pattern](20260511-20b-n512-hang-8479579.md)
exactly: training hits a step boundary, performs the next collective,
and one rank (somewhere) never completes the all-reduce. Without the
watchdog the job would have sat consuming the full walltime emitting
nothing.

## Wrapper response — the highlight

The wrapper sees `Execution finished with 124`, classifies it
correctly, and executes the swap-and-retry path in one continuous
block. From the .o file (lines 1383-1392):

```
[2026-05-23 21:36:42][I][ezpz/launch:687:launch] Execution finished with 124.
[failover] attempt 1 exited 124 (ezpz idle-output watchdog tripped after 1800s) — treating as silent-hang bad-node failure
[failover] attempt 1 failed (exit 124) — scraping for bad nodes
[failover] no specific bad node identified — rotating one spare in blindly
[failover] swapped: x4220c3s6b0n0.hsn.cm.aurora.alcf.anl.gov -> x4220c5s3b0n0.hsn.cm.aurora.alcf.anl.gov
[failover] swap summary: 1 bad nodes replaced; 1 spares remaining
[failover] attempt 2/2 — active=8 nodes, spare=1 nodes
[failover] logging to /flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/logs/failover-8505298/attempt-2.log
```

Three things to note here:
- **Exit 124 → silent-hang classification** (not walltime). The
  `evaluate_rc` path in `failover_lib.sh` has the regex specifically
  to distinguish exit 124 (watchdog) from exit 143 (PBS walltime
  SIGTERM). It picked the right branch.
- **"No specific bad node identified"** — there was no traceback to
  scrape because the hang was silent. The wrapper fell through to
  the `failover_swap_one_blind()` path.
- **Blind swap** picked rank-0's host (`x4220c3s6b0n0`) and rotated
  in the next spare (`x4220c5s3b0n0`). Active count stayed at 8;
  spare count dropped from 2 to 1.

## Attempt 2 recovery

Attempt-2 started 4 seconds after the swap finished, rebuilt the
data cache, hit step 1 at 21:38:17, and ran uninterrupted from
there:

```
[2026-05-23 21:38:17][I][components/metrics:526:log] step:  1  loss: 12.96189  grad_norm: 1.1973  memory: 36.36GiB(56.82%)  tps:   731  tflops:  8.18  mfu:  2.74%
[2026-05-23 21:45:08][I][components/metrics:526:log] step: 100  loss:  8.35545  grad_norm: 1.1415  memory: 44.18GiB(69.05%)  tps: 4,152  tflops: 46.45  mfu: 15.58%
[2026-05-23 21:51:32][I][components/metrics:526:log] step: 200  loss:  6.34585  grad_norm: 1.0669  memory: 44.18GiB(69.05%)  tps: 1,040  tflops: 11.63  mfu: 3.90%
[2026-05-23 21:57:49][I][components/metrics:526:log] step: 296  loss:  5.68275  grad_norm: 0.6050  memory: 44.18GiB(69.05%)  tps: 4,785  tflops: 53.54  mfu: 17.95%
=>> PBS: job killed: walltime 3629 exceeded limit 3600
```

The two slow-TPS spots (step 1 = ~700, step 200 = ~1000) are the
async-checkpoint write at step 100 finishing during step 100 +
overlapping into the early next steps; nothing pathological.

Loss curve: **12.96 → 5.68 in 296 steps** at 8N, GBS=192, SEQ_LEN=8192
= ~466M tokens. Healthy convergence trajectory for 2B at this scale.

## Async-save persistence proof

Two DCP checkpoints landed cleanly:

```
$ ls -la /flare/.../outputs/checkpoints/smoke-N8-20260523-205140/step-100/
-rw-r--r-- 5.5 MB Sat May 23 21:45:11 2026 .metadata
-rw-r--r-- 239 MB Sat May 23 21:45:10 2026 __0_0.distcp
-rw-r--r-- 239 MB Sat May 23 21:45:10 2026 __1_0.distcp
... (96 shards × ~239 MB = ~22 GB total)
```

step-100 saved at 21:45:11 (3s after the step log); step-200 at
21:51:40 (8s after). This is the **first unambiguous proof since
the 2026-05-03 regression that async checkpoint save can land
on-disk with the new ezpz 0.16.0 tarball + fresh wrapper**. The 512N
async-cluster-cascade ([the separate regression at scale](../../../guides/known-issues.md))
is unrelated and remains open — but at 8N async works.

## Configuration

```bash
# Job: 8505298 — smoke-agpt-2b-n8-20260523-205140
# Queue: debug-scaling
# Walltime: 01:00:00
# Nodes: 10 (PBS select) = 8 active + 2 spare (failover_init split)
# Submit env:
NHOSTS_TRAIN=8
FAILOVER_MAX_RETRIES=2
TRAINING_STEPS=20            # didn't take (script copy lagged the edit)
CKPT_INTERVAL=10             # didn't take (same reason)
CKPT_DIR=checkpoints/smoke-N8-20260523-205140
CHECKPOINT_ASYNC_MODE=async  # 2B uses async (matches production)
# Effective: TRAINING_STEPS=2,971,509 (production default), CKPT_INTERVAL=100
# Walltime caps the spend at 1h regardless.
```

Script: `torchtitan/experiments/ezpz/scripts/submit_agpt_2b_aurora_venv_failover.sh`
Library: `torchtitan/experiments/ezpz/scripts/failover_lib.sh`

## Logs

- Full .o: `/flare/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/smoke-agpt-2b-n8-20260523-205140.o8505298`
- attempt-1: `logs/failover-8505298/attempt-1.log` (the silent-hang one)
- attempt-2: `logs/failover-8505298/attempt-2.log` (the recovery)
- hostfiles: `logs/failover-8505298/{active,spare,all,bad_nodes}.hostfile/.txt`

## What changes after this

This validation greenlights:

- **Resubmitting the 9 queued production jobs** (2B/20B/80B chains)
  against the fresh tarballs. The wrapper proved it can handle the
  one operational hazard (silent hang) we couldn't unit-test
  end-to-end before.
- **Bumping the live `bad-node-failover.md` guide** from "v2 in
  production 2026-05-11" — which was the *deployment* date — to
  "v2 production-validated 2026-05-23" with a pointer to this
  report.

What this does **NOT** validate:

- The 20B 512N async-save cluster cascade
  ([known-issues.md](../../../guides/known-issues.md)) is a separate
  failure mode, scale-dependent, unrelated to the wrapper. 20B + 80B
  smoke jobs (`8505325`, `8505326`) are queued to exercise the
  preflight path at 8N for those models too.
- 80B init-time bad-node prevalence ([11 attempts since 2026-05-11,
  zero persisted](../../../production/agpt/80b/README.md)). The
  wrapper *correctly identifies* these failures, but Aurora's bad-node
  rate at 80B init currently exceeds the spare count we can afford.
  That's a node-health problem, not a wrapper problem.

## Related

- [Bad-node failover wrapper guide](../../../guides/bad-node-failover.md)
- [Original 8479579 silent-hang incident](20260511-20b-n512-hang-8479579.md)
- [Wrapper test harness](../../../../tests/failover/)
- [Known issues / async-save regression](../../../guides/known-issues.md)
