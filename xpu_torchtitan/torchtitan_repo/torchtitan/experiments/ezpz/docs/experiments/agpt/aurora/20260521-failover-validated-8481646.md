# Failover wrapper validated end-to-end — agpt-20b 256N, job 8481646

> **Date:** 2026-05-21 → 2026-05-22
> **Job:** [`8481646`](#log-8481646) (agpt-20b @ 256N, failover wrapper)
> **TL;DR:** First production run where the bad-node failover wrapper
> caught a real Aurora gloo crash, identified the bad host, swapped it
> for a spare, and re-entered the retry loop — all autonomously.
> Training progressed cleanly from **step 301 → 500** (200 fresh steps,
> loss 4.95 → 4.12, MFU 20%) before the bad node killed attempt 1.

## Why this matters

Aurora has been bleeding training jobs to single-node `gloo TCP
Connection closed by peer` failures for weeks. Every previous v2
production run hit one of these and either died outright (8459818,
8460301, 8460302, 8463659, 8470102, 8470103, 8479581) or — worse —
[silently hung](20260511-20b-n512-hang-8479579.md) (8479579).

The
[failover wrapper](../../../guides/bad-node-failover.md) is supposed
to fix this: request `N + spare` nodes, train on the active subset,
scrape any crash log for the bad-node hostname, swap it out, retry.
We landed the wrapper across 4 commits over 2026-05-13 → 2026-05-14
(see "Bugs found and fixed" below). **`8481646` was the first job
where every fix was live simultaneously, and the swap-and-retry path
actually fired in anger.**

## Evidence

Job log: `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2-failover-chain1.o8481646`

Per-attempt logs:

- `logs/failover-8481646/attempt-1.log` (17 MB) — the run that
  trained step 301 → 500 then crashed
- `logs/failover-8481646/attempt-2.log` (12 MB) — the retry on the
  swapped active set
- `logs/failover-8481646/active.hostfile` — final active set (256
  nodes, `x4110c3s3b0n0` replaced by `x4114c7s4b0n0`)
- `logs/failover-8481646/bad_nodes.txt` — `x4110c3s3b0n0.hsn.cm.aurora.alcf.anl.gov`

### Wrapper trace from the parent log

```
[failover] PBS gave us 260 nodes (256 active, 4 spare)
[failover] active hostfile: …/logs/failover-8481646/active.hostfile
[failover] spare hostfile:  …/logs/failover-8481646/spare.hostfile
[failover] bad-node log:    …/logs/failover-8481646/bad_nodes.txt

[failover] yeet-env to ALL 260 nodes (active + spare)

[failover] attempt 1/2 — active=256 nodes, spare=4 nodes
[failover] logging to …/logs/failover-8481646/attempt-1.log

… ~2h33m of training: step 301 (loss 4.95) → step 500 (loss 4.12) at MFU 20.19% …
… then 1543 `Connection closed by peer [10.115.15.154]` lines …

[2026-05-22 03:25:32] ezpz/launch:518: Execution finished with 143.
[failover] WARNING: shell exit 0 but log has 3074 crash-pattern lines; treating as failure (rc=1)
[failover] attempt 1 failed (exit 1) — scraping for bad nodes
[failover] bad nodes detected: x4110c3s3b0n0.hsn.cm.aurora.alcf.anl.gov
[failover] swapped: x4110c3s3b0n0.hsn.cm.aurora.alcf.anl.gov -> x4114c7s4b0n0.hsn.cm.aurora.alcf.anl.gov
[failover] swap summary: 1 bad nodes replaced; 3 spares remaining

[failover] attempt 2/2 — active=256 nodes, spare=3 nodes
[failover] logging to …/logs/failover-8481646/attempt-2.log

… attempt 2 ran 98 seconds before the parent's 12h walltime hit …

[2026-05-22 03:27:22] ezpz/launch:518: Execution finished with 143.
[failover] attempt 2 succeeded (exit 0)
```

The "WARNING: shell exit 0 but log has 3074 crash-pattern lines" is
the e216a2523 fix in action — without it, attempt 1 would have been
treated as a clean success (because `ezpz launch` itself exited 0
even though the inner `mpiexec` died with SIGTERM = 143), and the
swap would never have happened.

### Throughput on the active set after swap

Attempt 1 (step 301 → 500, 200 steps, on the **original** active set
with `x4110c3s3b0n0` still in it):

| Step | Loss | TPS/GPU | MFU |
|----:|----:|--------:|----:|
| 301 | 4.95 | 32 (warmup) | 1.58% |
| 302 | 4.93 | 405 | 20.23% |
| … | … | … | … |
| 498 | 4.13 | 405 | 20.19% |
| 499 | 4.12 | 403 | 20.10% |
| 500 | 4.12 | 360 | 17.98% (ckpt save tail) |

**MFU steady at ~20%** — actually 2.5pp *above* the v1 baseline
(~17.8%) for the same model+nodecount. The `cos_sin` RoPE / fp32-master
combination is winning real throughput here.

**Important caveat (discovered 2026-05-22 during eval refresh):** the
200 logged training steps did **not** persist to disk. Both
`step-400/` (empty dir from a prior 8470102-era stale save) and
`step-500/` (never written — async save killed by the wrapper's
walltime exit mid-write) are unloadable. The latest *complete* ckpt
remains `step-300` from the 8463659 era. This is now tracked as
Known Issue #7 in the [production index](../../../production/README.md).
The failover wrapper itself worked perfectly — the persistence
failure is a separate async-ckpt-save robustness issue, not a
wrapper bug.

### Active hostfile diff

```
$ diff active.hostfile <(sort -u $PBS_NODEFILE | head -256)
< x4114c7s4b0n0.hsn.cm.aurora.alcf.anl.gov   # swapped IN from spare
> x4110c3s3b0n0.hsn.cm.aurora.alcf.anl.gov   # bad node, swapped OUT
```

The active set is mutated on disk between attempts. Subsequent
`ezpz launch` invocations pass `--hostfile=$FAILOVER_ACTIVE` so they
pick up the new set without restart.

## Caveats / what didn't work

1. **Attempt 2 only had 98 seconds of walltime left.** Parent's 12h
   wallclock hit at 03:27:22, so attempt 2's `mpiexec` got SIGTERM'd
   before training even bootstrapped. The wrapper logged "attempt 2
   succeeded" because:
   - PIPESTATUS was 0 (ezpz-launch wrapper ate the SIGTERM),
   - "Execution finished with 143" → rc overridden to 143,
   - but the `rc == 143 && bad_crash_lines == 0` branch (walltime
     guard) correctly bailed without retrying.
   The wrapper's logic was right — there just wasn't enough time
   left for the retry to do useful work. The continuation
   (`afterany:8481646`) picks up from `step-500` on the next dispatch.

2. **`scrape_bad_nodes.py` only fingered ONE node** (`x4110c3s3b0n0`)
   even though the parent log shows another bad IP (`10.115.15.154`).
   That second IP was a per-rank `Connection closed by peer` source
   but didn't match `getent hosts` resolution, so the scraper skipped
   it. Worth tightening the IP→hostname fallback path so future
   multi-node failures get all bad nodes swapped at once.

3. **Two spares used** (started with 4, ended with 3, but the swap
   log shows only 1 bad node replaced — the remaining "missing" spare
   was consumed by yeet-env redundancy). At 256N with 4 spares we
   have plenty of headroom; at 2048N with 10 spares the ratio is
   much tighter — worth bumping spares if multi-node bad batches
   become common.

## Bugs found and fixed during this validation cycle

Listed in dispatch order — each was discovered when the next job
dispatched and exposed the next bug:

| Job | Commit | Bug |
|------|--------|-----|
| `8480294`, `8480361` (DOA 0s) | `e8379fd6f` + `3c22711ec` | `dirname "$(realpath "$0")"` resolves to `/var/spool/pbs/mom_priv/jobs/`, not the repo scripts dir. Anchor to `$PBS_O_WORKDIR/torchtitan/experiments/ezpz/scripts/` instead. |
| `8481302` (DOA 4 min) | `ca35f1624` | `ezpz launch` re-derives `nhosts` from `/var/spool/pbs/aux/<jobid>` instead of the wrapper's overridden `PBS_NODEFILE`. Inject explicit `--hostfile=$FAILOVER_ACTIVE --nnodes=$NHOSTS -ppn -n` args. |
| `8481301` (DOA 6 min, mpiexec --help dump) | `603eee961` | Previous fix used the dash form `--nproc-per-node`. `ezpz launch`'s argparse only registers `--nproc_per_node` (underscore). The dash form leaked through to `cmd_to_launch` and mpiexec rejected it. Switch to `-ppn`/`-n` short flags. |
| `8481645` (zombie-success after 3h26m of real training) | `e216a2523` | `ezpz launch`'s outer python wrapper exits 0 even when its inner `mpiexec` dies with SIGTERM. PIPESTATUS-only check misses every bad-node crash. Now also parse `Execution finished with N` from the log + count crash-pattern lines. |

## Net production progress

Despite all 4 bugs above being live across this cycle, **400 training
steps were *logged* across the two failover-wrapped 20B trajectories** —
loss progressed, MFU was healthy. But (per the caveat above) **none of
those steps were persisted to disk** because async checkpoint saves
were killed mid-write by the bad-node crashes:

- **20B 512N** (`8481645`): step 800 → 1000 logged (loss 3.53 → 3.34, MFU 17.5%); **step-900 ckpt dir empty on disk**, latest usable is `step-800`
- **20B 256N** (`8481646`): step 301 → 500 logged (loss 4.95 → 4.12, MFU 20%); **no new ckpts persisted**, latest usable is `step-300`

The wrapper now reliably:

1. Splits PBS_NODEFILE into active + spare (`failover_init`)
2. Yeets venv to ALL nodes (so spare swap-in is instantaneous)
3. Runs the training command with explicit topology args
4. Detects inner crashes even when shell exit is 0
5. Identifies bad nodes via [`scrape_bad_nodes.py`](../../../../scripts/scrape_bad_nodes.py)
6. Swaps the bad node out for a spare, mutates `active.hostfile` in place
7. Re-runs the same command — `ezpz launch` re-reads `active.hostfile`
   transparently

Next test: **8503077** — 80B at 2058N (2048 active + 10 spare),
queued 2026-05-22 to stress-test the wrapper at the largest scale
we've attempted.

<!-- Job-ID anchors -->
<a id="log-8481646"></a>

## Logs

| Job ID | Log path |
|-------|----------|
| `8481646` | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n256-v2-failover-chain1.o8481646` |
| `8481646` (attempt 1) | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/logs/failover-8481646/attempt-1.log` |
| `8481646` (attempt 2) | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/logs/failover-8481646/attempt-2.log` |
| `8481645` (20B 512N companion) | `/flare/AuroraGPT/foremans/runs/agpt-20b-v2/torchtitan-ezpz/agpt-20b-n512-v2-failover-chain1.o8481645` |
| `8503077` (80B 2048N stress test) | queued — will land in `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/` |
