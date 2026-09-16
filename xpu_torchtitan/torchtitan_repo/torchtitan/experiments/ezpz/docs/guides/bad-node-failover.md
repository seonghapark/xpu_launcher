# Bad-node failover for production training

> **Status (2026-06-26): the current path is native `ezpz launch
> --auto-retry`.** New runs should use the portable
> `submit_agpt_{2b,20b,80b}_autoretry.sh` scripts. The older bash
> `failover_lib.sh` wrapper (`*_aurora_venv_failover.sh`) is **retained
> but legacy** -- kept for reproducing earlier Aurora production runs and
> as the original reference for the failure taxonomy, not recommended for
> new work. See [Evolution](#evolution) for how we got here.

ALCF compute nodes go bad mid-run: a node dies (`signal 9`), a gloo
connection drops, init OOMs, or training silently hangs. On a 256/512N
job a single bad node throws away hours of post-checkpoint compute and
the walltime is then up. Both implementations below solve this the same
way -- request `N + spare` nodes, train on `N`, and on a bad-node failure
swap in a spare and retry -- and share the same
[failure taxonomy](#failure-taxonomy). They differ only in *who* runs the
retry loop.

## Current: native `ezpz launch --auto-retry`

`ezpz launch --auto-retry` (ezpz >= 0.17.1, PR #170) owns the whole
split + scrape + swap + retry loop internally. The submit scripts
[`submit_agpt_{2b,20b,80b}_autoretry.sh`](../../scripts/) are portable:
**Aurora PBS headers by default** (`AuroraGPT` / `prod` / `home:flare`,
data list `olmo-mix-1124`), Sunspot via qsub overrides
(`-A datascience -q workq -l filesystems=flare:home`, data list `books`).
The data list auto-selects by `ezpz_get_machine_name`, so only the
account/queue/filesystem flags need overriding off-Aurora.

### Usage

```bash
# Aurora (default headers): 512 active + 10 spare (select = NHOSTS_TRAIN + spares)
qsub -l select=522 -l walltime=12:00:00 -v NHOSTS_TRAIN=512 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_2b_autoretry.sh

# Sunspot: qsub flags override the #PBS Aurora defaults
qsub -A datascience -q workq -l filesystems=flare:home \
    -l select=14 -l walltime=12:00:00 -v NHOSTS_TRAIN=12 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_2b_autoretry.sh

# 80B (Aurora): TP=4/LBS=1/AdamW default. Keep dp_degree (=NGPUS/TP) <= ~186
# -- the safe corner is validated at 62 active nodes (dp=186, GBS=372 via
# GAS=2). The script WARNS if dp_degree exceeds 186. Cap retries (each
# 80B retry pays ~5-15 min init):
qsub -l select=64 -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=62,MAX_FAILOVER_RETRIES=2,GAS=2 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_80b_autoretry.sh
```

`--spare-nodes auto` => spares = `total_pbs_nodes - NHOSTS_TRAIN`, so the
submitter just sets `select = NHOSTS_TRAIN + desired_spares`. All other
env knobs (`LBS`, `GAS`, `OPTIMIZER`, `LR`, `CKPT_DIR`,
`CHECKPOINT_ASYNC_MODE`, `VALIDATOR_ENABLE`, ...) work as documented in
each script's header.

### Validated training config per model

| Model | Config | TP | LBS | Optimizer | LR | Notes |
|---|---|---|---|---|---|---|
| 2B  | `agpt_2b_real` | 1 | 2 | sophiag | 2.28e-5 | `_real` cos_sin RoPE default (compile-lowerable); compile ON. Validator on. |
| 20B | `agpt_20b_real` | 1 | 2 | sophiag | 2.28e-5 | `_real` RoPE default; compile ON; `DATASET` knob (blendcorpus/HF). Validator on. |
| 80B | `agpt_80b` | 4 | 1 | adamw | 1e-6 | bf16-compute/fp32-master, AC=full, compile OFF. Plain (non-`_real`) config -- compile is off so the `_real` win is moot. Supersedes the legacy TP=2 default (NaN-prone). **Warns when `dp_degree>186`.** |

The `CONFIG_SUFFIX` knob toggles the RoPE flavor (`_real` default for
2B/20B; set `CONFIG_SUFFIX=` for the plain complex flavor). 80B stays on
the plain numerically-validated `agpt_80b`. See
[`production/agpt/80b/README.md`](../production/agpt/80b/README.md) for
the TP=4/LBS=1 stable-corner derivation and the `dp_degree<=186` ceiling.

### What the submit script still must do

`ezpz launch --auto-retry` handles the split + retry loop but does **not**
broadcast the venv. So the `*_autoretry.sh` scripts:

1. Leave `PBS_NODEFILE` **whole** (do NOT pre-split -- ezpz needs the full
   list to carve active + spare itself).
2. Run `ezpz yeet --src .venv.tar.gz` against that whole nodefile, so
   `/tmp/.venv` lands on active **and** spare nodes -- a swapped-in spare
   is then a filesystem no-op.
3. Compute `--nproc` and `GBS` from the **active** count
   (`NHOSTS_TRAIN * 12`), NOT from `ezpz_setup_job`'s `$NGPUS` (which sees
   the full allocation). The active count is constant across retries (a
   swap replaces a node in-place by index), so this is valid for the
   whole run; at runtime `os.environ["WORLD_SIZE"]` also equals it.
4. Call `ezpz launch --nproc <active> --nproc_per_node 12 --auto-retry
   --spare-nodes auto --timeout 1800 -- python3 -m ...train ...`.

No separate preflight smoke: ezpz's `STUCK_PRE_TRAINING` guard bails
(without burning spares) if init crashes twice with zero training
progress -- which is what the old bash preflight was for.

### How many spares?

Rule of thumb: **~2% spare, minimum 4**. For 512N use 10 spares (522
total); for 1024N use 20 (1044 total). At 2% the queue penalty is small
and you can survive 3-4 distinct bad-node hits in one walltime window.
The scheduler treats the job as `select = N + spare`, so wider
allocations queue with the wider class -- make sure your account has the
queue/limit headroom before bumping spare counts.

## Failure taxonomy

Both implementations classify a non-zero exit the same way. The scrape
patterns were empirically tuned against production crash logs from
April-May 2026.

| Pattern | Detection | Action | Production examples |
|---|---|---|---|
| `<host>: shepherd died from signal 9` | regex -> emit hostname | swap that host | 8459818, 8460301, 8460302, 8463659 |
| `RuntimeError: ... Connection closed by peer [IP]:port` | regex -> reverse-resolve via `getent hosts` -> normalize to `.hsn.cm.aurora.alcf.anl.gov` | swap that host | 8470102, 8470103, 8479581 |
| `set_determinism` `std::bad_alloc` init crash | not matched -- no specific host | blind-rotate first active host | 8463182, 8463183, 8466848 |
| **silent hang** (no output for `--timeout`s) | watchdog -> exit 124 | blind-rotate, retry | 8479579, 8505298 |
| exit 143 (SIGTERM / walltime) | not a bad-node failure | do NOT retry; chain a continuation | -- |

### Why we DON'T match `signal {11,15}`

`rank N died from signal 11` (SIGSEGV) / `15` (SIGTERM) are almost always
**cascading** deaths from a primary kill on a different node. Matching
them would falsely tag innocent nodes. Verified against 8466848: the
crash was `std::bad_alloc` on rank 2413, but 8 other ranks died from
signal 11/15 as a downstream effect -- none were the actual bad node.

### Silent hangs (the `--timeout` watchdog)

If the launched process emits no output for `--timeout` seconds (default
1800 = 30 min), `ezpz launch` SIGTERMs the inner mpiexec and exits 124,
treated as a bad-node failure: scrape (typically no specific host, since
the hang IS the silence), blind-rotate, retry. 1800s was chosen because
the longest observed *legitimate* quiet period was a 19-min async ckpt
save at 20B 512N. First real-world recovery:
[`8505298`](../experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)
(hung at step 37, watchdog tripped at 30 min, swapped + retried, trained
to walltime landing step-100/200 checkpoints). Originating incident:
[`8479579`](../experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md).

### Things to know

- **NaN / genuine bugs**: any non-zero exit triggers a retry, but a
  deterministic bug (NaN, bad config) will recur on every retry. The
  retry cap is the circuit breaker; after exhaustion the job exits with
  the original failure code, surfacing the bug.
- **Compile cost amplifies on retry**: each retry re-runs `torch.compile`
  (2B/20B at 256N: 7-15 min; 80B at TP=4: 4+ hours). Cap retries for 80B
  so one bad-node hit doesn't burn the walltime on recompiles.
- **Async checkpointing helps**: with `--checkpoint.async-mode=async` a
  retry resumes from the last *successfully-saved* ckpt, losing < 100
  steps (a gloo failure during a sync save aborts the save -- this is
  what killed 8479581 at step 500).

## Why this exists

Recurring Aurora bad-node failures killed at least 9 production jobs in a
two-week window (April-May 2026):

| Job ID | Trajectory | Failure mode |
|---|---|---|
| 8459818 | 2B 256N v2 | `shepherd died from signal 9` after step 2070 |
| 8460301 | 2B 512N v2 | `shepherd died from signal 9` after step 1387 |
| 8460302 | 20B 512N v2 | `shepherd died from signal 9` at end of walltime |
| 8463659 | 20B 256N v2 | `shepherd died from signal 9` after step 364 |
| 8470102 | 20B 256N v2 | gloo TCP `Connection closed by peer` after ~3h |
| 8470103 | 20B 256N v2 | gloo TCP timeout after ~3h |
| 8466848 | 20B 512N v2 | `set_determinism` `MemoryError: std::bad_alloc` at startup |
| 8479581 | 20B 256N v2 | gloo TCP timeout during DCP ckpt save at step 500 |
| 8479579 | 20B 512N v2 | **silent hang** at step 803 (no exit, killed manually) |

## Legacy: bash `failover_lib.sh` wrapper (retained)

Before native auto-retry, the retry loop lived in a bash library that the
`*_aurora_venv_failover.sh` scripts sourced. It is **retained but not
recommended for new runs** -- kept for reproducing earlier Aurora
production trajectories (which all ran under it) and as the
fixture-tested reference implementation of the failure taxonomy above.

### How the bash wrapper maps to native

| | **Legacy** (`failover_lib.sh`) | **Native** (`ezpz launch --auto-retry`) |
|---|---|---|
| Retry loop | `failover_run` in bash | inside `ezpz launch` |
| Nodefile split | `failover_init` (pre-split, narrow `PBS_NODEFILE`) | ezpz splits internally from `--nproc`; do not pre-split |
| Venv broadcast to spares | `failover_yeet_all` | still the submit script's job (`ezpz yeet` on the whole nodefile) |
| Scrape patterns | `scrape_bad_nodes.py` | same taxonomy, inside ezpz |
| Retry cap | `FAILOVER_MAX_RETRIES` (default 3) | `--max-failover-retries` (default unbounded) |
| Idle watchdog | `FAILOVER_IDLE_TIMEOUT` (default 1800) | `--timeout` (default 1800) |

### Legacy mechanics (`failover_lib.sh`)

- **`failover_init`** (startup): read `PBS_NODEFILE` (N+spare lines),
  split into `active.hostfile` (first N) + `spare.hostfile` (rest) under
  `$FAILOVER_LOG_DIR` (default `$(pwd)/logs/failover-<JOBID>/`), override
  `PBS_NODEFILE -> active.hostfile` so `ezpz_setup_job` sees only the
  training subset, truncate `bad_nodes.txt`.
- **`failover_yeet_all`** (after split): temporarily point
  `PBS_NODEFILE -> all.hostfile`, run `ezpz yeet-env` so the venv lands
  on active + spare, restore `PBS_NODEFILE -> active.hostfile`.
- **`failover_run`** (per attempt): run training, redirect to
  `attempt-N.log`; on exit, classify rc per the taxonomy --
  `failover_swap_in HOSTS...` (`sed`-replace each bad host with a popped
  spare, append to `bad_nodes.txt`) for an identified host, or
  `failover_swap_one_blind` (rotate the first active host out) when no
  specific host is found; sanity-check the active count still equals
  `NHOSTS_TRAIN`; loop up to `FAILOVER_MAX_RETRIES`.

### Legacy usage

```bash
# 2B canonical chain at 512 active + 10 spare nodes
qsub -q prod -l select=522 -l walltime=12:00:00 -v NHOSTS_TRAIN=512 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_2b_aurora_venv_failover.sh

# 80B at 512 active + 10 spare with reduced retry budget
qsub -q prod -l select=522 -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=512,FAILOVER_MAX_RETRIES=2 \
    torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
```

Before any edit to `failover_lib.sh`, run `bash tests/failover/run_tests.sh`
-- all 9 synthetic log fixtures (one per known failure mode) must pass.

### Postmortem files (both implementations)

After a job runs, `$FAILOVER_LOG_DIR` (or ezpz's `logs/failover-<JOBID>/`)
contains `active.hostfile` (final active set), `spare.hostfile`,
`all.hostfile`, `bad_nodes.txt` (one host per swap event -- **the
canonical bad-node record; copy it to a long-lived file before the
allocation expires** if feeding an ALCF ticket), and `attempt-N.log` per
attempt.

## Evolution

1. **Bash wrapper, v1-v2 (2026-05-13 -> 2026-05-23)**: `failover_lib.sh`
   + `scrape_bad_nodes.py`. v2 added the silent-hang `--timeout` watchdog
   (`eefccfc9d`), ANSI-aware exit-code parsing (`94a8fda66`), a unified
   walltime+crash regex (`0d93a1e91`), and the fixture test harness under
   [`tests/failover/`](../../tests/failover/). First real-world silent-hang
   recovery: job 8505298 (2026-05-23).
2. **Native auto-retry (2026-06-24)**: ezpz PR #170 moved the
   split/scrape/swap/retry loop into `ezpz launch --auto-retry`. The
   `*_autoretry.sh` scripts replaced the bash wrapper as the maintained
   path, dropped the manual nodefile split, and became portable across
   Sunspot/Aurora. The non-failover `submit_agpt_{2b,20b}_aurora_venv.sh`
   were removed; the `*_aurora_venv_failover.sh` scripts were kept as
   legacy.
3. **80B default corrected (2026-06-24)**: the 80B native script defaults
   to the validated TP=4/LBS=1/AdamW corner with a `dp_degree>186` NaN
   warning, superseding the legacy TP=2 default.

## Files

### Current (native)

| Path | Purpose |
|---|---|
| [`scripts/submit_agpt_2b_autoretry.sh`](../../scripts/submit_agpt_2b_autoretry.sh) | 2B native auto-retry submit script (portable Sunspot/Aurora; `_real` RoPE default). |
| [`scripts/submit_agpt_20b_autoretry.sh`](../../scripts/submit_agpt_20b_autoretry.sh) | 20B native auto-retry (adds the `DATASET` blendcorpus/HF knob). |
| [`scripts/submit_agpt_80b_autoretry.sh`](../../scripts/submit_agpt_80b_autoretry.sh) | 80B native auto-retry (TP=4/LBS=1/AdamW default; `dp_degree>186` NaN warning). |

### Legacy (retained)

| Path | Purpose |
|---|---|
| [`scripts/failover_lib.sh`](../../scripts/failover_lib.sh) | Bash library: `failover_init`, `failover_yeet_all`, `failover_swap_in`, `failover_swap_one_blind`, `failover_run`. |
| [`scripts/scrape_bad_nodes.py`](../../scripts/scrape_bad_nodes.py) | Extracts bad-node hostnames from a training log. |
| [`scripts/submit_agpt_2b_aurora_venv_failover.sh`](../../scripts/submit_agpt_2b_aurora_venv_failover.sh) | 2B production submit script with the bash failover wrapper. |
| [`scripts/submit_agpt_20b_aurora_venv_failover.sh`](../../scripts/submit_agpt_20b_aurora_venv_failover.sh) | 20B production submit script with failover. |
| [`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../scripts/submit_agpt_80b_aurora_venv_failover.sh) | 80B production submit script with failover (AdamW LR=1e-6, TP=2, AC=full, compile=OFF). |
| [`scripts/FAILOVER.md`](../../scripts/FAILOVER.md) | Brief code-adjacent quickref. **This page is the canonical reference.** |

## See also

- [`docs/experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md`](../experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)
  -- first real-world silent-hang recovery
- [`docs/experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md`](../experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md)
  -- the originating silent-hang incident
- [`docs/production/agpt/80b/README.md`](../production/agpt/80b/README.md)
  -- 80B TP=4/LBS=1 stable corner + `dp_degree<=186` ceiling
- [`scripts/FAILOVER.md`](../../scripts/FAILOVER.md) -- code-adjacent quickref
- [`docs/experiments/agpt/aurora/20260630-failover-restart-economics.md`](../experiments/agpt/aurora/20260630-failover-restart-economics.md)
  -- log-mined restart economics: 7 confirmed successful restarts, ~11% raw
  recovery rate (higher for genuinely node-local failures), 132 spare swaps
  across 133 failover-wrapped jobs (2026-06-30 analysis)
