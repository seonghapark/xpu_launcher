# Failover wrapper — code-adjacent quickref

> **Canonical reference:**
> [`docs/guides/bad-node-failover.md`](../docs/guides/bad-node-failover.md).
> This file is just a pointer + one-liner usage so people poking at
> the scripts can find the docs.

## Files in this directory

| File | Purpose |
|---|---|
| `failover_lib.sh` | Bash library: `failover_init`, `failover_yeet_all`, `failover_swap_in`, `failover_swap_one_blind`, `failover_run`. Source from a submit script. |
| `scrape_bad_nodes.py` | Extracts bad-node hostnames from a training log. Used by `failover_run` to decide which node to swap out. |
| `submit_agpt_2b_aurora_venv_failover.sh` | 2B production submit script with failover (bash `failover_lib.sh`). |
| `submit_agpt_20b_aurora_venv_failover.sh` | 20B production submit script with failover. |
| `submit_agpt_80b_aurora_venv_failover.sh` | 80B production submit script with failover (AdamW LR=1e-6, TP=2, AC=full, compile=OFF). |
| `submit_agpt_{2b,20b,80b}_autoretry.sh` | Submit scripts using **native** `ezpz launch --auto-retry` (no `failover_lib.sh`). Portable Sunspot/Aurora. 80B defaults to the TP=4/LBS=1/AdamW stable corner + warns at `dp_degree>186`. |

## One-liner usage

```bash
# 2B/20B: 512 active + 10 spare = 522 nodes, default 3 retries
qsub -q prod -l select=522 -l walltime=12:00:00 -v NHOSTS_TRAIN=512 \
    submit_agpt_{2b,20b}_aurora_venv_failover.sh

# 80B: same, but cap retries at 2 (each retry pays 5-15 min init cost)
qsub -q prod -l select=522 -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=512,FAILOVER_MAX_RETRIES=2 \
    submit_agpt_80b_aurora_venv_failover.sh

# Native auto-retry (no failover_lib.sh). Sunspot default headers;
# select = active + spare, --spare-nodes auto. 12 active + 2 spare:
qsub -l select=14 -l walltime=12:00:00 -v NHOSTS_TRAIN=12 \
    submit_agpt_{2b,20b}_autoretry.sh

# 80B native auto-retry: TP=4/LBS=1/AdamW default; keep dp_degree<=186
# (62 active nodes, GBS=372 via GAS=2). Script warns past that.
qsub -l select=64 -l walltime=12:00:00 \
    -v NHOSTS_TRAIN=62,MAX_FAILOVER_RETRIES=2,GAS=2 \
    submit_agpt_80b_autoretry.sh
```

See [`docs/guides/bad-node-failover.md`](../docs/guides/bad-node-failover.md)
for the full design, detected failure modes, limitations (silent
hangs are NOT handled — manual intervention required), and
postmortem-file reference.
