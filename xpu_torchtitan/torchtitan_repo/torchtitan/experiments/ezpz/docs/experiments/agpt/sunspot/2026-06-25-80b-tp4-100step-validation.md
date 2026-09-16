# 80B TP=4 stable-corner validation, 100 steps (Sunspot, 2026-06-25)

First real (non-smoke) run of the 80B TP=4/LBS=1/bf16 "stable corner"
via `scripts/submit_agpt_80b_autoretry.sh`, taken to 100 steps to confirm
the corner holds NaN-free well past the prior 30-step evidence.

**Result: clean success.** 100/100 steps, zero NaN, loss 12.93 -> 7.72,
step-100 checkpoint saved, rc=0.

## Config

- Job `12469551`, Sunspot `workq`, 62 active + 2 spare (select=64),
  `submit_agpt_80b_autoretry.sh` (native `ezpz launch --auto-retry`).
- `agpt_80b` (complex RoPE), TP=4, LBS=1, GAS=2 -> GBS=372,
  dp_degree=186 (at the validated <=186 ceiling), AdamW LR=1e-6,
  bf16-compute / fp32-master, AC=full, compile=OFF.
- Data: `data-lists/sunspot/books.txt` (blendcorpus), warm index cache
  (pre-built; no build-race -- see the cache-build-race fix below).
- `--np=744`, ~76 s/step, total wall 7927 s (~2h12m incl. init + a
  163 s final checkpoint).

## Trajectory (every 10 steps)

```
step   1  loss 12.93  grad_norm 7.81
step  10  loss 12.67  grad_norm 7.90
step  20  loss 11.83  grad_norm 7.74
step  30  loss 10.80  grad_norm 9.01
step  40  loss 10.13  grad_norm 9.90
step  50  loss  9.44  grad_norm 8.57
step  60  loss  8.81  grad_norm 5.24
step  70  loss  8.42  grad_norm 2.88
step  80  loss  8.15  grad_norm 2.70
step  90  loss  7.93  grad_norm 2.04
step 100  loss  7.72  grad_norm 5.98
```

- **Loss:** 12.93 -> 7.72 (-5.2 nats), monotone modulo small noise.
- **grad_norm:** bounded throughout (peak ~9.9 early, settling to ~2-6);
  no spikes toward inf, no NaN one-step-before-loss (the TP=2/LBS>1
  failure signature). Verified zero NaN/inf in any loss/grad_norm value
  across all 100 steps.
- **MFU:** steady ~9.8% (matches the documented TP=4 cost); memory flat
  at 32% (20.5 GiB/tile) -- comfortable headroom.
- **Checkpoint:** `outputs/checkpoints/agpt-80b-adamw-books-n62-gbs372/step-100`
  (906 GiB), saved in 163 s; `[auto-retry] FAILOVER STOP: success`.

## Significance

This supersedes the prior 80B TP=4 evidence, which only reached 30 steps
(jobs 12469494/12469509, loss ~9.7). 100 clean steps at GBS=372 / 62N
confirms the **TP=4/LBS=1/bf16 corner is the production-ready path** --
not a 30-step knife-edge. The underlying TP=2/LBS>1 grad-path overflow
remains an open upstream-worthy bug; TP=4/LBS=1 is the working production
recommendation.

## Notes

- The native `submit_agpt_80b_autoretry.sh` worked end to end: split
  62 active / 2 spare, `--np=744`, yeet, and `FAILOVER STOP: success` on
  attempt 1 (no bad-node swaps needed this run).
- Two earlier attempts (12469548, 12469550) crashed in dataset init on a
  **cold blendcorpus index cache** (TP>1 build race); fixed at the source
  (global barrier, blendcorpus `debfff5` / PR #8) and operationally via
  `scripts/prewarm_blendcorpus_cache.sh`. This run used the warm cache and
  saw no race. See the 2026-06-25 journal entry.
- This run predates the `--validator.enable` script change, so no
  validation pass ran here (validator was configured but not enabled).
