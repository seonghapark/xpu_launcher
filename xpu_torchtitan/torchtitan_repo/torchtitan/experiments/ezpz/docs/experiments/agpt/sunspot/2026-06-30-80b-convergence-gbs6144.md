# 80B head-to-head convergence at the production batch (GBS=6144) -- Sunspot, 2026-06-30

The follow-up the [2026-06-27 production-batch finder](2026-06-27-80b-lr-finder-production-batch.md)
called for: run each optimizer at its finder-recommended **constant** LR for
more steps than the 15-step finder, to confirm the ranking (mano > sophiag >
AdamW) holds past the finder's early-step window before committing a production
optimizer.

**Headline (a negative result that overturns the finder's optimism):** at
GBS=6144, **all three optimizers diverge to NaN within 5-12 steps** at their
finder-recommended constant LRs. Each descends cleanly for a few steps, then
`grad_norm` creeps up and explodes, and loss NaNs one step later. The 15-step
finder's ramping LR masked an instability that a constant LR at the same batch
exposes immediately.

## Setup

- `scripts/submit_80b_convergence.sh` -> `scripts/run_80b_convergence.sh`,
  one PBS job per optimizer, Sunspot `workq`, **64 nodes (dp_degree=186 at
  TP=4)**, LBS=1 / GAS=32 -> **GBS=6144**, seq_len=8192, compile=OFF, AC=full,
  pure FSDP -- the same validated stable corner the finder used.
- **Constant LR**: warmup 5 steps then flat (`decay_ratio=0`,
  `min_lr_factor=1.0`). This is the key difference from the finder, which ramps
  LR across the whole sweep.
- Finder-recommended LRs: **mano 3e-6** (min/5), **sophiag 1e-6** (min/2.5),
  **AdamW 5e-7** (under the 7.4e-7 cliff).
- Requested 50 steps; all three hit their 8h wall around step ~22 (see
  "Cost" below) but every run had already NaN'd by step 12, so the truncation
  is irrelevant.
- git HEAD `1b554df71`. Jobs: mano 12469910, sophiag 12469911, adamw 12469912.
- W&B: [mano](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/1ph6c4yd),
  [sophiag](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ls56x69j),
  [adamw](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/55l4i6d6).

## Result: all three diverge

| optimizer | LR | clean-descent steps | last finite loss | grad_norm run-up | NaN at |
|-----------|-----|--------------------:|-----------------:|------------------|-------:|
| **mano**    | 3e-6 | 1-4  | 11.92 (step 5) | 8.0 -> 9.3 -> nan            | grad step 5, loss step 6 |
| **sophiag** | 1e-6 | 1-11 | 11.20 (step 11)| 8 -> 14 -> 31 -> 65 -> nan   | grad step 12, loss step 13 |
| **AdamW**   | 5e-7 | 1-8  | 11.37 (step 8) | 8 -> 9 -> 10.5 -> nan        | grad step 9, loss step 10 |

Full trajectories:

```
mano @ 3e-6                sophiag @ 1e-6             adamw @ 5e-7
step loss     grad         step loss     grad         step loss     grad
  1  12.907   7.96           1  12.964   7.91           1  12.897   8.09
  2  12.734   7.99           2  12.843   7.93           2  12.835   8.11
  3  12.386   8.03           3  12.600   8.06           3  12.712   8.16
  4  12.132   9.26           4  12.234   8.44           4  12.542   8.25
  5  11.918   nan            5  11.753   8.12           5  12.277   8.49
  6  nan      nan            6  11.593  14.53           6  11.995   7.91
  ...                        7  11.695  31.41           7  11.776   9.15
                             8  11.791  38.27           8  11.573  10.54
                             9  12.088  64.90           9  11.372   nan
                            10  11.711  45.42          10  nan      nan
                            11  11.497  39.15           ...
                            12  11.205   nan
                            13  nan     nan
                            ...
```

## Reading the result

1. **The finder ranking does NOT survive a constant-LR run.** The finder
   ranked mano safest (broadest U-min); here mano dies **first** (step 5) --
   the opposite of the finder's optimism. The finder measures early-step
   descent under a *ramping* LR; it does not measure sustained stability at a
   *held* LR, and the two disagree sharply.

2. **The failure mode is shared across all three optimizers.** All three show
   the identical signature -- a few clean descent steps, then `grad_norm`
   run-up, then NaN -- despite completely different optimizer math (AdamW's
   moment estimates, mano's manifold normalization, sophiag's Hessian
   diagonal). A failure common to all three at the same batch/corner is
   unlikely to be an optimizer-tuning problem; it points to a **shared
   instability at the 80B / GBS=6144 / dp=186 corner** (consistent with the
   documented 80B grad-path fragility and bf16 sensitivity at dim=9216),
   independent of the optimizer.

3. **`grad_norm` is the leading indicator, not loss.** In every case the
   gradient norm blew up (or NaN'd) one step *before* the loss did -- the loss
   was still finite and descending when the gradient had already diverged
   (mano: loss 11.92 at the step its grad_norm went NaN). sophiag is the
   clearest: grad_norm 8 -> 14 -> 31 -> 65 over steps 5-9 while loss looked
   fine. Any 80B production run should watch grad_norm, not just loss.

4. **sophiag lasted longest and reached the lowest loss** (11.20 at step 11)
   but with a violently rising grad_norm the whole way -- not "stable," just
   slower to cross the NaN threshold. AdamW @ 5e-7 (the doc's "safe if staying
   on AdamW" number) also dies, at step 9.

## Implications for 80B production

- **None of the finder-recommended constant LRs is production-safe as-is** at
  GBS=6144. The finder gave a usable *starting* estimate but not a stable
  *sustained* LR. Do not lift a constant production LR directly from the
  finder minimum / (min/N) heuristic for 80B at this batch.
- **A warmup schedule that keeps effective LR low well past the grad-spike
  window is likely mandatory** -- the 5-step warmup here was far too short; the
  divergence hits 4-12 steps in, i.e. right where a real 200-step linear
  warmup would still be holding LR low. The 2B/20B v2 configs use a 200-step
  warmup; the 80B needs at least that, probably more, and likely also tighter
  gradient clipping.
- **Next step:** re-run the leading candidate (sophiag or mano) with (a) a long
  linear warmup (>=200 steps) and (b) explicit grad clipping, and watch
  grad_norm -- the question is whether the corner is stabilizable with
  schedule + clipping, or whether it is a harder numerical wall (bf16 at
  dim=9216) that needs an fp32 grad path or a smaller per-step effective LR.

## Cost / logistics

- Per-step wall at this corner is **~20 min/step** (mano step 1 -> step 2:
  23:57:07 -> 00:17:25), plus ~20 min startup and a ~20 min first step.
  50 steps would have needed ~17 h; the 8h wall capped each job near step ~22.
  Because all three NaN'd by step 12, the wall cap did not cost any signal.
- This is a very expensive corner to iterate on (64N, ~20 min/step). The
  smoke-first discipline paid off (a config bug -- `env: ezpz` exit 127 from a
  missing yeet-env preamble -- was caught and fixed in a throwaway 10-step
  smoke, job 12469908, before these runs). Future 80B convergence probes
  should batch schedule/clipping variants into one submission wave rather than
  iterate serially.
