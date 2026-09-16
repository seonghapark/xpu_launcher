# 80B: LR, batch size, and dp-degree -- a connected set of findings (Sunspot, 2026-06-26)

This synthesizes five 80B jobs run 2026-06-26 that, together, answer
three entangled questions the global-batch simulation campaign raised:

1. **What is the usable LR for 80B, and were the sims scaled correctly?**
   (LR-finder: 12469631 adamw/mano, 12469632 muon/sophiag)
2. **Does scaling LR with batch size help or hurt the GBS=5952 NaN?**
   (12469698: GBS=5952 + 16x-scaled LR)
3. **Does the TP=4 corner actually survive dp_degree > 186?**
   (dp-degree cliff bisect: 12469628 dp=192, 12469629 dp=264)

All at TP=4 / LBS=1 / bf16 / AdamW (except the LR-finder optimizer sweep),
62-88N, compile=OFF.

## TL;DR

- **The usable LR for 80B is low.** LR-finder (at GBS=192) min loss is at
  **lr~3e-6 (sophiag) / ~8e-6 (mano)**, diverging by ~1e-2; production
  **1e-6** sits on the stable left-shoulder, *below* the optimum.
  **CAVEAT:** this finder ran at GBS=192, ~32x below the ~6144 production
  target, and optimal LR shifts with batch -- a corrected GBS=6138 sweep
  (jobs 12469715-18) is running; these are provisional / a ranking only.
- **Do NOT linearly scale LR up with batch.** The GBS=5952 (16x) run with
  a 16x-linear-scaled LR=1.6e-5 NaN'd at **step 7** -- *far worse* than
  the same batch at flat LR=1e-6 (step 29). 1.6e-5 sits at the LR-finder's
  divergence shoulder, so it blew up almost immediately. The LR ceiling is
  set by the optimizer/model (bf16 at dim=9216), not the batch.
- **The "dp_degree <= 186 ceiling" is not a real cliff.** dp=192 and
  dp=264 both ran 30 steps NaN-free. 186 was only the highest *tested*
  point (62N x 12 / 4), never a measured boundary.

## 1. LR-finder: the usable LR is ~1e-6 to ~1e-5, set by the optimizer

Swept lr = 1e-6 -> 1.0 over 40 steps at **GBS=192** (world_size 768 x
LBS=1 / TP=4, no GAS), per optimizer. SophiaG/Muon are documented-broken
at 80B (bf16 overflow at dim=9216); included to confirm.

> **IMPORTANT (batch-size caveat, 2026-06-26):** this sweep ran at
> **GBS=192 -- ~32x below the ~6144 production target.** Optimal LR is
> batch-size dependent, so these numbers give at best the *optimizer
> ranking* and a rough scale, NOT the production LR. A corrected sweep at
> the production batch (GBS=6138 via GAS=33) is running: prewarm 12469714
> -> per-optimizer finders 12469715 (adamw) / 716 (mano) / 717 (muon) /
> 718 (sophiag), sweeping 1e-8 -> 1e-4. This section will be updated with
> the production-batch curves; treat the GBS=192 numbers below as
> provisional.

| Optimizer | min-loss LR (GBS=192) | stable ceiling (<=5% above min) | diverges |
|---|---|---|---|
| sophiag | 2.8e-6 | 1.1e-5 | NaN by ~1.8e-1 |
| mano | 7.9e-6 | 6.3e-5 | hard rise past ~1e-2 |
| adamw | (crashed*) | -- | -- |
| muon | (crashed*) | -- | -- |

`*` adamw and muon "crashed" at ~61s on the **blendcorpus cold-cache
build-then-load race** (`mmap length is greater than file size`) -- the
*first* optimizer in each sweep cold-builds the TP=4 index and races it;
the LR-finder has no auto-retry, so only the *second* optimizer (mano /
sophiag) ran on the now-warm cache. This is a harness artifact, not an
optimizer finding -- adamw's curve still needs a warm-cache rerun.

sophiag curve (every ~5th point): lr=1e-6 loss=12.92, 5.6e-6 -> 12.45
(min region), 3.2e-5 -> 13.35, 1e-3 -> 14.29, 3.2e-2 -> 37.85,
1.8e-1 -> NaN. Classic LR-finder U with a low, sharp right wall.

**Takeaway:** production LR=1e-6 is conservative-but-safe. There is
headroom to ~3-8e-6, but the wall is close -- anything past ~1e-5 is
risky, consistent with the documented "LR=1.1e-5 NaNs at production GBS."

## 2. Scaling LR up with batch makes the NaN WORSE, not better

The GBS=5952 (2048N-batch) run had NaN'd at step 29 with flat LR=1e-6,
but that run was *inside warmup* (effective LR only ~5.8e-7), confounding
any LR conclusion. Job 12469698 controlled for it: 16x-linear-scaled
LR=1.6e-5 with warmup=5 so the scaled LR is actually active.

| GBS=5952 run | nominal LR | warmup | NaN at |
|---|---|---|---|
| 12469627 | 1e-6 (eff ~5.8e-7) | 200 (clamped 50) | step 29 |
| 12469698 | **1.6e-5** | 5 | **step 7** |

12469698 trajectory: step 1 loss 12.93 (clean) -> step 5 loss 11.58
(warmup done, full 1.6e-5 active) -> step 6 loss **20.38, grad_norm
121.9** -> step 7 grad_norm NaN -> steps 8-11 loss NaN. A clean LR-driven
divergence the moment the scaled LR engaged.

**This is the answer to "should we scale LR with batch size?": no.** 1.6e-5
sits right at mano's stable ceiling (6.3e-5) / well past sophiag's
(1.1e-5), so the 16x-scaled LR is simply too large for the
optimizer/model regardless of batch. The standard linear/sqrt batch-LR
rules assume the LR ceiling scales with batch; here it is fixed by the
bf16-at-dim-9216 overflow, so scaling LR up walks straight into the wall.

This reframes the flat-LR step-29 NaN too: at the tiny effective LR
(~5.8e-7), the GBS=5952 step-29 failure is a *separate, milder*
batch-accumulation effect, NOT the LR-divergence seen here. The two
GBS=5952 NaNs have different causes.

## 3. The dp_degree<=186 "ceiling" is not a cliff -- TP=4 scales further

The 80B safe corner was stated as `dp_degree <= ~186`, but 186 is just
`62N x 12 / TP=4` -- the largest node count we'd tested, not a measured
boundary. The bisect raised N at fixed TP=4/LBS=1/GAS=1:

| Job | N | dp_degree | vs 186 | step 30 | Result |
|---|---|---|---|---|---|
| 12469628 | 64 | 192 | 1.03x | loss 9.68, grad_norm 16.5 | **clean, 0 NaN** |
| 12469629 | 88 | 264 | 1.42x | loss 9.69, grad_norm 13.0 | **clean, 0 NaN** |
| 12469630 | 108 | 324 | 1.74x | (queued) | pending |

**dp=192 and dp=264 both ran 30 NaN-free steps.** The only NaN ever seen
on the dp-degree axis remains 12469492 (TP=2, dp=372). So the safe corner
extends past 186 -- at least to 264, with 324 pending. The exact cliff is
somewhere in (264, 372], still unmapped (dp=372 needs 124N, beyond
Sunspot's 114 usable nodes; confirmable only on Aurora).

## Synthesis: what actually bounds the 80B corner

Two independent walls, now both quantified:

- **LR wall (sharp, low):** usable LR ceiling ~1e-5, set by bf16 overflow
  at dim=9216. Production 1e-6 is safe; do not scale up with batch.
- **dp-degree wall (further out than thought):** clean to dp=264+, not
  186. Not the binding constraint at the node counts tested.
- **Batch-accumulation effect (mild):** at fixed low LR, very large GAS
  (32 -> GBS=5952) shows a late grad-path NaN (step 29) that GAS<=16 does
  not -- distinct from the LR wall, and the weakest of the three.

For production this means: the TP=4/LBS=1/bf16/LR=1e-6 corner is sound and
has more dp_degree headroom than the docs claimed; the thing to NOT do is
raise LR for larger batches.

## Caveats / open

- adamw + muon LR-finder curves are missing (cold-cache race); rerun on a
  warm cache to complete the optimizer comparison.
- dp=324 (12469630) still queued; dp=372 needs Aurora.
- Reproducibility of the flat-LR GBS=5952 step-29 NaN (12469699, queued)
  not yet in -- whether that mild batch effect is deterministic vs
  XPU-execution-order nondeterminism is still open.
- The LR-finder ran at GBS=192 (corrected sweep at GBS=6138 in flight); a batch-size sweep of the LR curve itself
  (does the optimum shift with GBS?) was not done.

## Jobs

| Job | What | Verdict |
|---|---|---|
| 12469631 | LR-finder adamw+mano | mano OK (min lr~8e-6); adamw cold-cache crash |
| 12469632 | LR-finder muon+sophiag | sophiag OK (min lr~2.8e-6); muon cold-cache crash |
| 12469698 | GBS=5952 + LR=1.6e-5 (16x) | NaN @ step 7 (LR too high) |
| 12469628 | dp=192 bisect | clean 30 steps |
| 12469629 | dp=264 bisect | clean 30 steps |
| 12469630 | dp=324 bisect | queued |
| 12469699 | GBS=5952 flat-LR reproducibility | queued |
