# LR Finder -- agpt 80B

Learning-rate-finder results for the dense **agpt 80B** model. For how the
finder works see the [methodology README](../../README.md); for the
cross-model recommendation table and findings see the
[agpt index](../README.md). Figures in [`figures/`](figures/).

**TL;DR for production (GBS~6144):** do NOT run AdamW at LR=1e-6 -- it is on
the NaN cliff. Use **mano ~3e-6** (safest), **sophiag ~1e-6** (lowest loss,
narrower band), or AdamW ~5e-7. The small-batch "AdamW 1.1e-5" number does
not transfer to the production batch.

> **UPDATE 2026-06-30 -- these finder LRs do NOT hold as CONSTANT LRs.** A
> [head-to-head convergence run](../../../agpt/sunspot/2026-06-30-80b-convergence-gbs6144.md)
> at GBS=6144 found **all three optimizers NaN within 5-12 steps** at their
> finder-recommended constant LRs (mano @ 3e-6 dies first, step 5; AdamW @
> 5e-7 step 9; sophiag @ 1e-6 step 12). grad_norm runs up then explodes,
> loss NaNs one step later -- a shared instability at this corner that the
> finder's ramping LR masked. A long warmup (>=200 steps) + grad clipping
> is required before any of these is a usable production LR.

---

## 2026-06-27 -- 80B at the PRODUCTION batch (GBS=6144, Sunspot)

> **This supersedes the small-batch (GBS=192) recommendation.** The earlier
> 80B finder (2026-04-21) ran at the 2-node default batch (**GBS=192**) and
> gave "AdamW = 1.1e-5". At the real production batch (**GBS=6144**, 32x
> larger) that LR diverges: AdamW's usable ceiling collapses to **~7e-7**,
> and the production default LR=1e-6 sits **on the NaN cliff**. mano -- marked
> "N/A / broken" in the old table -- is actually the **best-behaved optimizer
> at this batch**.
>
> **Two-part correction to the "SophiaG/Muon broken at 80B" claim:** at this
> batch **sophiag is NOT broken** -- it runs all 15 steps finite with a real
> minimum at lr~2.5e-6 and the *lowest loss of all four optimizers* (12.60).
> Only **muon** is broken as documented (NaN from step 7). The old "both NaN
> regardless of LR" line came from the GBS=192 finder; the larger batch
> smooths sophiag's Hessian `grad*grad` term below the bf16 overflow
> threshold. (mano stays the safest production pick -- see below.)

### Why redo the finder at GBS=6144

Optimal LR is batch-size dependent, and 80B production runs at GBS~6144
(512N-equiv) -- 32x the GBS=192 the old finder used. Calibrating from the
small batch mis-predicts the production LR by ~14x. This sweep runs the
finder **at the production batch** (TP=4 / LBS=1 / GAS=32 -> GBS=6144,
dp_degree=192), one job per optimizer.

### Environment

| Field | Value |
|-------|-------|
| Date | 2026-06-27 |
| Machine | Sunspot |
| Nodes | 64 (768 XPU tiles), TP=4 / LBS=1 / GAS=32 |
| GBS | **6144** (production batch) |
| Compile | disabled (80B AC+TP regression) |
| Seq len | 8192 |
| LR range | 1e-8 -> 1e-4, 15 steps (adamw 1e-6 -> 1e-3) |
| Jobs | adamw 12469723, mano 12469724, muon 12469725, sophiag 12469726 |

### Headline result: all four optimizers at the production batch

![80B LR finder at GBS=6144, all optimizers](figures/sunspot_80b_gbs6144_all_optimizers.png)

(mano and sophiag both have real loss minima; AdamW only descends to a wall
then NaNs -- no minimum. muon is omitted from the plot: it NaNs from step 7
regardless of LR. The earlier `adamw_vs_mano` figure predated the sophiag
sweep and is superseded by this all-optimizer version.)

| Optimizer | behavior @ GBS=6144 | usable LR | min loss | NaN | verdict |
|-----------|---------------------|-----------|----------|-----|---------|
| **mano** | clean broad **U-curve** | ~1.6e-5 (min-bounded) | 12.62 @ 1.6e-5 | 0/15 | **best (safest) production pick** |
| **sophiag** | real **U-min**, blow-up onset ~4.6e-6 | ~2.5e-6 (min-bounded) | **12.60 @ 2.5e-6** (lowest of all four) | 0/15 | **NOT broken at this batch** |
| **AdamW** | NaN **cliff** (no minimum in range) | ~7.4e-7 (cliff-bounded) | 12.78 @ 7.4e-7 | 7/15 | **1e-6 is past the cliff** (NaN @ 1.36e-6) |
| muon | NaN from step 7 (~4e-7) | -- | (12.91) | 9/15 | broken as documented (bf16 dim=9216) |

- **mano**: a textbook LR-finder U -- descends to a real minimum at
  lr=1.6e-5 (loss 12.62) then rises gently, no divergence across the whole
  1e-8 -> 5.4e-5 sweep. **~20x more LR headroom than AdamW with a soft top.**
- **sophiag**: descends cleanly to a real minimum at lr=2.5e-6 (loss 12.60,
  the lowest of all four), then blows up sharply (grad_norm 14 -> 85 -> 132
  over steps 10 -> 12). **Directly contradicts the GBS=192 finding that
  sophiag NaNs by step 7 regardless of LR** -- at GBS=6144 it has a usable
  region and the best minimum. Narrower safe band than mano, though.
- **AdamW**: loss descends monotonically to lr=7.4e-7 (12.78), then NaNs
  at 1.36e-6 -- no minimum, just a wall. The production LR=1e-6 lands
  *between* the last stable point and the NaN, explaining the
  nondeterministic NaNs observed on GBS~6000 production attempts.
- **muon**: NaN from step 7 (lr~4e-7) onward, exactly as the GBS=192 finder
  reported -- the Newton-Schulz `A @ A` (9216x9216) overflow is not relieved
  by the larger batch the way sophiag's Hessian term is.

> **Data note:** the GBS=6144 AdamW numbers here come from the figures /
> per-experiment record below, not from the live `.../80B/adamw/
> lr_finder_data.csv` -- that CSV is keyed by `(model, optimizer)` only, so
> the later GBS=2304 and GBS=288 trend probes overwrote the GBS=6144 AdamW
> rows in place. The mano/sophiag/muon CSVs are unique (their optimizer ran
> only at GBS=6144) and are stamped `global_batch_size=6144, world_size=768`.

![AdamW LR-finder GBS=6144](figures/sunspot_80b_gbs6144_adamw.png)

(The finder's auto `lr_vs_loss.png` for this run is misleading -- it drops
the NaN sweep points and its derivative annotations misfire on the
near-flat 80B floor. These figures are rebuilt from the CSV to show the
real cliff.)

### Batch dependence (the reason the old number was wrong)

| Finder batch | AdamW usable-LR ceiling |
|---|---|
| GBS=192 (2026-04-21) | ~1.1e-5 (clean blow-up) |
| **GBS=6144 (this run)** | **~7.4e-7 (NaN cliff)** |

~14x lower ceiling at the production batch, AND a change of failure mode
(clean blow-up -> hard NaN cliff). A small-batch finder cannot be trusted
to set a large-batch production LR for AdamW at 80B.

### Production recommendation

1. **Do not run AdamW at LR=1e-6 at GBS=6144** -- it is on the NaN cliff.
   If staying on AdamW, use **~5e-7** (under the 7.4e-7 stable point).
2. **Strongly consider mano as the 80B production optimizer at this batch**
   -- no cliff, ~20x LR headroom, soft top. Suggested mano LR **~3e-6**
   (min/5). mano is the *safest* pick: the widest margin between a usable LR
   and divergence.
3. **sophiag is a viable second** (no longer "broken" at this batch): it has
   the lowest minimum loss (12.60) but a narrower safe band -- blow-up onset
   is only ~2x above its minimum (2.5e-6 -> 4.6e-6), vs mano's gentle rise.
   If trying sophiag, stay conservative: LR **~1e-6** (min/2.5) with tight
   grad clipping.
4. **Head-to-head convergence run DONE (2026-06-30) -- all three diverge.**
   Running each at its finder LR as a *constant* LR (GBS=6144, 64N), all
   three NaN'd within 5-12 steps: mano @ 3e-6 first (step 5), AdamW @ 5e-7
   (step 9), sophiag @ 1e-6 (step 12). Each descends a few steps, then
   grad_norm explodes and loss NaNs. The finder's early-step ranking does
   NOT predict sustained stability. **No finder LR is production-safe as a
   constant LR here** -- a long warmup (>=200 steps, vs the 2B/20B configs'
   200) plus grad clipping is needed, and it may be a bf16-at-dim=9216 wall
   requiring an fp32 grad path. Full trajectories:
   [2026-06-30 convergence report](../../../agpt/sunspot/2026-06-30-80b-convergence-gbs6144.md).

Raw per-experiment record:
[`docs/experiments/agpt/sunspot/2026-06-27-80b-lr-finder-production-batch.md`](../../../agpt/sunspot/2026-06-27-80b-lr-finder-production-batch.md).

### LR-ceiling vs GBS trend (AdamW)

The GBS=192 vs GBS=6144 jump above is two endpoints of a continuous trend.
To see *how* AdamW's usable LR collapses as the batch grows -- and where the
clean U-min turns into the hard NaN cliff -- we swept AdamW at a ladder of
batch sizes (TP=4 throughout; small-GBS points at 8N/dp=24, larger points at
64N/dp=192).

![AdamW LR-ceiling and min-loss vs GBS](figures/sunspot_80b_adamw_lr_ceiling_vs_gbs.png)

The underlying loss-vs-LR curves, one line per batch size, show the
transition directly: the small-batch U-minima (144-1152) sit near lr~1.5e-5,
while the GBS=6144 curve has slid left to lr~1e-6 and just descends to the
NaN cliff with no minimum at all:

![80B AdamW loss vs LR, all GBS](figures/sunspot_80b_adamw_loss_vs_lr_by_gbs.png)

| GBS | nodes | usable/min LR | min loss | sweep | shape |
|----:|------:|---------------|---------:|-------|-------|
| 144  | 8  | 1.0e-5  | 11.73 | 1e-6 -> 1e-3 (15/15) | clean U-min |
| 288  | 8  | 1.6e-5  | 11.81 | 1e-6 -> 1e-3 (15/15) | clean U-min |
| 576  | 8  | 1.6e-5  | 11.83 | 1e-6 -> 1e-3 (15/15) | clean U-min |
| 1152 | 64 | 1.6e-5  | 11.65 | 1e-6 -> 1e-3 (15/15) | clean U-min |
| 2304 | 64 | 4.6e-6  | 12.58 | 1e-8 -> 1e-4 (15/15) | U-min (pre-cliff shoulder) |
| 4608 | 64 | 4.6e-6  | 12.54 | 1e-8 -> 1e-4 (15/15) | U-min (pre-cliff shoulder) |
| 6144 | 64 | ~7.4e-7 | 12.78 | 1e-8 -> 1e-4 (15/15) | **NaN cliff (no min)** |

Reading the trend:

- **Usable LR falls monotonically with batch** -- from ~1.6e-5 (GBS<=576)
  toward ~7e-7 (GBS=6144), roughly a ~20x drop across a 10x batch increase.
- **The failure mode changes, not just the number.** Small batches blow up
  cleanly with a real loss minimum in-range; the production batch has *no*
  minimum -- loss descends to a wall and NaNs (the "usable LR" there is a
  ceiling, not a U-bottom).
- **The plateau holds, then steps down before the cliff.** Usable LR is flat
  at ~1.5e-5 through GBS=1152, eases to ~4.6e-6 at GBS=2304 and 4608 (a clean
  U-min "shoulder", still 0 NaN), then collapses to the ~7.4e-7 NaN cliff at
  6144. The 2304/4608 shoulder is the last clean-U stage before divergence
  swallows the minimum.
- **Min-loss is non-monotonic** (right panel): nearly flat (~11.7-11.8)
  through 1152, then steps up to ~12.5-12.6 at the 2304/4608 shoulder and
  ~12.78 at the cliff -- once the usable LR drops, the 15-step sweep can't
  descend as far.

All seven points are final (15/15, 0 NaN except the 6144 cliff). The
1152/2304/4608 points were re-run at 64N/dp=192 after the first attempts hit
walltime/watchdog limits at 16N; the original GBS=2304 partial (11/15, the
"11.24" basin reported earlier) was superseded by the clean 64N rerun
(min-LR 4.6e-6, loss 12.58).

A 2B reproduction of this same trend (16N, dp=192, GBS 192..24576) is running
-- see [agpt 2B](../2b/README.md).

---

<details closed>
<summary><b>Small-batch debug experiment (2026-04-21, GBS=192, 2-node) -- superseded by the production sweep above</b></summary>

### 2026-04-21 -- 80B + GAS sweep (Sunspot, small batch GBS=192)

First empirical 80B LR finder (previously extrapolated only), 2 nodes / 24 XPU
tiles, torch 2.13, compile disabled, seq_len=8192, LR 1e-6 -> 1.0 over 100 steps.

> **Small-batch results -- do NOT use for production.** Everything in this
> section is at GBS=192 (2-node default). The optimal LR is batch-dependent,
> and at the production batch (GBS=6144) all of these numbers change: AdamW's
> "1.13e-5 / clean" collapses to a ~7e-7 NaN cliff, and SophiaG stops being
> broken. See the [2026-06-27 production section](#2026-06-27----80b-at-the-production-batch-gbs6144-sunspot)
> and the [LR-ceiling-vs-GBS trend](#lr-ceiling-vs-gbs-trend-adamw) for the
> values that actually apply at scale. "Suggested LR" below is the
> conservative blow-up/10 heuristic (so 1.13e-5 = blow-up 1.13e-4 / 10); the
> trend section reports the loss-minimum LR directly (~1.6e-5 at these small
> batches) -- consistent, just a different definition.

| Optimizer | Suggested LR (blow-up/10) | Blow-up | NaN Count | Status @ GBS=192 |
|-----------|-------------|---------|-----------|--------|
| **AdamW** | **1.13e-5** [small batch only] | 1.13e-4 | 0/100 | Clean sweep -- but cliffs at production, see note |
| **Muon** | N/A | NaN at step 7 | 93+/100 | Broken (bf16 overflow), at any batch |
| **SophiaG** | N/A | NaN at step 7 | 93+/100 | Broken **at GBS=192 only** (see note) |

**Muon/SophiaG bf16 overflow** -- both produce NaN regardless of LR at 80B
(dim=9216): Muon's Newton-Schulz `A @ A` (9216x9216 matmul) overflows bf16,
SophiaG's Hessian estimate `grad * grad` overflows bf16. Confirmed
LR-independent (NaN even at LR=1e-8). fp32 Newton-Schulz delays Muon NaN to
step 16 but is 6x slower -- not viable. Overflow is model-size-specific: 20B
(dim=5120) works fine for both.

> **Correction (2026-06-27): the SophiaG "broken at 80B" verdict is
> batch-specific.** At the production batch GBS=6144, SophiaG runs all 15
> finder steps finite with a real minimum at lr~2.5e-6 (loss 12.60) -- the
> larger batch smooths its Hessian `grad*grad` estimate below the bf16
> overflow threshold. **Muon stays broken regardless of batch.** See the
> [2026-06-27 production-batch section](#2026-06-27----80b-at-the-production-batch-gbs6144-sunspot)
> above.

> **Correction (2026-06-27): the AdamW "1.13e-5 / clean" result does NOT
> transfer to production.** That LR is the small-batch (GBS=192) usable
> ceiling. The LR-ceiling-vs-GBS trend shows AdamW's usable LR holds ~1.5e-5
> only through GBS~1152, eases to ~4.6e-6 by 2304/4608, then collapses to a
> ~7.4e-7 NaN cliff at the production batch GBS=6144 -- where LR=1e-6 sits
> *past* the last stable point. Do not set 80B production AdamW LR from this
> 1.13e-5 number; use ~5e-7 (or prefer mano/sophiag). Same batch-dependence
> the GBS=192 finder could not have seen.

![80B small-batch finder](figures/sunspot_80b.png)

(The 2B/20B verification runs and the 2B AdamW GAS sweep that shared this
2026-04-21 job are recorded on the [2B](../2b/README.md) and
[20B](../20b/README.md) pages.)

</details>

---

## Reports index

| Date | Machine | GBS | Optimizers | Nodes | Key Result |
|------|---------|-----|-----------|-------|------------|
| [2026-06-30](../../../agpt/sunspot/2026-06-30-80b-convergence-gbs6144.md) | Sunspot | 6144 (prod) | mano, sophiag, AdamW | 64 | convergence: ALL 3 NaN in 5-12 steps at finder LRs (constant LR); needs long warmup + clipping |
| [2026-06-27](#2026-06-27----80b-at-the-production-batch-gbs6144-sunspot) | Sunspot | 6144 (prod) | AdamW, mano, muon, sophiag | 64 | AdamW cliffs @ ~7e-7; mano U-min 1.6e-5 (best); sophiag NOT broken |
| trend | Sunspot | 144..6144 | AdamW | 8-64 | usable LR falls ~20x; U-min -> NaN cliff transition |
| [2026-04-21](#2026-04-21----80b-gas-sweep-sunspot-small-batch-gbs192) | Sunspot | 192 | AdamW, Muon, SophiaG | 2 | AdamW 1.1e-5 (small batch); Muon/SophiaG NaN |
