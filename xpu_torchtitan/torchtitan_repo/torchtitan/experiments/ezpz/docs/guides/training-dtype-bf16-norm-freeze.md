# `training.dtype = bfloat16` silently freezes RMSNorm weights

> **Severity:** all completed agpt production training runs are
> affected. Loss curves look real but every RMSNorm parameter is
> stuck at its 1.0 init, and downstream lm-eval is flat at the
> random baseline regardless of token count. See "Empirical evidence"
> below for the v1-vs-v2 comparison.
>
> Discovered: 2026-04-29. Default flipped 2026-04-30.

## Summary

When `training.dtype = "bfloat16"` (the default in
`agpt/config_registry.py` and `moe/config_registry.py` until commit
`<TBD>`), the master parameter copy is bf16 (no fp32 master). For
`RMSNorm.weight` initialized to `1.0`, the bf16 ULP at scale 1.0 is
`2^-7 ≈ 7.8e-3`, but the per-step optimizer update for those
parameters is much smaller (`lr * exp_avg / sqrt(hessian) ≈ 1.6e-5`
for the 2B SophiaG runs). Every update rounds to zero in bf16, so
`norm.weight` never moves from `1.0` for the entire training run.

This applies to all 25 RMSNorm parameters in the 2B model
(`norm.weight`, `layers.{i}.attention_norm.weight`,
`layers.{i}.ffn_norm.weight` for `i = 0..11`). All are exactly `1.0`
in every position from step 100 through step 17,400 of the
`agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072` checkpoints.

The 20B and any other agpt/moe runs with the same default dtype
have the same bug.

## How to confirm on a checkpoint

```python
import torch, torch.distributed.checkpoint as dcp

ckpt = "outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs3072/step-1000"
sd = {"norm.weight": torch.zeros(2048)}
dcp.load(sd, checkpoint_id=ckpt)
print(sd["norm.weight"][:5])     # -> tensor([1., 1., 1., 1., 1.])
print(torch.allclose(sd["norm.weight"], torch.ones(2048)))  # -> True
```

Even at step 17,400 the same tensor is uniform 1.0. By contrast,
`layers.0.attention.qkv_linear.wq.weight` shows real changes
(max abs diff ~1.6e-2 = 2 bf16 ULPs at scale ~0.001).

## Why other parameters update fine

Linear layers initialize at small scales (`std ≈ 0.02` for q/k/v,
`std ≈ 0.005` for o). A bf16 ULP at scale `0.005` is `~3.8e-5`,
roughly the same magnitude as the per-step update — so updates can
register, especially after a few steps of accumulation. RMSNorm
weights are 100x larger (1.0 vs 0.01), so their ULP is 100x coarser
relative to the same update size.

## Why optimizer state shows non-zero exp_avg / hessian

The optimizer DOES receive gradients and DOES accumulate exp_avg /
hessian. The update is computed correctly. The failure is in the
final write to the parameter: the ~1.6e-5 update rounds to zero
when added to a bf16 1.0.

We confirmed `optimizer.state.norm.weight.exp_avg ≈ 2.78e-04` and
`hessian ≈ 1.52e-07` at step 1000 — both larger than the
`qkv_linear.wq.weight` equivalents (`5.15e-06` and `2.45e-10`).

## Fix

`agpt/config_registry.py` and `moe/config_registry.py` now default
`dtype = "float32"`. With FSDP `MixedPrecisionPolicy(param_dtype=bf16,
reduce_dtype=fp32)`:

- Master params: fp32 (sub-ulp accumulation works)
- Forward / backward all-gather: bf16 (memory-efficient)
- Gradient reduce: fp32 (numerical stability)
- Optimizer step: on fp32 master

Memory cost: ~1 GB extra master at 2B, ~10 GB at 20B. Well under
budget at production scale.

## Existing checkpoints

The 2B step-33,700 and 20B step-4,100 checkpoints inherit the
problem — every RMSNorm.weight is exactly 1.0. Resuming with the
new default doesn't retroactively fix them; the optimizer state
still has stale `exp_avg` / `hessian`. Two paths forward:

1. **Continue training from existing checkpoint with `dtype=float32`** —
   norm weights start at 1.0 (already-trained value, same as init),
   then begin accumulating real updates from this point on. Loss
   trajectory should slowly diverge from what we'd see resuming with
   `dtype=bf16`.

2. **Restart from scratch** — only justified if we conclude that
   17K (resp. 4K) steps of stuck-norm training is so degenerate
   that resuming makes no sense.

We chose **path 2** for the 2B/20B production runs — see "Empirical
evidence" below for why a clean restart was the right call.

## Empirical evidence (v1 vs v2)

After flipping the default to `float32`, we restarted 2B and 20B
production from scratch as **v2** runs. The v1 (bf16-master, broken)
and v2 (fp32-master, fixed) trajectories share the same architecture,
optimizer (SophiaG), dataset (blendcorpus olmo-mix-1124), and seed.
The only difference is the master-weight dtype — making this a
controlled A/B for the bug's downstream impact.

### Training-loss overlays (W&B)

Both v1 and v2 *training loss* curves descend, because attention/FFN
weights are unfrozen and the loss is dominated by them. The bug only
shows up when you actually exercise the model on held-out data:

- 2B overlay:
  [`docs/production/agpt/2b/figures/overlay_2b_v1_vs_v2.png`](../production/agpt/2b/figures/overlay_2b_v1_vs_v2.png)
  ([source page](../production/agpt/2b/README.md))
- 20B overlay:
  [`docs/production/agpt/20b/figures/overlay_20b_v1_vs_v2.png`](../production/agpt/20b/figures/overlay_20b_v1_vs_v2.png)
  ([source page](../production/agpt/20b/README.md))

Generated by `python3 torchtitan/experiments/ezpz/utils/plot_production_wandb.py --overlay {2b,20b}`.

### Downstream lm-eval (the smoking gun)

The benchmark-accuracy comparison is where the freeze becomes
unmistakable. v1 hovers within ~1pp of the random baseline on every
task across all 18K steps (~453B tokens) — the model has no
trainable normalization, so it cannot learn the structured
representations that downstream tasks need. v2 descends visibly once
it has enough tokens.

- 2B v1-vs-v2 figure:
  [`docs/evals/agpt/2b/figures/eval_overview.png`](../evals/agpt/2b/figures/eval_overview.png)
  ([full writeup](../evals/agpt/2b/README.md))
- 20B v1-vs-v2 figure:
  [`docs/evals/agpt/20b/figures/eval_overview.png`](../evals/agpt/20b/figures/eval_overview.png)
  ([full writeup](../evals/agpt/20b/README.md))

Plot scripts: `docs/evals/agpt/{2b,20b}/plot_eval_overview.py`.

### Headline result — 2B at 503B tokens

Per the [2B production page](../production/agpt/2b/README.md):

> at 503B tokens v2 is **+19.8pp** on ARC-Easy and **+15.4pp** on
> HellaSwag vs v1's flat baseline at any token count.

For context, v1's mean across steps 1K–18K (≈453B tokens) was
ARC-Easy 27.23 and HellaSwag 25.20 (both percentages, both within
~2pp of the random baseline). See the
[v1 results table](../evals/agpt/2b/README.md) for the per-step
breakdown — v1 oscillates in noise and never escapes the
~25-27% range.

v1's flat trajectory across hundreds of billions of tokens is the
clearest signature of the bug — a model with frozen normalization
cannot improve on the things lm-eval measures, no matter how much
data you give it. v2 reaches numbers that v1 never approaches in
ten times the tokens.

### Why we chose restart-from-scratch

Path 1 (continue from the bf16-tainted checkpoint with `dtype=fp32`)
would let RMSNorm.weight finally start accumulating updates, but
every other parameter has been trained against frozen-norm gradients
for tens of thousands of steps. The optimizer state and the param
manifold are jointly degenerate. The lm-eval evidence above shows
that 17K-33K steps of frozen-norm training produces a *qualitatively*
different model that doesn't learn — it would have been
indistinguishable from a model that's just slow to converge until we
ran the v2 control. Restarting was cheaper than trying to recover.

## Related

- The "DCP -> HF converter has an embedding bug" diagnosis in earlier
  versions of `docs/evals/agpt/2b/README.md` was wrong. The converter
  is innocent; the original observation (`tok_embeddings.weight[0]`
  identical across checkpoints) was a frozen padding-token row, not
  a generic stuck-embedding issue. Embeddings DO update for
  non-padding tokens.

- bf16 master weights in distributed training is a recognized
  general issue; see torchtitan #600 and the FSDP2 docs on
  `MixedPrecisionPolicy`.
