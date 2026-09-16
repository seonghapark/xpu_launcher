# Evaluation Results — agpt 2B (Megatron-DeepSpeed SophiaG)

> **Scope:** Megatron-DeepSpeed AuroraGPT-2B SophiaG continuation
> evaluated at 5K-step intervals. One continuous 140K-step run; the
> three "stage" suffixes in the source path (`ntok4673B`,
> `ntok7064B`, `ntok7770B`) reflect data-mix transitions inside the
> single trajectory but all resolve via symlinks to the same physical
> checkpoint dir.
>
> Training-side plots (loss / grad_norm / TFLOPS / TPS) live at
> [`docs/production/agpt/2b-mds/`](../../../production/agpt/2b-mds/README.md).
>
> Last updated: 2026-05-03

## Setup

| Field | Value |
|-------|-------|
| Model | AuroraGPT-2B (Llama, 1.99B params) |
| Tokenizer | google/gemma-7b (vocab_size=256128) |
| Optimizer | SophiaG, LR=2.17e-5 (Megatron-DeepSpeed) |
| GBS | 6144 | seq_len 8192 |
| Eval tasks | hellaswag, arc_easy, arc_challenge, winogrande (0-shot, acc_norm) |
| Eval backend | lm-eval 0.4.10 + transformers 4.57.6 (HF backend, XPU, bf16) |
| Conversion | MDS `mp_rank_00_model_states.pt` → HF safetensors via `eval/mds_to_hf.py` |
| Source | `/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/optimizer-experiments/Megatron-DeepSpeed/checkpoints/...sophiag-lr2.17e-5..._ntok4673B_..._flash` |
| Steps evaluated | global_step5000, 10000, ..., 140000 (28 checkpoints) |

The data-mix stages (per the W&B
[3-stage pretraining report](https://api.wandb.ai/links/aurora_gpt/ux408056)):

| Stage label | Data slice |
|-------------|-----------|
| `ntok4673B` | first 4.673T tokens of the AdamW-trained parent |
| `ntok7064B` | next 2.391T tokens (4.673T → 7.064T) |
| `ntok7770B` | final 0.706T tokens (7.064T → 7.770T) |

All three labels point at the same on-disk SophiaG checkpoint dir, so
the lm-eval sweep ran each of the 28 steps three times — **84 result
files = 28 unique steps × 3 measurement replicates**. The replicates
differ at the ~1pp level because XPU lm-eval is not bitwise
deterministic; we average them.

## Eval Results (mean across 3 replicates)

![2B MDS Eval Results](figures/eval_2b-mds.png)

| Step | HellaSwag | ARC-Easy | ARC-Chall | Winogrande |
|-----:|----------:|---------:|----------:|-----------:|
|   5,000 | 0.3598 | 0.4822 | 0.2654 | 0.5014 |
|  10,000 | 0.4575 | 0.5219 | 0.2753 | 0.5101 |
|  15,000 | 0.5030 | 0.5533 | 0.3040 | 0.5409 |
|  20,000 | 0.5231 | 0.5634 | 0.3046 | 0.5567 |
|  25,000 | 0.5421 | 0.5800 | 0.3131 | 0.5522 |
|  30,000 | 0.5531 | 0.5821 | 0.2935 | 0.5617 |
|  35,000 | 0.5561 | 0.5919 | 0.3148 | 0.5659 |
|  40,000 | 0.5640 | 0.5819 | 0.3120 | 0.5717 |
|  45,000 | 0.5692 | 0.6052 | 0.3257 | 0.5583 |
|  50,000 | 0.5727 | 0.6000 | 0.3294 | 0.5806 |
|  55,000 | 0.5784 | 0.6083 | 0.3333 | 0.5801 |
|  60,000 | 0.5801 | 0.5960 | 0.3299 | 0.5733 |
|  65,000 | 0.5885 | 0.6013 | 0.3353 | 0.5727 |
|  70,000 | 0.5838 | 0.6230 | 0.3342 | 0.5714 |
|  75,000 | 0.5887 | 0.6055 | 0.3436 | 0.5725 |
|  80,000 | 0.5911 | 0.6093 | 0.3333 | 0.5691 |
|  85,000 | 0.5900 | 0.6180 | 0.3336 | 0.5688 |
|  90,000 | 0.5887 | 0.6244 | 0.3362 | 0.5780 |
|  95,000 | 0.6004 | 0.6660 | 0.3623 | 0.5770 |
| 100,000 | 0.5937 | 0.6675 | 0.3746 | 0.5859 |
| 105,000 | 0.5944 | 0.6953 | **0.3845** | **0.6004** |
| 110,000 | 0.5921 | 0.6637 | 0.3817 | 0.5777 |
| 115,000 | 0.5897 | 0.6937 | **0.3914** | 0.5946 |
| 120,000 | 0.5910 | **0.6985** | 0.3862 | 0.5864 |
| 125,000 | 0.5920 | 0.6921 | 0.3808 | 0.5827 |
| 130,000 | 0.5947 | 0.6918 | 0.3840 | 0.5846 |
| 135,000 | 0.5932 | 0.6514 | 0.3732 | 0.5877 |
| 140,000 | 0.5924 | 0.6658 | 0.3712 | 0.5806 |

## Observations

- **All four tasks show meaningful learning**, in contrast to our DCP
  eval pipeline (steps 1K–18K of the torchtitan-trained agpt 2B v1)
  which produced near-random scores at every checkpoint. The MDS
  trajectory uses a separate converter
  (`eval/mds_to_hf.py`) and was *not* affected by the bf16-master
  RMSNorm-freeze bug — see
  [`docs/guides/training-dtype-bf16-norm-freeze.md`](../../../guides/training-dtype-bf16-norm-freeze.md).

- **HellaSwag** has the steepest early ramp: 0.36 → 0.59 by step 60K,
  then plateaus around 0.59. Strong signal for commonsense narrative
  completion early.

- **ARC-Easy** keeps climbing throughout: 0.48 → 0.70. Peak 0.6985 at
  step 120K.

- **ARC-Challenge** has the slowest but still meaningful ramp: 0.27 →
  0.39. Random baseline 0.25. Peak 0.3914 at step 115K.

- **Winogrande** drifts upward 0.50 → 0.60 with high variance. Peak
  0.6004 at step 105K.

- **Replicate variance is small** — error bars on the figure are
  barely visible (typically ≤ 0.005 across 3 replicates), well below
  the per-task lm-eval stderr of ~0.014.

## Reproducing

```bash
# Convert + eval one MDS checkpoint
python3 torchtitan/experiments/ezpz/eval/mds_to_hf.py \
    --mds_checkpoint <path>/global_step{N}/mp_rank_00_model_states.pt \
    --output_dir outputs/evals/agpt-2b-mds/<stage>/step-{N}/hf

# Full sweep (12h walltime; submit two in parallel for ~half wall-clock)
qsub torchtitan/experiments/ezpz/scripts/eval/eval_mds_sweep.sh

# Aggregate
python3 torchtitan/experiments/ezpz/eval/aggregate_evals.py --model 2b-mds
```

> Note: the canonical eval charts are regenerated by
> `scripts/update_all_charts.sh` (via `refresh_all.sh`);
> `aggregate_evals.py` is a manual fallback.
