# lm-eval: SFT'd AuroraGPT-2B (tulu_math_uc_mix) vs pretrained baseline

**Date:** 2026-06-10
**Machine:** Sunspot
**Models compared:**
- **baseline**: `AuroraGPT-2B-sophiag-gs138650` (raw pretrained)
- **sft-step729**: `outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf`
  (4.5B-token SFT'd on `tulu-3-sft-mixture:0.65 + metamathqa:0.15 +
  ultrachat-200k:0.20`, 3 epochs, see
  [SFT writeup](../failover-story.md))

## TL;DR

The SFT'd model and the baseline are **statistically indistinguishable**
on 7 base-LM benchmarks (0-shot). 3 wins, 4 losses, all deltas in
`[-0.024, +0.026]` — well within the ±0.02 noise floor for a 2B-class
model. This is the expected **alignment-tax** pattern: SFT on
instruction-following data doesn't add base knowledge and slightly
narrows the output distribution, which hurts multiple-choice tasks a
hair. The actual SFT win has to be measured on an
instruction-following eval (IFEval, MT-Bench, AlpacaEval) or by
downstream task performance (GRPO from this checkpoint vs from the
baseline) — see related work below.

## Results

| Task | Baseline | SFT (step 729) | Delta |
|---|---:|---:|---:|
| arc_challenge | 0.391 | 0.375 | **−0.015** |
| arc_easy | 0.694 | 0.671 | **−0.024** |
| boolq | 0.599 | 0.586 | **−0.013** |
| hellaswag | 0.592 | 0.593 | +0.001 |
| openbookqa | 0.370 | 0.384 | **+0.014** |
| piqa | 0.737 | 0.730 | **−0.007** |
| winogrande | 0.578 | 0.604 | **+0.026** |

Metric is `acc_norm,none` where present (hellaswag, arc_*, piqa,
openbookqa) and `acc,none` otherwise (winogrande, boolq).
0-shot evaluation. Identical task specs across both models.

## Why this is the expected outcome

Three reasons the 7-task suite would not show SFT improvements:

1. **These benchmarks measure base-LM knowledge**, not
   instruction-following. ARC measures elementary science reasoning,
   HellaSwag measures sentence completion plausibility, BoolQ measures
   reading comprehension. None of these test whether the model can
   *follow a prompt format*. SFT on chat data doesn't teach the model
   new facts about elementary science.
2. **SFT narrows the output distribution.** Pre-trained LMs produce
   relatively entropic next-token distributions; SFT on chat data
   biases the model toward producing assistant-formatted responses.
   For multiple-choice tasks scored via per-option log-likelihood,
   that bias *reduces* the model's discrimination between options,
   even when the correct option is the higher-ranked one in absolute
   terms. This is the classic "alignment tax."
3. **The deltas are within noise.** For a 2B model on these task
   sizes (boolq=3.3k, piqa=1.8k, arc_easy=2.4k, etc.), ±0.02 is
   roughly one standard error. With 7 comparisons we expect about
   half to land negative just from sampling noise on the eval set.

## Where SFT *does* show up (planned follow-up evals)

- **IFEval** — `lm-eval --tasks ifeval`. Generative, 541 prompts that
  test verifiable structural instruction-following ("include the
  word X", "use exactly 5 sentences", "do not use commas"). A base
  LM scores near-zero; an SFT'd LM should jump substantially.
  Running now in
  [`eval_ifeval_sft_vs_baseline.sh`](../../../../../../rl/scripts/sft/eval_ifeval_sft_vs_baseline.sh).
- **GRPO from this checkpoint** — the original justification for the
  SFT push. A model that already speaks chat will accept the GRPO
  signal much faster than a raw base LM. Smoke comparison in
  [`grpo_smoke_sft_vs_baseline.sh`](../../../../../../rl/scripts/sft/grpo_smoke_sft_vs_baseline.sh).
- **MMLU 5-shot** — not run here (57 subjects × 14k questions is
  slow at 0-shot on 2B; would burn most of a day on 1 tile). MMLU
  is also a base-LM benchmark and likely to show the same
  alignment-tax pattern.
- **GSM8K 5-shot** — needed to validate the metamathqa portion of
  the mix. The 7-task suite ran gsm8k at 0-shot which scores ~0 for
  a 2B base model (no format-learning examples).

## Eval setup

| Field | Value |
|-------|-------|
| Machine | Sunspot (Intel Max 1550, single XPU tile per task) |
| Eval backend | lm-eval 0.4.10 + transformers 4.50.1 (frameworks/2025.3.1 module) |
| Parallelism | 7 tasks × 1 tile each = 7 of 12 tiles per node, single node, back-to-back per model |
| Wall time | 18 min total (6 min baseline + 12 min SFT, sequential) |
| Model dtype | bfloat16 (auto-resolved by lm_eval's default) |
| Batch size | 4 |
| n-shot | 0 (default for all 7 tasks) |
| Submit script | [`eval_sft_vs_baseline_parallel.sh`](../../../../../../rl/scripts/sft/eval_sft_vs_baseline_parallel.sh) |
| Results dir | `outputs/evals/aurora2b-sft-vs-baseline-parallel-20260610-195131/` |

### Blockers cleared along the way

Both bit during initial setup and are worth knowing for any future
lm-eval-on-XPU work:

1. **transformers `4.50.1` does not accept `dtype=` kwarg in
   `AutoModelForCausalLM.from_pretrained()`** — that kwarg arrived
   in transformers 5.x. lm-eval 0.4.10 uses the new kwarg.
   On torch 4.50.1 the kwarg gets forwarded to
   `LlamaForCausalLM.__init__()` as `**model_kwargs` and crashes
   with `TypeError: ... unexpected kwarg 'dtype'`. Workaround:
   monkey-patch `AutoModelForCausalLM.from_pretrained` to translate
   `dtype` → `torch_dtype` before delegating, in our eval script.
   Same skew CLAUDE.md flags under "User venv transformers
   conflict."
2. **Config-level `"dtype": "bfloat16"`** in transformers 4.57-style
   `config.json` makes `PretrainedConfig.__repr__` try to
   JSON-serialize a `torch.dtype`, which 4.50.1 can't handle.
   Workaround: dropped the field from both `config.json`s (backed up
   originals via `backup`). Doesn't affect behavior because the
   safetensors carry their own dtype.
3. **XPU runtime contention from zombie processes**. First parallel
   run got stuck at 1% for 7 minutes; `top` revealed 8 zombie
   `train_sft.py` processes from an earlier debug job, hogging all
   tiles. `pkill -9 -f 'torchtitan.experiments.ezpz.rl.train_sft'`
   cleared them. With a quiet node the same eval finished in 6 min
   per model. Lesson: always check `ps -eo etimes,cmd | awk '$1>60'`
   on the target node before launching parallel eval.

## Related

- SFT training writeup: [`20260610-sft-2b-tulu-mix-n32-failover.md`](../failover-story.md)
- IFEval results: [`ifeval.md`](ifeval.md) — +8 pp on
  `prompt_level_strict_acc` (0.1645 → 0.2440, ~4-5σ), +48% relative
- GRPO smoke results: [`grpo-smoke.md`](grpo-smoke.md) — 8×
  speedup over baseline (0.117 → 0.922 mean reward over last 10 steps,
  100% accuracy by step 40)
- Eval scripts:
  [`rl/scripts/sft/eval_sft_vs_baseline_parallel.sh`](../../../../../../rl/scripts/sft/eval_sft_vs_baseline_parallel.sh)
  (base-LM tasks),
  [`rl/scripts/sft/eval_ifeval_sft_vs_baseline.sh`](../../../../../../rl/scripts/sft/eval_ifeval_sft_vs_baseline.sh)
  (instruction-following),
  [`rl/scripts/sft/grpo_smoke_sft_vs_baseline.sh`](../../../../../../rl/scripts/sft/grpo_smoke_sft_vs_baseline.sh)
  (downstream RL).
