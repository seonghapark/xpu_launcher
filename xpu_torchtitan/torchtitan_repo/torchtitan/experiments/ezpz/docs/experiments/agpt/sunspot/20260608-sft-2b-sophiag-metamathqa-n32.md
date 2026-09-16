# 32N SFT: AuroraGPT-2B-sophiag-138650 + metamathqa

**Date:** 2026-06-08
**Machine:** Sunspot
**Job:** (filled in after submission)
**Config:** 4N → 32N scaleup of the verified 50-step SFT smoke
([12468212 path](20260607-sft-smoke-n4-aurora2b-gsm8k.md), if that doc exists).

## Goal

Produce a real SFT checkpoint over the
`AuroraGPT-2B-sophiag-gs138650` pre-trained model so the existing
GRPO loop (`--task arithmetic`, etc.) has a stronger
instruction-following starting point than the raw pretraining ckpt.
The 4N gsm8k smoke earlier (12468212) only ran 50 steps and was
purely a wiring test; gsm8k's 7473 examples is also too small to
saturate 32N (would be <10 steps/epoch).

Switch to **meta-math/MetaMathQA** (~395k augmented GSM8K+MATH
examples) — at 32N × `per_dev_bsz=1` × `packing=True` that's
~260 steps/epoch, so 3 epochs ≈ **~780 training steps** total.
Estimated wall time: ~15-25 min training + ~2 min init.

## Config

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset metamathqa \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --output_dir outputs/sft/aurora2b-sophiag-metamathqa-32n \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 100 \
    --save_total_limit 8 \
    --report_to wandb
```

Submitted via
[`rl/scripts/sft/aurora2b_metamathqa_32n.sh`](../../../../rl/scripts/sft/aurora2b_metamathqa_32n.sh).

## Pre-launch knowns

- **Pipeline already validated**: same `train_sft.py` + chat-template
  + FSDP wiring as the 4N gsm8k smoke (job 12468212, 2026-06-07).
  That run hit `Training complete.` at 50 steps with
  `mean_token_accuracy=0.871` and `loss=0.582`. Nothing changes
  between that and this run except dataset + node count.
- **`assistant_only_loss=True`** is the default in
  `EzpzSFTConfig.__post_init__`; works because all 3 chat-template
  fallbacks (gemma/chatml/plaintext) wrap the assistant turn in
  `{% generation %}` markers per
  [commit `5446e1d74`](../../../../rl/train_grpo.py).
- **Packing** is on (`SFTConfig` default for the
  `EzpzSFTConfig` subclass). MetaMathQA responses are typically
  <500 tokens so 1024-token packing should pack 2-4 per sequence.
- **Checkpoint saves**: every 100 steps + final. ~8 intermediate
  ckpts at ~4 GB each (bf16 model + optim state via sharded state
  dict). Final checkpoint at `outputs/sft/.../final/`.
- **No streaming on TRL SFTTrainer**: same as GRPOTrainer
  (see [trl#3213](https://github.com/huggingface/trl/issues/3213)).
  metamathqa is materialized via HF `datasets` `.map()` — first
  build is cached.

## What I'm watching for

1. **`Training complete.` line** + the final `train_loss` /
   `mean_token_accuracy` from the wrap-up dict.
2. **Loss trajectory** — should descend from ~1.5-2 (cold start on
   the new dataset) to <0.5 by end of epoch 1 if the SFT is
   actually adapting the model.
3. **Checkpoint sequence** — `step-100`, `step-200`, ...,
   `step-700`, `final/` all on disk. Each save should land in
   <60 s for the bf16 2B model.
4. **Chat-template behavior in the W&B logs** — `train_loss`
   should be strictly lower than the 4N gsm8k smoke's
   `train_loss=0.582` since metamathqa is a much bigger dataset
   and the model can actually generalize.

## Post-run

Once the run completes:

1. **Evaluate** the final ckpt by pointing
   `train_grpo --task arithmetic` at it as
   `--model_name_or_path
   outputs/sft/aurora2b-sophiag-metamathqa-32n/final` and
   running the same 25-step verification harness from
   [`20260607-sft-smoke-n4-aurora2b-gsm8k.md`](20260607-sft-smoke-n4-aurora2b-gsm8k.md)
   (if it exists; otherwise use the prompt format from the
   parent `docs/rl/README.md` "Sample completions" section).
2. **Compare** the step-1 completion table from the SFT'd ckpt vs
   the raw sophiag ckpt — should see far fewer "What is the load
   bearing wall?" cold-start derailments.
3. **Append actual results** to this doc:
   - final `train_loss` / `train_runtime` / `samples_per_second`
   - `mean_token_accuracy` final
   - W&B URL
   - any failure modes

## Result

(to be filled in after the run completes)
