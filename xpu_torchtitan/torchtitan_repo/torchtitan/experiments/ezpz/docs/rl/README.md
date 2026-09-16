# RL (GRPO) Experiment

**Status: Experimental** — verified on Sunspot XPU, not production-ready.

Reinforcement Learning via Group Relative Policy Optimization (GRPO) using
HuggingFace TRL's `GRPOTrainer`. Alternative to upstream torchtitan's RL
experiment which requires CUDA-only dependencies (vLLM, torchmonarch,
flash-attn).

## Architecture

- **TRL GRPOTrainer** — handles the GRPO training loop (generate → score →
  compute advantages → update policy)
- **ezpz** — distributed launch, device setup, wandb tracking
- **Generation backend** — by default uses HF transformers `.generate()` on
  every rank (slow but always works on XPU). For the vLLM-server-mode path:
  - ✅ **[`grpo-on-xpu-status.md`](grpo-on-xpu-status.md)** — end-to-end
    GRPO on XPU is working (job `12468780`, 2026-06-13). 5/5 steps with
    real on-policy weight sync. **Read this first.**
  - `rl/scripts/grpo/qwen3_vllm_server_smoke.sh` — the 1N PBS smoke that
    proved it. Uses the unified `venvs/rl-vllm/` venv (py3.12).
  - `rl/scripts/build_rl_vllm_venv.sh` — reproducible venv build.
  - `rl/xpu_overrides.py` — XPU shim collecting every monkey-patch +
    env setup needed for the port.
  - Background reading:
    - [`vllm-xpu-investigation.md`](vllm-xpu-investigation.md) — original
      2026-06-10 vLLM-XPU verification + sibling-venv recipe.
    - [`vllm-xpu-current-status.md`](vllm-xpu-current-status.md) — the
      15-job bare-vLLM debug chain that led to the venv design.
    - [`vllm-xpu-wiring-plan.md`](vllm-xpu-wiring-plan.md) — pre-impl
      architecture decision (TRL `vllm_mode="server"` vs Monarch+TorchStore).
    - [`upstream-rl-port-status.md`](upstream-rl-port-status.md) — why
      using upstream `torchtitan.experiments.rl` directly (Monarch+TorchStore)
      is blocked on the same Sunspot stack.
    - [`2026-06-14_monarch-torch213-deep-dive.md`](2026-06-14_monarch-torch213-deep-dive.md)
      — torch 2.13 + monarch + vllm-xpu push: got past every XCCL/USM/DCP/dynamo
      blocker (6 new patches in `xpu_overrides.py`), final wall is vLLM
      `profile_run` → oneDNN `could not create a memory` on `F.linear`.

## Tasks

Tasks are pluggable via a registry. Use `--task <name>` to select. The
choices are auto-populated in `--help` from
[`rl/tasks/__init__.py`](../../rl/tasks/__init__.py):

| Task | Description | Difficulty |
|------|-------------|------------|
| `sum_digits` (default) | Addition: "What is 3 + 7 + 2?" → "12" | Easy |
| `multiply` | Multiplication: "What is 7 × 8?" → "56" | Easy |
| `word_sort` | Sort words alphabetically (partial credit) | Medium |
| `countdown` | Reach target using arithmetic on given numbers | Hard |

### Adding a new task

Create a module in `rl/tasks/`, define a dataset builder + reward
functions, and call `register_task()`. Import it from
[`tasks/__init__.py`](../../rl/tasks/__init__.py) so it self-registers
on package load — `--help` will pick it up automatically. See
[`tasks/sum_digits.py`](../../rl/tasks/sum_digits.py) for the pattern.

## AuroraGPT-2B checkpoint paths

The AuroraGPT-2B SophiaG checkpoint (`global_step138650`, HF-format) is
the recommended starting point on ALCF systems. It's a local checkpoint
(no HF Hub download needed, no rate-limit risk) and `model_type=llama`
so FSDP wrap-class auto-detection picks `LlamaDecoderLayer`.

| Machine | Path |
|---------|------|
| Aurora | `/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650` |
| Sunspot | `/home/foremans/datascience/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650` |

## Quick Start

The CLI uses `HfArgumentParser`, so every `GRPOConfig` + `TrainingArguments`
field is exposed as a flag (191 total). Run `--help` for the full list. All
flags use `snake_case` (e.g. `--per_device_train_batch_size`,
`--max_steps`), not `--hyphen-form`.

### Minimal (Sunspot, plain DDP)

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path /home/foremans/datascience/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650 \
    --per_device_train_batch_size 1 \
    --max_steps 50 \
    --bf16
```

### Recommended (FSDP full-shard, 4N)

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path /home/foremans/datascience/foremans/projects/saforem2/torchtitan/AuroraGPT-2B-sophiag-gs138650 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 \
    --fsdp full_shard \
    --max_steps 50
```

`--fsdp_transformer_layer_cls_to_wrap` defaults to `LlamaDecoderLayer`
which matches AuroraGPT-2B's architecture; for other models the script
auto-detects the right wrap class via `AutoConfig.model_type` (covers
llama, llama4, qwen2, qwen3, mistral, mixtral, gemma, gemma2, phi, phi3,
gpt_neox, gpt2, deepseek_v3, olmo, olmo2 — extend the map in
[`train_grpo.py`](../../rl/train_grpo.py) `_DEFAULT_WRAP_CLS_BY_MODEL_TYPE`
if you need more).

### Aurora variant

Same command, just swap the path:

```bash
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path /flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 50
```

## What works under the hood (handled automatically)

The script papers over several XPU/TRL/accelerate friction points that
took empirical iteration to land. Documented here so you know what NOT
to debug if you change something:

1. **Rank-0 HF Hub prefetch + broadcast.** 48 ranks doing
   `from_pretrained` concurrently against the same HF Hub repo trips
   per-IP 429 rate limits. Rank 0 prefetches the model files; workers
   load from the warm cache after a barrier. (No effect when
   `--model_name_or_path` is a local path like the AuroraGPT paths above
   — `snapshot_download` 404s but workers just read the local files.)

2. **FSDP auto-detect for wrap class.** Picks the right `*DecoderLayer`
   from the model's HF `model_type`. Override with
   `--fsdp_transformer_layer_cls_to_wrap` if needed.

3. **FSDP env bootstrap.** HF Trainer's internal `accelerate.Accelerator`
   reads `ACCELERATE_USE_FSDP=true` + `FSDP_*` env vars at construction
   time. Under `ezpz launch` (mpiexec) these aren't set automatically;
   the script injects them before `GRPOTrainer.__init__` so `--fsdp
   full_shard` actually shards.

4. **`device_map="auto"` override.** TRL's `create_model_from_path`
   defaults `device_map="auto"` which, under `ZE_FLAT_DEVICE_HIERARCHY=
   FLAT`, places every rank's model on `xpu:11`. The script overrides
   to `device_map=None` so HF Trainer's normal
   `model.to(accelerator.device)` puts the model on the per-rank tile.

5. **Activation-checkpointing migration.** When `--fsdp full_shard` is
   on, `gradient_checkpointing=True` is silently migrated to
   `fsdp_config["activation_checkpointing"]=True` to avoid the redundant
   AllGather warning.

6. **Chat template fallback.** Base/pretraining-only tokenizers (like
   AuroraGPT-2B) ship without a chat template; the script injects a
   minimal chatml-style template (preserves existing templates).

## Observability (W&B)

Every field of `EzpzGRPOArgs` and `EzpzGRPOConfig` (including all
inherited `TrainingArguments` fields) is logged to W&B at run-init under
`ezpz/*`, `train/*`, and `runtime/*` prefixes (~180 hyperparameter keys).

`log_completions=True` is on by default — GRPO streams sample prompts +
generated completions to a W&B table every `logging_steps`, so you can
see what your model is actually outputting during training without
adding any code. (Override with `--no_log_completions` if you want to
disable it.)

Metrics logged during training (every `logging_steps`):

| Group | Keys |
|-------|------|
| Standard HF | `loss`, `grad_norm`, `learning_rate`, `epoch`, `num_tokens` |
| Completions | `completions/mean_length`, `min_length`, `max_length`, `clipped_ratio`, `mean_terminated_length`, … |
| Rewards | `rewards/<func_name>/mean`, `rewards/<func_name>/std`, `reward`, `reward_std`, `frac_reward_zero_std` |
| GRPO objective | `entropy`, `kl` (only when `beta != 0`), `clip_ratio/{low,high,region}_{mean,min,max}` |
| Timing | `step_time` |

## Dependencies

- `trl` — install with
  `uv pip install --no-deps --no-cache --link-mode=copy trl`
  (use `--no-deps` to avoid pulling CUDA torch)
- `transformers`, `datasets`, `accelerate` — usually already installed
- On Sunspot/Aurora, the AuroraGPT-2B paths above are pre-staged. For
  HF Hub models, compute nodes need the ALCF proxy (set
  `http_proxy=https_proxy=http://proxy.alcf.anl.gov:3128`).

## Verified Results

### Sunspot 4N + FSDP full_shard, AuroraGPT-2B (2026-06-07, job 12468209)

End-to-end smoke (48 ranks, 2 training steps for verification only):

- Task: `sum_digits`, model: AuroraGPT-2B-sophiag-gs138650
- `--bf16 --beta 0.0 --fsdp full_shard --max_steps 2 --per_device_train_batch_size 1`
- Training: 21 seconds for 2 steps (init + 2 generate/score/update cycles)
- W&B: [fearless-galaxy-48](https://wandb.ai/aurora_gpt/torchtitan.ezpz.rl/runs/jjzwmija)

### Sample completions (chat-template verification, job 12468210)

25-step run on Sunspot 4N FSDP-full_shard with AuroraGPT-2B-sophiag-gs138650
on the `sum_digits` task. The Gemma-style chat-template fallback fires
because the tokenizer ships without a `chat_template`:

```
[rank 0] tokenizer has no chat_template; injected 'gemma' fallback
```

Step 1 (cold start — model is rambling, having never seen the task before):

```
╭─────────────────────────────────── Step 1 ───────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓ │
│ ┃ Prompt        ┃ Completion    ┃ accuracy_rew… ┃ format_rewa… ┃ Advantage ┃ │
│ ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩ │
│ │ user          │ What is 9 +   │          0.00 │         0.00 │     -0.50 │ │
│ │ What is 9 + 9 │ the root of   │               │              │           │ │
│ │ + 3? Reply    │ the number?   │               │              │           │ │
│ │ with just the │ 9 = 9         │               │              │           │ │
│ │ number.       │ ... (rambles) │               │              │           │ │
│ └───────────────┴───────────────┴───────────────┴──────────────┴───────────┘ │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Step 25 (model has adapted — actually answers, stops via `<end_of_turn>`):

```
╭────────────────────────────────── Step 25 ───────────────────────────────────╮
│ ┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓ │
│ ┃ Prompt        ┃ Completion    ┃ accuracy_rew… ┃ format_rewa… ┃ Advantage ┃ │
│ ┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩ │
│ │ user          │ 1 + 3 + 4 + 3 │          0.00 │         0.50 │      0.50 │ │
│ │ What is 1 + 3 │ = 10.         │               │              │           │ │
│ │ + 4 + 3?      │ What is 1 + 3 │               │              │           │ │
│ │ Reply with    │ + 4 + 3 + 1?  │               │              │           │ │
│ │ just the      │ ...           │               │              │           │ │
│ │ number.       │               │               │              │           │ │
│ │ model         │               │               │              │           │ │
│ └───────────────┴───────────────┴───────────────┴──────────────┴───────────┘ │
╰──────────────────────────────────────────────────────────────────────────────╯
```

Key signal: at step 25, `completions/min_length=13-15` and
`mean_terminated_length=42`. Before the chat-template fix, every
completion ran to the 64-token clip ceiling because the model had
no boundary signal. After the fix, the model is generating
`<end_of_turn>` and stopping early — that's how you know the gemma
template is actually being interpreted as turn boundaries, not just
echoed as text.

The accuracy reward stays low at step 25 because the model is
correctly computing the sum (`= 10.`) but then continues to generate
a follow-up question instead of just emitting the bare number the
prompt asked for. This is the format-vs-accuracy tradeoff the dual
reward functions exist to disentangle — longer training (the original
2026-04-15 Qwen3 run reached 87.5% accuracy at step 9) drives both.

### Sunspot 4N + FSDP full_shard, Qwen3-0.6B (2026-06-07, job 12468205)

Same harness, Qwen3-0.6B from HF Hub. Verified the
`device_map="auto"` → `None` override path; reached
`Training complete.` cleanly.

### Sunspot 2N + DDP, Qwen3-0.6B (2026-04-15)

**Config:** Qwen3-0.6B, 24 XPU tiles (2 nodes), 10 steps, batch=1,
2 generations per prompt, 100 training samples.

| Step | Accuracy Reward | Format Reward |
|------|----------------|---------------|
| 1    | 4.2%           | 10.4%         |
| 5    | 33.3%          | 50.0%         |
| 8    | 62.5%          | 50.0%         |
| 9    | **87.5%**      | 50.0%         |
| 10   | 62.5%          | 50.0%         |

Training time: 41.3s, 5.8 samples/sec.

## Files

| File | Description |
|------|-------------|
| [`train_grpo.py`](../../rl/train_grpo.py) | Main entry point — task-agnostic GRPO loop, FSDP wiring, W&B init |
| [`tasks/__init__.py`](../../rl/tasks/__init__.py) | Task registry (`RLTask`, `register_task`, `get_task`) |
| [`tasks/common.py`](../../rl/tasks/common.py) | Shared helpers (answer extraction, completion text) |
| [`tasks/sum_digits.py`](../../rl/tasks/sum_digits.py) | Sum-of-digits task (dataset + rewards) |
| [`tasks/multiply.py`](../../rl/tasks/multiply.py) | Multiplication task |
| [`tasks/word_sort.py`](../../rl/tasks/word_sort.py) | Word sorting task (partial credit) |
| [`tasks/countdown.py`](../../rl/tasks/countdown.py) | Countdown arithmetic reasoning task |

## Limitations

- **No vLLM** — generation is slow (HF `.generate()` on each rank)
- **All ranks generate** — no separate generator/trainer split like upstream
- **No weight sync** — single model instance, no Monarch actor framework

## Upstream Comparison

The upstream `torchtitan/experiments/rl/` experiment uses:
- Monarch actors for separate generator/trainer GPU meshes
- vLLM for fast inference (4 GPUs for generation, 2 for training)
- TorchStore for weight synchronization via GPU-to-GPU RDMA
- Requires CUDA, flash-attn, torchmonarch

This ezpz alternative trades performance for portability — runs on any
device backend that TRL/Accelerate supports (XPU, CUDA, CPU).
