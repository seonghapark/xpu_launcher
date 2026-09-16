# Production GRPO

> GRPO-tuned checkpoints derived from SFT'd or pre-trained AuroraGPT
> models. Each entry is a recipe + checkpoint pair: a specific task
> applied to a specific starting model.

## Index

| Base | Recipe | Status | Reward (last 10 mean) | Steps | Checkpoint | Trajectory |
|---|---|---|---:|---:|---|---|
| `aurora2b-sft-tulu-mix-step729` | [sum_digits arithmetic](aurora2b/sft_arithmetic/README.md) | **complete** | TBD | 1000 | `outputs/grpo/aurora2b-sft-arithmetic-8n/checkpoint-1000/` | 8N, sum_digits, lr=1e-6, 2026-06-11 |
