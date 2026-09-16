# MoE Training Configs

## Model Configs

All configs use `vocab_size=256128` (Gemma tokenizer) and `seq_len=8192`.

| Config | Total Params | Active Params | Layers | Experts | Top-K | Dim |
|--------|-------------|--------------|--------|---------|-------|-----|
| `moe_debugmodel` | 0.05B | 0.04B | 6 | 8 | 3 | 256 |
| `moe_500m` | 0.25B | 0.14B | 12 | 16 | 3 | 512 |
| `moe_2b` | 1.61B | 0.49B | 18 | 24 | 3 | 1024 |
| `moe_4b` | 2.89B | 0.81B | 22 | 24 | 3 | 1536 |
| `moe_7b` | 7.54B | 1.57B | 24 | 36 | 3 | 2048 |
| `moe_10b_2b` | 9.41B | 1.98B | 27 | 36 | 3 | 2048 |

### EP-enabled variants

Use `AllToAllTokenDispatcher` for expert parallelism (EP > 1):

| Config | Base | Default EP |
|--------|------|-----------|
| `moe_debugmodel_ep` | debugmodel | 2 |
| `moe_2b_ep` | 2B | 2 |
| `moe_7b_ep` | 7B | 2 |
| `moe_10b_2b_sdpa_ep` | 10B_2B_sdpa | 2 |

Override EP: `--parallelism.expert_parallel_degree=N`

Valid EP values must divide `num_experts`. At 24 XPU tiles:
- 8 experts: EP = 2, 4, 8
- 24 experts: EP = 2, 3, 4, 6, 8, 12, 24
- 36 experts: EP = 2, 3, 4, 6, 12

### Special variants

- `moe_10b_2b_sdpa` — uses SDPA instead of FlexAttention (avoids fp32 autocast on XPU)
- `moe_*_from_json` — applies JSON overrides from `TT_CONFIG_JSON` env var

## Quick Start

```bash
# Basic training
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_7b --checkpoint.no-enable --training.steps 100

# With EP
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_7b_ep --checkpoint.no-enable --training.steps 100

# LR finder
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module ezpz.moe --config moe_7b --checkpoint.no-enable \
  --lr_finder.enable --lr_finder.init_lr 1e-6 --lr_finder.max_lr 1.0
```

## JSON Config Overrides

Set `TT_CONFIG_JSON` to a JSON file for environment-driven overrides:

```bash
TT_CONFIG_JSON=torchtitan/experiments/ezpz/moe_runs/deepseek_v3_10b2b_ep12_2nodes_smoke.json \
  ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.moe --config moe_10b_2b_from_json --checkpoint.no_enable
```

Pre-configured JSON overrides in `moe_runs/`:
- `*_smoke.json` — quick 40-step test
- `*_perf.json` — throughput measurement
- `*_prod_sim*.json` — production simulation configs
- `*_128nodes*.json` — large-scale configs

See `moe_runs/README.md` for the full list.

## Known Limitations

- **AC incompatible with MoE 7B+** — routing is non-deterministic, causing
  shape mismatches on AC recomputation. Use `--activation_checkpoint.mode none`.
- **FlexAttention on XPU** — `torch.autocast(dtype=fp32)` in MoE router crashes
  on XPU. Use `_sdpa` variants instead.
- **10b_2b_sdpa OOMs at seq_len=8192** on 2 nodes with EP=1. Use EP=2+ or
  reduce seq_len.
- **Muon very slow on small MoE** — Newton-Schulz overhead dominates for
  debugmodel/500M.

## Experiment Reports

See [experiments/moe/README.md](../experiments/moe/README.md) for benchmark
reports, and [experiments/lr-finder/moe/](../experiments/lr-finder/moe/) for
LR finder results.

## LR Finder Summary (Sunspot, torch 2.13)

| Config | AdamW | Muon | SophiaG |
|--------|-------|------|---------|
| debugmodel | stable | slow | stable |
| 500M | stable | slow | stable |
| 2B | stable | stable | stable |
| 4B | stable | stable | 1 NaN (high LR) |
| 7B | stable | stable | 3-5 NaN (high LR) |
| 10b_2b_sdpa (EP=2) | stable | stable | — |

All optimizers numerically stable on MoE (dim ≤ 2048, below bf16 overflow
threshold). SophiaG has mild instability at high LR only.
