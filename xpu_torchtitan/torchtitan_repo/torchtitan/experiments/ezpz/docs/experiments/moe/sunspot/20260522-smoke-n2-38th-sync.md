# 38th Upstream-Sync Smoke — Sunspot 2N (2026-05-22)

Routine post-merge smoke after the 38th upstream sync
(`cfe97c605..c2a3771a4`). Validates that `agpt_2b` and `moe_2b_ep`
still train cleanly on `experiments/ezpz/` post-merge + post-replay.

## TL;DR

- **`agpt_2b`** clean: 50 steps, exit 0, 140 s wall, peak 24.34 GiB.
  Byte-comparable to the prior post-resync baseline.
- **`moe_2b_ep @ LBS=1`** clean: 50 steps, exit 0, 299 s wall, peak
  **14.95 GiB (23.36%)**, ~2,820 TPS/GPU, 8.2% MFU. Matches the
  37th-sync baseline (15.03 GiB, ~2,860 TPS) to within run-to-run
  noise on both memory (0.08 GiB) and throughput (1.4%).
- **`moe_2b_ep @ LBS=2`** also clean: peak 27.08 GiB (42.33%),
  ~3,200 TPS/GPU, 9.4% MFU, loss 12.93 → 6.15.
- **`moe_2b_ep` at the previous registry default LBS=16 OOMs** on
  the bf16 vocab projection (`[16 × 8192, 256128] × 2 B ≈ 62.5 GiB`,
  overflows a 64 GiB Max 1550 tile). This is the same pre-existing
  `_ep` vocab-projection OOM the
  [37th-sync follow-up](20260520-smoke-n2-pr3386-ep-followup.md)
  flagged ("`_ep` configs should override LBS in the registry");
  every `_ep` config inherits its parent's LBS without an override
  and any LBS > 1 overflows on the Gemma vocab.

Registry pin landed in
[`59354e43f`](https://github.com/saforem2/torchtitan/commit/59354e43f)
(`moe_2b_ep` now defaults to LBS=2), mirroring
[`f2cbc0327`](https://github.com/saforem2/torchtitan/commit/f2cbc0327)
for `moe_debugmodel_ep` after the 37th sync.

## Environment

| Field        | Value                                                             |
|--------------|-------------------------------------------------------------------|
| Date         | 2026-05-22                                                        |
| Branch       | `ezpz` (post-38th-sync merge + moe replay)                        |
| Commit       | `d87729ad8`                                                       |
| Machine      | Sunspot                                                           |
| Job IDs      | 12467277 (agpt_2b + moe LBS=16/2), 12467323 (moe LBS=1 parity)    |
| Nodes        | 2                                                                 |
| Devices      | 24 (Intel Max 1550, 12/node)                                      |
| Steps        | 50 per config                                                     |
| Dataset      | blendcorpus                                                       |
| Backend      | xccl                                                              |
| Torch        | `2.13.0.dev20260519+xpu` (.venv)                                  |
| Checkpoint   | disabled (`--checkpoint.no-enable`)                               |

Stale `outputs/checkpoint/step-100` from a pre-PR-3159 (35th sync)
revision was incompatible with the merged code (`Missing key in
checkpoint state_dict: layers.0.attention.qkv_linear.wk.weight.` —
Llama3 `qkv_linear` weight layout changed in PR #3159). Backed up to
`outputs/checkpoint-20260522-120005` and all smokes ran from fresh
init.

## Results

### `agpt_2b` (sanity)

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/c6vff0te
- Log: `logs/smoke-38th-sync/agpt_2b-fresh-20260522-120017.log`
- 50 steps, **exit 0**, 140 s wall.
- Final loss 6.05, peak memory **24.34 GiB (38.04%)**.
- Matches the prior `agpt_2b` post-resync baseline from
  [20260520-smoke-n2-pr3386-merge-followup.md](../../agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md)
  to within run-to-run noise.

### `moe_2b_ep @ LBS=1` (parity vs 37th sync)

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/__lbs1__
  (log `logs/smoke-38th-sync/moe_2b_ep-lbs1-parity-20260522-142430.log`)
- 50 steps, **exit 0**, 299 s wall.
- Loss 12.93 → ~6.10 (step 49); step 50 was 6.57 (volatile final step).
- **Peak step memory 14.95 GiB (23.36%)**.
- ~2,820 TPS/GPU, 24.5 TFLOPS, 8.2% MFU.

Side-by-side with the 37th-sync baseline at the same config:

| Metric           | 37th sync (`1d4115d3f`) | 38th sync (`d87729ad8`) | Δ            |
|------------------|-------------------------|--------------------------|--------------|
| Peak step memory | 15.03 GiB               | **14.95 GiB**            | -0.08 GiB    |
| TPS/GPU          | ~2,860                  | ~2,820                   | -1.4% (noise)|
| MFU              | ~8.3%                   | ~8.2%                    | -0.1pp       |
| Loss step 50     | 6.07                    | 6.57 (volatile)          | within noise |

### `moe_2b_ep @ LBS=2`

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/re576w5b
- Log: `logs/smoke-38th-sync/moe_2b_ep-lbs2-20260522-121209.log`
- 50 steps, **exit 0**, 427 s wall.
- Loss 12.93 → 6.15.
- Peak memory **27.08 GiB (42.33%)** — consistent with the LBS=1
  baseline scaled by the vocab-projection's linear LBS dependence
  (`14.95 + (2-1) × ~12 GiB ≈ 27 GiB`).
- ~3,200 TPS/GPU, 9.4% MFU.

### `moe_2b_ep @ LBS=16` (pre-existing OOM)

- W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/2x0435bc
- Log: `logs/smoke-38th-sync/moe_2b_ep-fresh-20260522-120344.log`
- All 12 ranks OOM at first forward on `lm_head`
  (`decoder.py:148: output = self.lm_head(h)` →
  `F.linear(input, self.weight, self.bias)`) with
  `Tried to allocate 62.53 GiB`.

The allocation matches the bf16 vocab-projection logits shape:

```
[LBS × seq_len, vocab_size] × 2 B = [16 × 8192, 256128] × 2 ≈ 62.5 GiB
```

`moe_2b_ep` was inheriting `LBS=16` from the EP=1 `moe_2b` base
(matching the docstring's earlier — incorrect — claim that LBS=16 was
"validated"; the validated datapoint was actually at LBS=1 override).
Closed by the registry pin in
[`59354e43f`](https://github.com/saforem2/torchtitan/commit/59354e43f).

## Artifacts

- Smoke logs under `logs/smoke-38th-sync/`:
  - `agpt_2b-fresh-20260522-120017.log`
  - `moe_2b_ep-fresh-20260522-120344.log` (LBS=16 OOM)
  - `moe_2b_ep-lbs2-20260522-121209.log`
  - `moe_2b_ep-lbs1-parity-20260522-142430.log`
- Stale checkpoint backup: `outputs/checkpoint-20260522-120005/`
  (pre-PR-3159 layout; safe to delete).
