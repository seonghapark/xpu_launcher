# Torch 2.12 Benchmark -- Sunspot 2-node (2026-04-18)

## Environment

| Field        | Value                     |
|--------------|---------------------------|
| Date         | 2026-04-18                |
| Commit       | 525d0540                  |
| Machine      | Sunspot (x1921c1s2b0n0)   |
| Job ID       | 12465061                  |
| Nodes        | 2                         |
| Devices      | 24 (Intel Max 1550)       |
| Devices/Node | 12                        |
| Steps        | 10                        |
| Dataset      | blendcorpus (books)       |
| Backend      | xccl                      |
| Compile      | enabled                   |
| Torch        | 2.12.0.dev20260415+xpu    |
| Venv         | .venv (standalone)        |
| Env          | ZE_FLAT_DEVICE_HIERARCHY=FLAT |

## Results

### Dense (agpt)

| Config          | TP | Memory            | TPS    | TFLOPS | MFU    | W&B | Status |
|-----------------|----|-------------------|--------|--------|--------|-----|--------|
| agpt_debugmodel | 1  | 2.59GiB (4.05%)   | 43,949 | 10.15  | 3.40%  | [hp9dxmrc](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/hp9dxmrc) | OK |
| agpt_2b         | 1  | 24.04GiB (37.58%) | 6,122  | 68.49  | 22.97% | [4dkdu46y](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/4dkdu46y) | OK |
| agpt_20b        | 1  | 32.07GiB (50.13%) | 459    | 68.28  | 22.90% | [66r66pt4](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/66r66pt4) | OK |
| agpt_80b        | 2  | —                 | —      | —      | —      | — | CRASH (compile) |
| agpt_80b (no compile) | 2 | 61.03GiB (95.38%) | 85 | 46.54 | 15.61% | [p3dsyfai](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/p3dsyfai) | OK |

### MoE

| Config          | TP | Memory            | TPS    | TFLOPS | MFU    | W&B | Status |
|-----------------|----|-------------------|--------|--------|--------|-----|--------|
| moe_debugmodel  | 1  | 3.01GiB (4.70%)   | 28,155 | 16.58  | 5.56%  | [u2fs9s65](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/u2fs9s65) | OK |
| moe_2b          | 1  | —                 | —      | —      | —      | — | CRASH |
| moe_7b          | 1  | 33.58GiB (52.48%) | 1,165  | 15.79  | 5.30%  | [op5j83i4](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/op5j83i4) | OK |
| moe_10b_2b_sdpa | 1  | 39.42GiB (61.61%) | 159    | 2.23   | 0.75%  | [7o5xrkkh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/7o5xrkkh) | OK |

TPS, TFLOPS, and MFU are from step 10.

## Torch 2.12 vs 2.10 Comparison

### Dense Models

| Config    | Metric  | Torch 2.10 | Torch 2.12 | Change |
|-----------|---------|-----------|-----------|--------|
| debugmodel| TPS     | 65,274    | 43,949    | -33% |
|           | TFLOPS  | 15.07     | 10.15     | -33% |
|           | MFU     | 5.05%     | 3.40%     | -1.65pp |
|           | Memory  | —         | 2.59 GiB (4%) | — |
| agpt_2b   | TPS     | 5,497     | 6,122     | **+11%** |
|           | TFLOPS  | 61.50     | 68.49     | **+11%** |
|           | MFU     | 20.63%    | 22.97%    | **+2.3pp** |
|           | Memory  | 46.80 GiB (73%) | 24.04 GiB (38%) | **-49%** |
| agpt_20b  | TPS     | 355       | 459       | **+29%** |
|           | TFLOPS  | 52.79     | 68.28     | **+29%** |
|           | MFU     | 17.70%    | 22.90%    | **+5.2pp** |
|           | Memory  | 44.54 GiB (70%) | 32.07 GiB (50%) | **-28%** |
| agpt_80b  | TPS     | 83* (compile) | CRASH (compile) / 85 (no compile) | regression (compile) |
|           | TFLOPS  | 45.31*    | 46.54 (no compile) | ~same |
|           | MFU     | 15.20%*  | 15.61% (no compile) | ~same |
|           | Memory  | 58.84 GiB (92%)* | 61.03 GiB (95%) | +2 GiB |
|           | Notes   | *from 04-13 run, compile+AC | No compile+AC; compile+AC crashes (DeviceMesh) | — |

### MoE Models

| Config    | Metric  | Torch 2.10 | Torch 2.12 | Change |
|-----------|---------|-----------|-----------|--------|
| moe_debug | TPS     | 10,354    | 28,155    | **+172%** |
|           | TFLOPS  | 17.82     | 16.58     | -7% |
|           | MFU     | 5.98%     | 5.56%     | -0.4pp |
|           | Memory  | —         | 3.01 GiB (5%) | — |
| moe_2b    | TPS     | 3,345     | CRASH     | regression |
|           | Notes   | OK on 2.10 | tensor does not have a device | — |
| moe_7b    | TPS     | 1,066     | 1,165     | **+9%** |
|           | TFLOPS  | 19.27     | 15.79     | -18% |
|           | MFU     | 6.46%     | 5.30%     | -1.2pp |
|           | Memory  | 33.40 GiB (52%) | 33.58 GiB (52%) | ~same |
| moe_10b   | TPS     | 979       | 159       | -84% |
|           | Notes   | On 2.10 (sdpa) | Recompilation warnings, very slow | — |

## Key Findings

### Improvements on Torch 2.12

1. **agpt_2b: +11% throughput, -49% memory** — 6,122 vs 5,497 tps, 24 GiB vs
   47 GiB. The most significant improvement. The memory reduction means LBS=2+
   should now be feasible without OOM.

2. **agpt_20b: +29% throughput, -28% memory** — 459 vs 355 tps, 32 GiB vs
   44.5 GiB. The largest throughput gain of any config. MFU jumped from 17.7%
   to 22.9%.

3. **moe_debugmodel: +172% TPS** — 28,155 vs 10,354 tps. However TFLOPS/MFU
   are slightly lower, suggesting the TPS improvement is from reduced overhead
   rather than more compute.

4. **moe_7b: +9% throughput** — 1,165 vs 1,066 tps. Modest improvement.

### Regressions on Torch 2.12

1. **agpt_80b (TP=2) with compile: CRASH** — `AssertionError: expected all
   tensors_saved_with_vc_check to be Tensors, got types: [..., DeviceMesh]`.
   AOT autograd saves a `DeviceMesh` object during AC recomputation where it
   expects only tensors. This is a torch 2.12 regression in compile + AC + TP.
   Does not affect TP=1 configs.
   - `compile + AC=full + TP=2` → DeviceMesh assertion
   - `compile + AC=none + TP=2` → OOM (61.18/63.98 GiB)
   - `compile + AC=none + TP=2 + gc_freq=1` → same OOM
   - **Workaround: `--compile.no-enable`** — eager mode with AC=full works at
     85 tps / 15.6% MFU / 95% memory. Comparable throughput to torch 2.10
     compile (83 tps / 15.2% MFU) since compile provided minimal benefit for
     80B with TP.

2. **moe_2b: CRASH** — `RuntimeError: tensor does not have a device`. New torch
   2.12 issue in the MoE forward path.

3. **moe_10b_2b_sdpa: -84% throughput** — 159 vs 979 tps. Dynamo recompilation
   warnings suggest torch.compile is struggling with this config on 2.12.
   Multiple recompilation cycles (5/8) severely degrade performance.

4. **agpt_debugmodel: -33% throughput** — 43,949 vs 65,274 tps. The debugmodel
   is very small (0.02B params) so overhead dominates; this may not be
   representative.

### Environment Notes

- `ZE_FLAT_DEVICE_HIERARCHY=FLAT` is required for torch 2.12 to see 12 tiles
  per node (without it, only 6 devices are visible)
- The `.venv` must be standalone (not based on the frameworks Python) to avoid
  IPEX version mismatch errors
- Required additional installs in `.venv`: `torchdata`, `sentencepiece`,
  `blendcorpus`, `deepspeed` (blendcorpus dependency)

## Expert Parallelism (EP) Sweep — torch 2.13

EP was previously blocked on torch 2.10 by missing `ShardPlacementResult`.
Now available on torch 2.12+ via lazy import in the FSDP EP > 1 path.

### debugmodel (8 experts)

| EP | DP | TPS | TFLOPS | MFU | Memory |
|----|-----|-----|--------|-----|--------|
| 1 (no EP) | 24 | 28,155 | 16.58 | 5.56% | 3.01 GiB |
| 2 | 12 | 17,582 | 10.35 | 3.47% | 2.99 GiB |
| **4** | **6** | **19,058** | **11.22** | **3.76%** | **2.99 GiB** |
| 8 | 3 | 11,351 | 6.69 | 2.24% | 2.99 GiB |

EP=4 is the sweet spot for 8 experts — all-to-all volume is minimized (2
experts per EP group). EP=8 (1 expert per rank) has maximum communication.
EP=1 is fastest overall since no all-to-all is needed, but EP will become
beneficial at higher node counts where FSDP communication dominates.

### 7b (36 experts)

| EP | DP | TPS | TFLOPS | MFU | Memory |
|----|-----|-----|--------|-----|--------|
| 1 (no EP) | 24 | 1,165 | 15.79 | 5.30% | 33.58 GiB |
| 2 | 12 | **1,552** | **21.04** | **7.05%** | 29.80 GiB |
| 3+ | — | — | — | — | 30+ min compile warmup |

EP=2 gives **+33% TPS** and **-11% memory** over EP=1 for the 7b model.
Higher EP values have prohibitively long compile warmup times (30+ min
per config) due to the different EP mesh creating new compilation graphs.

### Blockers

- **moe_2b with EP**: crashes with `RuntimeError: tensor does not have a
  device` — same torch 2.12/2.13 regression as the non-EP moe_2b config
- **EP=1 + AllToAllTokenDispatcher**: `ExpertParallel` plan asserts EP > 1.
  Use `LocalTokenDispatcher` (default) for EP=1 configs
- **Compile warmup**: Each EP degree creates a new compile graph. First
  run takes 10-30+ min for 7b+ models. Subsequent runs with the same EP
  should use cached compilations

## Logs

Wandb project: [torchtitan.ezpz.train](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train)
