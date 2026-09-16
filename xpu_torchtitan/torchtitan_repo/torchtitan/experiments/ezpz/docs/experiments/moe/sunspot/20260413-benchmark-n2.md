# MoE Benchmark -- Sunspot 2-node (2026-04-13)

## Environment

| Field        | Value                  |
|--------------|------------------------|
| Date         | 2026-04-13             |
| Commit       | 8aac3598               |
| Machine      | Sunspot (x1921c1s4b0n0)|
| Job ID       | 12464285               |
| Nodes        | 2                      |
| Devices      | 24 (Intel Max 1550)    |
| Devices/Node | 12                     |
| Steps        | 20                     |
| Dataset      | blendcorpus (books)    |
| Backend      | xccl                   |
| Compile      | enabled (per-block)    |

## Results

| Config        | Total Params | Active Params | LBS | TPS    | TFLOPS | MFU    | Memory            | AC   | Status |
|---------------|-------------|---------------|-----|--------|--------|--------|-------------------|------|--------|
| moe_debugmodel| 0.05B       | 0.04B         | 8   | 16,796 | 28.91  | 9.70%  | —                 | full | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/rb5by48n) | OK |
| moe_500m      | 0.25B       | 0.14B         | 4   | 7,228  | 27.16  | 9.11%  | —                 | full | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ay8ea2xq) | OK |
| moe_2b        | 1.61B       | 0.49B         | 16  | 3,569  | 25.93  | 8.70%  | —                 | full | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/oazxwiv3) | OK |
| moe_4b        | 2.89B       | 0.81B         | 16  | 2,414  | 21.00  | 7.04%  | —                 | full | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/y4b5rg8a) | OK |
| moe_7b        | 7.54B       | 1.57B         | 2   | 1,099  | 19.86  | 6.66%  | 33.18GiB (51.85%) | none | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/tb31cp8u) | OK |
| moe_10b_2b    | 9.41B       | 1.98B         | 1   | —      | —      | —      | —                 | none | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/0tar231b) | CRASH (timeout) |
| moe_10b_2b_sdpa| 9.41B      | 1.98B         | 2   | 987    | 17.22  | 5.77%  | 35.78GiB (55.92%) | none | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/z0xmmnqa) | OK |

## Key Findings

1. **MFU peaks around 9–10%** for small MoE models (debugmodel, 500m), dropping to 5.7% for 10b
2. **moe_10b_2b timed out** — FlexAttention compile warmup exceeds 1800s timeout. The SDPA variant (`moe_10b_2b_sdpa`) works and is faster to start
3. **AC=none required for 7b+** — full activation checkpointing is incompatible with MoE routing (recomputation routes different tokens to different experts, producing shape mismatches)
4. **Memory is well-behaved** up to 7b (52%), 10b_2b_sdpa at 56%

## Comparison with Aurora (2-node)

| Config      | Sunspot TPS | Aurora TPS | Sunspot MFU | Aurora MFU |
|-------------|-------------|------------|-------------|------------|
| moe_500m    | 7,228       | 11,200     | 9.11%       | 8.46%      |
| moe_2b      | 3,569       | 6,625      | 8.70%       | 11.11%     |
| moe_4b      | 2,414       | 5,127      | 7.04%       | 11.39%     |
| moe_7b      | 1,099       | 1,136      | 6.66%       | 5.16%      |
| moe_10b_2b  | CRASH       | 975        | —           | 4.59%      |

Aurora outperforms Sunspot on MoE 500m–4b (compile disabled on Aurora gives better throughput for MoE). For 7b+ the gap narrows since both are compute-bound.

## LBS Scaling (from earlier optimization, 2026-04-12)

Optimal LBS values found on Sunspot 2-node:

| Config   | LBS=1  | LBS=2  | LBS=4  | LBS=8  | LBS=16  | Best |
|----------|--------|--------|--------|--------|---------|------|
| moe_2b   | 3,857  | 5,419  | 6,319  | 6,738  | **7,012** | 16 |
| moe_4b   | 1,469  | 2,606  | 3,804  | 4,773  | **5,236** | 16 |
| moe_7b   | 652    | **979**| OOM    | OOM    | OOM     | 2  |

## Scaling Study (1–64 nodes)

See [scaling/moe.md](../../../scaling/moe.md) for full results.

| Model  | 1N TPS | 64N TPS | Weak Scaling Efficiency |
|--------|--------|---------|------------------------|
| moe_2b | 7,121  | 3,376   | 47.4%                  |
| moe_7b | 1,841  | OOM     | OOM at 32N+            |

MoE models scale poorly past 16 nodes due to all-to-all routing communication overhead. Memory grows from 15 GiB (2N) to 38 GiB (64N) for moe_2b.
