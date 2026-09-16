# 80B Throughput Leaderboard -- Aurora 2-node

Best throughput results for ~80B AGPT models on Aurora (2 nodes, 24 XPUs).

## Leaderboard Chart

```
TPS
 68 |                                                 * ← 80B_wide TP=4 compile
 63 |                                              *
 53 |                                        *
 50 |                                     *
 48 |                                  *
 45 |                            *  *
 44 |                         *
 41 |                      *
 40 |                   *
 31 |             *
 29 |          *
 23 |       *
 20 |    *
    +--+--+--+--+--+--+--+--+--+--+--+--+--+--+
       1  2  3  4  5  6  7  8  9 10 11 12 13 14
```

## All Results (sorted by TPS)

| Rank | TPS | TFLOPS | MFU    | Variant      | TP | Compile | seq_len | LBS | Memory            | W&B | Status |
|------|-----|--------|--------|--------------|-----|---------|---------|-----|-------------------|-----|--------|
| --   | 71  | 34.72  | 11.64% | 80B_wide     | 4   | on      | 4096    | 2   | 43.2GiB (67.4%)   | [wxmb242s](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/wxmb242s) | best TPS (short seq) |
| 1    | **68** | **35.15** | **11.79%** | **80B_wide** | **4** | **on** | **8192** | 1 | 44.2GiB (69.1%) | [o81amwdg](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/o81amwdg) | **BEST @ seq=8k** |
| --   | 68  | 35.06  | 11.76% | 80B_wide     | 4   | on (no loss parallel) | 8192 | 1 | 46.8GiB (73.1%) | [oieijdbx](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/oieijdbx) | no gain, +mem |
| --   | 66  | 34.14  | 11.45% | 80B_wide     | 4   | on+async| 8192    | 1   | 44.2GiB (69.1%)   | [q2n43d5l](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/q2n43d5l) | no gain |
| --   | 63  | 32.46  | 10.89% | 80B_wide     | 4   | loss-only | 8192  | 1   | 49.5GiB (77.3%)   | [lx61y7eo](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/lx61y7eo) | = no-compile |
| 2    | 63  | 32.21  | 10.80% | 80B_wide     | 4   | off     | 8192    | 1   | 49.5GiB (77.3%)   | [iuzf6mbh](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/iuzf6mbh) | OK |
| 3    | 53  | 27.28  | 9.18%  | 80B_wide     | 6   | on      | 8184    | 1   | 40.5GiB (63.3%)   | [b7tqqkd2](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/b7tqqkd2) | OK |
| 4    | 50  | 25.58  | 8.58%  | 80B_wide     | 6   | off     | 8184    | 1   | 41.4GiB (64.7%)   | [gqnbyrux](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/gqnbyrux) | OK |
| 5    | 45  | 24.53  | 8.23%  | 80B_alt      | 6   | on      | 8184    | 1   | 39.5GiB (61.8%)   | [vukb3nil](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/vukb3nil) | OK |
| 6    | 45  | 24.34  | 8.16%  | 80B_alt      | 6   | on+async| 8184    | 1   | 39.5GiB (61.8%)   | [bm3itfex](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/bm3itfex) | no gain |
| 7    | 44  | 23.91  | 8.02%  | 80B_deep_alt | 6   | on      | 8184    | 1   | 41.6GiB (65.0%)   | [ia3jel8w](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ia3jel8w) | OK |
| 8    | 41  | 22.52  | 7.55%  | 80B_alt      | 6   | off     | 8184    | 1   | 44.2GiB (69.0%)   | [irgxd2su](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/irgxd2su) | OK |
| 9    | 40  | 21.85  | 7.33%  | 80B_deep_alt | 6   | off     | 8184    | 1   | 41.2GiB (64.4%)   | [qxmz25qb](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/qxmz25qb) | OK |
| 10   | 38  | 19.60  | 6.57%  | 80B_wide     | 12  | on      | 8184    | 1   | 37.4GiB (58.4%)   | [0722m3nx](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/0722m3nx) | OK |
| 11   | 31  | 16.68  | 5.59%  | 80B_deep_alt | 12  | on      | 8184    | 1   | 35.5GiB (55.5%)   | [n11hu2yr](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/n11hu2yr) | OK |
| 12   | 29  | 14.97  | 5.02%  | 80B_wide     | 3   | on      | 8190    | 1   | 52.8GiB (82.4%)   | [mq8b418t](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/mq8b418t) | OK |
| 13   | 23  | 12.46  | 4.18%  | 80B_alt      | 3   | on      | 8190    | 1   | 51.6GiB (80.7%)   | [7br9j58x](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/7br9j58x) | OK |
| 14   | 20  | 11.25  | 3.77%  | 80B_alt      | 3   | off     | 8190    | 1   | 52.4GiB (81.9%)   | [52vrxoc0](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/52vrxoc0) | OK |

All runs use LBS=1, AC=full. TPS/TFLOPS/MFU from step 10 (steady-state).

## Failed Experiments

| Variant      | TP | Compile | seq_len | Memory Peak       | Failure |
|--------------|-----|---------|---------|-------------------|---------|
| 80B          | 2   | off     | 8192    | OOM at init       | driver UR_RESULT_ERROR_OUT_OF_RESOURCES |
| 80B          | 2   | on      | 8192    | OOM at init       | same |
| 80B          | 2   | off     | 8192    | OOM at init       | +cpu_offload, still OOM |
| 80B_wide     | 2   | on      | 8192    | 60.75GiB (94.94%) | OOM step 2 (missed by 70 MiB) |
| 80B_wide     | 2   | off     | 8192    | 60.75GiB (94.94%) | OOM step 2 (missed by 70 MiB) |
| 80B_alt      | 6   | on      | 8184    | 61.56GiB (96.22%) | AC=selective OOM |
| 80B_alt      | 6   | on      | 8184    | 56.07GiB (87.63%) | reshard=never OOM |
| 80B_alt      | 4   | on      | 8192    | --                | compile hung 4+ hours |
| 80B_alt      | 6   | on      | 8184    | --                | memory_budget AC graph break |

## Key Insights

### Architecture: 80B_wide is the winner

| Variant      | Dim   | Layers | Best TPS | Best TP |
|-------------|-------|--------|----------|---------|
| **80B_wide** | 10752 | 48     | **68**   | 4       |
| 80B_alt     | 9216  | 84     | 45       | 6       |
| 80B_deep_alt| 7680  | 96     | 44       | 6       |

Fewer, wider layers = faster training, faster compilation, lower memory.

### TP scaling for 80B_wide

| TP | TPS (compile) | TPS (no compile) | Memory | DP |
|----|--------------|-----------------|--------|-----|
| 2  | OOM          | OOM             | 94.9%  | 12  |
| 3  | 29           | --              | 82.4%  | 8   |
| 4  | **68**       | 63              | 69.1%  | 6   |
| 6  | 53           | 50              | 63.3%  | 4   |

TP=4 is the sweet spot: enough sharding to fit comfortably, enough DP
for efficient gradient aggregation, and large enough per-rank matmuls
to keep XPU tiles busy.

### torch.compile impact

| Variant | TP | No Compile | Compile | Speedup |
|---------|-----|-----------|---------|---------|
| 80B_wide | 4  | 63 TPS    | 68 TPS  | +8%     |
| 80B_wide | 6  | 50 TPS    | 53 TPS  | +6%     |
| 80B_alt  | 3  | 20 TPS    | 23 TPS  | +15%    |
| 80B_alt  | 6  | 41 TPS    | 45 TPS  | +10%    |
| 80B_deep | 6  | 40 TPS    | 44 TPS  | +10%    |

Compile consistently helps 6-15%. Larger benefit at higher TP (more fusion opportunities).

### TP=2 blocked on Aurora

All 80B variants OOM at TP=2 on Aurora with `aurora_frameworks-2025.3.1`.
The benchmark results (85 TPS for 80B, 43 TPS for 80B_wide) were on Sunspot.
80B_wide TP=2 gets 75 TPS on step 1 before OOMing on step 2 (missed by 70 MiB).

## Best Config for Production

```
Model:    80B_wide (dim=10752, 48 layers, heads=84, kv_heads=12)
TP:       4
Compile:  on
seq_len:  8192
AC:       full
LBS:      1

→  68 TPS  |  35.15 TFLOPS  |  11.79% MFU  |  44.2 GiB (69.1%)
```
