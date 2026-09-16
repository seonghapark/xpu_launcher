# agpt Benchmark -- Sunspot 2-node (2026-04-13)

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
| Compile      | enabled                |

## Results

| Config          | Params | TP | LBS | Memory            | TPS    | TFLOPS | MFU    | W&B | Status |
|-----------------|--------|----|-----|-------------------|--------|--------|--------|-----|--------|
| agpt_debugmodel | 0.02B  | 1  | 8   | —                 | 49,103 | 11.33  | 3.80%  | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/0q44a9ve) | OK |
| agpt_2b         | 1.99B  | 1  | 1   | 46.80GiB (73.14%) | 5,609  | 62.76  | 21.05% | — | OK |
| agpt_7b         | 7.4B   | 1  | 2   | —                 | —      | —      | —      | — | CRASH |
| agpt_8b         | 8.7B   | 1  | 8   | —                 | —      | —      | —      | — | CRASH |
| agpt_20b        | 21.5B  | 1  | 1   | 44.54GiB (69.61%) | 351    | 52.29  | 17.53% | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/dbjmn0u1) | OK |
| agpt_50b        | 47.5B  | 1  | 1   | —                 | —      | —      | —      | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/s1pxhdem) | OOM |
| agpt_80b        | 80.8B  | 2  | 1   | —                 | —      | —      | —      | — | CRASH |
| agpt_80b_alt    | 80.8B  | 2  | 1   | —                 | —      | —      | —      | — | CRASH |
| agpt_80b_wide   | 80.8B  | 2  | 1   | 60.75GiB (94.94%) | 44     | 22.47  | 7.54%  | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/qzkgwxvi) | OOM |
| agpt_80b_deep   | 80.8B  | 2  | 1   | 58.84GiB (91.95%) | 83     | 45.31  | 15.20% | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/gm1xl33f) | OK |
| agpt_80b_deep_alt | 80.8B | 2 | 1   | 58.95GiB (92.14%) | 77     | 42.33  | 14.20% | [link](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/6oz4n5yt) | OK |

TPS, TFLOPS, and MFU are from step 20.

## Key Findings

1. **agpt_2b** is the best-performing dense model at 2 nodes: 5,609 TPS, 21% MFU
2. **agpt_7b/8b crash** due to default LBS being too high (LBS=2 and LBS=8 respectively). Would need LBS=1 override
3. **agpt_50b needs TP=2** — OOMs without tensor parallelism at 2 nodes
4. **80B variants at TP=2 on 2 nodes**: only `80b_deep` and `80b_deep_alt` fit. The deep variants use fewer heads per layer, reducing the SDPA attention matrix size
5. **80b_wide at 95% memory** gets a few steps in (44 TPS) before OOM — borderline, needs TP=3+
6. **80b and 80b_alt crash immediately** at TP=2 on 2 nodes (insufficient memory for SDPA MATH backend)

## Comparison with Aurora (2-node)

| Config    | Sunspot TPS | Aurora TPS | Sunspot MFU | Aurora MFU |
|-----------|-------------|------------|-------------|------------|
| debugmodel| 49,103      | 28,987     | 3.80%       | 2.24%      |
| 2b        | 5,609       | 5,326      | 21.05%      | 19.99%     |
| 20b       | 351         | 356        | 17.53%      | 17.74%     |
| 80b_deep  | 83          | —          | 15.26%      | —          |

Sunspot slightly outperforms Aurora on small models (debugmodel, 2b) and matches on 20b.

## Scaling Study (1–64 nodes)

See [scaling/](../../../scaling/) for the full weak scaling analysis across 1–64 nodes on Sunspot.

| Model    | 1N TPS | 64N TPS | Weak Scaling Efficiency |
|----------|--------|---------|------------------------|
| agpt_2b  | 6,457  | 4,783   | 74.1%                  |
| agpt_20b | 406    | 353     | 86.9%                  |
| agpt_80b | CRASH  | CRASH   | — (OK at 4–32N)        |

## 80B Throughput (historical, 2026-03-30)

Earlier 80B TP sweep on Sunspot (2 nodes, 5 steps):

| Config        | TP | TPS | TFLOPS | MFU    | Memory | Status |
|---------------|----|-----|--------|--------|--------|--------|
| 80B           | 2  | 85  | 46.31  | 15.53% | 59.82GiB (93%) | OK |
| 80B_alt       | 2  | 81  | 44.34  | 14.87% | 59.82GiB (93%) | OK |
| 80B_alt       | 3  | 23  | 12.40  | 4.16%  | 51.61GiB (81%) | OK |
| 80B_alt       | 6  | 43  | 23.30  | 7.82%  | 39.54GiB (62%) | OK |
| 80B_wide      | 3  | 29  | 15.16  | 5.08%  | 52.75GiB (82%) | OK |
| 80B_wide      | 6  | 51  | 26.35  | 8.84%  | 40.47GiB (63%) | OK |
| 80B_deep      | 2  | 83  | 45.58  | 15.29% | 58.84GiB (92%) | OK |
| 80B_deep_alt  | 2  | 77  | 41.87  | 14.04% | 58.95GiB (92%) | OK |
| 80B_deep_alt  | 3  | 23  | 12.56  | 4.21%  | 47.43GiB (74%) | OK |
| 80B_deep_alt  | 6  | 41  | 22.43  | 7.52%  | 41.63GiB (65%) | OK |
| 80B_deep_alt  | 12 | 28  | 15.40  | 5.16%  | 35.48GiB (55%) | OK |

TP=2 gives the best throughput on Sunspot 2-node (unlike Aurora where TP=2 OOMs).
