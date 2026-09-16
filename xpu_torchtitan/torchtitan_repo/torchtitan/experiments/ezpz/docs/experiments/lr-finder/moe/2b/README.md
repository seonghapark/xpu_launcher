# LR Finder -- moe 2B (24 experts)

Part of the [moe LR-finder index](../README.md); methodology in the
[parent README](../../README.md). Figures in [`figures/`](figures/).

**Status:** small-batch only -- the single 2026-04-21 GBS=192 sweep below is
all current data; no production-GBS MoE finder yet (see the
[index status note](../README.md#status-small-batch-only-no-production-gbs-finder-yet)).

<details closed>
<summary><b>Small-batch debug sweep (2026-04-21, GBS=192, 2-node)</b></summary>

### 2026-04-21 (Sunspot, 2N, torch 2.13, compile on, seq_len=8192, LBS=1)

| Optimizer | NaN | Suggested LR | Blow-up | Status |
|-----------|-----|-------------|---------|--------|
| AdamW   | 0 | 1.72e-7* | 1.72e-6* | clean |
| Muon    | 0 | -- | -- | ok |
| SophiaG | 0 | -- | -- | ok |

*The very low AdamW suggested LR is likely a derivative-analysis artifact --
early noise in the loss curve triggers false blow-up detection. The 4B/7B
values are more representative of the real MoE optimum.

![moe 2B finder](figures/lr_finder_2B.png)

</details>
