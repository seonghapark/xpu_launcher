# LR Finder -- moe 500M (16 experts)

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
| AdamW   | 0 | 4.17e-2 | 4.17e-1 | clean |
| Muon    | -- | -- | -- | expired (Newton-Schulz overhead too slow) |
| SophiaG | 0 | -- | -- | clean |

![500M finder](figures/lr_finder_500M.png)

</details>
