# MoE (Mixture of Experts) Benchmarks

MoE training benchmarks using DeepSeek-style MLA + MoE architecture across ALCF machines.

## Reports

### Aurora

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-002800-smoke-n2.md) | 500M, 2B, 4B, 7B, 10B_2B_sdpa | 2 | All pass; AC incompatible with 7B+ routing |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-033047-smoke-n2.md) | debugmodel | 2 | OK; 25.8k TPS, 5.11% MFU |
| 2026-04-12 | [Smoke test (n2)](aurora/20260412-035648-smoke-n2.md) | debugmodel, 2b | 2 | OK; 2b at 11.4% MFU, 34 TFLOPS |
| 2026-04-13 | [Throughput (n2)](aurora/20260413-143400-throughput-n2.md) | 500m-10b sweep | 2 | 2b best MFU (11.1%); compile hurts MoE |

### Sunspot

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Scaling Study](../../scaling/moe.md) | moe_2b, moe_7b at 1-64N | 1-64 | 2b 47% efficiency at 64N; 7b OOM at 32N+ |
| 2026-04-13 | [Benchmark (n2)](sunspot/20260413-benchmark-n2.md) | All 7 MoE configs | 2 | 9.7% MFU (debugmodel); compile hurts MoE |
| 2026-04-15 | [Full Benchmark (n2)](../agpt/sunspot/20260415-benchmark-n2.md) | All 7 MoE configs | 2 | 6/7 pass; moe_10b_2b Inductor crash |
| 2026-04-18 | [Torch 2.12 Benchmark (n2)](../agpt/sunspot/20260418-torch212-benchmark-n2.md) | 4 MoE configs + EP sweep | 2 | EP unblocked; 7b EP=2 +33% TPS; moe_2b crash |
| 2026-04-21 | [LR Finder (n2)](../lr-finder/moe/README.md#2026-04-21----moe-sweep-summary-sunspot) | 5 configs x 3 opts | 2 | All stable; 0 NaN for AdamW/Muon; 10b OOM at 8192 |
| 2026-05-12 | [`for_loop` backend smoke (n8)](sunspot/20260512-for-loop-smoke-n8.md) | moe_500m | 8 | PR #13 validated: for_loop fallback fires on XPU; loss 12.90→6.66 over 50 steps; 8,694 TPS / 13% MFU |
| 2026-05-20 | [Post-resync smoke (n2)](sunspot/20260520-smoke-n2-postresync.md) | debugmodel, 2b (LBS=1) | 2 | PR #3159 replay verified: debugmodel 12.96→7.01, moe_2b 12.91→6.16 in 50 steps each |
| 2026-05-20 | [PR #3386 replay smoke (n2)](sunspot/20260520-smoke-n2-pr3386-replay.md) | debugmodel, 2b (LBS=1) | 2 | PR #3386 (MoE clean DTensor boundaries) replay verified: debugmodel 12.92→7.00 (Δ-0.01), moe_2b 12.95→6.11 (Δ-0.05) vs baselines |
| 2026-05-20 | [PR #3386 EP follow-up smoke (n2)](sunspot/20260520-smoke-n2-pr3386-ep-followup.md) | debugmodel_ep, 2b_ep | 2 | EP=2 wiring validated at 2B scale (12.94→6.07 in 50 steps, +0.06 GiB vs EP=1); `_ep` registry configs need LBS override (OOM at defaults); LBS=2 debugmodel_ep hung at step 41/50 (uninvestigated) |
| 2026-05-22 | [38th upstream-sync smoke (n2)](sunspot/20260522-smoke-n2-38th-sync.md) | agpt_2b sanity + moe_2b_ep (LBS=1, 2, 16) | 2 | agpt_2b clean (24.34 GiB, baseline-matched). moe_2b_ep LBS=1 numerically equivalent to 37th-sync baseline (14.95 vs 15.03 GiB, TPS within 1.4%). LBS=2 clean (27.08 GiB). LBS=16 OOMs on the pre-existing `_ep` vocab-projection bug; registry pinned to LBS=2 in `59354e43f`. |
| 2026-05-27 | [40th upstream-sync smoke (n2)](sunspot/20260527-smoke-n2-40th-sync.md) | agpt_2b + moe_2b_ep (default + `--debug.deterministic`) | 2 | agpt_2b + moe_2b_ep numerically identical to 2026-05-22 baselines post-merge. `--debug.deterministic` on MoE+XPU **still fails** with the same `_histc_xpu` error from 2026-05-21 — PR #3146 was supposed to fix it but the merged diff is missing the `histc → bincount` swap (only the SAC save-list change landed). Upstream issue to file. |
| 2026-06-02 | [xccl split_group workaround smoke (n2)](sunspot/20260602-smoke-n2-xccl-split-workaround.md) | moe_2b_ep | 2 | Validates new `xccl_split_group_workaround.py` (commit `8031d1d3a`). Without the workaround, `ParallelDims.build_mesh` raised `RuntimeError: No backend ... does not support splitting` at trainer init. With the workaround: exit 0, 10 steps, EP sparse mesh built cleanly (`'ep'`, `'efsdp'` active), loss 12.95→8.27, ~3,050 TPS / 8.85% MFU, 58.28 GiB peak. Also fixed two latent bugs in `submit_moe_smoke.sh` (`PBS_O_WORKDIR` clobber by `ezpz_setup_job` + deprecated `ezpz yeet-env`). |

### Polaris

| Date | Report | Configs | Nodes | Key Result |
|------|--------|---------|-------|------------|
| 2026-04-12 | [Smoke test (n2)](polaris/20260412-185037-smoke-n2.md) | debugmodel, 500M, 2b, 4b, 7b, 10b_2b_sdpa | 2 | All pass; 2b best at 22.5% MFU; 7B+ memory-bound |
