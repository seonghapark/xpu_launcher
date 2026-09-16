---
author: Sam Foreman
date: 2026-03-15
---

# Pre-Training AuroraGPT with TorchTitan + 🍋 `ezpz`

> Living documentation for the `experiments/ezpz/` work. Sections
> ordered by importance (live → reference → outbound). Within each
> section, rows are sorted newest first by last-commit date.
>
> **Looking for something specific?** See [`TREE.md`](./TREE.md)
> for a single-page annotated tree of every directory and file
> under `docs/`, with descriptions of what goes where.

## Recently Updated

The 25 most-recently-changed docs by git commit date (across all 136
docs, not just the curated tables below). Auto-generated -- do not edit
by hand; run `utils/refresh_docs_readme_table.py` (or `refresh_all.sh`).

<!-- BEGIN recently-updated (auto-generated) -->
| Modified | Doc |
|---------:|-----|
| 2026-07-01 | [Upstream Sync Log](./upstream-sync.md) |
| 2026-07-01 | [Production Training — Dense (agpt) Models](./production/agpt/README.md) |
| 2026-07-01 | [Production Training — agpt 80B](./production/agpt/80b/README.md) |
| 2026-07-01 | [Production Training Runs — Aurora](./production/README.md) |
| 2026-07-01 | [Development Journal](./journal.md) |
| 2026-07-01 | [LR Finder -- agpt 80B](./experiments/lr-finder/agpt/80b/README.md) |
| 2026-07-01 | [80B head-to-head convergence at the production batch (GBS=6144) -- Sunspot, 2026-06-30](./experiments/agpt/sunspot/2026-06-30-80b-convergence-gbs6144.md) |
| 2026-07-01 | [Failover / auto-retry "restart economics" -- log-mining analysis](./experiments/agpt/aurora/20260630-failover-restart-economics.md) |
| 2026-07-01 | [80B v2 production launch: SophiaG, constant-LR, scale brackets 512N/1024N/2048N](./experiments/agpt/aurora/20260628-80b-sophiag-constant-lr-512-1024-2048.md) |
| 2026-07-01 | [Claude Session Log](./claude-sessions.md) |
| 2026-07-01 | [Pre-Training AuroraGPT with TorchTitan + 🍋 ezpz](./README.md) |
| 2026-06-30 | [Bad-node failover for production training](./guides/bad-node-failover.md) |
| 2026-06-30 | [LR Finder -- agpt (Dense) -- index](./experiments/lr-finder/agpt/README.md) |
| 2026-06-30 | [LR Finder -- agpt 2B](./experiments/lr-finder/agpt/2b/README.md) |
| 2026-06-30 | [LR Finder -- agpt 20B](./experiments/lr-finder/agpt/20b/README.md) |
| 2026-06-29 | [Summaries](./summaries/README.md) |
| 2026-06-29 | [INCITE Quarterly Report — Q2 2026 (Apr 1 – Jun 30)](./summaries/2026-Q2-incite.md) |
| 2026-06-29 | [Production Training — agpt 2B @ 512 nodes](./production/agpt/2b/n512/README.md) |
| 2026-06-29 | [Production Training — agpt 2B @ 256 nodes](./production/agpt/2b/n256/README.md) |
| 2026-06-29 | [Production Training — agpt 2B](./production/agpt/2b/README.md) |
| 2026-06-29 | [Production Training — agpt 20B @ 512 nodes](./production/agpt/20b/n512/README.md) |
| 2026-06-29 | [Production Training — agpt 20B @ 256 nodes](./production/agpt/20b/n256/README.md) |
| 2026-06-29 | [Production Training — agpt 20B](./production/agpt/20b/README.md) |
| 2026-06-29 | [AuroraGPT Sync — Meeting Notes](./meeting-notes/agpt-sync.md) |
| 2026-06-29 | [AuroraGPT Sync — 2026-06-29 (Sam Foreman)](./meeting-notes/2026-06-29.md) |

<details>
<summary>Next 25 (#26-50)</summary>

| Modified | Doc |
|---------:|-----|
| 2026-06-29 | [LR Finder -- moe debugmodel (8 experts)](./experiments/lr-finder/moe/debugmodel/README.md) |
| 2026-06-29 | [LR Finder -- moe (Sparse) -- index](./experiments/lr-finder/moe/README.md) |
| 2026-06-29 | [LR Finder -- moe 7B (36 experts)](./experiments/lr-finder/moe/7b/README.md) |
| 2026-06-29 | [LR Finder -- moe 500M (16 experts)](./experiments/lr-finder/moe/500m/README.md) |
| 2026-06-29 | [LR Finder -- moe 4B (24 experts)](./experiments/lr-finder/moe/4b/README.md) |
| 2026-06-29 | [LR Finder -- moe 2B (24 experts)](./experiments/lr-finder/moe/2b/README.md) |
| 2026-06-29 | [Evaluation Results — agpt 2B](./evals/agpt/2b/README.md) |
| 2026-06-28 | [blendcorpus index-cache race: atomic-rename fix landed (041d015f) + one follow-up TOCTOU bug](./upstream-issues/blendcorpus-atomic-rename-index-fix.md) |
| 2026-06-28 | [Two-Week Summary: 2026-04-12 → 2026-04-27](./summaries/2026-04-12_to_2026-04-27.md) |
| 2026-06-28 | [Scaling Tests & Production Runs — Aurora (2026-04-18 to 2026-04-21)](./production/scaling-performance.md) |
| 2026-06-28 | [The "validator CCL deadlock at 80B TP=4" was a phantom -- two unrelated bugs](./guides/known-bugs/validator-tp4-at-80b.md) |
| 2026-06-28 | [Pre-#3623 checkpoints can't resume on current code: optimizer state-dict format migration](./guides/known-bugs/pre3623-optim-statedict-resume.md) |
| 2026-06-28 | [Blendcorpus EOFError Race in _build_index_mappings](./guides/known-bugs/blendcorpus-eoferror-race.md) |
| 2026-06-28 | [MoE (Mixture of Experts) Benchmarks](./experiments/moe/README.md) |
| 2026-06-28 | [Learning Rate Finder](./experiments/lr-finder/README.md) |
| 2026-06-28 | [80B autoretry verification + blendcorpus cold-cache race fix (end-to-end)](./experiments/agpt/aurora/20260628-80b-autoretry-blendcorpus-race-fix.md) |
| 2026-06-28 | [Breaking 512N queue starvation: sneak jobs, the multi-chain umbrella, and walltime-aware checkpointing](./experiments/agpt/aurora/20260626-512n-sneak-umbrella-walltime.md) |
| 2026-06-28 | [agpt (Dense AuroraGPT) Benchmarks](./experiments/agpt/README.md) |
| 2026-06-27 | [Production Queue-Wait Analysis (Aurora small)](./production/queue-wait-analysis.md) |
| 2026-06-27 | [80B LR-finder at the production batch (GBS=6144) -- Sunspot, 2026-06-27](./experiments/agpt/sunspot/2026-06-27-80b-lr-finder-production-batch.md) |
| 2026-06-26 | [2026-06-12 to 2026-06-26 -- Two-Week Summary](./summaries/2026-06-12_to_2026-06-26.md) |
| 2026-06-26 | [80B: LR, batch size, and dp-degree -- a connected set of findings (Sunspot, 2026-06-26)](./experiments/agpt/sunspot/2026-06-26-80b-lr-batch-dpdegree-findings.md) |
| 2026-06-26 | [80B GBS=5952 2048N-batch simulation (Sunspot, 2026-06-26)](./experiments/agpt/sunspot/2026-06-26-80b-gbs5952-2048N-sim.md) |
| 2026-06-26 | [80B GBS=2976 1024N-batch simulation (Sunspot, 2026-06-26)](./experiments/agpt/sunspot/2026-06-26-80b-gbs2976-1024N-sim.md) |
| 2026-06-26 | [Evaluation Results — agpt 2B (Megatron-DeepSpeed SophiaG)](./evals/agpt/2b-mds/README.md) |

</details>
<!-- END recently-updated (auto-generated) -->

## Production Training (live)

The canonical place for "what's training right now, and how is it
going?" Tracking is per-model and per-node-count.

| Page | Notes | Modified |
|------|-------|---------:|
| [Production Index](./production/README.md) | Top-level snapshot of every active trajectory | 2026-07-01 |
| [Dense (agpt) Production](./production/agpt/README.md) | 2B / 20B / 80B chains, v1-vs-v2 overlays | 2026-07-01 |
| [2B 256N](./production/agpt/2b/n256/README.md) | step-**92,859** (4.674T tokens, 100.0% of 4.67T), loss 2.6511. | 2026-06-29 |
| [2B 512N](./production/agpt/2b/n512/README.md) | step-**30400** (3.06T tokens, 65.5% of 4.67T). | 2026-06-29 |
| [20B 512N](./production/agpt/20b/n512/README.md) | step-**4,400** (442.9B tokens, 9.5% of 4.67T). | 2026-06-29 |
| [20B 256N](./production/agpt/20b/n256/README.md) | step-**2,100** (105.7B tokens, 2.3% of 4.67T), loss 4.8092. | 2026-06-29 |
| [agpt 80B](./production/agpt/80b/README.md) | 11 attempts since 2026-05-11; zero persisted. Wrapper detects failures correctly, but Aurora bad-node prevalence at init is too high for current spare count. | 2026-07-01 |
| [80B 256N](./production/agpt/80b/n512/README.md) | Latest config: AdamW LR=1e-6, TP=2, AC=full, compile=OFF, fp32-master. Most recent attempt `8505221` used 20 spares + 5 retries, still died in init. | 2026-05-28 |
| [20B 1024N](./production/agpt/20b/n1024/README.md) | First attempt (8463183) crashed at startup; not retried | 2026-06-24 |
| [2B 1024N](./production/agpt/2b/n1024/README.md) | First attempt (8463182) crashed at startup; not retried | 2026-06-24 |
| [agpt 2B](./production/agpt/2b/README.md) | All 2B trajectories + v1-vs-v2 overlay | 2026-06-29 |
| [agpt 2B-MDS](./production/agpt/2b-mds/README.md) | Pre-torchtitan Megatron-DeepSpeed reference baseline | 2026-05-03 |
| [Production Scaling Report](./production/scaling-performance.md) | Apr 18-21 experiments (historical) | 2026-06-28 |

## Evaluation (lm-eval results)

The smoking gun for the bf16-master fix: v2 ARC-Easy / HellaSwag /
ARC-Challenge / Winogrande vs the (frozen-norm) v1 baseline.

| Page | Notes | Modified |
|------|-------|---------:|
| [agpt 2B evals](./evals/agpt/2b/README.md) | v2 256N async sweep step 36K-45.5K (plateau at ARC-Easy ~0.645). v2 512N sync sweep step 14K-25K. v2 512N full sweep step 1K-13K + 256N-vs-512N per-batch. v2 ARC-Easy **0.6115** at step-13K (+33pp vs v1). | 2026-06-29 |
| [agpt 20B evals](./evals/agpt/20b/README.md) | **🏁 20B 512N sync full sweep step 900-3,200: ARC-Easy 0.463→0.665 (+20pp), HellaSwag norm 0.296→0.574 (+28pp). Now beating 2B 256N async per token.** v1 vs v2 step 100-800 (ARC-Easy 0.27 → 0.44) + 256N-vs-512N comparator. | 2026-06-10 |
| [agpt 2B-MDS evals](./evals/agpt/2b-mds/README.md) | Pre-torchtitan reference scores | 2026-06-26 |
| [Eval Index](./evals/README.md) | Top-level eval landing page | 2026-06-26 |

## Big Findings (post-mortems and live workarounds)

Landmark issues that shape current production. Always check the
relevant guide before suggesting work that touches one of these.

| Page | Notes | Modified |
|------|-------|---------:|
| [Bad-node failover wrapper](./guides/bad-node-failover.md) | **🏁 v2 production-validated 2026-05-23** ([incident report 8505298](./experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md)). Production submit scripts that request N+spare nodes, swap bad nodes for spares on crash, retry. Silent-hang watchdog (`--timeout=1800`) caught its first real production hang at step 37, blind-swapped, recovered cleanly. Test harness at [`tests/failover/`](../tests/failover/) — 9 fixtures, all passing. | 2026-06-30 |
| [Known Issues / Operational Notes](./guides/known-issues.md) | **Top entry (2026-05-23)**: `--checkpoint.async-mode=async` kills the cluster at 20B 512N+ — root cause of 3 weeks of lost persisted progress. Workaround: `CHECKPOINT_ASYNC_MODE=disabled`. | 2026-05-23 |
| [bf16-master RMSNorm freeze](./guides/training-dtype-bf16-norm-freeze.md) | Root cause of v1 → v2 restart; `dtype=float32` is now default | 2026-06-10 |
| [TP > 1 loss reporting off by `dp_world_size`](./guides/loss-reporting-tp-dist-reduce.md) | **Resolved upstream 2026-05-18** (PR #3159, commit `d64eabcce`). Doc preserved as historical context for affected 80B W&B traces. | 2026-06-12 |
| [XPU Attention Issues](./guides/xpu-attention-issues.md) | SDPA, FlexAttention, Triton on Intel Max 1550 | 2026-04-26 |

## Day-by-day Work

| Page | Notes | Modified |
|------|-------|---------:|
| [Development Journal](./journal.md) | Session-by-session log of what happened, with findings and incidents | 2026-07-01 |
| [AuroraGPT Sync Notes](./meeting-notes/agpt-sync.md) | Recurring agendas + action items | 2026-06-29 |
| [Meeting Notes Index](./meeting-notes/README.md) | Top-level meeting index | 2026-05-04 |
| [Summary 2026-04-12 → 2026-04-27](./summaries/2026-04-12_to_2026-04-27.md) | 2-week retrospective | 2026-06-28 |
| [Periodic Summaries Index](./summaries/README.md) | Index of 2-week / monthly retros | 2026-06-29 |

## Setup & Reference

| Page | Notes | Modified |
|------|-------|---------:|
| [Running with Newer PyTorch (≥ 2.10)](./guides/running-with-newer-pytorch.md) | torch 2.13 venv setup + at-scale yeet (8N → 4096N) | 2026-06-09 |
| [Reference Baselines](./baselines/README.md) | Training curves and benchmarks | 2026-04-29 |
| [Dense Model Configs](./configs/dense.md) | 2B / 20B / 50B / 80B | 2026-04-26 |
| [MoE Variants](./configs/moe.md) | 500M-10B | 2026-04-26 |

## Scaling Studies

| Page | Notes | Modified |
|------|-------|---------:|
| [Scaling Index](./scaling/README.md) | Top-level scaling landing page | 2026-06-06 |
| [agpt 2B scaling](./scaling/agpt-2b.md) | Per-N TPS / MFU | 2026-06-13 |
| [agpt 20B scaling](./scaling/agpt-20b.md) | Per-N TPS / MFU | 2026-06-13 |
| [agpt 80B scaling](./scaling/agpt-80b.md) | Per-N TPS / MFU | 2026-04-26 |
| [MoE scaling](./scaling/moe.md) | Per-N TPS / MFU | 2026-06-13 |
| [Per-run Experiment Reports](./experiments/README.md) | Raw smoke tests, LR-finder sweeps, benchmark logs | 2026-04-12 |

## Sandboxes / Side-channels

| Page | Notes | Modified |
|------|-------|---------:|
| [Optimizer Speedrun Competitions](./competitions/README.md) | [W&B link](https://api.wandb.ai/links/aurora_gpt/hda3milo) | 2026-04-28 |
| [RL (GRPO) Experiment](./rl/README.md) | TRL-based GRPO on XPU (experimental) | 2026-06-13 |

## Outbound (upstream)

| Page | Notes | Modified |
|------|-------|---------:|
| [Upstream Sync Log](./upstream-sync.md) | What we pulled from `pytorch/torchtitan` and replayed onto agpt/moe | 2026-07-01 |
| [`_dist_reduce` skips DTensor reduction (PR #3204)](./upstream-issues/dist_reduce_dtensor_skip.md) | **Closed as superseded 2026-06-12** — upstream landed `to_local()` fix via PR #3159 (commit `d64eabcce`, 2026-05-18). | 2026-06-12 |
| [`StateDictStager` bug](./upstream-issues/STATE_DICT_STAGER_ISSUE.md) | Repro for upstream filing | 2026-05-01 |

## Planning

| Page | Notes | Modified |
|------|-------|---------:|
| [TODO](./TODO.md) | Open work items | 2026-05-05 |
