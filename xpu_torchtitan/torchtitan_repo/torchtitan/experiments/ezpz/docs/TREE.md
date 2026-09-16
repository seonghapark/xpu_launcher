# `docs/` tree map

> One-page index of every directory and key file under
> `torchtitan/experiments/ezpz/docs/`. When you're not sure where
> something lives or where new content should go, scan this page
> instead of `cd`'ing around.
>
> See [`README.md`](./README.md) for the prioritized landing-page
> view (sections ordered by importance, with last-modified dates).
> This file is the structural map.

## At a glance

| Directory | Contents | When to look here |
|---|---|---|
| `production/` | Live training-job tracking — per-model, per-node-count | "How is the canonical chain doing right now?" |
| `evals/` | lm-eval results — per-model, with v1-vs-v2 plots | "What benchmark scores has the v2 run hit?" |
| `guides/` | Big-finding writeups, operational notes, how-to docs | "How do I do X?" or "Why does Y break?" |
| `experiments/` | Per-machine smoke / benchmark / LR-finder reports | "What did we measure on Sunspot 8N?" |
| `scaling/` | Per-model scaling-study results (TPS/MFU vs N) | "How does throughput scale at 64N vs 256N?" |
| `competitions/` | Optimizer speedrun leaderboards | "Which optimizer won the GBS=384 sprint?" |
| `meeting-notes/` | AuroraGPT sync agendas + action items | "What did we agree to in last week's sync?" |
| `summaries/` | 2-week / monthly retrospectives | "What happened in the past 2 weeks?" |
| `upstream-issues/` | Repros + patches we're filing back to `pytorch/torchtitan` | "What PRs are we trying to land upstream?" |
| `baselines/` | Reference training curves + benchmarks | "What's the AdamW baseline at GBS=48?" |
| `configs/` | Model config docs (architecture, registered names) | "What's the dim/n_heads of `agpt_50b_wide`?" |
| `rl/` | GRPO experiment notes (TRL-based, experimental) | RL/GRPO work |
| `journal.md` | Day-by-day session log | "What did we do on date X?" |
| `TODO.md` | Open work items | "What's pending?" |
| `upstream-sync.md` | Log of upstream merges + replays | "What did the last merge from main pull in?" |
| `README.md` | Prioritized landing page | First-time landing |

## Full tree

```
docs/
├── README.md              ← Prioritized landing page (8 sections, ordered by importance)
├── TREE.md                ← (this file) — structural map of the docs tree
├── journal.md             ← Day-by-day session log; most recent first
├── TODO.md                ← Open work items
├── upstream-sync.md       ← What we pulled from pytorch/torchtitan main; what was replayed onto agpt/moe
│
├── production/            ★ LIVE training tracking. Updated every session.
│   ├── README.md                  ← Top-level snapshot: canonical chains + active 256N + other
│   ├── scaling-performance.md     ← Apr 18-21 production scaling experiments (historical)
│   ├── agpt/                      ← Dense (agpt) production
│   │   ├── README.md              ← Index: 2B/20B/80B canonical chain tables, v1-vs-v2 overlays
│   │   ├── 2b/                    ← agpt 2B trajectories
│   │   │   ├── README.md          ← All 2B node counts at a glance
│   │   │   ├── figures/           ← v1-vs-v2 overlay PNGs
│   │   │   ├── n256/README.md     ← 2B 256N (active continuation chain — step 12,889+, loss 2.81)
│   │   │   ├── n512/README.md     ← 2B 512N CANONICAL chain (step 13,279+, loss 2.79, 1.34T tokens)
│   │   │   └── n1024/README.md    ← 2B 1024N (1st attempt 8463182 crashed at startup, not retried)
│   │   ├── 20b/                   ← agpt 20B trajectories
│   │   │   ├── README.md          ← All 20B node counts at a glance + v1-vs-v2 overlay
│   │   │   ├── figures/           ← v1-vs-v2 overlay PNGs
│   │   │   ├── n256/README.md     ← 20B 256N (8479581 active, gloo TCP didn't reproduce)
│   │   │   ├── n512/README.md     ← 20B 512N CANONICAL chain (8479579 hung at step 803, killed)
│   │   │   └── n1024/README.md    ← 20B 1024N (1st attempt 8463183 crashed at startup, not retried)
│   │   ├── 80b/                   ← agpt 80B trajectories (v1 NaN'd, v2 first attempt 2026-05-11)
│   │   │   ├── README.md          ← v2 attempt header + collapsed v1 history
│   │   │   └── n512/README.md     ← 80B 512N v2 first attempt (8480361 Q, AdamW LR=1e-6, compile=OFF)
│   │   └── 2b-mds/README.md       ← Pre-torchtitan Megatron-DeepSpeed 2B SophiaG baseline
│   └── moe/                       ← MoE production (sparse)
│       └── 10b_2b_sdpa_ep/        ← 10B-2B MoE SDPA + EP experiment
│
├── evals/                 ★ lm-eval results. Smoking gun for the bf16 fix.
│   ├── README.md                  ← Top-level eval index
│   └── agpt/
│       ├── 2b/README.md           ← 2B v1 vs v2 + 256N-vs-512N per-batch comparison
│       ├── 2b/figures/            ← v1-vs-v2 plot, scaling plots
│       ├── 20b/README.md          ← 20B v1 vs v2, steps 100-800 (ARC-Easy 0.27 → 0.44)
│       ├── 20b/figures/           ← eval_overview.png + per-task scaling plots
│       ├── 2b-mds/README.md       ← MDS 2B SophiaG reference scores
│       └── 2b-mds/figures/
│
├── guides/                ★ Big findings + operational notes. Check before suggesting work.
│   ├── bad-node-failover.md       ← Failover wrapper v2 (silent-hang watchdog, fixture tests). See ../../tests/failover/
│   ├── training-dtype-bf16-norm-freeze.md  ← Root cause of v1 → v2 restart
│   ├── loss-reporting-tp-dist-reduce.md    ← TP > 1 loss off by dp_world_size (PR #3204)
│   ├── known-issues.md            ← Catch-all live workarounds
│   ├── running-with-newer-pytorch.md       ← torch 2.13 venv setup + at-scale yeet
│   ├── xpu-attention-issues.md    ← SDPA / FlexAttention / Triton on Intel Max 1550
│   └── known-bugs/                ← Per-bug deep dives
│       ├── blendcorpus-megatron-aliasing.md  ← 7 leftover Megatron-style aliases in dataloader
│       └── blendcorpus-eoferror-race.md      ← Cache-build race causing mass EOFError at init (80B 8N, 80B 522N)
│
├── experiments/           ← Per-run reports. Every job that produced data should land a report here.
│   ├── README.md                  ← Top-level experiment index
│   ├── agpt/                      ← Dense (agpt) per-machine reports
│   │   ├── README.md
│   │   ├── aurora/                ← Aurora reports
│   │   │   ├── 80b-throughput-leaderboard.md       ← 80B TPS/MFU table across configs
│   │   │   ├── 20260412-035148-smoke-n2.md         ← Initial 2N smoke
│   │   │   ├── 20260412-193100-throughput-80b-n2.md
│   │   │   ├── 20260413-143800-throughput-20b-n2.md
│   │   │   ├── 20260414-production-20b-n512.md
│   │   │   ├── 20260418-80b-tp2-restored.md
│   │   │   ├── 20260511-20b-n512-hang-8479579.md   ← Silent-hang incident. Now handled by --timeout watchdog (2026-05-23).
│   │   │   ├── 20260523-failover-silent-hang-recovery-8505298.md  ← 🏁 First real silent-hang recovery validation (job 8505298 PASS).
│   │   │   ├── 20260524-80b-256n-sigsegv-cascade-8505222.md       ← 80B SIGSEGV-cascade post-mortem (wrapper bailed after 6 attempts).
│   │   │   └── figures/
│   │   ├── polaris/               ← Polaris (A100) reports
│   │   │   ├── 20260412-160749-smoke-n2.md
│   │   │   └── 20260414-020243-benchmark-n2.md
│   │   └── sunspot/               ← Sunspot (Intel Max 1550) reports
│   │       ├── 20260413-benchmark-n2.md
│   │       ├── 20260415-benchmark-n2.md
│   │       ├── 20260418-torch212-benchmark-n2.md
│   │       └── 20260425-scaling-2b-venv-torch213.md
│   ├── lr-finder/                 ← LR-finder sweeps
│   │   ├── README.md
│   │   ├── agpt/{aurora,polaris,sunspot}/  ← Per-machine LR-finder reports + figures
│   │   └── moe/sunspot/           ← MoE LR-finder
│   └── moe/                       ← MoE per-machine smoke + benchmarks
│       ├── README.md
│       └── {aurora,polaris,sunspot}/
│
├── scaling/               ← Per-model scaling-study results (TPS/MFU vs node count)
│   ├── README.md
│   ├── agpt-2b.md
│   ├── agpt-20b.md
│   ├── agpt-80b.md
│   ├── moe.md
│   └── yeet_env/README.md         ← yeet-env tarball broadcast scaling (8N → 4096N)
│
├── competitions/          ← Optimizer speedrun leaderboards (W&B link in each)
│   ├── README.md
│   ├── agpt2b-n2-1000steps/       ← 2B, 2N, GBS=48, 1000 steps
│   ├── agpt2b-n2-gas8-1000steps/  ← 2B, 2N, GAS=8 (GBS=384), 1000 steps
│   ├── agpt2b-n8-10BT/            ← 2B, 8N, 10B tokens
│   └── agpt2b-n8-r5/              ← 2B, 8N, repeat 5
│
├── meeting-notes/         ← Recurring meeting agendas + action items
│   ├── README.md
│   └── agpt-sync.md               ← AuroraGPT sync (one stable file with ## YYYY-MM-DD sections)
│
├── summaries/             ← 2-week / monthly retrospectives
│   ├── README.md
│   └── 2026-04-12_to_2026-04-27.md
│
├── upstream-issues/       ← Repros + patches we're filing back to pytorch/torchtitan
│   ├── dist_reduce_dtensor_skip.md      ← TP loss-reporting bug (PR #3204 filed)
│   └── STATE_DICT_STAGER_ISSUE.md       ← StateDictStager bug repro
│
├── baselines/             ← Reference training curves + benchmarks
│   └── README.md
│
├── configs/               ← Model config docs (architecture, registered names)
│   ├── dense.md                   ← 2B / 20B / 50B_wide / 80B
│   └── moe.md                     ← 500M-10B MoE variants
│
└── rl/                    ← RL/GRPO work (experimental, TRL-based)
    └── README.md
```

## Where to put new content

| Type of content | Goes under |
|---|---|
| New training run that produced data | `experiments/<module>/<machine>/<YYYYMMDD>-<purpose>.md` |
| New active production trajectory | `production/<module>/<model>/n<NODES>/README.md` |
| New lm-eval result for a checkpoint | Append to `evals/<module>/<model>/README.md` results table |
| New big finding (post-mortem, root cause writeup) | `guides/<finding>.md` (or `guides/known-bugs/<bug>.md` for narrower scope) |
| Per-day status update | Append top of `journal.md` |
| Upstream PR draft / repro | `upstream-issues/<PR-name>.md` + log in `upstream-sync.md` |
| Anything ephemeral (temp diagnostics, scratch notes) | NOT here — use `~/scratch/` or a TODO; don't litter the docs tree |

## Conventions

- **Per-day artifacts** use `YYYY-MM-DD` filenames (or `YYYYMMDD-HHMMSS-<purpose>` for experiments).
- **Per-recurring-meeting docs** use a stable filename with `## YYYY-MM-DD` sections inside (see `meeting-notes/agpt-sync.md`).
- **Figures** live under `<page>/figures/` next to the page that references them.
- **Cross-links**: production READMEs link to evals READMEs and vice versa; the bf16 freeze guide links to both training overlays and lm-eval figures.
- **Append-only** for production progress tables — don't rewrite history when a new chain link finishes.
- **Collapse stale content** into `<details>` blocks rather than deleting (e.g. v1 trajectories under each per-trajectory README).
