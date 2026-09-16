# INCITE Quarterly Report — Q2 2026 (Apr 1 – Jun 30)

> **Project:** AuroraGPT — A Large-Scale Foundation Model for Advancing Science
> (INCITE-2026, Allocation Year 2)
> **Author:** Sam Foreman
> **Period covered:** 2026-04-01 through 2026-06-30
> **Scope:** torchtitan + ezpz training work on Aurora (primary), with
> Sunspot (Intel Max 1550) and Polaris (NVIDIA A100) used for
> development, scaling, and evaluation.

This report synthesizes the five two-week retrospectives written during
the quarter ([index](README.md)) plus the live
[production](../production/README.md) and [eval](../evals/README.md)
trackers. Section anchors link to the underlying per-trajectory pages so
every claim is traceable to data on disk or in W&B.

---

## 1. Executive Summary

This quarter the project brought a **PyTorch-native training stack
(torchtitan + `ezpz`) to production maturity on Aurora's Intel XPU
fabric** and used it to complete the first full base pre-training run of
the post-bug-fix era.

The four headline outcomes:

1. **AuroraGPT-2B base pre-training is COMPLETE** — 4.674T tokens (100%
   of the 4.67T olmo-mix-1124 budget), final loss 2.652. This is the
   first AuroraGPT model trained end-to-end through the new fp32-master
   ("v2") torchtitan pipeline. ([detail](#41-agpt-2b-dense-complete))
2. **AuroraGPT-20B is converging dramatically faster per token** —
   the 20B 512N chain beats the 2B chain on *every* downstream benchmark
   per token (ARC-Easy 0.46 -> 0.66, HellaSwag-norm 0.30 -> 0.64 over
   the first ~440B tokens). ([detail](#5-evaluation-results))
3. **AuroraGPT-80B production launched for the first time** after a
   multi-week numerical-stability investigation root-caused the 80B NaN
   and identified a stable training corner. ([detail](#42-agpt-80b-dense-launched))
4. **A reinforcement-learning post-training path now runs end-to-end on
   Intel XPU** (SFT -> GRPO via TRL `vllm-serve`), de-risking the
   post-training milestones. ([detail](#62-reinforcement-learning-sft--grpo))

A critical correctness bug — RMSNorm weights frozen at 1.0 under bf16
master weights — was caught and fixed at the start of the quarter
(2026-04-30 "v2" restart, default `dtype=float32`). Every production
trajectory in this report is a clean, post-fix run; the evaluation
curves are the smoking gun that the fix worked.

The dominant *cost* this quarter was **not** training throughput but
**Aurora `small`-queue contention**: the two 512-node canonical chains
sat queued for 20-25 days each. Allocation burn is healthy and well
within budget (see below).

---

## 2. Resource Usage & Allocation

| Item | Value | Source |
|---|---|---|
| Year-2 Aurora budget | 6,620K node-hours | INCITE-2026 renewal milestone table |
| Year-2 Polaris budget | 150K node-hours | renewal milestone table |
| Aurora burn ratio (as of 2026-06-26) | **0.29** (~29% of Year-2 Aurora) | PBS, via [queue-wait-analysis](../production/queue-wait-analysis.md) |
| Implied Aurora node-hours used | ~1.9M of 6,620K (derived from ratio) | derived |

Burn is on a healthy trajectory: at roughly the half-year mark, ~29% of
the Year-2 Aurora allocation has been consumed, with the largest runs
(80B at 512-2048 nodes) only just launching at quarter end — i.e. the
heaviest consumption is ahead, as planned.

**Queue contention, not allocation, is the binding constraint.** PBS
confirmed the 512N chains are losing the scheduling race for the `small`
queue to other projects despite thousands of physically free nodes and a
healthy burn ratio. This is an operational/policy issue (candidate
levers: a project reservation, consolidating on 256N, or a PI->ALCF
priority discussion), documented in full at
[queue-wait-analysis.md](../production/queue-wait-analysis.md). Worth
raising at the program level.

---

## 3. Milestone Status

> **Mapping note.** The INCITE-2026 renewal milestone table names a
> Megatron-DeepSpeed pipeline and a model lineup of 70B dense / 300B MoE
> / 7B & 36B CPT. This quarter's production work runs on the
> **torchtitan + `ezpz`** PyTorch-native stack with dense AuroraGPT
> models at 2B / 20B / 80B. The 2B and 20B runs are the validation /
> de-risking ladder for the large dense milestone; the 80B run is the
> closest production analog to the Y2:M1 dense target. The mapping below
> is therefore approximate and is offered to connect this quarter's
> deliverables to the proposal's structure.

| Milestone | Proposal target | Q2 status | Notes |
|---|---|---|---|
| **Y2:M1** | Dense ~70B, pretrain + CPT + post-train + eval | **In progress** | 80B dense production **launched** 2026-06-28 (512N/1024N/2048N). 2B base run **complete** (4.674T tokens); 20B in active training. Stable-training corner for 80B established. |
| **Y2:M2** | 300B MoE, pretrain + post-train + eval | **Infrastructure ready; not yet in production** | MoE stack isolated + sped up (PR #14, 1.94x), expert-parallelism validated EP=1-12, MoE LR-finder done. No production MoE pretraining started. |
| **Y2:M3** | 7B dense CPT | **Plumbing ready** | Constant-LR (decay_ratio=0) continued-pretraining path built and pre-checked for the 80B launch; applies directly to dense CPT. 2B completed base run is a CPT candidate. |
| **Y2:M4 / M5** | 36B MoE CPT + dev time | **Dev time consumed** | Counts as development/validation time: MoE EP, optimizer, and dataloader infrastructure built and validated this quarter. |

**Cross-cutting deliverable not itemized in the table:** standing up the
torchtitan+`ezpz` stack on Aurora XPU as a validated alternative to the
Megatron-DeepSpeed pipeline (the pre-torchtitan 2B-MDS run remains the
[reference baseline](../evals/agpt/2b-mds/README.md)). This is the
enabling work behind all four milestones above.

---

## 4. Production Training Campaign

Full-scale pre-training on the
[olmo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)
corpus (4.67T-token budget). All runs are "v2" (post-bf16-fix,
`dtype=float32` master weights). Live tracker:
[production/README.md](../production/README.md).

### 4.1 agpt 2B (dense) — COMPLETE

The first AuroraGPT model trained end-to-end on the v2 torchtitan stack.

| Metric | Value |
|---|---|
| Final checkpoint | step-92,859 |
| Tokens | **4.674T (100.0% of 4.67T target)** |
| Final loss | 2.652 (grad_norm ~0.056) |
| Throughput | ~3,700 TPS/GPU, ~14% MFU |
| Optimizer | SophiaG, LR=2.28e-5, fp32-master |
| Nodes | 256 (async checkpointing) |

The chain ran across ~13 PBS dispatches since the 2026-04-30 restart,
surviving repeated bad-node failures via the failover wrapper (Section
6.3). The canonical **512N** 2B chain (large-batch trajectory) reached
step-30,400 / 3.06T tokens (65.5%) but is queue-stalled. Detail:
[2b/n256](../production/agpt/2b/n256/README.md),
[2b/n512](../production/agpt/2b/n512/README.md).

**Next:** convert the final DCP checkpoint to HF and run the full
lm-eval suite for the end-of-pretraining scorecard (top post-maintenance
action).

### 4.2 agpt 20B (dense) — in training, leading per-token

| Trajectory | Persisted step | Loss | Tokens | Status |
|---|---:|---:|---:|---|
| 512N (canonical) | 4,400 | 2.51 | 442.9B (9.5%) | queue-stalled + init-crash exposure |
| 256N | 2,100 | 2.85 | 105.7B (2.3%) | advancing (~21.8% MFU); resumes post-PM |

The 20B chain is the per-token efficiency story of the quarter (Section
5). Two operational issues constrain it: the same 512N queue starvation,
and a `set_determinism` `std::bad_alloc` init crash that intermittently
kills 512N restarts. Detail: [20b/README](../production/agpt/20b/README.md).

### 4.3 agpt 80B (dense) — LAUNCHED

First real 80B v2 production dispatches, queued 2026-06-28 to start after
the Aurora PM maintenance window, submitted simultaneously at three
scales as a combined production-and-scaling experiment:

| Bracket | Active+spare nodes | GBS | ~steps to budget |
|---|---|---:|---:|
| 512N | 510 + 6 | 6120 | 93,223 |
| 1024N | 1022 + 14 | 6132 | 93,041 |
| 2048N | 2046 + 26 | 6138 | 92,950 |

Common config: **SophiaG LR=1e-6, constant LR after warmup** (CPT-ready),
TP=4 / LBS=1, compile OFF, AC=full, validator ON. The launch was gated on
the numerical-stability work in Section 6.1. Two open risks are flagged
deliberately: (a) the optimizer choice (SophiaG vs mano) is a live team
decision, and (b) no v2 run has yet succeeded above ~512N, so the
1024N/2048N brackets may crash at init — that is treated as data, not
regression. Plan:
[20260628-80b-sophiag-constant-lr](../experiments/agpt/aurora/20260628-80b-sophiag-constant-lr-512-1024-2048.md).

---

## 5. Evaluation Results

Benchmarked with
[lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness)
on a 7-task suite. Live tables: [evals/README.md](../evals/README.md).

**The 20B 512N chain converges qualitatively faster per token than 2B**,
monotonically across 35+ consecutive checkpoints (step-100 -> step-4,400,
~442B tokens):

| Benchmark | 20B start | 20B @ step-4,400 | Gain |
|---|---:|---:|---:|
| ARC-Easy (acc) | 0.463 | **0.664** | +20pp |
| HellaSwag (acc_norm) | 0.296 | **0.635** | +34pp |
| ARC-Challenge (acc_norm) | 0.224 | **0.380** | +16pp |
| Winogrande (acc) | 0.493 | **0.586** | +9pp |

At ~442B tokens the 20B chain already matches what the 2B chains reach
around ~2T tokens, and beats the 2B 256N chain (step-69,900, ~3.5T
tokens, HSn 0.555) on every benchmark — confirming the larger model is
converging faster per token, not just per FLOP.

These curves are also the **smoking gun for the bf16-master fix**: the v2
(fp32-master) chains climb steadily on ARC / HellaSwag / Winogrande
where the v1 frozen-RMSNorm runs were flat. Detail:
[evals/agpt/20b](../evals/agpt/20b/README.md),
[evals/agpt/2b](../evals/agpt/2b/README.md).

---

## 6. Technical Highlights

### 6.1 80B numerical stability — root cause + stable corner

The multi-week 80B NaN was root-caused via a controlled multi-node sweep
to **two independent triggers in the gradient path** (not a single
global-batch threshold): (1) large `dp_degree` (> ~186) and (2) local
batch size `LBS > 1`. The **safe corner is TP=4 / LBS=1 / dp_degree <=
~186**, reaching target global batch by adding gradient accumulation
(GAS) instead of data-parallel width. Stability was confirmed 4/4 clean
and stress-tested NaN-free to 4x-16x batch via 512N/1024N/2048N
global-batch simulations.

A production-batch (GBS=6144) LR-finder on 2026-06-27 then showed **AdamW
sits on a NaN cliff** at production scale (usable ceiling ~7e-7), while
**mano (~3e-6) and SophiaG (~1e-6) train clean** — which is what made the
80B launch (Section 4.3) possible. Detail:
[80b/README](../production/agpt/80b/README.md). The underlying TP=2 /
LBS>1 grad-path overflow remains an open, upstream-worthy bug with two
cheap reproducers.

### 6.2 Reinforcement learning (SFT + GRPO) on XPU

A full post-training path now runs end-to-end on Intel XPU:

- **SFT**: 2B x `tulu_math_uc_mix` completed (729 steps x 3 epochs,
  GBS=6144). IFEval confirms the SFT'd model beats base on
  instruction-following.
- **GRPO**: production run completed at 8N (1000 steps, 2B-SFT x
  arithmetic); the SFT'd init converges ~8x faster than base. GRPO then
  ran **end-to-end on Sunspot XPU** (real on-policy weight sync) via TRL
  `vllm-serve`. The enabling breakthrough was a **non-PMIx TCP rendezvous
  for oneCCL**, letting the vLLM server and trainer form one rank group
  across two process trees.
- **Monarch** (upstream RL path) was ported far into XPU bring-up but
  remains parked at the fork-spawn / oneCCL-rendezvous boundary; the TRL
  path is the working alternative.

Detail: [rl/README](../rl/README.md),
[grpo-on-xpu-status](../rl/grpo-on-xpu-status.md).

### 6.3 Production resilience infrastructure

Aurora's recurring single-bad-node `NODE_FAIL` pattern made unattended
multi-day runs impossible without tooling. Delivered this quarter:

- **Bad-node failover wrapper** (production-validated 2026-05-23: caught a
  real silent hang at step 37, swapped the bad node, recovered cleanly),
  later replaced by **native `ezpz launch --auto-retry`** submit scripts
  for 2B/20B/80B. ([bad-node-failover](../guides/bad-node-failover.md))
- **Silent-hang watchdog** (`--timeout`) so a stalled collective aborts
  at a deadline instead of burning the full PBS walltime.
- **Walltime-aware checkpointing** — force a final checkpoint before the
  walltime SIGKILL, surviving failover retries.
- **Multi-chain "umbrella" job** packing all four canonical chains into a
  single allocation routed to the uncontended `medium` queue.

### 6.4 Upstream & open-source contributions

The project tracks `pytorch/torchtitan` main continuously — **~30+
upstream syncs absorbed and replayed** this quarter (sync log:
[upstream-sync.md](../upstream-sync.md)) — and contributed fixes back:

- **pytorch/pytorch #184767** (xccl `_set_pg_timeout` dispatch) — filed;
  closed in deference to the overlapping Intel-side PR #183625. A local
  `ezpz` workaround is the production fix in the meantime.
- **pytorch/torchtitan #3436** (deterministic MoE routing via
  `bincount`/`scatter_add_`) — filed with a powered statistical A/B;
  likely superseded by upstream draft #3450.
- **TP loss-reporting bug** (loss off by `dp_world_size` at TP>1) —
  resolved upstream via PR #3159.
- **blendcorpus index-cache race** — fixed at source (atomic write +
  poll) in `saforem2/blendcorpus`, removing the manual cache-prewarm
  requirement for fresh 80B runs.
- **PR #14 (Isolate ezpz MoE)** merged after a multi-day review with full
  A/B numerics (EP=1-12) and a monotonic 1.00x -> 1.94x speedup.

### 6.5 Performance, scaling & research platform

- **torch 2.13 XPU venv**: +23% throughput over torch 2.10 (2B: 7,142
  TPS/GPU at 2N, 27.6% MFU); near-perfect weak scaling to 8N, 94%
  efficiency at 64N.
- **`ezpz yeet-env` tarball broadcast**: venv distribution to compute
  nodes in 70-420s across 8N-4096N (replacing per-file rsync that
  saturated Lustre).
- **Optimizer / architecture research platform**: implemented Mano,
  SPAM, and TorchMuon optimizers plus QK-Norm, logit-softcapping, and
  ReLU^2 architecture variants, run as a 40+ experiment competition with
  live tracking. ([competitions](../competitions/README.md))

---

## 7. Platform / XPU Findings

Recurring Intel-XPU-specific constraints documented this quarter (full
list in [guides/known-issues.md](../guides/known-issues.md)):

- **torch 2.14 (py313)** `torch.compile` segfaults in the Triton 3.7.2
  XPU backend — stay on torch 2.13 (eager works, compile does not at 80B).
- **DeviceMesh-in-saved-tensors** AOT-autograd assertion forces
  `compile=OFF` for the 80B family.
- **FlexAttention** falls back to eager on XPU (~4x slower) — Triton-XPU
  cannot codegen `tanh`/complex ops.
- **`torch.compile` OOM at 512N** (2B on GPU, 80B on CPU) — use
  `--compile.no-enable` for 512+ node jobs.

---

## 8. Risks & Blockers

1. **Aurora 512N queue starvation** — the single biggest throughput
   drag; the 512N chains lost 3+ weeks of wall-clock to scheduling
   contention. Candidate fix: a project reservation or PI->ALCF priority
   discussion. ([analysis](../production/queue-wait-analysis.md))
2. **80B scale > 512N unproven** — 1024N/2048N have a documented init
   crash (`set_determinism` at 12,288+ ranks); real 2048N training is not
   yet submit-ready (would force pipeline parallelism, never validated for
   agpt_80b).
3. **80B grad-path overflow** (TP=2 / LBS>1) — root-caused and worked
   around, but the underlying bug is still open and upstream-worthy.
4. **80B optimizer decision open** — SophiaG (lowest loss, narrow band)
   vs mano (wider stability margin) for the long unattended run.

---

## 9. Priorities for Q3 (Jul – Sep 2026)

1. **Score the completed 2B base model** — full lm-eval scorecard on the
   step-92,859 checkpoint (post-maintenance #1 action).
2. **Drive the 80B production run** — confirm the brackets start, settle
   the optimizer decision, and characterize behavior above 512N.
3. **Advance the 20B chain** past the queue/init-crash blockers toward a
   meaningful token count.
4. **Resolve 512N scheduling** at the program level (reservation /
   priority).
5. **Begin MoE production pretraining** to open the Y2:M2 milestone, now
   that the MoE stack is isolated, sped up, and EP-validated.

---

## Appendix: Source Material & Counts

This report is assembled from the quarter's two-week retrospectives:

| Period | Headline |
|---|---|
| [2026-04-12 -> 04-27](2026-04-12_to_2026-04-27.md) | 291 commits — LR finder + scaling study + production training + optimizer competition platform |
| [2026-05-08 -> 05-22](2026-05-08_to_2026-05-22.md) | 51 commits — first upstream PyTorch PR, 4 syncs, 80B bad-node failover wrapper |
| [2026-05-22 -> 05-29](2026-05-22_to_2026-05-29.md) | 80 commits — 2 upstream PRs, 5 syncs, 20B HSn 0.579->0.635, failover hardening |
| [2026-06-05 -> 06-12](2026-06-05_to_2026-06-12.md) | ~140 commits — SFT complete, GRPO 8N done, PR #14 review, 6 syncs |
| [2026-06-12 -> 06-26](2026-06-12_to_2026-06-26.md) | 151 commits — 80B NaN root-caused, GRPO on XPU, native auto-retry, 4 syncs, PR #14 merged |

**Approximate Q2 activity** (sum of the documented windows; excludes
undocumented gaps): **700+ commits** on the `ezpz` branch, **hundreds of
PBS jobs** across Aurora/Sunspot, **~30+ upstream syncs**, **3 upstream
PRs** (pytorch/pytorch #184767, pytorch/torchtitan #3436, saforem2 PR
#14), and **1 critical correctness bug** (bf16 RMSNorm freeze) caught and
fixed.

Branch: [`ezpz`](https://github.com/saforem2/torchtitan/tree/ezpz) ·
Docs root: [README.md](../README.md) ·
Live production: [production/README.md](../production/README.md)
