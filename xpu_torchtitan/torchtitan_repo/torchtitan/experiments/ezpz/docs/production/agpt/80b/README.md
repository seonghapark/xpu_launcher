# Production Training — agpt 80B

> Last updated: 2026-07-01

> **🟡 LAUNCH ATTEMPTED 2026-07-01 -- 2048N crashed at init; 512N + 1024N
> running.** The 6 80B v2 dispatches (**SophiaG @ LR=1e-6, constant-LR for
> CPT, validator on**, at **512N + 1024N + 2048N**:
> `8574385`/`8574386`/`8574387` heads + 3 `afterany` conts) all stayed
> queued through the 06-29 maintenance -- **no head ran pre-PM.** Post-PM the
> **2048N head (8574387) started first** at ~15:00 UTC (5th attempt after 4
> exec-server rejects during post-maintenance node-release flapping) but
> **SIGSEGV'd in `set_determinism`** at 24,864 ranks (`F`, rc=143, 14:47
> walltime). This is the **documented init-crash class** (previously 2B/20B
> at 1024N/12,288 ranks), now confirmed for 80B at 2048N. Its continuation
> (8574390) was `qhold`'d (would recrash deterministically). **512N
> (8574385, proven-safe) and 1024N (8574386, the untested 80B data point)
> are the live brackets** -- 1024N will bracket exactly where the 80B init
> ceiling sits (512N=dp1530 proven, 2048N=dp6138 crashes, 1024N=dp3066 is
> the missing measurement). Also found: auto-retry misclassified the SIGSEGV
> (rc=143) as a walltime stop and skipped its 2 retries -- a classifier gap
> (moot for a deterministic init crash, but real for swappable bad-node
> SIGSEGVs). The old AdamW "step-2 NaN" was a production-batch LR problem
> (LR-finder GBS=6144: AdamW NaN-cliff ~7e-7; mano ~3e-6 / sophiag ~1e-6
> clean). **TEAM DECISION OPEN: SophiaG vs mano** for the base optimizer.
> Full plan + launch log:
> [20260628-80b-sophiag-constant-lr-512-1024-2048.md](../../experiments/agpt/aurora/20260628-80b-sophiag-constant-lr-512-1024-2048.md).
> LR-finder: [lr-finder/agpt/80b](../../experiments/lr-finder/agpt/80b/README.md).

## 80B trajectory chart

**No live 80B production chart yet** — the production brackets (above) are
queued but haven't started persisting ckpts (machine in PM). The chart
will populate once they run. The per-trajectory stub
[`n4/README.md`](n4/README.md) holds the 4N validation details.

For the cross-model view (2B + 20B together — 80B is not on it yet for
the reasons above), see [`../README.md`](../README.md).

## v2 status -- stable TP=4/LBS=1 corner found; production-restart pending

**Headline**: a stable 80B training corner (TP=4 / LBS=1 / bf16 / AdamW
LR=1e-6, GAS-to-GBS) was identified 2026-06-24 and has held NaN-free from
GBS=372 up to **GBS=2976 (8x batch)**; the 16x rung (GBS=5952) NaN'd at
step 29 (see the simulation campaign below, with its LR/warmup caveat) --
this supersedes the long "256N blocked on NaN" status. The original
end-to-end stack validation was on Aurora **2026-06-08** via an
interactive 4N smoke that completed through step-10 sync-checkpoint save
cleanly
([n4/README.md](n4/README.md): 904 GB / 48 .distcp shards / .metadata —
matches the Sunspot 12468197 reference exactly). Required clearing 5
stacked bugs: repo 229 commits behind, broken venv symlink, blendcorpus
init-barrier deadlock at 4N+, missing `ZE_FLAT_DEVICE_HIERARCHY=FLAT`,
and an interactive-launch env block. The same software stack was then
tried at 256N.

**256N attempts (2026-06-08 / 06-09)**:

| Job ID | Date | LR | Result | Notes |
|--------|------|-----|--------|-------|
| 8530891 | 2026-06-08 | 1e-6 | **Loss NaN at step 2** | Step 1 clean (loss 12.94), step 2 grad_norm NaN, step 3+ loss NaN. Even at scheduler-clamped LR ≈ 1.8e-8, NaN persists. Open hypotheses: bf16 overflow at GBS=1536, TP=2 loss-reduction bug, fp32 second-moment overflow. |
| 8531345 | 2026-06-08 | 1e-7 | OOM / failover-exhausted | Retry with smaller LR; bad-node hit storm, never reached step 1. |
| 8531721 | 2026-06-09 | 1e-7 | `std::bad_alloc` at model construction | Ranks 401-528 (contiguous block on 11-12 nodes) — failover ran out of spares. Never reached step 1. |

### grad_norm NaN: two independent triggers (LBS>1 and large dp-degree), 2026-06-24

A controlled 5-run sweep at 80B (LR=1e-6, the working recipe, q_BLNH
fix in) maps the grad_norm-first NaN. **It is NOT simply GBS-driven, nor
purely TP-driven** -- an early "TP=4 fixes it" read was disproved by the
LBS=2 run. The real picture, with `dp_degree = NGPUS / TP` (the rank
count the gradient all-reduce spans) and LBS = per-forward local batch:

| Job ID | TP | LBS | GAS | dp_degree | GBS | Result |
|--------|----|----|----|-----------|-----|--------|
| 12469486 | 2 | 1 | 1 | 168 | 168 | **clean** → loss 10.38 @ step20 |
| 12469492 | 2 | 1 | 1 | 372 | 372 | **NaN** @ step 3 |
| 12469494 | 4 | 1 | 2 | 186 | 372 | **clean** → loss 10.31 @ step20 |
| 12469495 | 4 | 1 | 1 | 186 | 186 | **clean** |
| 12469499 | 4 | **2** | 1 | 186 | 372 | **NaN** @ step 4 |

Reading the matrix:

- **GBS alone is not the trigger.** GBS=372 is clean (12469494) *and*
  NaN (12469492, 12469499) depending on how it's composed.
- **Two distinct things each cause the NaN:**
  1. **Large `dp_degree`** -- 12469492 (dp=372) NaNs at LBS=1; the three
     clean runs all have dp ≤ 186. Bigger DP all-reduce -> grad blowup.
  2. **LBS > 1** -- 12469499 NaNs with dp=186 (same as the clean runs)
     and GAS=1, differing from the clean 12469494 *only* by
     LBS=2 vs LBS=1+GAS=2. So a larger per-forward batch independently
     triggers it, even at a "safe" dp_degree.
- The safe corner is **LBS=1 and dp_degree ≤ ~186**; you reach a target
  GBS by adding GAS (sequential microbatches), not by raising LBS or
  dp_degree past that. TP=4 helps only because, at fixed NGPUS, it halves
  dp_degree (744/4=186 vs 744/2=372) -- it's not a TP fix per se.
- Signature throughout: **grad_norm goes NaN one step before loss**, so
  the blowup is in the gradient/optimizer path (DP grad all-reduce, TP
  loss-parallel grad, clipping, or fp32 second-moment), not the forward.

**Performance cost of the safe corner (TP=4, 62N, GBS=372):**

| Config | LBS | GAS | MFU | TFLOPS/gpu | Memory |
|--------|----|----|-----|-----------|--------|
| TP=2, GBS=168 (baseline) | 1 | 1 | 18.7% | 55.7 | 65.9% |
| TP=4, GBS=372 (GAS=2) | 1 | 2 | 9.85% | 29.4 | 32.0% |
| TP=4, GBS=372 (LBS=2) | 2 | 1 | ~14-15.8% | ~42-47 | 48.0% |

LBS=2 is markedly faster than GAS=2 (better comm amortization, ~15% vs
~10% MFU) and fits memory comfortably (48%) -- **but it NaNs**, so it's
not usable until the LBS>1 trigger is fixed. The viable stable config is
TP=4 + LBS=1 + GAS to reach GBS, at ~9.85% MFU -- roughly **half** the
TP=2 baseline throughput. The penalty is TP=4 communication (confirmed
GAS-independent: GAS=1 and GAS=2 both ~9.8%).

**Status / next:** a stable 80B path exists (TP=4, LBS=1, GAS-to-GBS) but
costs ~2x throughput. The high-value fix is root-causing the grad-path
NaN -- two reproducers now exist that are far cheaper than 256N: the
dp-degree trigger (12469492, 62N) and the LBS trigger (12469499, 62N).
Both show grad_norm NaN first. Likely an upstream DTensor/loss-parallel
or FSDP grad-reduction numerical issue worth an upstream report. The
batch-size ramp (`FaultTolerantTrainer.batch_ramp_steps`, commit
`09f2d243b`) ramps GAS only (LBS/dp fixed), so it mitigates the
dp-degree onset but NOT the LBS trigger.

#### Stability of the TP=4/LBS=1/bf16 path — CONFIRMED 4/4 clean (2026-06-24)

The clean 80B runs could have been single-shot luck: the failure mode is
partly **nondeterministic** (see the n32 diagnosis doc reconciliation
below: the *same* config can be clean or NaN run-to-run depending on XPU
execution order during the step-15-17 grad spike). So we ran 4
independent repeats of the exact config (TP=4, LBS=1, GAS=2, GBS=372,
bf16, LR=1e-6):

| Job ID | Steps | Result |
|--------|-------|--------|
| 12469494 | 20 | clean, loss 12.94 -> 10.31, 0 NaN |
| 12469509 | 30 | clean, loss 12.94 -> 9.69, 0 NaN |
| 12469510 | 30 | clean, loss 12.94 -> 9.69, 0 NaN |
| 12469511 | 30 | clean, loss 12.94 -> 9.70, 0 NaN |

**4/4 clean, all traversing the step-15-20 danger window that NaN'd every
TP=2 GBS>=192 run; three landed at the identical final loss (9.69-9.70).**
This is not the nondeterministic knife-edge -- TP=4/LBS=1/bf16 is a
genuinely stable 80B path at GBS=372, in pure bf16 (no fp32-acts, no
determinism flag). It costs ~half the TP=2 MFU (~9.85% vs 18.7%), which
is the price of TP=4 communication.

**Production recommendation (supersedes the n32 doc's fp32-acts default):**
run 80B at **TP=4, LBS=1, bf16, GAS to reach target GBS**. Cheaper than
fp32-acts (~3-5x) and determinism (~50%, and determinism doesn't even
scale past n=32). Caveat: validated to 30 steps / GBS=372 / 62N; a longer
production chain should still be watched for late-training instability,
and the underlying TP=2 / LBS>1 grad-path overflow is still an open
upstream-worthy bug (two cheap 62N reproducers recorded above).

#### Corner holds to 8x batch, breaks at 16x (global-batch simulation campaign) -- 2026-06-25/26

To check the corner isn't specific to the small GBS=372 batch, a series
of runs raised GAS at fixed 62N (dp_degree=186) to reach the *global
batch* an Aurora 512N/1024N/2048N run would see -- on the safe side of
the dp_degree<=186 ceiling (GAS scales GBS without touching dp_degree).

| Job ID | TP | LBS | GAS | dp_degree | GBS | Simulates | Steps | Result |
|--------|----|----|----|-----------|-----|-----------|-------|--------|
| 12469551 | 4 | 1 | 2 | 186 | 372 | native | 100 | clean, 12.93 -> 7.72, 0 NaN |
| 12469609 | 4 | 1 | 8 | 186 | 1488 | 512N | 46 (walltime) | clean, 12.92 -> 8.84, 0 NaN |
| 12469626 | 4 | 1 | 16 | 186 | 2976 | 1024N | 34 (walltime) | clean, 12.92 -> 9.84, 0 NaN |
| 12469627 | 4 | 1 | 32 | 186 | 5952 | 2048N | NaN @ step 29 | 28 clean, grad_norm NaN @ 29 |

**Holds to 8x (GBS=2976), breaks at 16x (GBS=5952).** All rungs ride the
same step-22-27 grad transient (peaks ~21-23); 1488/2976 recover and run
past step 29 clean, but 5952 fails to come down off it and NaNs at step
29 -- batch-dependent at *matched step count*.

**Key caveat (do not over-read):** LR was flat 1e-6 for every rung (NO
batch scaling), and the runs were *inside warmup* (clamped to
total_steps), so effective LR at the 5952 NaN was only ~5.8e-7. So the
16x NaN is batch-dependent at matched step + effective-LR, but its
dependence on the full nominal / batch-scaled LR is unknown. Follow-ups:
`12469698` (LR=1.6e-5 16x-scaled + warmup=5), `12469699` (LR=1e-6,
warmup=200, reproducibility). Full report:
[`gbs5952 2048N`](../../../experiments/agpt/sunspot/2026-06-26-80b-gbs5952-2048N-sim.md).

This campaign establishes the corner's stability is largely batch-size
independent up to 8x. Reports:
[`gbs1488 512N`](../../../experiments/agpt/sunspot/2026-06-25-80b-gbs1488-512N-sim.md),
[`gbs2976 1024N`](../../../experiments/agpt/sunspot/2026-06-26-80b-gbs2976-1024N-sim.md),
[`gbs5952 2048N`](../../../experiments/agpt/sunspot/2026-06-26-80b-gbs5952-2048N-sim.md).

#### The dp_degree<=186 "ceiling" is not a cliff -- corner scales past it (2026-06-26)

186 was only `62N x 12 / TP=4`, the highest *tested* point -- not a
measured boundary. A node-count bisect at fixed TP=4/LBS=1/GAS=1 raised
dp_degree directly:

| Job | N | dp_degree | vs 186 | Result |
|---|---|---|---|---|
| 12469628 | 64 | 192 | 1.03x | **clean** 30 steps, 0 NaN |
| 12469629 | 88 | 264 | 1.42x | **clean** 30 steps, 0 NaN |
| 12469630 | 108 | 324 | 1.74x | queued |

**dp=192 and dp=264 both ran NaN-free.** The only dp-axis NaN ever seen
remains 12469492 (TP=2, dp=372). So the safe corner extends to at least
dp=264; the real cliff is somewhere in (264, 372], unmapped (dp=372 needs
124N > Sunspot's 114 usable nodes -- confirmable only on Aurora). **The
script's `dp_degree>186` warning is therefore over-conservative** (left in
place pending the exact cliff, but 192-264 are now known-safe).

#### LR is the sharp wall, not batch or dp-degree (2026-06-26)

An LR-finder sweep (lr 1e-6->1.0, GBS=372) puts the min-loss LR at
**~3e-6 (sophiag) / ~8e-6 (mano)**, diverging by ~1e-2 -- production
LR=1e-6 sits safely on the left shoulder. Confirming the wall is LR not
batch: GBS=5952 with a 16x-linear-scaled LR=1.6e-5 NaN'd at **step 7**
(vs step 29 at flat 1e-6) -- scaling LR up walks straight into the
divergence shoulder. **Do not scale LR with batch size**; the ceiling is
fixed by the bf16/dim-9216 overflow. Full analysis:
[`lr/batch/dp-degree findings`](../../../experiments/agpt/sunspot/2026-06-26-80b-lr-batch-dpdegree-findings.md).

## Working 4N stack (Aurora + Sunspot)

The working 80B config has been replicated four times across the
two clusters with bit-equivalent numerics:

| Job ID | Cluster | Date | Stack | Loss (step 1 → 20) | MFU |
|--------|---------|------|-------|---------------------|-----|
| 12466025 | Aurora | 2026-05-05 | torch 2.13, pre-xccl-workaround | 12.98 → 10.46 | ~17.8% |
| 12467825 | Sunspot | 2026-06-02 | + xccl_split_group workaround (commit 8031d1d3a) | 12.94 → 10.39 | ~17.8% |
| 12468157 | Sunspot | 2026-06-06 | + 47th upstream sync (RoPE refactor PR #3458, mixed-optimizer PR #3269) | 12.97 → 10.41 | ~17.8% |
| 8530800 (r4 smoke) | Aurora | 2026-06-08 | + blendcorpus barrier fix + venv-symlink fix + `ZE_FLAT_DEVICE_HIERARCHY=FLAT` | step-10 ckpt saved (904 GB, 48 shards) | — |
| 12469486 (**28N**) | Sunspot | 2026-06-24 | + 57th/58th upstream sync (AC policy hierarchy, disable_loss_parallel removal, **q_BLNH attention fix** for TP>1 local_map) | 12.95 → 10.38 | ~18.7% |

The first four are 4N and land within ±0.08 nat of one another at
step 20 (88.94% peak memory). **`12469486` is the first multi-node
(28N) validation** -- loss 12.95 → 10.38 matches the 4N baseline
within noise at ~18.7% MFU / 65.86% memory, `step-20` ckpt saved.
It also flushed out a TP>1-only regression from the 57th sync: the
ezpz attention forks used bare `q/k/v` forward args while upstream's
`set_gqa_inner_attention_local_map` now keys `in_dst_shardings` by the
shape-suffixed `q_BLNH/k_BLNH/v_BLNH`; the local_map contract check
asserts under TP>1. Fixed in `74c7452f6`; `sync_smoke.sh` gained a TP=2
entry (`13c09ddf9`) so future syncs catch it. See
[../../../upstream-sync.md](../../../upstream-sync.md) 57th-sync
follow-up. See [n4/README.md](n4/README.md) for the 4N smokes.

**Working config**: AdamW LR=1e-6, TP=2, AC=full, compile=OFF,
fp32-master, sync checkpoint mode (`CHECKPOINT_ASYNC_MODE=disabled`).
Submit script:
[`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../../../scripts/submit_agpt_80b_aurora_venv_failover.sh).
Production clone: `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/`.

## Per-trajectory detail

- [n4/](n4/README.md) — 4N validation smokes (Aurora + Sunspot,
  proven-stable end-to-end including sync ckpt save 2026-06-08)
- [n512/](n512/README.md) — 80B 512N (not yet attempted — 256N still
  blocked)

## Next steps

1. **Diagnose the 256N NaN.** Possible angles:
   - Try TP=4 to halve effective per-replica GBS (current TP=2, GBS=1536)
   - Try with `--validator.enable` off to rule out validator-loss-path issues
   - Dump per-tensor stats at step 1 to localize which weight first goes NaN
   - Bisect on bf16-vs-fp32 master at 256N (4N is fp32-master, working)
2. **256N spare headroom**: every 256N retry has been killed by bad
   nodes long before reaching meaningful training. The wrapper rotates
   spares correctly but the hit-rate at 256N is high; try `select=296`
   (40 spares vs current 10).
3. **Replay 80B once 2B 512N stalls clear** so we're not competing for
   the same 256N+ slots in `small` while debugging.

## Historical

- Pre-2026-06-08 80B 256N production: 11+ failed dispatches starting
  2026-05-11. Failure-mode writeup:
  [20260524-80b-256n-sigsegv-cascade-8505222.md](../../../experiments/agpt/aurora/20260524-80b-256n-sigsegv-cascade-8505222.md).
  Distinct from the 80B 8N smoke failure (`blendcorpus` EOFError race —
  fixed by the init-barrier removal that landed in 2026-06-08's r4 smoke):
  [blendcorpus-eoferror-race.md](../../../guides/known-bugs/blendcorpus-eoferror-race.md).
- Historical v1 (NaN'd, bf16-master) runs:
  [../historical/v1-bf16/](../historical/v1-bf16/README.md).
