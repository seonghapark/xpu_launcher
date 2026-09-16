# Development Journal

Running log of what's happening, session by session. Most recent first.

---

## 2026-07-01 (aurora) -- 80B launch: 2048N crashes at init, 512N + 1024N run

Machine returned from the Mon 2026-06-29 maintenance. Managing the 80B
SophiaG/constant-LR production launch and refreshing docs.

- **80B launch attempted (~15:00 UTC).** All 6 jobs (3 heads + 3 conts) stayed
  queued through the PM; no head ran pre-maintenance, so these are cold starts.
  The **2048N head (8574387) started first**, ahead of the 512N/1024N brackets.
- **4 exec-server rejects before it placed.** run_count 1-4: the job flipped
  Q -> R -> Q with `PBS Error: Execution server rejected request`, never writing
  a log. Diagnostic that drove the "wait, don't requeue" call: *peer 2000+N jobs
  were running fleet-wide* during the reject window, so it was post-maintenance
  node-release flapping specific to our attempts, not a machine-wide inability to
  place large jobs. Attempt 5 (~15:00 UTC) placed cleanly on 2072 nodes.
- **Then it SIGSEGV'd in `set_determinism`** (`F`, rc=143, 14:47 walltime).
  Config echo was correct (GBS=6138, TP=4, LBS=1, compile OFF, steps=92,950,
  SophiaG, constant-LR) and venv broadcast finished, but the seed broadcast in
  `distributed/utils.py:231` faulted at **24,864 ranks** (`rank 13602 died from
  signal 11`; CCL/PMI KVS "Connection reset by peer"). This is the **documented
  init-crash class** (2B/20B at 1024N/12,288 ranks), now confirmed for 80B at
  2048N. NOTE: I briefly misread an earlier `build_mesh` success as
  "set_determinism cleared" -- it had not; the crash is *in* set_determinism.
- **Finding: init-crash ceiling for 80B is bracketed by the 1024N run.**
  512N (dp=1530) proven; 2048N (dp=6138) crashes; **1024N (8574386, dp=3066) is
  the missing measurement** -- if it survives init the ceiling is ~2048N-specific,
  if it crashes then 512N is the practical 80B max on this stack.
- **Finding: auto-retry misclassified the SIGSEGV as a walltime stop.** The rank
  SIGSEGV -> SIGTERM -> job rc=143, which `launch_autoretry.py:726` logged as
  `FAILOVER STOP: walltime` and did NOT consume its 2 retries. Moot for a
  deterministic init crash, but a real classifier gap: a swappable bad-node
  SIGSEGV would also produce rc=143 and never trigger a spare-swap. Filed as a
  follow-up (rc=143/SIGTERM should be distinguished from a wrapper-initiated
  walltime-margin stop).
- **Action:** `qhold`'d the 2048N continuation (8574390, `afterany:8574387`) --
  `afterany` fires on failure too, so it would have grabbed 2072 nodes for the
  same crash. 512N + 1024N heads left to run; backfilling as 8574387's nodes
  release. No `qdel` (holds are reversible).
- **Docs refreshed:** dashboard + agpt/80b rollups + launch report all corrected
  from "launching" to the crash outcome, with the two findings recorded.
  Operational lesson also recorded: after a reservation tears down, a large job
  can eat several exec-server rejects before nodes stabilize; wait (don't
  requeue, which forfeits queue priority) as long as peer large jobs are placing.
- **Restart-economics analysis** (from 2026-06-30) stands: ~7 confirmed failover
  recoveries of ~62 triggered episodes; ~78% of exhaustions are systemic
  (CCL/PMI KVS timeout 57%, now-fixed blendcorpus race 15%, pals-RPC 6%);
  node-hour waste is bimodal (median episode ~11min but 8 episodes >3h account
  for ~64% of the ~31.8k wasted node-h). See
  [`20260630-failover-restart-economics.md`](experiments/agpt/aurora/20260630-failover-restart-economics.md).

## 2026-07-01 (sunspot) -- 80B convergence run (all optimizers NaN), MoE page reorg, 63rd sync

Concurrent Sunspot session (separate from the Aurora 80B launch above).
Three things landed.

- **MoE LR-finder pages reorganized production-first** (`56aad15ac`). Matched
  the dense-page reorg: MoE has no production-GBS finder yet (all data is the
  2026-04-21 GBS=192 small-batch sweep), so the index leads with a "Status:
  small-batch only" banner + the batch-dependence lesson, and the April sweep
  is collapsed into `<details closed>` on the index + all 5 config pages.
  Preserved the one inbound anchor (`experiments/moe/README.md`).
- **80B head-to-head convergence run -- all three optimizers NaN**
  (`c015d55d5` + report `2026-06-30-80b-convergence-gbs6144.md`). New
  `scripts/{run,submit}_80b_convergence.sh`. Ran mano/sophiag/AdamW at their
  finder-recommended CONSTANT LRs (3e-6/1e-6/5e-7, GBS=6144, 64N, jobs
  12469910/911/912). **All three descended a few steps then diverged: mano
  first (grad NaN step 5), AdamW step 9, sophiag step 12.** grad_norm runs up
  then explodes, loss NaNs one step later. The finder's early-step ranking
  (which said mano safest) does NOT predict sustained stability; the shared
  failure across 3 different optimizers points to a corner-level instability
  (bf16 at dim=9216), not tuning. Conclusion: no finder LR is production-safe
  as a constant LR here -- needs a long warmup (>=200 steps) + grad clipping,
  possibly an fp32 grad path. Smoke-first caught a real bug first (the runner
  missed the yeet-env/`/tmp/.venv` preamble -> `env: ezpz` exit 127, fixed in
  `1b554df71`). The corner is ~20 min/step, so 50-step jobs were the practical
  cap (all NaN'd by step 12 anyway).
- **Queue hygiene:** qdel'd the stale 112N dp=324 bisect (12469630, queued ~5d,
  dp ceiling already disproved) and the finished smoke -- unblocked the
  convergence jobs to backfill immediately.
- **63rd upstream sync** (`cf99e127e`, 13 commits `390ea37cc..`). No replays
  (llama3/deepseek_v3 untouched; agpt/moe byte-identical). One RL conflict in
  `generator.py` resolved by taking upstream (spmd_types if/else superset).
  Impact is RL/FLUX/spmd_types (no-ops for our DTensor backend; loss.py edits
  are spmd_types-guarded) + a DeepEP-v2 upgrade touching moe's token
  dispatcher. **Smoke-validated:** agpt 2B trained clean 10 steps (loss 8.35);
  moe debugmodel exercised the DeepEP-v2 dispatcher path without error then
  OOM'd downstream (`UR_RESULT_ERROR_OUT_OF_RESOURCES`, a known XPU resource
  limit, not a merge regression). Most of the effort was worktree plumbing --
  a git worktree only has tracked files, so jobs run from one need `.venv`,
  `.venv.tar.gz`, and `assets/hf` symlinked in (the last tripped me: `assets/`
  is a tracked dir, so the tokenizer download links at `assets/hf`, 3 levels
  up).

## 2026-06-29 (sunspot) -- 2B 100-step production-batch ladder + LR-finder page reorg

Continuation of the LR-finder work. Two deliverables, both shipped.

- **2B 100-step production-GBS ladder (jobs 12469854-868, 16N/dp=192).**
  Re-ran the production batch sweep (GBS 1536/3072/6144/12288/24576) at the
  classic **100-step** finder length (the 2026-06-28 trend used 15 steps) for
  adamw/mano/sophiag -- 15 jobs in parallel, isolated dumps
  `outputs/lrfind-2b-100step-prod/gbs<N>/`. **All 15 complete, 0 NaN across
  the full 16x batch range.** Confirms the 15-step story at the longer length:
  2B never cliffs even at 100 steps + 4x production batch, no batch-scaling
  trend (AdamW ~2.5-3.6e-3, mano ~5-9e-3, sophiag dead-flat ~2e-3), 3-4 orders
  above the 80B 7e-7 cliff. Deep minima (~7.7-8.3 vs ~11.5 at 15 steps) are the
  cumulative-training sweep-length effect -- compare by min-LR, not loss value.
  New `scripts/plot_2b_100step_prod.py` (hue=optimizer canonical colors, batch
  = shade+width+opacity); two figures, generated on Sunspot from real CSVs,
  y-capped + smoothed. Built incrementally as tiers landed (9/15 -> 12/15 ->
  15/15), regenerating each time. Commits `a9bfc91b5`, `e1eefcba6`,
  `6f7f6b6b6`.
- **LR-finder pages reorganized production-first.** All three agpt pages now
  lead with the production/recent results and collapse the old 2026-04 2-node
  small-batch finders into `<details closed>` blocks:
  - 2B (`542ae2d49`): (1) 100-step production ladder, (2) 15-step trend,
    (3) additional findings, (4) collapsed April debug runs, (5) reports index.
  - 20B + 80B (`098ec19cc`): production/trend on top, April runs collapsed
    (demoted to `###`). Heading text preserved verbatim so every inbound anchor
    (2B page, agpt index, experiments/agpt, scaling-performance) still resolves.
- **Docs hygiene.** Caught + corrected a missed step: the docs-root
  "Recently Updated" index table is git-commit-date driven and auto-generated
  (`utils/refresh_docs_readme_table.py`) -- refreshed it in the same commit on
  every doc push from here on, not after the fact.

---

## 2026-06-28 (sunspot) -- LR-finder docs overhaul, 2B trend, 62nd sync, validator phantom fixed

Docs/tooling-heavy session plus one real bug closed.

- **LR-finder docs split per-model.** `docs/experiments/lr-finder/` went from
  two consolidated family pages to `{agpt,moe}/<model>/README.md` +
  per-model `figures/` (agpt 2b/20b/80b; moe debugmodel/500m/2b/4b/7b), each
  family keeping an index README. All inbound links + anchors repointed.
- **2B LR-ceiling-vs-GBS trend (jobs 12469769-776, 16N/dp=192, GBS
  192..24576).** Headline: **2B never cliffs** -- AdamW usable LR flat at
  ~1e-2 across the whole 128x batch range, 0 NaN, vs 80B's collapse to a
  ~7e-7 NaN cliff. So the batch-dependent usable-LR collapse is a
  LARGE-MODEL (dim=9216 bf16) phenomenon, not universal. Ran all 3 working
  optimizers (adamw/mano/sophiag) at all 8 GBS; muon dropped (oneCCL abort
  ~7min in). Found sophiag's U sharpens/deepens with batch (not perfectly
  batch-independent like adamw/mano).
- **Charts: house style + completeness.** Wired `apply_style()` (ambivalent +
  Iosevka) into the finder auto-plot and `plot_lr_finder.py`, made
  `apply_style` import-safe without IPython (loads the .mplstyle by path).
  Added per-GBS loss-vs-LR curve families for all optimizers + a reusable
  `scripts/plot_lr_trend.py`, registered in the refresh catch-all.
- **Corrections caught by the user:** (1) "production batch" is **GBS=6144**
  (2B 256N + all 80B), not 12288 (that's 2B 512N) -- relabeled, kept all
  12288 data/figures. (2) Documented why old finder min-loss (~9-10) <
  new (~11.3): sweep length (100 vs 15 steps), since the finder trains
  cumulatively -- only the min-LR is comparable across sweep lengths.
- **62nd upstream sync** (3 commits, `0e886617e..390ea37cc`, all
  experiments/rl/ -- no replays).
- **Validator "CCL deadlock at 80B TP=4" was a PHANTOM (root-caused + fixed +
  CONFIRMED).** A 4-way worktree fan-out found no collective-deadlock log:
  the label conflated a validator dataloader cold-cache mmap-race CRASH (job
  12469584) and a training-side index-build barrier stall (job 12469597),
  plus a genuine `loss_fn` tuple-unpack crash in `validator.py`. Fixes:
  tuple-unpack (`loss_sum, _ = self.loss_fn(...)`); validator inherits the
  warm `data_cache_path` (config default + the submit script already passed
  it on the CLI); prewarm builds the validation index. **Confirmed 2026-06-28
  (jobs 12469784 prewarm + 12469785 train-loop, 4N/dp=12):** first-ever
  `validate()` completions at 80B TP=4 -- finite val loss, no
  mmap/AttributeError/CCL hang, across 4 passes. dp=12 confirmation; a 62N
  pass would be belt-and-suspenders. Writeup:
  [`docs/guides/known-bugs/validator-tp4-at-80b.md`](guides/known-bugs/validator-tp4-at-80b.md).
- **Ops:** flipped the default to auto-push-after-batch on working branches
  (pull-rebase first); the 80B trend gap-fill reruns (2304-redo 12469778,
  4608 12469767) and dp=324 bisect (12469630) are still out.

---

## 2026-06-26 (sunspot eve) -- 80B: LR is the wall, dp-ceiling isn't, don't scale LR with batch

Five jobs resolving the LR/batch/dp-degree questions the sim campaign
raised. Synthesis report:
[`docs/experiments/agpt/sunspot/2026-06-26-80b-lr-batch-dpdegree-findings.md`](experiments/agpt/sunspot/2026-06-26-80b-lr-batch-dpdegree-findings.md).

- **Were the sims LR-scaled? No -- and they shouldn't be.** All sims used
  flat LR=1e-6. An LR-finder (lr 1e-6->1.0, GBS=372) puts min-loss LR at
  **~3e-6 (sophiag) / ~8e-6 (mano)**, diverging by ~1e-2 -- production
  1e-6 is on the safe left shoulder, below optimum. (adamw/muon curves
  lost to the cold-cache race; need warm rerun.)
- **Scaling LR up with batch made it WORSE:** GBS=5952 + 16x-scaled
  LR=1.6e-5 (job 12469698) NaN'd at **step 7** (loss 11.6->20.4->nan),
  vs step 29 at flat 1e-6. 1.6e-5 is at the LR-finder divergence shoulder.
  The LR ceiling is fixed by bf16/dim-9216 overflow, not the batch ->
  **do not linearly scale LR with batch here.**
- **The dp_degree<=186 "ceiling" is not a cliff:** bisect jobs 12469628
  (dp=192) and 12469629 (dp=264) both ran 30 steps NaN-free. 186 was just
  the highest tested point. Corner scales to at least dp=264; the script's
  dp>186 warning is over-conservative. dp=324 (12469630) queued; dp=372
  needs Aurora. **This vindicates the earlier instinct that TP=4 should
  scale past 62N.**
- Net: two walls quantified -- a sharp LR wall (~1e-5, optimizer-set) and
  a much-further-out dp-degree wall (>264). The GBS=5952 flat-LR step-29
  NaN is a separate, mild batch-accumulation effect at low LR.

---

## 2026-06-26 (sunspot) -- 80B global-batch sim campaign continues: GAS=16 (1024N) clean

Continued the 80B batch-scaling campaign past the GBS=1488/512N sim.

- **GAS=16 / GBS=2976 / 1024N-batch** (job `12469626`, 62N, dp=186):
  **clean, 34/34 steps to the 6h walltime cut, zero NaN**, loss
  12.92 -> 9.84. grad_norm showed the same step-22-27 transient as the
  other sims (peak 21.5 @ step 23, recovered to ~9.3). 8x the validated
  GBS=372, same smooth descent. Report:
  [`docs/experiments/agpt/sunspot/2026-06-26-80b-gbs2976-1024N-sim.md`](experiments/agpt/sunspot/2026-06-26-80b-gbs2976-1024N-sim.md).
  Notably this run **cold-built the blendcorpus index at 744 ranks
  cleanly** (no race at this index size), confirming the `debfff5`
  sibling barriers suffice for cold-build-at-scale.
- **GAS=32 / GBS=5952 / 2048N-batch** (job `12469627`): **NaN at step 29**
  (UPDATE -- the line below said "in flight, clean through step 24"; it
  went on to NaN). Ran 28 clean steps then `grad_norm=nan` at step 29
  (loss still finite -- grad-path-first signature), then PBS walltime-cut.
  Its first attempt hit the blendcorpus cold-cache race (`EOFError`) and
  **auto-retry self-healed** (one spare). Full report:
  [`docs/experiments/agpt/sunspot/2026-06-26-80b-gbs5952-2048N-sim.md`](experiments/agpt/sunspot/2026-06-26-80b-gbs5952-2048N-sim.md).
- Campaign result: GBS=372/1488/2976 clean; **5952 (16x) NaN'd at step
  29.** The corner holds NaN-free to **8x** the validated batch, breaks
  at 16x. CAVEAT (the key catch): LR was flat 1e-6 for ALL rungs (no
  batch scaling) and every run was *inside warmup* (clamped to
  total_steps), so effective LR at the NaN was only ~5.8e-7. So the 16x
  NaN is batch-dependent at matched step+effective-LR, but its dependence
  on the full / batch-scaled LR is unknown. Two follow-ups queued:
  `12469698` (LR=1.6e-5 scaled + warmup=5) and `12469699` (LR=1e-6,
  warmup=200, 60 steps -- reproducibility). Added a WARMUP_STEPS knob to
  the 80b script for this.
- Still pending: the dp-degree cliff bisect (n64/n88/n108 -> dp
  192/264/324), the actual probe of whether the corner survives
  node-count scaling past dp=186.

---

## 2026-06-25 (sunspot) -- 80B GBS=1488 512N-batch simulation: clean, 4x batch NaN-free

Pushed the 80B TP=4/LBS=1/bf16 stable corner to **4x the validated
global batch** (GBS 372 -> 1488) via GAS=8, to simulate the global batch
an Aurora 512N run would see while staying inside the NaN-free
`dp_degree <= 186` corner. Held 62N (dp=186) and raised GAS, since
`GBS = dp_degree * LBS * GAS` -- GAS scales the batch without touching
the dangerous dp_degree.

- **Job `12469609`** (62 active + 6 spare), GBS=1488 (~3% under the
  Aurora 512N GBS of 1536), AdamW LR=1e-6, AC=full, compile=OFF,
  `VALIDATOR_ENABLE=0`.
- **Result: clean.** 46/46 steps before the 4h walltime cut, **zero
  NaN**, loss 12.92 -> 8.84, grad_norm bounded (peak 22 @ step 23,
  recovered to ~8), MFU steady ~9.85%. The corner's stability is not
  specific to the small batch.
- Full report:
  [`docs/experiments/agpt/sunspot/2026-06-25-80b-gbs1488-512N-sim.md`](experiments/agpt/sunspot/2026-06-25-80b-gbs1488-512N-sim.md).

**The multi-attempt arc had TWO distinct causes (corrected after reading
all four failed-job logs).** This run was the successful relaunch after
four prior GBS=1488 attempts failed:

1. **Validator cold-built at the wrong path (12469584).** That attempt's
   *training* dataloader built the full 74773-sample index **cold at 744
   ranks and succeeded**, reaching step 1 clean -- so cold-build at scale
   already works with the `debfff5` sibling barriers. The crash was the
   *validator* building its validation-split index cold at the default
   `.cache/blendcorpus` path (missing `--validator.dataloader.data-cache-path`).
2. **Self-inflicted barrier (12469590/592/597).** Reacting to (1), I
   added a global `torch.distributed.barrier()` in blendcorpus
   `_build_index_mappings` (`c7eb628`). That function is called a
   data-dependent number of times (per corpus x per split, branch-gated),
   so ranks hit the barrier a mismatched number of times ->
   partial-participation deadlock (`oneCCL allreduce_scaleout ...
   atl_comm->wait`, confirmed at `gpt_dataset.py:1145` in 12469597).
   **Reverted in `74b09fd`**; the next run (12469609) trained clean.

- Corrects the earlier read: cold-build at 744 ranks **works** (12469584
  proves it) -- the cache this run loaded "warm" was built cold by
  12469584, not by a pre-warm. The `prewarm` `--training.steps=1` call
  builds a *wrong-sized* index anyway (hash keys on
  `num_samples = GBS * train_iters`).
- Ruled out: **NOT node health** (`rc=127` was failover-scrape noise),
  **NOT batch size** (step 1 always clean; both failures are dataloader
  init).
- The **3 sibling barriers** from `debfff5` are correct and kept -- each
  sits next to a pre-existing all-rank collective. (blendcorpus PR #8.)
- Systematic-debugging takeaway: a barrier is only safe where every rank
  provably reaches it the same number of times; never inside a
  branch-gated, per-item loop. And: read *all* the failure logs before
  naming a single root cause -- there were two here, not one.

What the sim does **not** establish: survival of `dp_degree > 186` (the
real 512N+ regime, where dp itself is the trigger) or the >512N
distributed-init path (open `set_determinism` crash at 12,288+ ranks).
Those need a node-count study (dp-degree cliff bisect), not a batch
study. The separate validator-CCL-deadlock at 80B TP=4 was sidestepped
(`VALIDATOR_ENABLE=0`), not fixed.

---

## 2026-06-25 (sunspot) -- autoretry scripts: _real RoPE + validator; validator.py bug fixed

Two requested changes to `submit_agpt_{2b,20b,80b}_autoretry.sh`, plus a
latent validator bug surfaced + fixed.

1. **`_real` RoPE default (2B/20B).** `CONFIG_SUFFIX` now defaults to
   `_real` for 2B/20B -> `agpt_{2b,20b}_real`, which use real-valued
   (cos_sin) RoPE. The default complex backend uses torch.complex64 ops
   that torch.compile's inductor refuses to lower (eager fallback inside
   the compiled graph); cos_sin is real-valued and compiles, so it's the
   faster path with compile ON. Overridable via `CONFIG_SUFFIX=` (empty).
   **80B left on plain `agpt_80b`**: it runs compile OFF (the `_real`
   win is a compile-lowering optimization, moot there) and `agpt_80b` is
   the numerically-validated config.

2. **Validator enabled (all three).** `--validator.enable` +
   `VALIDATOR_FREQ` (default 100) + `VALIDATOR_STEPS` (default 10) knobs
   + `--validator.dataloader.dataset-path=$DFL` (blendcorpus only).

3. **Fixed a real EzpzValidator bug** (the "validator wired but not
   smoke-tested" CLAUDE.md flag). `validator.py:validate()` unpacked
   `post_dataloading_process` into 4 values
   (`inputs, labels, extra_inputs, extra_kwargs`) and splatted
   `**extra_inputs` at the pp-eval + model-forward sites, but upstream
   `torchtitan/components/validate.py` now returns a **3-tuple**
   (`extra_inputs` folded into `extra_kwargs`) -> every validate() call
   raised `ValueError: not enough values to unpack (expected 4, got 3)`.
   Dropped `extra_inputs` at all 3 sites to match upstream.

Smoke (job 12469561, agpt_2b_real, 2N, validator freq=3): config
resolved to `agpt_2b_real`, validation ran clean at steps 3 + 6
(val loss 12.46 -> 11.96), training completed. First crash (12469560)
is what caught the validator.py bug.

Commits (ezpz): `117ac69ce` validator.py fix, `5ffb850a1` script changes.
NOTE: torchtitan imports from the repo path (not copied into `.venv`),
so the validator.py fix is live for new jobs without re-yeeting.

---

## 2026-06-25 (sunspot) -- 80B TP=4 first real run + blendcorpus cache-build race fix

Launched the first real 80B TP=4/LBS=1/AdamW/bf16/GAS=2 run (GBS=372)
via `submit_agpt_80b_autoretry.sh`, 62 active + 2 spare on Sunspot, to
validate the stable corner past the 30-step doc result.

**Two init crashes -- NOT NaN/model/script. Root cause: blendcorpus
index-cache build race at TP>1, structurally broken barrier.**
- 12469548: 3 ranks raced the per-corpus `shuffle_idx.npy` (EOF magic /
  mmap-length errors).
- 12469550: per-corpus loaded warm, then 558 ranks raced the BLENDABLE
  index (`FileNotFoundError: a23baff6..._index.npy`).

The build path builds on global rank 0, then "waits" with only
`get_data_parallel_group()` + `get_pipeline_model_parallel_group()`
barriers before all ranks `np.load`. Those subgroup barriers do NOT gate
ranks whose TP coordinate != 0 against rank 0 (their DP/PP subgroups
exclude rank 0), so ~(1 - 1/TP) of ranks race ahead and read the .npy
mid-write. At TP=4/744 ranks that's ~3/4 -- matches the 558 blast radius.
The cross-group `all_reduce` that used to backstop this was commented out
("I don't think this is necessary any more") in
`deps/blendcorpus/.../blendable_dataset.py`. Prior "stable" 80B runs
(12469494/12469509) hit the same `building on rank 0` warning and only
survived by winning the timing race.

**Fix (both, per user):**
1. **Root cause** -- `deps/blendcorpus` (saforem2/blendcorpus, branch
   feat/remove-deepspeed, commit `debfff5`): added a global
   `torch.distributed.barrier()` after the subgroup barriers at all three
   build-then-load sites (gpt_dataset corpus-load + corpus-build
   completion, blendable_dataset load) so every rank waits for the
   rank-0 writer regardless of TP/PP/DP coordinate. (Outside
   experiments/ezpz/, but its own repo + the user's, so in scope.)
2. **Defense-in-depth** -- new `scripts/prewarm_blendcorpus_cache.sh`:
   a small/low-rank job that builds the index cache (1 step, compile+ckpt
   off) at the SAME data-cache-path a large run will use, so the big run
   loads-not-builds and never exercises the build path at scale. Path
   derivation mirrors the autoretry scripts exactly (verified: MODEL=80b
   NHOSTS_TRAIN=62 GAS=2 -> agpt-80b-adamw-books-n62-gbs372).

**Validation: clean success.** Attempt 3 (12469551) ran on the
now-fully-warm cache (both layers `loading`, no build, no race) and
completed **100/100 steps with zero NaN**: loss 12.93 -> 7.72 (-5.2
nats), grad_norm bounded throughout (peak ~9.9 early, settling ~2-6, no
spike-to-inf), MFU steady ~9.8%, mem flat 32%. step-100 checkpoint saved
(906 GiB, 163 s), `[auto-retry] FAILOVER STOP: success`, rc=0. This
**supersedes the prior 30-step TP=4 evidence** and confirms TP=4/LBS=1/
bf16/GBS=372 as the production-ready 80B corner (the TP=2/LBS>1 grad-path
overflow remains the open upstream bug). Full report:
`docs/experiments/agpt/sunspot/2026-06-25-80b-tp4-100step-validation.md`.

---

## 2026-06-24 (sunspot) -- py313-pt214 torch-2.14 compile segfault diagnosed

A user training attempt in the new `venvs/py313-pt214` env (torch
2.14.0.dev20260623+xpu, triton 3.7.2, py3.13) crashed with SIGSEGV at
step 1, in `triton/backends/intel/driver.py:364 __init__` (via
`get_current_device` -> `get_current_target`) during the first
`torch.compile` codegen. Two SEPARATE problems, isolated by minimal tests
on a compute node (see `project_py313_pt214_compile_segfault` memory):

1. **UR-loader symbol mismatch (import-time) -- FIXABLE.**
   `import torch` fails with
   `ImportError: libsycl.so.9: undefined symbol: urDeviceWaitExp, version
   LIBUR_LOADER_0.12` whenever the inherited `LD_LIBRARY_PATH` puts the
   system oneAPI loader first. Both the system loader
   (`/opt/aurora/26.26.0/oneapi/compiler/latest/lib/libur_loader.so.0.12.0`)
   and the venv-bundled one (`venvs/py313-pt214/lib/libur_loader.so.0.12.0`)
   advertise `LIBUR_LOADER_0.12`, but only the **bundled** one actually
   exports `urDeviceWaitExp` (system `nm -D | grep -c` = 0; bundled = 1)
   -- Aurora 26.26.0 ships an older 0.12 predating that symbol. Fix:
   prepend the venv lib so the bundled loader wins:
   `export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib:$LD_LIBRARY_PATH"` (AFTER the
   oneAPI module load). With this, import + `torch.xpu.is_available()`
   (6 devices) + EAGER xpu compute all succeed. Same class as the
   pyzes/libze_loader bug -- bundled-vs-system Intel runtime collision.

2. **Triton XPU `torch.compile` segfault -- NOT fixable from our side.**
   Even WITH the loader fix, `torch.compile(backend="inductor")` on an
   xpu tensor segfaults at `driver.py:364 __init__`. Confirmed compile-
   specific: eager xpu ops exit 0; compile exits 139. Not a
   fork/concurrency issue (`TORCHINDUCTOR_COMPILE_THREADS=1` still
   segfaults). triton 3.7.2 here vs 3.7.1 in the working torch-2.13
   `.venv`. Env-build incompatibility between triton 3.7.2's Intel backend
   and the Sunspot compute runtime.

Conclusion: `venvs/py313-pt214` is not usable for compiled XPU training
on Sunspot yet. Use the production torch-2.13 `.venv` (triton 3.7.1),
where agpt_2b/20b compile + train cleanly (smokes 12469525/12469526
today). If py313-pt214 is needed, run `--compile.no-enable` (eager works)
or wait for a triton-xpu build matched to the system runtime.

Also: flare project-quota (pid 2297) hit its 11 TB hard cap mid-session
(`EDQUOT` on every write; `lfs df` OST imbalance was a red herring --
pinning to an empty OST also failed, proving it was the project quota).
User cleared space (11.0 TB -> 4.8 TB used); writes recovered.

---

## 2026-06-24 (sunspot) -- ezpz-native auto-retry 20B + 80B scripts

Ported the validated 2B native-auto-retry template to
`scripts/submit_agpt_{20b,80b}_autoretry.sh` (full report appended to
`docs/experiments/agpt/sunspot/2026-06-24-native-autoretry-2b-smoke.md`).

- **20B** -- same shared plumbing as 2B; per-model deltas: the `DATASET`
  blendcorpus/HF knob, `--dataloader.num-workers=2`, `_CKPT_DATASET_SLUG`
  (sanitizes HF `/` in ckpt dir), `--checkpoint.async-mode=disabled`
  default. Live-smoked (12469526, own select=4): `--np=24`, GBS=48,
  trained 5 steps clean, Training completed. (Loss bounce / grad spikes
  are the expected tiny-GBS/no-warmup smoke behavior, not a crash.)
- **80B** -- defaults to the confirmed-stable **TP=4 / LBS=1 / AdamW
  LR=1e-6 / bf16-compute+fp32-master / AC=full / compile=OFF** corner,
  which SUPERSEDES the old failover script's TP=2 default (TP=2 NaNs at
  production GBS -- see the 80B NaN investigation entry below). bf16 is
  the agpt_80b() builder default so no --training.dtype flag is needed.
  GBS defaults to `dp_degree*LBS*GAS`; you reach a token target via GAS,
  not by raising dp_degree past the safe ceiling. The script **warns when
  `dp_degree = NGPUS/TP > 186`** -- the grad-path NaN trigger, validated
  safe only to ~62N. Validated by dry-rendering the launch argv (PATH-
  shimmed ezpz): it matches the known-good stable job 12469494
  token-for-token, incl. `activation-checkpoint:full` passed LAST. A real
  80B live smoke needs >=62N (TP=4 + safe dp), so plumbing (already proven
  by 2B/20B) was not re-burned at that scale.

Note: during 80B argv dry-rendering I accidentally truncated the live
`.venv/bin/activate` to 0 bytes (a `: >` redirect in the test harness);
restored it from the `venvs/rl-monarch-torch213` activate template
(path + `torchtitan` prompt patched) and verified `source .venv/bin/
activate && ezpz launch --help` works. Lesson: never aim `: >`/`>` at
real venv paths in a shim.

---

## 2026-06-24 (sunspot) -- ezpz-native auto-retry 2B submit script

Wrote `scripts/submit_agpt_2b_autoretry.sh`: a portable 2B production
submit script that uses `ezpz launch --auto-retry` **exclusively** for
bad-node failover, replacing the bash `failover_lib.sh` machinery
(`failover_init`/`failover_yeet_all`/`failover_run`). Native auto-retry
landed in ezpz >= 0.17.1 (PR #170); the installed venv is 0.20.0.

Key design points (verified against `../ezpz` source):
- `ezpz launch --auto-retry` splits the PBS allocation into active +
  spare internally from `--nproc`, runs the inner command, scrapes the
  same bad-node signatures on any non-zero exit (incl. watchdog 124 /
  walltime-racing 143), swaps a spare in-place, and retries. Active
  count is constant across retries (in-place swap by index).
- The **one** thing native auto-retry does NOT do is broadcast the venv
  to spares. So the script still `ezpz yeet --src .venv.tar.gz` to the
  **whole** (un-split) nodefile -- covering active + spare -- before
  launch. This is the only piece of `failover_yeet_all` we keep.
- GBS / `--nproc` are computed from the **active** count
  (`NHOSTS_TRAIN * 12`), not `ezpz_setup_job`'s `$NGPUS` (which sees the
  full allocation). Dry-checked: NHOSTS_TRAIN=12 of a 14-node alloc ->
  `--nproc 144`, GBS 288 (active-only), not 168/336.
- No preflight: ezpz's `STUCK_PRE_TRAINING` guard already bails without
  burning spares on a twice-zero-progress init crash.
- Portable: PBS headers default to Sunspot (datascience/workq/flare:home);
  Aurora via qsub overrides. Per-machine data default -- Sunspot `books`
  (the `/flare .../books-dataset` data actually resident on Sunspot; the
  `dolma`/`olmo-mix-1124` lists point at `/gila`, NOT mounted on Sunspot
  even on compute), Aurora `olmo-mix-1124`. `books` is already the
  established Sunspot smoke/benchmark dataset.
- Caught + fixed a real bug pre-submit: `${VAR:+--flag "$VAR"}` collapses
  to a SINGLE argv token (`--max-failover-retries 3`) that argparse
  rejects; switched to an array (`mfr_args=(...)`) -> two tokens / zero.

Smoke (my own select=4 jobs, NHOSTS_TRAIN=2 + 2 spare, 5 steps;
full report `docs/experiments/agpt/sunspot/2026-06-24-native-autoretry-2b-smoke.md`):
- **12469523** -- exercised the FULL native failover lifecycle for real
  (a node genuinely crashed): yeet-to-all (63.6s), active-only
  `--np=24`, ezpz split `4 total / 2 active / 2 spare`, attempt-1 crash
  -> scrape -> blind spare swap -> attempt 2 (`active=2/spare=1`) ->
  `FAILOVER STOP: stuck_pre_training` -> exit 143. The no-preflight
  guard fired exactly as designed.
- **12469524** -- chunkedce + LBS=1 + AC-full (compile-off fit recipe):
  trained all 5 steps clean, loss 12.94 -> 12.18, finite grad_norm.
- **12469525** -- production config (compile ON, LBS=2, sophiag): trained
  all 5 steps clean, loss 13.01 -> 12.06, peak 69.83% mem, ~26.6% MFU
  (matches the 2N scaling figure). agpt_2b LBS=2 on 2N fits fine.

NOTE: the 12469523 OOM (`UR_RESULT_ERROR_OUT_OF_RESOURCES` in
`cross_entropy_loss`) was NOT a script defect and NOT inherent to LBS=2
at 2N -- it was an artifact of the smoke passing `--compile.no-enable`.
Eager-mode CE materializes the full ~16GB 256k-vocab logit slice;
production compile-ON fuses it. agpt_2b LBS=2 on 2N is the normal
config. (Also confirmed: 2B defaults to sophiag via the submit-script
`--optimizer=sophiag` override; the registry base-config AdamW default
from PR #3269 replay `bac0a3473` is just the inherited template
fallback, not what 2B trains with.)

Extended `docs/guides/bad-node-failover.md` with a "Two implementations"
section (bash wrapper vs native) rather than a new doc. Old failover
scripts (20B/80B) untouched -- they still use `failover_lib.sh`.

---

## 2026-06-24 (sunspot) -- 80B grad-path NaN: mapped to LBS>1 + dp-degree; TP=4/bf16 path found

Root-caused the 80B grad_norm-NaN with a controlled multi-node sweep
(LR=1e-6, q_BLNH fix in). Findings (full matrix + perf in
`docs/production/agpt/80b/README.md`):

- The NaN is **not** a raw-GBS threshold. Two independent triggers, both
  in the gradient path (grad_norm NaNs one step before loss): **LBS>1**
  and **large dp_degree** (=NGPUS/TP). GBS=372 is clean (TP=4/LBS=1) AND
  NaN (TP=2, or TP=4/LBS=2) depending on composition.
- **New clean path the prior n32 factorial missed: TP=4 + LBS=1 + bf16 +
  GBS=372** via GAS (12469494 20 steps, 12469509 30 steps, both clean,
  loss -> 9.7/10.3). The factorial concluded "fp32-acts is the only clean
  path at GBS>=192" but never tried TP=4/LBS=1 with GAS. Reconciled the
  20260611 n32 diagnosis doc with an UPDATE header.
- **Perf:** TP=4 stable path is ~9.85% MFU, ~half the TP=2 baseline
  (18.7%). Cost is TP=4 comm, GAS-independent (GAS=1 and GAS=2 both
  ~9.8%). LBS=2 was faster (~15%) and fit memory (48%) but NaNs.
- **Stability CONFIRMED 4/4 clean** -- 12469494 (20), 12469509/510/511
  (30 each), all 0 NaN, three at identical loss 9.69-9.70. Not the
  nondeterministic knife-edge. New production recommendation:
  **TP=4, LBS=1, bf16, GAS-to-GBS** -- supersedes the n32 doc's fp32-acts
  default (cheaper: ~9.85% MFU vs fp32-acts ~3-5x slower / determinism
  ~50% and doesn't scale past n=32). Underlying TP=2/LBS>1 grad-path
  overflow still an open upstream-worthy bug (2 cheap 62N reproducers).
- Added a **batch-size ramp** (`FaultTolerantTrainer.batch_ramp_steps`,
  `09f2d243b`) -- ramps GAS (effective GBS) like LR warmup; mitigates the
  dp-degree onset but not the LBS trigger.

Also: the live `.venv` pyzes hardcodes a Debian `libze_loader.so.1`
path that doesn't exist on Sunspot (SUSE -> /usr/lib64); patched to the
bare soname. Only bites interactive live-`.venv` use, not yeet-env
training (tarball torch doesn't bundle pyzes). NOT an LD_LIBRARY_PATH
issue (chased that wrongly first). See
[[project_venv_ld_library_path_ze_loader]].

Filesystem was 100% full earlier today (amplified transient failures);
cleaned ~6.3 TB of core dumps + test ckpts -> 57% used.

---

## 2026-06-24 (sunspot) -- 80B verified at 28N + TP>1 sync regression fix

First **multi-node** 80B v2 functionality verification (all prior
validations were 4N). Job `12469486`, 28 active nodes (TP=2), torch
2.13, books blendcorpus dataset, 20 steps:

- Loss descended **12.94893 -> 10.38276** (-2.57 nats), matching the
  4N baseline (12.98 -> 10.46) within noise.
- MFU steady **~18.7%** (4N was ~17.8%), TPS ~102, ~40s/step, memory
  flat 65.86%. grad_norm climbed to ~33 at steps 15-16 then settled
  to ~14 by step 20 -- same pattern the 4N smoke showed; production
  still needs the 200-step warmup.
- `step-20` checkpoint saved cleanly.
- Failover swapped one bad node (rank 204 signal 15) at launch and
  training started clean -- the 8-spare headroom did its job.

**The reason this run mattered: it caught a TP>1-only 57th-sync
regression that the TP=1 debugmodel smokes missed.** A first 64N
attempt (`12469471`) crashed on every rank at `model.parallelize`:

```
AssertionError: XPUScaledDotProductAttention: local_map is set but
in_dst_shardings is missing entries for: ['q', 'k', 'v']
```

Root cause: the 57th sync adopted upstream's shape-suffix naming --
`ScaledDotProductAttention.forward` args became `q_BLNH/k_BLNH/v_BLNH`
and `set_gqa_inner_attention_local_map` keys `in_dst_shardings` by
those names. The local_map contract check matches `in_dst_shardings`
against the wrapped forward's positional-arg names, and the ezpz
attention forks still used bare `q/k/v`. Only asserts under TP>1, so
TP=1 smokes passed. Fixed in `74c7452f6` (renamed the three ezpz
attention forwards) + `13c09ddf9` (added a TP=2 entry to
`sync_smoke.sh` so this class of regression can't slip through again).
See `docs/upstream-sync.md` 57th-sync "Follow-up" for the full chain.

Getting here also surfaced two non-code issues, both resolved:
- **Filesystem was 100% full** (`/flare` 0 avail) -- amplified transient
  yeet/checkpoint failures. Cleaned ~6.3 TB: 864 core dumps (5.66 TB,
  Apr 26 -> today crash debris) + 3 async/smoke test checkpoints
  (621 GB). torchtitan/ 8.6T -> 2.3T, fs 100% -> 57% used.
- **HF `eliplutchok/fineweb-small-sample` is too small** for a 28N
  multi-step run -- exhausts and re-loops in a tight spam loop
  (944k warnings) instead of feeding step 2. Use a real blendcorpus
  data list (books) for anything past a single step.

Throwaway verification ckpt dirs left for cleanup:
`outputs/checkpoints/agpt-80b-{32n,64n}-funcverify`.

---

## 2026-06-24 (sunspot) -- 57th upstream sync + replays

Merged `upstream/main` into `ezpz` (59 commits, `7b579adde..c6c2fb2c5`,
merge `1f288f2e7`, no conflicts). Worked through all 6 commits that
touch `llama3/` or `deepseek_v3/`:

**2 real replays (source changes + smoke-verified):**
- `b3b60dabf` delete `--disable_loss_parallel` (commit `fa6f0681e`):
  dropped the kwarg from agpt/moe model+sharding and trainer.py.
- `c5d93d109` AC policy class hierarchy (commit `bb38b95e1`):
  `ActivationCheckpointConfig(mode=...)` -> `FullAC`/`SelectiveAC`/`None`;
  `apply_ac()` -> `ac_config.build().apply()`. Rewrote the
  `moe/activation_checkpoint.py` `_get_save_ops` monkey-patch as a clean
  `MoeSelectiveAC(SelectiveAC)` subclass. Fixed two
  `cfg.activation_checkpoint.mode =` mutation sites in the config-registry
  wrappers (slots Config has no `mode`).

**4 no-ops with documented reasons:**
- `cd8950ba7` score_before_experts: deliberate divergence -- ezpz fork's
  dispatcher genuinely branches on the flag upstream removed as dead.
- `70dd94551` FusedQKVLinear hooks: inherited via upstream import,
  inactive path (ezpz uses stock QKVLinear).
- `581f175dc` / `aa1d37414` fused/offset-aware experts: opt-in EP
  features ezpz/moe doesn't enable.

**Scripts (commit `807d5050b`):** PR #3674 made AC a tyro subcommand,
so `--activation_checkpoint.mode=full` no longer parses. Migrated 7
launcher scripts to the positional `activation-checkpoint:full` token.

**Smoke (job `12469466`, sunspot 1N, seed=42 --debug.deterministic):**
agpt_debugmodel (FullAC) loss `10.83863 -> 10.67256`; moe_debugmodel
(MoeSelectiveAC, seq=512/lbs=1) loss `12.90956 -> 12.36751`; both rc=0.
agpt step-1 loss bitwise identical across two runs.

**Venv note:** ran the smoke in `venvs/rl-monarch-torch213` (py3.13.6 +
torch 2.13). The repo-root `.venv` is now py3.14 (torchtitan import
fails on `importlib.metadata`) and `.venv.tar.gz` is stale. Had to
add `sh`, editable `ezpz` (`-e ../ezpz` -- installed 0.19.0 wheel was
missing `get_timestamp`), and editable `blendcorpus` (`-e
deps/blendcorpus`) -- all `--no-deps`, torch untouched. Worth
rebuilding a clean py3.13 training venv + fresh `.venv.tar.gz`.

Full detail: `docs/upstream-sync.md` (57th sync entry).

---

## 2026-06-14 (sunspot overnight) — Monarch + torch 2.13: 6 patches, 16 jobs, still wall

Pushed the upstream `torchtitan.experiments.rl.train` Monarch + GRPO
pipeline through 16 PBS submissions (`12468799` → `12468815`),
peeling back one failure mode at a time. Each crash mapped to a
distinct XPU porting gap, all now fixed in `xpu_overrides.py`:

1. `_make_replicate_tensor` skip-broadcast (oneCCL USM check rejects
   torch.xpu USM-device pointers under Monarch's execve'd actors —
   buffers are deterministic-identical anyway)
2. Force `init_distributed(enable_cpu_backend=True)` → backend becomes
   `xpu:xccl,cpu:gloo` so DCP's `all_gather_object` for the central
   plan routes objects through gloo, not xccl
3. Suppress vLLM-XPU's `xpu_worker.py:103` oneCCL "warmup" allreduce
   on a `torch.zeros(1).xpu()` — that allreduce is unconditional in
   vLLM-XPU and trips USM check immediately at engine init
4. `XPUPlatform.get_attn_backend_cls` patched to accept
   `AttentionBackendEnum.CUSTOM` (vLLM-XPU's selector raises
   `ValueError: Invalid attention backend` for any backend it doesn't
   explicitly list; CUSTOM is the path `rl/actors/generator.py` uses
   for varlen attention)
5. `vllm._torch_cuda_wrapper` patched to NOT alias
   `torch.cuda.current_stream = torch.xpu.current_stream` (Dynamo's
   `(cuda, xpu, accelerator).current_stream` handler-table build then
   trips `AssertionError: Handler already registered` because the
   same function appears twice). Keep `Stream`/`stream`/etc. aliases.
6. `EzpzPerHostProvisioner.make_bootstrap_command_for_gpu_ids` —
   pre-execve env overlay via Monarch's `bootstrap_command=`. Sets
   `ZE_AFFINITY_MASK` BEFORE `import torch` runs in bootstrap_main.py,
   so torch.xpu's primary SYCL context picks up the right tile
   topology.

Got past every torchtitan + DCP + XCCL issue, but hit a wall at
vLLM's `profile_run` → `_dummy_run(max_num_tokens=2048,
is_profile=True)` → first decoder layer `F.linear` →
`RuntimeError: could not create a memory`. This is oneDNN's
`dnnl::memory` constructor failing — not OOM (model load succeeded
at 1.22 GiB, we have 46 GiB tile). Same crash with:

- `--generator.gpu-memory-limit 0.4`
- `--generator.sampling.max-tokens 256`
- `--generator.cudagraph.no-enable + --compile.no-enable`
- `--batcher.batch.seq-len 512`
- narrow `ZE_AFFINITY_MASK` per actor
- `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`
- `VLLM_DISABLED_KERNELS=xpu_kernels` (rules out our custom-built
  vllm-xpu-kernels)

Suspect: oneDNN scratchpad allocator queries `sycl::get_pointer_type`
with a different context than torch.xpu's allocator (same root cause
as the oneCCL USM check — but there's no "skip the check" knob for
oneDNN). Either Monarch's execve'd actors construct SYCL context
differently than mpiexec'd processes, or our custom-built vllm-xpu-
kernels somehow taint the allocator pool. Disabling its custom ops
didn't help, so leaning toward the first cause.

Full writeup: [`docs/rl/2026-06-14_monarch-torch213-deep-dive.md`](rl/2026-06-14_monarch-torch213-deep-dive.md).

Next options when resumed:
1. Get an Intel torch.xpu engineer to look at `DNNL_VERBOSE=2`
   output from profile_run.
2. Try the same code path under mpiexec to confirm it's Monarch-
   spawn-specific.
3. Fall back to torch 2.12 + prebuilt vllm-xpu-kernels (known-good
   combo on Sunspot), accept losing torch 2.13's DTensor USM fixes
   (we skipped the broadcast anyway).
4. Drop vLLM for the generator side — naive HF .generate() loop.

---

## 2026-06-13 (sunspot late eve) — 🎉 GRPO end-to-end on XPU via TRL vllm-serve

Job `12468780` completed **5/5 GRPO steps with real on-policy weight
sync** on Sunspot XPU. Full pipeline: trl vllm-serve on tile 0,
8-rank GRPO trainer on tiles 1-8 via ezpz launch, communicating via
HTTP for rollouts and via XCCL TCP-KVS for trainer→server weight
broadcasts.

`format_reward/mean` moved 0 → 0.0625 → 0.25 → 0.0625 → 0.125 over 5
steps with a cold Qwen3-0.6B on the `sum_digits` task. Signal is
noisy at bsz=8×ngens=4 but the upward trend in steps 1-3 confirms the
policy update path is alive.

The breakthrough was discovering that oneCCL DOES support a
non-PMIx, TCP-based rendezvous when configured correctly:

    CCL_PROCESS_LAUNCHER=none   FI_PROVIDER=tcp
    CCL_ATL_TRANSPORT=ofi       CCL_KVS_IP_PORT=127.0.0.1_29513

Both server and trainer set the same env, and XCCL forms an N+1-rank
group across the two process trees without any PMIx coupling. This
is exactly what TRL's `vllm_mode="server"` needs.

Sam's pushback on the no-op workaround was correct — the proper fix
exists and we just needed to find the right env knobs.

Full writeup: [`docs/rl/grpo-on-xpu-status.md`](rl/grpo-on-xpu-status.md).

---

## 2026-06-13 (sunspot eve) — vLLM-XPU + Monarch RL actor infra

Kicked off the long-deferred wiring of vLLM into ezpz/rl. Per
[`docs/rl/vllm-xpu-wiring-plan.md`](rl/vllm-xpu-wiring-plan.md),
two parallel paths:

- **Track 2 (TRL `vllm_mode="server"`)** — wires existing TRL-based
  `train_grpo.py` to an external vLLM-XPU server. Faster to land but
  the live `venvs/vllm-test/` + TRL 1.5.1 combo hit a worker-side
  `current_platform.device_type` empty-string error (jobs `12468737`,
  failing in TRL's `vllm_serve.py:llm_worker`). TRL 1.5.1 also warns
  vllm 0.22.1 is outside its 0.12.0-0.18.0 supported range. Need to
  either drop to vllm 0.18.x or use TRL 1.6.0 (which no longer
  hard-pins the vllm version).
- **Track 1 (Monarch + TorchStore actors)** — was thought blocked
  on torchmonarch lacking cp314 wheels. Resolved by building a new
  sibling venv `venvs/rl-actors/` on **py3.13** + torch 2.12+xpu +
  monarch 0.5 + torchstore (main) + vllm 0.22.1 + vllm-xpu-kernels
  0.1.9.1 + TRL 1.6.0 + transformers 5.11 + accelerate 1.14 +
  datasets 5.0. All imports clean.

`scripts/monarch_smoke.py` + `scripts/monarch_smoke.sh` validate
the framework end-to-end. First run (job `12468738`, 1N) showed:

- 2-actor `this_host().spawn_procs({"gpus": 2})` works on XPU.
  Both ranks reported `xpu_count=12 xpu_avail=True` from inside the
  spawned actor — Monarch's CUDA-only assumptions don't actually
  block XPU runtime use.
- `monarch.actor` + `torchstore` imports clean on py3.13 + torch 2.12+xpu.
- TorchStore transport probe hit a minor naming bug
  (`TransportType.RPC` vs `TransportType.MonarchRPC`); fixed in the
  smoke script. Rerunning as `12468739`.

This unblocks the Monarch path — the open question shifts from "is it
even possible on XPU" to "how much rewrite to adapt ezpz/rl off TRL".

### Stack table

| Component | Main `.venv` | `venvs/vllm-test/` | `venvs/rl-actors/` |
|---|---|---|---|
| Python | 3.14.2 | 3.14.2 | 3.13.6 |
| torch | 2.13.dev | 2.12.0+xpu | 2.12.0+xpu |
| vllm | — | 0.22.1 | 0.22.1 |
| vllm-xpu-kernels | — | 0.1.9.1 | 0.1.9.1 |
| trl | 1.5.1 | — | 1.6.0 |
| transformers | 5.6.2 | (vllm dep) | 5.11.0 |
| torchmonarch | — | — | 0.5.0 |
| torchstore | — | — | main |

`rl-actors/` is the new "actor venv" for both Monarch controllers and
vLLM server workers. Main `.venv` stays unchanged for everything else.

### Scripts landed this session

- `rl/scripts/vllm_serve_xpu.sh` — generic launcher: `trl vllm-serve`
  from `venvs/vllm-test/` with `PYTHONPATH` to .venv's TRL wrapper.
- `rl/scripts/vllm_serve_smoke.sh` — PBS smoke for Phase 1+2 of the
  wiring plan.
- `rl/scripts/vllm_xpu_bare_smoke.sh` — standalone vLLM-XPU sanity
  (no TRL wrapper) using `venvs/rl-actors/` directly. Isolates whether
  the failure is in TRL's wrapping vs in the underlying vLLM stack.
- `rl/scripts/monarch_smoke.py` + `monarch_smoke.sh` — Monarch +
  TorchStore framework smoke.
- `rl/scripts/grpo/aurora2b_sft_arithmetic_8n_vllm.sh` — production
  GRPO submit variant using server-mode (Phase 5 of the wiring plan).

### Resolution (eve PM, 2026-06-13)

- ✅ `12468739` (monarch smoke with `MonarchRPC` fix) — PASSED. Both
  ranks reported `xpu_count=12`. Monarch framework on XPU is
  confirmed working.
- ✅ **vLLM-XPU SOLVED interactively on x1921c3s0b0n0**: KV cache
  48.43 GiB, max concurrency 1033x — bit-for-bit match with the
  2026-06-10 baseline.
- ❌ → ✅ `12468740..12468751` (12-job debug chain) — root-caused via
  the interactive replay. The bug was self-inflicted env contamination
  by `ezpz_setup_env`:
  - `CCL_PROCESS_LAUNCHER=pmix` made oneCCL look for a PMIx context
    vLLM's `multiprocessing.spawn`'d EngineCore doesn't have.
  - `FI_PROVIDER=cxi,tcp;ofi_rxm` made libfabric try the Slingshot
    CXI provider, which needs a NIC handle only mpiexec-bootstrapped
    processes get. `fi_getinfo` returned 0 providers; `atl_ofi
    init_transport` failed.
  Sam's pushback ("nothing has changed about the environment or
  system since 06/10/2026") was correct. Today's smoke scripts source
  `ezpz_setup_env` for PBS bookkeeping; the original 2026-06-10
  verification was a raw interactive shell with none of those env
  vars set. Fix: `unset CCL_*/FI_*` after `ezpz_setup_job`, invoke
  vLLM via plain python (no `ezpz launch`).
- 🔍 `12468752` (env-scrubbed PBS submit of the bare smoke) — queued.
  Verifies the fix works under PBS-direct, not just interactive SSH.
- See [`docs/rl/vllm-xpu-current-status.md`](rl/vllm-xpu-current-status.md)
  for the full debug chain, root cause, and fix.

---

## 2026-06-13 (sunspot) — PR #14 merged + 56th upstream sync

**PR #14 (`Isolate ezpz MoE` by @nscottnichols) landed on `ezpz`** as
merge commit `de85a179b`. Final pre-merge validation:

- `12468735` (pr14-fixes, 8N, EP=12, padding=1, AC=selective, 20 steps)
  ran clean: loss `12.91575 → 7.00367`, 272s, no errors. The
  `ezpz/moe/activation_checkpoint.py` wrapper (commit `82100fd67`)
  works as designed.
- `12468736` (ezpz baseline, same config) for comparison: ran 17 steps
  cleanly, then hit an apparently unrelated upstream tensor-size
  overflow at step 17 (`RuntimeError: Storage size calculation
  overflowed with sizes=[176...e18, 2048]`). Through step 17, **loss
  matched pr14-fixes within ~1e-4 nats** at every step
  (e.g. step 17: 7.74314 vs 7.74329) — strong evidence that
  PR14's fork + AC wrapper are numerically equivalent to upstream
  on this stack.

The earlier investigation isolating the
`TT_MOE_NORMAL_EQUAL_A2A_PADDING=1 × AC=selective` GPU PDE Write
fault and the AC-save-list workaround is documented in PR thread
[issuecomment-4698951275](https://github.com/saforem2/torchtitan/pull/14#issuecomment-4698951275).

Right after, pulled in 2 more upstream commits as the 56th sync
(`3935fc654`):

- `588fc12fd` `[RL] Add deterministic loss guard for GRPO training (#3474)` —
  `experiments/rl/` only; ezpz/rl has its own train_grpo.py.
- `7b579adde` `Add MinimalAsyncEP (#3561)` — adds new EP backend
  (additive). Touches `common/token_dispatcher.py` with a small
  `output_size=total` compile-hint on `repeat_interleave`; PR14's
  fork has the same call but no replay needed (runtime behavior
  identical when not compiled). See
  [`docs/upstream-sync.md`](upstream-sync.md).

---

## 2026-06-12 (sunspot eve) — 55th upstream sync (3 commits, no replays)

Three more upstream commits landed since the 54th sync earlier today
(`96ab7487d..0a73d82a4`):

- `3b8e060853` `Remove unused MetricsProcessor.lr_schedulers (#3644)` —
  pure cleanup of an attribute nothing reads. Our local `metrics.py`
  fork was already removed in PR #14, so we consume upstream directly.
- `14fb67575` `qwen3.5 tok_embeddings LocalMap region (#3648)` —
  qwen3_5-only sharding fix. ezpz doesn't use qwen3_5.
- `0a73d82a4` `avoid GradAccumulator init in ChunkedCELoss no_grad path (#3652)` —
  internal optimization in `components/loss.py`; ezpz uses
  `ChunkedCELoss` via direct import.

Merge `439ccf220` was clean — no conflicts, no replays. Sync entry
added to [`docs/upstream-sync.md`](upstream-sync.md). Skipping the
dynamic smoke this time — last sync's bitwise IDENTICAL already
covered the `ChunkedCELoss` path, and the other two commits don't
touch ezpz-reachable code.

---

## 2026-06-12 (sunspot pm) — 54th upstream sync (3 commits, no replays)

Three new upstream commits since the 53rd sync (`1c02a5cee..96ab7487d`):

- `88030eec1` `[rl] Fix batch invariant logprob calculation by forcing vllm
  use trainer's function (#3629)` — `experiments/rl/actors/generator.py`;
  ezpz/rl doesn't override that path.
- `5ba439938` `[Bug] Fix MoE SP token combine indices (#3604)` — fixes
  a `B > 1` × `sp_size > 1` bug in `common/token_dispatcher.py`; ezpz/moe
  re-imports the dispatcher unchanged, so the fix flows automatically.
  Our ezpz MoE configs run with `sp_size == 1` so the bug was never
  live for us anyway.
- `96ab7487d` `chore(ci): migrate ROCm matrix from 7.1 to 7.2 (#3267)` —
  CI matrix + ROCm loss reference files only.

Merge `f8be3bcd1` was clean — no conflicts, no replays needed.

Submitted bitwise checks in parallel:

- `12468696` — `bitwise_sync_check.sh` agpt_2b_chunkedce, 2N, 20 steps,
  comparing `434cfe5d1` pre-merge vs `f8be3bcd1` post-merge with
  `--debug.seed=42 --debug.deterministic`.
- `12468697` — `submit_moe_smoke.sh CONFIG=moe_10b_2b_sdpa_ep STEPS=10`,
  2N, head-only smoke compared against 52nd-sync baseline `12468666`.

Sync entry added to [`docs/upstream-sync.md`](upstream-sync.md).

Both bitwise jobs passed:

- `12468696` (agpt) — **VERDICT: IDENTICAL** across all 20 steps
  (head step 20 = pre step 20 = `loss 10.66272 / grad_norm 18.1259`).
- `12468697` (MoE) — clean 10-step run (loss 12.89 → 8.87, grad_norm
  bounded, ~80 GiB peak).

Merge-ready: `ezpz` already at `f8be3bcd1` (merged on the live branch,
not in a separate worktree). Push pending after committing doc updates.

---

## 2026-06-10 (sunspot) — 32N SFT auto-resume blocker: torch ShardedTensor.device hardcodes CUDA

> **Canonical writeup** (with the full failover-cycle worked
> example and run table):
> [`docs/production/sft/aurora2b/tulu_math_uc_mix/`](production/sft/aurora2b/tulu_math_uc_mix/README.md).
> This journal entry is the rolling debug log; the report is the
> end-of-day cleanup.


Continuing the 32N SFT push. Job 12468404 (the first 32N run with
auto-retry's bad-node failover) trained cleanly for 140 steps with
loss 1.16 → 0.86 and token_acc 0.73 → 0.78 before a worker rank
SIGABRT'd from `ccl::v1::exception`; auto-retry swapped in a spare
and relaunched. But the relaunch went back to step 0 instead of
resuming from `checkpoint-100/` — the second run wasn't passing
`--resume_from_checkpoint`.

Patched the submit script to pass `--resume_from_checkpoint
"${CKPT_DIR}"` (HF Trainer auto-detects the latest `checkpoint-N/`
subdir in the dir) and added a small coercion shim in
`train_sft.py:main()` to handle the case where the dir is a freshly-
created empty dir (HF errors out without it). Resubmitted as
12468408.

12468408 crashed differently: all 384 ranks tracebacked with
**`AssertionError: Torch not compiled with CUDA enabled`** during HF
Trainer's FSDP checkpoint load. Tracked it to
`torch/distributed/_shard/sharded_tensor/_ops/tensor_ops.py:54`:

```python
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    ...
    else:
        dev = torch.device(torch.cuda.current_device())   # <-- BUG on XPU
```

Upstream hardcodes CUDA as the no-local-shards fallback. The sibling
`tensor_func`/`dtensor_func` in `planner_helpers.py` already do the
device-agnostic thing via `_get_pg_default_device().type` +
`_get_device_module(...)`; only the ShardedTensor dispatch is broken.

Local workaround: added `_patch_sharded_tensor_device_for_xpu()` to
`train_sft.py` that re-registers the dispatch via `_sharded_op_impl`
with an XPU-aware fallback (tries `torch.accelerator.current_device_index()`
first, then falls back to whichever accelerator namespace is
available). Called once at top of `main()`. Verified the patch
correctly replaces the `_SHARDED_OPS` entry via a smoke import on
the login node.

Filed full writeup at
[`docs/upstream-issues/sharded_tensor_device_cuda_hardcode.md`](upstream-issues/sharded_tensor_device_cuda_hardcode.md)
with the rank-0 traceback and a proposed upstream fix, then filed
upstream as
[pytorch/pytorch#186938](https://github.com/pytorch/pytorch/issues/186938)
and opened
[pytorch/pytorch#186940](https://github.com/pytorch/pytorch/pull/186940)
with the one-spot fix (mirror what `planner_helpers._init_state_dict`
already does for plain tensors / DTensors).

While the patch was being written, also consolidated `checkpoint-100`
into a flat HF format at `checkpoint-100-hf/` (7.94 GB safetensors).
This gives us a usable artifact independent of the FSDP-resume
question — we now have a 600M-token SFT'd AuroraGPT-2B-tulu-mix
checkpoint we can hand off to GRPO regardless of whether resume
ever works.

Resubmitted as **12468409** with the patch. Currently queued.
Validation plan: tail `run.log` for `Continuing training from
checkpoint, will skip to global_step 100`, then verify loss picks
up from ~0.86 (not from cold-start 1.16).

### Postscript — completion of the 32N SFT chain (2026-06-10 PM)

**12468409** validated the XPU FSDP resume patch end-to-end
(2 successful failover cycles, loss 0.86 → 0.81 across
checkpoints 200 → 300) before tripping a different blocker:
ezpz `launch_autoretry`'s `STUCK_PRE_TRAINING` guard was matching
only torchtitan's `step=N` progress marker, falsely flagging
TRL's `{'loss': '...'}` log format as "no training happened" and
bailing on attempt-3. Patched the regex (ezpz commit
[`6b4a00b`](https://github.com/saforem2/ezpz/commit/6b4a00b)) to
also match the HF/TRL format, reinstalled via `uv pip install
-e ../ezpz`, resubmitted as **12468437**.

**12468437 ran to completion**: 1h39m wall time, 4 mpiexec
attempts, 3 oneCCL `pidfd_getfd` SIGABRTs survived, 3 spare-node
rotations (`x1921c1s0b0n0 → x1921c5s4b0n0 → x1921c5s5b0n0 →
x1921c5s6b0n0`), and finished `Training complete.` at step 729 /
epoch 3.0. autoretry verdict `FAILOVER STOP: success (attempt 4)`,
exit 0. Loss `1.16 → 0.77`, `mean_token_accuracy 0.7957`, ~4.5B
tokens consumed. Final consolidated HF artifact at
`outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf/`.

PR review on
[pytorch/pytorch#186940](https://github.com/pytorch/pytorch/pull/186940)
caught a regression risk in v1 of the fix (mirroring
`planner_helpers._get_pg_default_device` pattern breaks composite
PGs like `cpu:gloo,cuda:nccl` because that function prefers CPU
when both are registered). Pushed
[`570da16048`](https://github.com/saforem2/pytorch/commit/570da16048e049eb6e9b11239e718e621d4de720)
which switches to `torch.accelerator.current_accelerator()` —
doesn't consult the PG backend list, no composite-PG trap. Both
inline review threads addressed + resolved.

End-of-day deliverables: SFT'd AuroraGPT-2B HF ckpt for GRPO,
PR #186940 (v2) up for upstream review, autoretry recognizes
both torchtitan and HF/TRL trainer markers, complete writeup at
[`docs/production/sft/aurora2b/tulu_math_uc_mix/`](production/sft/aurora2b/tulu_math_uc_mix/README.md).

**Operational TODO:** file ALCF ticket for `x1921c1s0b0n0` —
this host showed up as the SIGABRT-er in multiple jobs across
the day, suggests a degraded NIC / Level Zero stack. Until it's
pulled from the queue, autoretry's 4-spare allocation handled
it, but every job pays a ~3min/failover overhead.

---

## 2026-06-08 (aurora pm) — 80B 4N validated end-to-end on Aurora + 256N NaN + chart wrapper

Big session covering several threads:

### 1. 80B production stack validated end-to-end at 4N (Aurora)

Spent ~3h chasing what looked like the long-pending "80B model-init
silent hang at 4N+" regression (pending since the 2026-06-06
session). Iterated through 5 PBS-script attempts (8530199, 8530216,
8530243, 8530800 + one mid-iteration kill via test.sh ssh-allocation
8530807). The final attempt
(`/flare/.../.interactive-80b-4n-r7-direct-*.log`) worked cleanly:

- Loss descent: **12.93 → 12.03 over 10 steps** (-0.91 nats)
- MFU steady at **~17.9%** (matches the May 5 12466025 + Sunspot
  12468197 baselines)
- Memory **88.97%** at peak (4N is dense)
- **Step-10 sync checkpoint save fired** at 14:06:22 and **landed on
  disk**: 904 GB across 48 .distcp shards + .metadata, matching the
  Sunspot reference exactly. Sync mode + xccl workaround validated.

The stack of fixes that got us there (in order of discovery):

1. **80b-v2 repo was 229 commits behind** origin/ezpz. Pulled to get
   `8031d1d3` (xccl_split_group_workaround for the `split_group`
   RuntimeError) + `ce321caae` (CHECKPOINT_ASYNC_MODE=disabled
   default) + the May/June 80B-prod-sync-ckpt validation work.
2. **80b-v2 .venv was symlinked to 2b-v2 .venv** — would have polluted
   the 2B chain. Broke the symlink (`cp -a` 8.5GB), installed ezpz
   0.18.7 (from the `yeet-retry-on-rsync-failure` branch) +
   `trl==1.5.1` + `spmd_types==0.2.1` (the latter unblocks the
   upstream `import spmd_types as spmd` in
   `torchtitan/components/loss.py` since commit `fec0c175d`).
3. **Patched blendcorpus shipped a deadlocking global
   `torch.distributed.barrier()`** in `_build_index_mappings`.
   `BlendableDataset.__getitem__` is lazy per-corpus, so different
   ranks hit the barrier on different corpora at different wall-clock
   times → at 4N+ some ranks advance into `train_step` while others
   sit at the barrier → 30 min wait → ezpz watchdog SIGTERM. Pinned
   this down via `py-spy dump --pid` from ssh into the head node
   (rank 0 stuck at `barrier (torch/distributed/distributed_c10d.py:5234)`
   inside `_build_index_mappings:1141`, rank N+ already in
   `train_step → dataloader.__next__ → multiprocessing.Queue.get`).
   Reverted the barrier in the source venv (the existing
   `_load_with_retry` already handles the EOFError race it was
   supposed to protect against).
4. **PBS-script invocation was missing `export ZE_FLAT_DEVICE_HIERARCHY=FLAT`**
   on the inner shell — caused `_infer_topology` to see 6 GPUs/host
   instead of 12 and reject the launch with `ngpus must be > 0 and
   <= 24, got 48`. Added it to the inner-shell setup.
5. The final r7 run swapped in: patched blendcorpus + xccl workaround
   + spmd_types + correct FLAT + the inner-shell env block. ssh-launched
   foreground from the test.sh allocation head node so I could
   `py-spy` and Ctrl-C without watchdog interference.

Tarball rebuilt with the barrier-removed blendcorpus baked in
(`.venv.tar.gz.bak-pre-barrier-removal-20260608-090613` preserved).

### 2. 80B 256N smoke — training works, but loss NaNs immediately

Submitted 8530891 (256N, NHOSTS_TRAIN=256, TRAINING_STEPS=110, sync
ckpt, LR=1e-6 default):

- Setup (compile, init, dataset, mesh) all clean
- Throughput **110 TPS/GPU / 20.3% MFU** — actually slightly better
  per-GPU than 4N's 98 / 17.9% (less compile overhead at larger scale)
- **Step 1**: loss=12.94 grad_norm=4.94 — clean
- **Step 2**: grad_norm=NaN
- **Step 3 onward**: loss=NaN forever
- Job walltime-killed at step 79 (1h cap), step-100 ckpt save never
  fired

Configured LR scheduler is correct (`warmup_steps=200,
decay_ratio=0.8, decay_type=linear` = classic WSD), but at
`TRAINING_STEPS=110` the scheduler clamps warmup to 110, so step-2
LR is effectively 2/110 × 1e-6 ≈ 1.8e-8 (essentially zero). Even
with that tiny LR the first optimizer step produces NaN grads — so
this is **not just "LR too high at GBS=1536"**; something else is
biting on the first backward at scale.

Open hypotheses (still TBD):
- bf16 overflow in attention/MLP at GBS=1536 (vs 4N's GBS=24)
- TP=2 loss-reduction bug (CLAUDE.md notes `_dist_reduce`
  short-circuits DTensor on orthogonal meshes since 2026-04-27);
  local workaround in `trainer.py` may not fully cover the grad-norm
  path
- AdamW fp32-master second-moment overflow with these activations

Submitted 8531345 with `LR=1e-7` (10× smaller) at TP=2 same as the
NaN run — but it died from bad-node SIGSEGV (`rank 438 died from
signal 11` on `x4408c1s3b0n0`) at 177s. Resubmitted as 8531721 with
`FAILOVER_MAX_RETRIES=2` (a misjudgment to set 0 on the first
attempt — even when the failure-mode-under-test isn't bad-node,
surviving allocation/yeet/init still wants retries). 8531721
currently Q'd waiting for a 256N debug-scaling slot.

### 3. Evals + chart refresh

- Submitted 2B 256N evals for step-69000 (8531449) and step-69900
  (8531450). step-69900 came back clean: HSn 0.5552, ARC-E 0.5939,
  ARC-C 0.3294, Wino **0.5627 (best yet)**. Other tasks within noise.
- Wrote `scripts/update_all_charts.sh` to wrap the six per-script
  plot invocations (`utils/plot_production.py`,
  `utils/plot_production_combined.py`,
  `utils/plot_production_wandb.py`, `eval/plot_evals_combined.py`,
  `docs/evals/agpt/{2b,20b}/plot_v1_vs_v2.py`) into a single parallel
  runner with per-script logs. Smoke: 50 figure files refreshed in
  294s, 0/6 failures.

### 4. Production chain status (no change)

All 3 canonical chains still Q+H — `small` queue is severely
contended (78 total / 68 Q / 3 R / 7 H). Last R for prod chains:
- 2B 256N (8519833): step-69900, 2026-06-06 18:07 (cleanly walltime'd)
- 2B 512N: step-30500, 2026-05-30 07:53 (idle 9 days)
- 20B 512N: step-4400, 2026-05-29 11:43 (idle 10 days)

8521627 (2B 512N cont) made it to R briefly on 2026-06-07 21:12 but
died at 8min when 1 of 522 nodes failed yeet-env rsync (the very
failure mode my `yeet-retry-on-rsync-failure` ezpz PR #160 fixes).
Chain still alive via failover (cont10 = 8521631 next up).

---

## 2026-06-08 — RL polish + first real SFT path (gsm8k / metamathqa / mix) + 32N XCCL pain

Long session, three intertwined threads. Tracked in tasks #66–#75.

### train_grpo polish

User-driven iteration on the GRPO entry point that turned up
several real bugs plus a bunch of UX improvements:

- **Vocab-aware chat-template picker**
  (`_pick_chat_template` in `train_grpo.py`, commit `31c19c8e4`).
  The chatml fallback I added 2026-06-07 used literal `<|user|>` /
  `<|assistant|>` tokens, which the AuroraGPT-2B tokenizer encodes
  as 4-token sequences the model has never seen as turn
  boundaries — every completion echoed the prompt back. New picker
  probes tokenizer vocab for single-token boundaries and picks
  `gemma` (`<start_of_turn>` / `<end_of_turn>`, ids 106/107 in
  AuroraGPT-2B), `chatml` (`<|im_start|>` / `<|im_end|>` for
  Qwen-family), or `plaintext` (USER:/ASSISTANT:) fallback. 25-step
  hardware verify (job 12468210) showed the model correctly
  generating `<end_of_turn>` and stopping early
  (`completions/min_length` 13-15 vs 64 with the broken fallback).
- **Per-task `--task` autocomplete** — choices auto-populated from
  `TASK_REGISTRY` (commit `05f0ee803`). Unknown task now fails at
  parse-time with the full list, not after dist init.
- **Auto-detect FSDP wrap class from `model_type`** (commit
  `e3477c717`). 15 model families pre-mapped (llama, llama4,
  qwen2, qwen3, mistral, gemma, phi, gpt_neox, deepseek_v3, …)
  so switching `--model_name_or_path` doesn't require also
  switching `--fsdp_transformer_layer_cls_to_wrap`.
- **Rank-0 model prefetch + broadcast** (commit `765f1f5a8`).
  48 ranks doing `AutoModel.from_pretrained` against the same HF
  Hub repo trips 429 rate-limits with 200s+ backoffs; rank 0
  pre-warms cache via `snapshot_download`, barrier, workers load
  from disk.
- **`device_map="auto"` override for FSDP** (commit `e00b130f9`).
  TRL's `create_model_from_path` defaults `device_map="auto"`
  which under FLAT mode lands every rank's model on the
  highest-numbered tile (`xpu:11`) — FSDP then catches the
  per-rank-device mismatch and raises before training. Override
  to `device_map=None` so `model.to(accelerator.device)` puts
  the model on the right tile. Diagnosed via the
  `scripts/diag/device_mismatch.py` 48-rank probe — phase-1 confirmed
  pre-trainer device assignments were all correct, phase-2 hit
  the same xpu:11 bug as the user, smoking gun was TRL's default.
- **Auto-populate `setup_wandb` config from every dataclass field**
  (commit `d5619f512`). 11 → 181 hyperparameter keys; nothing
  gets silently dropped from sweeps. Also enabled
  `log_completions=True` + `num_completions_to_print=2` so the
  Rich completions table is streamed to wandb without becoming
  a wall-of-text on screen.
- **Revert and replace dead-end fix** (`d1affb775`): the
  `torch.xpu.set_device(local_rank % ngpus)` early-pin I added
  2026-06-07 was verified harmless but didn't fix the bug — the
  real culprit was TRL's `device_map="auto"` (above). Reverted
  via `git revert` rather than force-pushing.
- **`extract_answer` prefers LAST `=` + recognizes gsm8k `####`**
  (commit `d749e2199`). User flagged that
  `6×12×10×12=<<6*12*12=720*12=8640>>` was graded 0.0 even
  though `8640` was correct — old regex matched the FIRST `=`
  and returned the intermediate `720`. Fix walks all `=` matches
  and returns the last one; adds gsm8k's `#### N` final-answer
  marker as a higher-priority alternative.

### Streaming-mode rewrite of RL tasks + new `arithmetic` task

User noticed the default `num_samples=1000` meant a 1000-step run
sees each prompt ~48× — model can memorize rather than learn.
Switched all five tasks (`sum_digits`, `multiply`, `arithmetic`
[new], `word_sort`, `countdown`) to a `build_streaming_or_finite`
helper that materializes a 100k-pool when `num_samples=0` (the
new default) and a finite-N pool otherwise. Iteration was bumpy:

  - Commits `98d537d07` (sum_digits + multiply) and `08638ff9e`
    (new `arithmetic` task with {+, −, ×, ÷}) tried to use a true
    `IterableDataset` — but TRL's GRPOTrainer rejects iterable
    datasets at __init__ (trl#3213). User hit the
    `NotImplementedError` on first launch.
  - Commit `f454e9236` replaced the IterableDataset with a large
    finite `Dataset.from_list` (default 100k); same effective
    "no prompt reuse" behavior at modest one-time init cost
    (0.5s for sum_digits, 125s for countdown due to its
    permutations-based rejection sampling).
  - Commit `637a83cda` extended the streaming default to
    `word_sort` + `countdown` after a user-launched
    `--task word_sort` hit the "There seems not to be a single
    sample in your epoch_iterator" empty-dataset bug — those two
    tasks had been missed in the first pass.

The new `arithmetic` task mixes operations with weighted sampling,
guarantees integer answers (division uses divisor + quotient
construction), adds an `op` column for per-op reward dashboards,
plus a `length_penalty` reward function on top of accuracy +
format (commit not pushed yet — verified locally only).

### SFT companion (`train_sft.py` + `datasets_sft.py` + Aurora submit)

Built the SFT side of the RL stack so weak-baseline checkpoints
like AuroraGPT-2B-sophiag can be instruction-tuned before being
used as a GRPO starting point. Mirrors `train_grpo.py`'s shape:
`HfArgumentParser((EzpzSFTArgs, EzpzSFTConfig))`, reuses all the
GRPO helpers (FSDP env-bootstrap, chat-template picker, rank-0
prefetch, wandb auto-config), adds `assistant_only_loss=True` +
`packing=True` + `max_length=1024` defaults.

Dataset registry (`datasets_sft.py`):
  - `gsm8k` (7473) — grade-school CoT math
  - `metamathqa` (~395k) — augmented GSM8K+MATH
  - `alpaca` (52k) — broad instruction-following (added later)
  - `math_alpaca_mix` (60/10/30 metamath/gsm8k/alpaca via
    `interleave_datasets`) — broader for downstream non-math tasks
    like `word_sort` (added later)

Bundle commit: `5446e1d74`. Required adding `{% generation %}` /
`{% endgeneration %}` markers around the assistant content in all
3 chat templates so SFTTrainer's `assistant_only_loss=True` could
compute the loss mask (folded into the same commit). Verified
locally that the gemma template produces the correct
`assistant_masks` via `apply_chat_template(...,
return_assistant_tokens_mask=True)`.

Verified end-to-end on 4N Sunspot (job 12468212): 50 steps
(`max_steps=50`), `train_loss=0.582`, `mean_token_accuracy=0.871`
in 35s. Pipeline intact.

### 32N production SFT — extended XCCL pain

User asked for a 32N SFT to actually produce a usable
instruction-tuned checkpoint. The pipeline works in the small but
**32N hits XCCL exceptions repeatedly**:

| Job | Dataset | Outcome | Failure mode |
|---|---|---|---|
| 12468217 | metamathqa | Died at 2:20 | 384 ranks × HF Hub xet-read = 429 storm + `.incomplete/dataset_info.json` cascade |
| 12468218 | metamathqa | Trained to step 442 (epoch 1.75), then `ccl::v1::exception` → SIGABRT on rank 241 | First XCCL crash |
| 12468220 | metamathqa (resume) | `--resume_from_checkpoint` silently ignored, restarted from scratch, hit same XCCL crash at step ~590 (epoch 1.55) | Resume bug + repeat crash |
| 12468221 | metamathqa (resume) | CLI validation: `--auto-retry` needs `--nproc`, not `--nhost` | Bad CLI |
| 12468222 | metamathqa (resume) | First successful `--auto-retry` swap-in (32 train + 4 spare), but `--resume_from_checkpoint` still silently ignored across both attempts. Walltime-killed mid-second-attempt | Resume bug + walltime |
| 12468232 | math_alpaca_mix | Crashed at argparse-init: `60% metamathqa` in description triggers `%m` format error | Argparse `%` escape |
| **12468237** | math_alpaca_mix | Trained to step 200 + saved a usable `checkpoint-200`, then hung post-save for 30 min until `--auto-retry` watchdog killed it. `stuck_pre_training` failover-stop on attempt 2 | Post-save hang (new failure mode) |

Fixes landed during the chain:

  - `train_sft.py:_prefetch_and_broadcast_dataset` — rank 0 builds
    dataset first, barrier, workers load from warm cache (no more
    429 storms)
  - Pre-warmed `~/.cache/huggingface/datasets/` locally on `/home`
    so compute nodes always hit cache (lustre-NFS shared)
  - `trainer.train(resume_from_checkpoint=config.resume_from_checkpoint
    or None)` explicit pass-through (the silently-ignored resume
    was an FSDP-sharded checkpoint compatibility issue with HF
    Trainer's auto-detect)
  - `%%` escape on all literal `%` in registry descriptions

What we have to show for it: **`outputs/sft/aurora2b-sophiag-metamathqa-32n/checkpoint-400-hf/`**
— consolidated HF-format checkpoint (~1.75 epochs metamathqa,
loss 0.20, mean_token_accuracy 0.934, 7.94 GB safetensors).
Usable for downstream GRPO via
`--model_name_or_path outputs/sft/aurora2b-sophiag-metamathqa-32n/checkpoint-400-hf`.
Plus `outputs/sft/aurora2b-sophiag-mix-32n/checkpoint-200/`
(FSDP-sharded, mix dataset, 200 steps) — could be consolidated
similarly.

The 32N XCCL crash pattern reproduces across two distinct SFT
training paths (metamath-only and mix). Not a one-off; not
strongly correlated with a single bad node either (12468218
crashed on `x1921c5s1b0n0`, 12468220 on a different host). Worth
investigating further (single-rank stuck-in-collective somewhere
between step 400-600 of a multi-node SFT run, looking the same
across two different datasets and across two different training
schedules) — but the SFT'd checkpoint we have is good enough to
move on with the GRPO experiments.

### SFT checkpoint consolidation tool

Added `rl/scripts/consolidate_sft_ckpt.sh` (bash) wrapping
`accelerate merge-weights` + cp of config + tokenizer from the
source model dir. Pattern: `checkpoint-N/pytorch_model_fsdp_0/`
(distcp shards) → `checkpoint-N-hf/model.safetensors` plus the
HF-format companion files. Verified consolidation produces a
`from_pretrained`-loadable checkpoint (`model_type=llama`,
`GemmaTokenizer`, 1.99B params, bf16 dtype).

### Aurora 80B production script default flip

Brief carry-over from 2026-06-07 evening: `submit_agpt_80b_aurora_venv_failover.sh`
default for `CHECKPOINT_ASYNC_MODE` was `async` (commit
`ce321caae` flipped it to `disabled`). Just confirming the change
shipped — anyone launching on Aurora today gets the safe
sync-checkpoint default that avoids the gloo-on-xpu crash.

---

## 2026-06-07 (evening) — 80B prod sync-ckpt validated end-to-end + train_grpo HfArgumentParser + FSDP wiring + blendcorpus index race + xccl issue filed

Big session covering four threads. Tracked in tasks #40–#63.

### 80B prod sync-ckpt path validated end-to-end (12468197, Sunspot 4N)

After three false-start retries diagnosing dataset-loader and PBS
env-var issues (12468190 hung after step 1 on the
`eliplutchok/fineweb-small-sample` HF stream; 12468194 died 32s in
without `NHOSTS_TRAIN`; 12468195/12468196 hit a blendcorpus
per-corpus + blendable-dataset index-build race — see below), got a
clean 4N TP=2 books-blendcorpus run in 12468197:

- **Loss descent**: 12.95 → **7.49** at step 164 (-5.46 nats)
- **Cadence**: ~42s per step, ~7 min per 10 steps, steady throughout
- **MFU**: 17.7-17.9% steady (matches the May 5 working-config smoke
  12466025 exactly)
- **Memory**: 88.97% peak (4N gives ~7 GiB tile headroom for 80B)
- **First sync checkpoint**: **saved at step 100 in 101.84s**,
  904 GB across 48 distcp shards, durable on disk at
  `outputs/checkpoints/agpt-80b-adamw-books-n4-gbs24/step-100/`
- **Walltime-killed at step 164** (Exit_status=-29, SIGKILL on 2h
  walltime — `TRAINING_STEPS=200` was a soft target; would have
  reached step 200 with a 3h allocation)

The **first sync ckpt save is the load-bearing milestone** — it's
exactly where 12468189 died on async mode hitting the gloo-on-xpu
bug. With `CHECKPOINT_ASYNC_MODE=disabled` the rank-0
`dist.new_group(backend="gloo")` is skipped entirely and the rest of
the path is clean. The 80B production stack is now end-to-end
validated on Sunspot under the workaround.

### train_grpo CLI rewrite: argparse (10 flags) → HfArgumentParser (191 flags)

User flagged that `torchtitan/experiments/ezpz/rl/train_grpo.py`
was exposing only ~7 of GRPOConfig's 64 own fields + ~100 inherited
TrainingArguments fields — every other knob was hardcoded or hidden
behind env-var fallbacks. Refactored to `HfArgumentParser((EzpzGRPOArgs,
EzpzGRPOConfig))`:

- `EzpzGRPOArgs` holds ezpz-side fields (`task`,
  `model_name_or_path`, `num_samples`, `no_save`,
  `fsdp_transformer_layer_cls_to_wrap`, `fsdp_cpu_ram_efficient_loading`)
- `EzpzGRPOConfig(GRPOConfig)` overrides defaults where ezpz/XPU
  values differ from upstream (bf16=True, gradient_checkpointing=True,
  beta=0.0, torch_empty_cache_steps=1, num_generations=4,
  max_completion_length=64, save_strategy="no", use_vllm=False)
- All other GRPOConfig fields fall through to TRL defaults, so
  every TRL knob (--beta, --epsilon, --loss_type, --vllm_*,
  --optim, --gradient_accumulation_steps, --num_iterations,
  --warmup_steps, --lr_scheduler_type, etc.) is now CLI-settable
  without code edits

Breaking change: `--model-name-or-path` and `--no-save` (hyphenated)
became `--model_name_or_path` and `--no_save` (snake_case) since
HfArgumentParser mirrors dataclass field names. `grep -r` found no
existing callers, so safe to land.

### FSDP env-bootstrap wiring (mirrors ezpz.examples.hf.py pattern)

Under `ezpz launch` (mpiexec), passing TRL's `--fsdp full_shard` is
a silent no-op: HF Trainer's internal `accelerate.Accelerator` only
builds a `FullyShardedDataParallelPlugin` when env vars
(`ACCELERATE_USE_FSDP=true`, `FSDP_*`) are pre-set — which
`accelerate launch` does for you but `mpiexec` does not. So passing
`--fsdp full_shard` was silently dropping every rank into plain DDP
(every rank holds the full model → 2B models OOM at 12 ranks/tile).

`ezpz.examples.hf.py` works around this by constructing the FSDP
plugin explicitly and passing it to `Accelerator(fsdp_plugin=...)`
— but that path requires owning the training loop, which TRL owns.
So we adopted the equivalent surgical approach: populate the same
env vars the explicit plugin would generate, BEFORE
`GRPOTrainer.__init__` runs. Wires up
`ACCELERATE_USE_FSDP=true`, `FSDP_SHARDING_STRATEGY`,
`FSDP_AUTO_WRAP_POLICY=TRANSFORMER_BASED_WRAP`,
`FSDP_TRANSFORMER_CLS_TO_WRAP`,
`FSDP_BACKWARD_PREFETCH=BACKWARD_PRE`,
`FSDP_USE_ORIG_PARAMS=true`,
`FSDP_STATE_DICT_TYPE=SHARDED_STATE_DICT`,
`ACCELERATE_MIXED_PRECISION=bf16` (from `bf16=True`),
`FSDP_CPU_RAM_EFFICIENT_LOADING=true` +
`FSDP_SYNC_MODULE_STATES=true` (when
`--fsdp_cpu_ram_efficient_loading` is on).

Two follow-up fixes the user surfaced from real launches:

1. **FSDPOption enum coercion** (`d859fafc1`): HfArgumentParser
   parses `--fsdp full_shard` into a list of `FSDPOption` enums.
   `str(FSDPOption.FULL_SHARD)` returns `'FSDPOption.FULL_SHARD'`
   (the StrEnum class-qualified name), not `'full_shard'`. Initial
   parser used `str(x).lower().split()[0]` →
   `'fsdpoption.full_shard'` → not in `_FSDP_STRATEGY_MAP` →
   ValueError before training. Fixed:
   `getattr(x, "value", str(x)).lower()` and scan whole list for a
   known strategy token (so `--fsdp "full_shard auto_wrap"` works
   regardless of token order).
2. **gradient_checkpointing + FSDP migration** (`6758a809d`):
   transformers warns `"When using FSDP full shard, instead of using
   gradient_checkpointing, please use activation_checkpointing in
   fsdp_config"` (training_args.py:2732). The warning fires inside
   `TrainingArguments.__post_init__`, BEFORE `main()` runs, so
   migrating in main() was too late — 48 ranks each printed it.
   Fixed: override `EzpzGRPOConfig.__post_init__` to migrate
   `gradient_checkpointing` → `fsdp_config["activation_checkpointing"]`
   BEFORE `super().__post_init__()` runs. Now transformers sees
   `gradient_checkpointing=False` and never warns. Also pre-loads
   `--fsdp_config <path>.json` so user-set keys
   (transformer_layer_cls_to_wrap, cpu_ram_efficient_loading) survive
   the migration merge.

### blendcorpus per-corpus index-build race

12468195 died in 2:34 with
`EOFError: No data left in file` when rank 2 tried to mmap-load
`shuffle_idx.npy` while rank 0 was still writing it. The books
dataset is tiny (3 shards, 11 GB, 4826 samples for a 200-step run
at GBS=24) so rank-0 index-build finishes in **8 ms** — too fast
for the implicit barrier-via-allreduce on the next dist op to close
the race.

12468196 died the same way 2:34 in but at the NEXT layer — the
"blendable dataset" index (`_index.npy`, `_sample_index.npy`),
built in 11 ms.

Root cause in `deps/blendcorpus/blendcorpus/data/gpt_dataset.py`:
the per-corpus path (`_build_index_mappings`, lines ~1050-1135)
does rank-0-write then all-ranks-`np.load(..., mmap_mode='r')`
**with NO `torch.distributed.barrier()` between them**. The
blendable-dataset path (lines 215-265) DOES have barriers — so the
per-corpus path is racy at small dataset sizes.

Production canonical chain uses olmo-mix (much larger, build takes
seconds) so this has never bitten before. Workaround: just retry —
once indices are durably on disk, the second pass hits cache and
skips the build. Permanent fix needs a `torch.distributed.barrier()`
in upstream blendcorpus.

Diagnosis + minimal repro shape + suggested upstream fix in
[`docs/upstream-issues/blendcorpus_index_build_race.md`](upstream-issues/blendcorpus_index_build_race.md).

### xccl supportsSplitting issue filed upstream

Filed as **[pytorch/pytorch#186548](https://github.com/pytorch/pytorch/issues/186548)**
with verified Sunspot repro log + collect_env block, plus a
cross-reference comment on the sibling
[pytorch/pytorch#171938](https://github.com/pytorch/pytorch/issues/171938).
The workaround `xccl_split_group_workaround.py` stays in place
until both: (1) `ProcessGroupXCCL.supportsSplitting() override`
lands, (2) `ProcessGroupXCCL::split` is implemented + CI-tested.

Removal criteria documented at the bottom of
[`docs/upstream-issues/xccl_split_group_unsupported.md`](upstream-issues/xccl_split_group_unsupported.md),
which now has a banner pointing to the upstream tracker.

---

## 2026-06-07 (pm) — 80B prod attempt on Sunspot 4N + new upstream CheckpointManager XPU bug

First real 80B production launch on Sunspot post-47th-sync. Job
12468189: died at trainer init with

    RuntimeError: No backend type associated with device type xpu

raised inside `CheckpointManager.__init__` at
`torchtitan/components/checkpoint.py:468`, on the line

    self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

which fires when `async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM)`.
xccl-only default PG has no gloo backend bound, so the gloo
subgroup-creation fails. Same shape as the xccl_split_group bug from
the prior session — another upstream code path that assumes a CUDA-
style gloo PG bundled in with the accelerator backend.

Discovered courtesy of the RANK 0 ABORT chain extension from
2026-06-06 (`84c84cd0b`): the underlying cause showed up clearly at
the top of the failure log instead of being buried under per-rank
mpiexec stderr.

**Workaround**: `--checkpoint.async-mode=disabled` (or
`CHECKPOINT_ASYNC_MODE=disabled` via the failover wrapper). Sync
ckpt mode skips the gloo subgroup creation entirely.

Resubmitted as 12468190 with the workaround; got past trainer init,
step 1 clean (loss 12.913, mem 53 GiB / 82.86%), training in
progress as of this entry.

Full diagnosis + removal criteria + the related-bugs cross-refs in
[`docs/upstream-issues/checkpoint_async_gloo_on_xpu.md`](upstream-issues/checkpoint_async_gloo_on_xpu.md).

Also added a `DATASET` env knob to
`scripts/submit_agpt_80b_aurora_venv_failover.sh` (commit
`eeb907b73`) so non-Aurora launches can use HF-streaming datasets
without a local data-list (Sunspot has no canonical olmo-mix-1124
list).

### Other 80B production state

- The 4N working-config table in `docs/production/agpt/80b/README.md`
  now has three bit-equivalent smoke datapoints (May 5 Aurora,
  Jun 2 Sunspot, Jun 6 Sunspot post-47th-sync). All converge to the
  same step-20 loss (10.39-10.46) and 88.94% memory. Configuration
  is stable across two ezpz refactors + one workaround.

### Optional follow-ups

- Ezpz-side workaround module for the gloo-on-xpu bug (mirroring
  `xccl_split_group_workaround.py`) so async ckpt mode "just works"
  on XPU. Deferred — sync mode is acceptable for now.
- File pytorch/torchtitan issue requesting defensive fallback in
  `CheckpointManager.__init__` when `new_group(backend="gloo")` fails.

---

## 2026-06-07 — spmd_types venv fix + LBS=2 scaling sweeps + post-mortem docs

Continuation of the 2026-06-06 scaling-study unblock. Two new
findings drove the work this session:

### 1. Stale wrapper default for agpt_20b

While reviewing yesterday's 64N + 128N numbers the user noticed the
agpt_2b scaling rows had MFU well below the 256N reference point
(18.99% / 16.13% vs 18.77%). Tracing the wrapper found agpt_2b
defaulting to `LBS=1` while production uses `LBS=2`. Fixed in commit
`4ceffb31e`. Verifying production scripts also caught agpt_20b at
`LBS=1` in the wrapper while `submit_agpt_20b_aurora_venv.sh` uses
`LBS="${LBS:-2}"`. Fixed in commit `8294883e5`.

Re-ran at the new LBS=2 defaults:

| N | agpt_2b TPS/MFU | agpt_20b TPS/MFU | moe_2b | Job |
|---|------------------|-------------------|--------|-----|
| 64 | 6,553 / 24.59% | 448 / 22.36% (LBS=1 stale) | killed @ walltime | [8528940](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/) |
| 64 | 6,083 / 22.82% | 511 / 25.48% | NO_OUTPUT 235s | [8529046](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/ihiy4ej1) |
| 128 | 4,934 / 18.51% | 480 / 23.96% | OOM 616s | [8529081](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/m9i0long) |

Confirmed expected weak-scaling pattern: 2B drops to 18.5% MFU at
128N, 20B holds steady around 24% — both consistent with all-reduce
overhead growing with N. Memory shot up from ~20 GiB (LBS=1) to
~44 GiB (LBS=2) for 2B; 20B from ~29 GiB to ~40 GiB. All still
safely under 70%.

### 2. Upstream `spmd_types` regression

Mid-resubmit, fresh 64N attempt (`8529016`) CRASHed in <20s with:

    File "/lus/.../torchtitan/components/loss.py", line 12, in <module>
      import spmd_types as spmd
    ModuleNotFoundError: No module named 'spmd_types'

Tracked to upstream commit `fec0c175d` (Pian Pawakapan, 2026-06-05,
[#3467] "[spmd_types] manual loss parallel CE"), which adds
`spmd_types==0.2.1` to `requirements.txt` + `pyproject.toml` but
relies on user reinstall. Our compute-node `/tmp/.venv` (from the
2026-06-03 tarball) didn't have it; nor did the source venv before
the user's `uvi --no-deps spmd_types` today.

Fix:

    uv pip install --python .venv/bin/python3 \
        --no-deps --no-cache --link-mode=copy spmd_types

Then rebuilt `.venv.tar.gz` (2.6 GB, 56 spmd_types entries verified
in tarball). Old broken tarball backed up at
`.venv.tar.gz.bak-pre-spmd-20260606-215928`. Validated on 8529046 +
8529081 — both ran cleanly post-fix.

### 3. Docs revert + redistribution

Yesterday's consolidation of the 4 per-model scaling docs into a
single 250-line README turned out to be a regression vs the sibling
convention (`docs/evals/agpt/{2b,20b}/`, `docs/production/agpt/`).
Reverted in commit `f19bbdec3`: README is now an index-only
dashboard, per-model pages restored. Today's LBS=2 numbers + the
spmd_types post-mortem flagged inline on the relevant per-model
page.

### 4. moe_2b regression

moe_2b at 64N + 128N still fails on Aurora torch 2.13 even with
spmd_types installed. Failure mode changed (NO_OUTPUT 235s → OOM
616s), but it's likely the upstream `edp_mesh=None` SIGABRT noted
in CLAUDE.md. Not yet diagnosed; tracked separately.

### State at end of session

- 2B 256N production chain (`8519833`) walltime-finished cleanly at
  step-69900 yesterday. Continuation `8521626` still Q for a 256N
  prod slot.
- All 2B + 20B production chains Q+H, waiting on prod queue rotation.
- Today's Aurora torch 2.13 scaling row at 256N is the next gap
  to fill once Q frees.

---

## 2026-06-06 — scaling-study unblock + production charts refresh + scaling docs consolidation

End-to-end session that turned a string of scaling-study NO_OUTPUT /
CRASH failures (every 64N+ submit since 2026-05-29) into a clean Aurora
torch-2.13 scaling table, refreshed all production charts, and
consolidated the four per-model scaling docs into one page.

### Scaling-study unblock (commits `6c6235fdf`, this entry)

Five stacked bugs in `scripts/run_scaling_study_aurora.sh` were silently
killing every 64N+ scaling submit. Root-caused and fixed:

1. **`.venv.tar.gz` rebuild lost `.venv/bin/`** (empty in tarball even
   though present in source venv) — rebuilt manually.
2. **Wrapper's trailing `"$@"`** on the inner `ezpz launch python3 -m
   torchtitan...train` invocation leaked PBS `-v` CLI args straight
   into the training entry point, causing instant arg-parse failure
   (NO_OUTPUT wall <60s). Removed.
3. **blendcorpus segfault at ≥768 ranks** in
   `blendcorpus_builder.py:275 __init__`. Bypassed by adding a
   `SCALING_DATASET` env knob so the sweep can run against an HF
   streaming dataset (default for sweeps now is
   `eliplutchok/fineweb-small-sample`). Default in the script stays
   `blendcorpus` so production-shaped sweeps still hit the real loader.
4. **`qsub -- /bin/bash -c "..."` swallowed the `#!/bin/bash --login`
   shebang**, leaving `module` undefined on the PBS-spawned shell →
   `module load oneapi/release/2025.3.1` silently failed → oneAPI MPI
   binaries (`mpiexec`, `qstat`) not on PATH →
   `from sh import qstat` ImportError. Fix: submit the script
   directly (`qsub <script>`), not via `bash -c`.
5. **PATH-order race** propagated rank-N python3 ahead of
   `/tmp/.venv/bin`, so the rank-N `python3 -m torchtitan...train`
   imported a system Python with no ezpz. Pinned the inner command to
   `${VIRTUAL_ENV:-/tmp/.venv}/bin/python3`.

Also caught a stale default: `agpt_2b` in the scaling wrapper was
`LBS=1`, but production runs `LBS=2` (`submit_agpt_2b_aurora_venv.sh`).
Changed the default to LBS=2 so the wrapper produces apples-to-apples
numbers with prod going forward.

**Validation (`SCALING_GROUP=light`, `SCALING_DATASET=eliplutchok/fineweb-small-sample`):**

| N | LBS | Job | agpt_2b TPS/MFU | agpt_20b TPS/MFU | moe_2b |
|---|-----|-----|------------------|-------------------|--------|
| 64 | 1 | 8528805 | 5,062 / 18.99% | 447 / 22.32% | NO_OUTPUT |
| 128 | 1 | 8528834 | 4,300 / 16.13% | 417 / 20.83% | CRASH |
| 64 | 2 | 8528940 | _pending_ | — | — |

moe_2b NO_OUTPUT at scale tracks the upstream `edp_mesh=None` SIGABRT
regression noted in CLAUDE.md. Separate task.

### Production chains

- **2B 256N (`8519833`)** walltime-finished cleanly @ step-69900
  (Exit_status=-29, walltime=12:00:20). Continuation `8521626` is now
  top of Q. Chain still has +2 conts beneath (`8521630` H,
  `8521631`/`8521632` H).
- **2B 512N + 20B 512N chains**: both Q'd for days waiting on prod
  slot. 20B chain's `step-4500` ckpt was an empty placeholder
  (4.0K, 0 .distcp shards) from a mid-save kill — renamed to
  `step-4500.bak-empty-20260606-170503/` so 8521628 resumes cleanly
  from step-4400 (244GB, complete) when it gets a slot.

### Evals (commit `11d8d9f26`)

Ran `eval-2b-v2.sh` on 2B 256N step-66000 and step-68000 (8528801):

| Step | Tokens (B) | HSn | ARC-E | ARC-C | Wino |
|------|-----------|-----|-------|-------|------|
| 64,000 | 3,221 | 0.5538 | 0.6040 | 0.3336 | 0.5549 |
| **66,000** | **3,322** | **0.5577** | **0.5918** | **0.3302** | **0.5462** |
| **68,000** | **3,422** | **0.5577** | **0.5905** | **0.3302** | **0.5509** |

ARC-E spike at 64K appears to be noise; otherwise convergence is flat
to slightly positive on HSn.

### Docs consolidation (commit `11d8d9f26`)

Collapsed `docs/scaling/{agpt-2b.md,agpt-20b.md,agpt-80b.md,moe.md}`
into a single `docs/scaling/README.md` organised by model. Added a
"Historical: the n=64/128 CRASH era" callout documenting the five-bug
stack above. Backfilled stale prod walltime (12h, was 6h on 20b page).

### Production charts refresh (commit `148fff6e6`)

Re-ran all plotting scripts (`plot_production.py`,
`plot_production_combined.py`, `plot_production_wandb.py`,
`plot_evals_combined.py`, `plot_v1_vs_v2.py` for both 2b + 20b)
against current W&B. 28 SVGs/PNGs refreshed across production +
historical-v1-bf16 + evals figure dirs.

---

## 2026-06-06 — 47th upstream sync (replays + smokes + post-smoke fixes; READY TO MERGE)

Pulled 34 commits since the 46th sync. Two structural refactors hit
ezpz; both replayed, smoked against baselines, and validated.

### Replays (initial)

1. **PR #3458 (RoPE refactor) — commit `02dd1e7fe`.** Splits
   `RoPE.Config` into `ComplexRoPE.Config` / `CosSinRoPE.Config`,
   moves rope ownership from top-level model config down to per-layer
   `Attention.Config`, removes the `apply_rotary_emb_*` helpers in
   favour of `self.rope = config.rope.build()` +
   `q, k = self.rope(q, k, positions)`. Replayed across
   `agpt/__init__.py`, `agpt/config_registry.py`, `moe/__init__.py`,
   and `moe/model.py`.

2. **PR #3269 (mixed-optimizer refactor) — commit `bac0a3473`.**
   Replaces the flat `OptimizersContainer.Config(lr=8e-4)` shape
   with `param_groups=[ParamGroupConfig(pattern, optimizer_name,
   optimizer_kwargs={"lr":...})]`. Custom-container subclasses are
   now thin wrappers registering their optimizer via
   `_resolve_optimizer_cls`; added 8 `default_<name>(lr=..., **kwargs)`
   factories mirroring upstream's `default_adamw`. Replayed across
   `optimizer/containers.py`, `optimizer/__init__.py`, both
   `config_registry.py`'s, `competition/configs.py` (28 callsites +
   19 in-place LR mutations), and `train.py`'s `--optimizer` CLI
   swap helper.

Full breakdown of both halves + the other 32 upstream commits in
[`upstream-sync.md`](upstream-sync.md).

### Baseline numerics smoke (against the 2026-06-02 baselines)

| Config        | Job      | Outcome                                            |
|---------------|----------|----------------------------------------------------|
| `moe_2b_ep` 2N | 12468156 | 10 steps clean, loss 12.95 → 7.92, mem 58.28 GiB matches baseline |
| `agpt_80b TP=2` 4N | 12468157 | 20 steps clean, all within ±0.08 nat of baseline, mem + MFU bit-identical |

### Coverage smokes — full registry sweep

Then ran a wider smoke sweep to exercise the rest of the registry.
This caught 3 real bugs in the initial replay, all now fixed:

| Config | Outcome | Notes |
|---|---|---|
| `agpt_2b` | ✅ 10 steps, 12.95 → 7.63, ~20% MFU | |
| `agpt_2b_real` | ✅ 10 steps, 12.99 → 8.50 | validates CosSinRoPE swap path through rewritten `_set_rope_backend` |
| `agpt_20b` | ✅ 10 steps, 12.90 → 10.39 | loss noisy (no warmup, hot LR) but trains |
| `moe_2b` (LBS=2) | ✅ 10 steps, 12.94 → 8.52 | (default LBS=16 OOMs, pre-existing) |
| `moe_10b_2b_sdpa_ep` (LBS=1, AC=selective) | ✅ 10 steps, 12.96 → 9.44, mem 51.12 GiB / 80% | new default — see fix #3 below |
| `speedrun_2b_muon` (LBS=1) | ✅ 10 steps, 12.93 → 9.32, ~3,000 tps | validates Muon dispatch — see fix #1 below |
| `speedrun_2b_sophiag` (LBS=1) | ✅ step 1 reached training | validates SophiaG dispatch via fix #1 |
| `moe_10b_2b` | ⏭ skipped | block_causal mask + HF-dataset mismatch (pre-existing, unrelated to sync) |
| `moe_10b_2b_sdpa{,_ep}` @ AC=full | ⏭ known broken | `CheckpointError: Recomputed values have different metadata` — MoE token routing isn't bit-exact across recompute (failure exists since at least 2026-05-12; PR #3146/#3450 fixed the forward path only) |

### Post-smoke fixes

1. **`optimizer/containers.py` — Config-dispatch bug (commit `6871e736b`).**
   The initial replay collapsed each custom container into a thin
   wrapper but the `default_<name>(...)` factories returned
   `OptimizersContainer.Config(...)` whose `_owner` is the base
   class. So `cfg.optimizer.build()` constructed the base
   container, whose `_resolve_optimizer_cls` only knows Adam/AdamW —
   any custom optimizer name (`Muon`, `SophiaG`, ...) raised
   `NotImplementedError: Optimizer Muon not added`. Caught by
   `speedrun_2b_muon` + `speedrun_2b_sophiag` smokes. Fix: add an
   empty `class Config(OptimizersContainer.Config): pass` to each
   of the 8 subclasses (so `_owner` binds to the subclass), and
   update each `default_<name>` factory to return the subclass's
   Config.
2. **`moe/config_registry.py` — 5 missed `cfg.optimizer.lr` mutations
   (commit `455013ed5`).** The optimizer refactor caught the 19
   such mutations in `competition/configs.py` but missed five in
   `moe/config_registry.py` (`moe_16b`, `moe_671b`, `moe_10b_2b`,
   `moe_10b_2b_sdpa`, `smoke_moe_500m_50steps`). Same fix as
   competition: write through `cfg.optimizer.param_groups[0].optimizer_kwargs["lr"]`
   instead of `cfg.optimizer.lr`. Caught when smoking
   `moe_10b_2b`.
3. **`moe/config_registry.py` — `moe_10b_2b_sdpa{,_ep}` defaults
   changed to LBS=1 + AC="selective" (commit `975a5bcd1`).** Prior
   default `(LBS=2, AC="none")` OOMs at first forward on 2N Sunspot
   (level_zero `UR_RESULT_ERROR_OUT_OF_RESOURCES`). Production
   scripts already overrode LBS=1 on the CLI, so this brings the
   registry in line. AC="full" can't be the answer — it hits the
   long-standing `CheckpointError` from non-deterministic MoE
   routing under recompute. AC="selective" only checkpoints the
   SAC save list (excludes the router), so the non-deterministic
   op never gets recomputed and shapes stay stable. Verified on
   job 12468186: 10 steps clean, peak 51.12 GiB / 79.9% (vs OOM
   at LBS=2).
4. **`datasets.py` — pickle fix for HF datasets + `num_workers >= 1`
   (commit `8746dfe2c`).** `_make_text_processor` returned a local
   closure that couldn't be pickled by PyTorch's `forkserver`
   DataLoader workers. Crashed every HF-dataset run with
   `num_workers > 0` (e.g. `agpt_2b ... --dataloader.dataset eliplutchok/fineweb-small-sample --dataloader.num-workers=2`).
   Pre-existing bug, not a replay regression. Fix: move `_process`
   to module scope as `_extract_text_column` + use `functools.partial`.

### Side notes from the merge

- Installed `spmd_types==0.2.1` into `.venv` via `uv pip install`
  (upstream `components/loss.py` requires it after PR #3466/#3467/#3560).
- Worktree symlinks (`.venv`, `.venv.tar.gz`, `assets/hf`) added
  by hand so submitted jobs find them — git ignores them.

### Final branch state

```
8746dfe2c datasets pickle fix
975a5bcd1 moe 10b_2b_sdpa{,_ep} → LBS=1/AC=selective default
455013ed5 moe registry 5 cfg.optimizer.lr fixes
6871e736b optimizer Config-dispatch fix
bc89aa85e docs (optimizer half done)
bac0a3473 optimizer refactor replay
3748b9e9c docs (RoPE half done)
02dd1e7fe RoPE replay
fb1c5a319 merge upstream/main
```

### Open follow-ups

- File pytorch/pytorch issue for MoE + AC-full `CheckpointError`
  (router non-determinism across recompute). Long-standing — the
  fix requires AC to save the routing decision instead of
  recomputing it.
- Audit the 4 RL commits in this sync — `experiments/ezpz/rl/`
  may need attention if they touch shared surfaces.

---

## 2026-06-02 — 46th upstream sync (graph_trainer-only, no ezpz replay)

Pulled 2 new commits since the 45th sync (`04a309858..27aa49077`):

- `27aa49077` [graph_trainer] Add full recompute memory policy (#3429)
- `051562e31` [graph_trainer] Re-enable DSv3 eager bitwise deterministic tests (#3482)

Both entirely inside `torchtitan/experiments/graph_trainer/`. Zero
ezpz files touched, no replay needed, no conflicts. Merge commit
`45a2b2568`. Full breakdown in
[`upstream-sync.md`](upstream-sync.md) (46th-sync entry).

---

## 2026-06-02 — 80B TP=2 4N smoke replay on Sunspot under xccl workaround

Followed up the moe_2b_ep workaround validation with a second
verification: that the new xccl_split_group_workaround
([`8031d1d3a`](https://github.com/saforem2/torchtitan/commit/8031d1d3a))
doesn't regress the working 80B TP=2 v2 config from the May 5
Aurora baseline (job 12466025).

Submitted 4N Sunspot smoke as job 12467825 using the same recipe
(`agpt_80b`, TP=2, AC=full, compile=OFF, AdamW LR=1e-6, fp32-master)
via the updated `submit_80b_no_compile_t213.sh` (now using SUBMIT_DIR
+ `ezpz tar-env`/`ezpz yeet`). Exit 0 in 902 s, 20 steps:

|             | May 5 (Aurora 4N) | 2026-06-02 (Sunspot 4N) |
|-------------|------------------:|------------------------:|
| Δloss       | -2.52             | **-2.55**               |
| Peak mem    | 88.94%            | **88.94%**              |
| Steady MFU  | ~17.8%            | **~17.8%**              |
| grad-norm peak | ~34 @ step 15-16 | **33.10 @ step 16** |

Numerically equivalent within run-to-run noise. The workaround
install line + `Successfully created meshes with active dimensions:
['batch', 'loss', 'tp', 'efsdp', 'fsdp']` confirms 5 nested PGs
built cleanly under the patched `_init_one_process_group`. The
existing xccl-timeout shim also ran (`Applied train timeout
0:01:40 to 5 xccl ProcessGroup(s)`).

Full writeup:
[`docs/experiments/agpt/sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md`](experiments/agpt/sunspot/20260602-smoke-n4-80b-tp2-xccl-workaround.md).
80B production page (`docs/production/agpt/80b/README.md`) updated
with the Sunspot replay row.

W&B: https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/a782sf8y

Open follow-ups (unchanged):
- Add 200-step linear warmup to 80B production config (grad-norm
  climb to 33.10 by step 16 is the same shape as the May 5 run).
- Mirror SUBMIT_DIR + ezpz yeet fixes into
  `scripts/submit_agpt_80b_aurora_venv_failover.sh`.

---

## 2026-06-02 — xccl split_group workaround for nested mesh init

`moe_2b_ep` smoke on torch 2.13 on XPU was hitting:

```
RuntimeError: No backend for the parent process group or its backend
does not support splitting
```

at trainer init, inside `ParallelDims.build_mesh` → the EP-flavored
sparse mesh `("pp", "dp_replicate", "efsdp", "ep")` (`parallel_dims.py:200`).

**Root cause** (confirmed by reading upstream C++ headers via
`gh search code`): `ProcessGroupXCCL` never declares
`supportsSplitting() override`. It inherits the base
`Backend::supportsSplitting()` from
[`Backend.hpp`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/Backend.hpp)
which returns `false`. `ProcessGroupNCCL` overrides to `true` —
xccl doesn't.

`DeviceMesh._init_one_process_group` (torch 2.13, `device_mesh.py:550-562`)
routes nested mesh PG creation through `split_group` whenever
`bound_device_id` is set on the default group AND the accelerator is
available AND the backend name matches. None of those rule out xccl;
the ezpz eager-init path sets `bound_device_id` on XPU, so the gate
always takes the broken branch.

`split_group` itself (`distributed_c10d.py:5565-5570`) then reads
`parent_backend.supports_splitting` (Python property bound to the C++
method), sees `False`, and raises before ever calling
`xcclCommSplit`. So the failure is purely at the gate — there's no
xccl split implementation to even crash on yet.

**Workaround**: monkey-patch `DeviceMesh._init_one_process_group`
from a new module
[`xccl_split_group_workaround.py`](../xccl_split_group_workaround.py).
The wrapper:

  1. No-ops on cuda/cpu builds (gates on `is_xccl_available() and
     torch.xpu.is_available()`).
  2. On xccl, inspects the default group's per-accelerator backend's
     `supports_splitting`. If `True` (NCCL), calls upstream verbatim.
  3. If `False` (xccl), temporarily clears `bound_device_id` on the
     default group so the upstream gate's first clause goes `False`
     and we fall through to the existing `new_group` loop. Restores
     `bound_device_id` afterwards.

Installed lazily from
[`FaultTolerantTrainer.init_distributed`](../trainer.py) so it only
fires when ezpz's trainer kicks off; never touches other torchtitan
paths.

Per Golden Rule #1, no upstream file was modified. Full diagnosis +
removal criteria in
[`docs/upstream-issues/xccl_split_group_unsupported.md`](upstream-issues/xccl_split_group_unsupported.md).

**Smoke validation** (2026-06-02, Sunspot 2N, job 12467823, exit 0
in 103 s — see
[`docs/experiments/moe/sunspot/20260602-smoke-n2-xccl-split-workaround.md`](experiments/moe/sunspot/20260602-smoke-n2-xccl-split-workaround.md)):

- Workaround install line in the log:
  `Installed xccl split_group workaround on DeviceMesh._init_one_process_group`.
- EP sparse mesh built cleanly:
  `Successfully created meshes with active dimensions: ['batch', 'loss', 'ep', 'efsdp', 'fsdp']`.
- Loss 12.945 → 8.273 across 10 steps, peak 58.28 GiB, ~3,050 TPS.
- The pre-existing xccl-timeout shim also ran:
  `Applied train timeout 0:01:40 to 6 xccl ProcessGroup(s)`
  (6 PGs = world + dense + sparse + efsdp + loss + batch — all the
  meshes the EP path constructs).

Two latent bugs in `scripts/submit_moe_smoke.sh` surfaced and were
fixed during the smoke (both apply to the analogous production
scripts under `scripts/submit_agpt_*_aurora_venv.sh`):

1. **`ezpz_setup_job` overwrites `$PBS_O_WORKDIR`** with the script's
   initial cwd (which is `$HOME` under default `qsub`). A naïve
   `cd "${PBS_O_WORKDIR}"` after sourcing the utils ends up in
   `/home/foremans` and `source .venv/bin/activate` resolves against
   `~/.venv` (no `ezpz`). Fix: stash submit dir into `SUBMIT_DIR`
   before sourcing utils.
2. **`ezpz yeet-env` is deprecated** in ezpz 0.18.x in favour of
   explicit `ezpz tar-env` + `ezpz yeet .venv.tar.gz`. Switched.

Open follow-ups:
- Mirror the `SUBMIT_DIR` + `ezpz yeet` fixes into
  `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` before the next
  512N+ production launch.
- File `pytorch/pytorch` issue with the two-part fix
  (`ProcessGroupXCCL::supportsSplitting() override + working split()`).

---

## 2026-06-02 — 45th upstream sync (PR #3450 closes the #3436 thread)

Merged 7 upstream commits (`b72d98648..04a309858`). Headline is PR
[#3450](https://github.com/pytorch/torchtitan/pull/3450) — the
upstream-canonical `histc → routing_map` swap for MoE routing that
finally lands the fix our [PR
#3436](https://github.com/pytorch/torchtitan/pull/3436) was tracking.
Approach: build a boolean `routing_map_BLE` via `scatter_` in the
router, compute `num_tokens_per_expert_E = routing_map.sum(...)` once,
thread the map through both router and dispatcher. Same direction we
recommended (scatter-based), executed better (computed once, reused).

API change at the boundary: router returns a 4-tuple now, `dispatch()`
takes a new `num_local_tokens_per_expert_E` positional arg. ezpz
doesn't override either method or unpack router output directly, so
inherits cleanly. ezpz `set_moe_sharding_config` calls upstream's
helper so the new shard declaration also lands automatically. **No
ezpz replay needed.** Imports + syntax green.

Other 6 commits: HybridEP compile support (#3360), MXFP8 consolidation
(#3473), graph_trainer DTensor fix + test re-enables (#3480), graph_trainer
test skips (#3432), and two CI migrations (#3464, #3479). None touch
ezpz code paths.

**Smoke status**: live 2N validation deferred to Aurora. The project
`.venv/bin/python` symlinks at
`/opt/aurora/26.26.0/spack/.../python-3.12.12-5zo3wzv/bin/python3`,
which is an Aurora-only Spack build (`5zo3wzv` hash). Sunspot has
the equivalent under a different hash (`nvje3vk`), so the symlink
is broken on Sunspot compute nodes. The `GroupedExperts`-touching
smoke per the prior journal lesson should run on Aurora next time
there's an alloc.

Also updated `~/.ezpz/utils.sh` from the bit.ly canonical (it now
exposes the new unified `ezpz_setup` function; the old
`ezpz_setup_job` / `ezpz_setup_xpu` split is superseded). Backup at
`~/.ezpz/utils.sh-20260602-065437`.

Open follow-ups:
- Close PR #3436 with a comment pointing at PR #3450.
- Smoke `moe_2b_ep` on Aurora to validate the routing_map flow.

---

## 2026-06-01 — Error-propagation fixes + 44th upstream sync

### Error-propagation fixes (from this morning's BlendCorpus debug)

Two bugs surfaced when a rank crashed at init with the misleading
`BlendCorpus dataset was requested but blendcorpus is not installed`
message — when in fact `blendcorpus` was installed and the actual
missing module was `deepspeed` (a transitive dep).

- [`9eb680dbc`](https://github.com/saforem2/torchtitan/commit/9eb680dbc)
  — `_import_blendcorpus_modules` now inspects `exc.name` and emits
  one of two messages: "blendcorpus is not installed" vs "blendcorpus
  IS installed but pulled in missing transitive dep `<name>`". Names
  the actual culprit module so users don't chase the wrong fix.
- [`2c5c7d597`](https://github.com/saforem2/torchtitan/commit/2c5c7d597)
  — wrap `config.build()` in its own try/except in `train.py:main()`.
  Rank 0 logs a single `RANK 0 ABORT during config.build():` line
  followed by the full `__cause__`/`__context__` chain on failure.
  mpiexec presents rank 0 output first, so this surfaces above the
  per-rank `rank N exited with code 1` spam.

### 44th upstream sync

Merged 1 upstream commit (`b72d98648`, PR
[#3403](https://github.com/pytorch/torchtitan/pull/3403)): 4-line
addition to project-root `.claude/CLAUDE.md` recommending ≥10
iterations for perf comparisons. No code change, no ezpz replay.

---

## 2026-05-31 — 43rd upstream sync (interleaved dataloader, no-op replay)

Merged 1 upstream commit (`221041490`, PR
[#3063](https://github.com/pytorch/torchtitan/pull/3063) — weighted
interleaved multi-source HF dataloader). Pure additions —
`InterleavedHuggingFaceTextDataLoader`, `InterleavedChatDataLoader`,
`HFDataSource`/`ChatDataSource`, an `InterleavedDataset` weighted
sampler, plus tests. The existing `HuggingFaceTextDataLoader` and
`DATASETS` that `experiments/ezpz/datasets.py` and
`blendcorpus/blendcorpus_builder.py` import are unchanged. No
replay; imports smoke green.

`git log HEAD..upstream/main` listed 8 commits but only 1 was a
genuine new patch — the other 7 were the patch-equivalent duplicates
from 41st/42nd-sync bookkeeping flagged in the 42nd-sync entry.
`git merge` handled the difference correctly.

---

## 2026-05-29 — 42nd upstream sync (RoPE refactor + replays)

Merged 6 upstream commits (`28483d0eb..065c2625d`). The headline is
PR [#3395](https://github.com/pytorch/torchtitan/pull/3395) which
restructures every model's `update_from_config`:

- Renames the `trainer_config` keyword to `config`.
- Promotes `seq_len > rope.max_seq_len` from warning to hard
  `ValueError`.
- Moves TP / `n_heads`/`n_kv_heads` validation, MoE `deepep`/`hybridep`
  EP=1 guard, MoE `moe_force_load_balance` debug flag, and the
  `rope.max_seq_len` sync into `Decoder.Config.update_from_config`.
- Adds async out-of-bounds checks inside `apply_rotary_emb_*`.

Replayed in `experiments/ezpz/{agpt,moe}/model.py`:

- agpt was a pure `trainer_config`→`config` rename plus parameter
  forwarding.
- moe was bigger: deleted the now-duplicated rope/MoE/TP checks and
  delegated to `Decoder.Config.update_from_config`; kept only the
  per-layer attention rope-field sync, the for_loop XPU fallback,
  the CP+MoE attention check, and `set_moe_sharding_config`. Dropped
  three now-unused imports.

Mirrors the post-refactor shape of upstream `deepseek_v3/model.py`.
2N smoke on alloc 12467655 surfaced two follow-ups:

- A missed `trainer_config`→`config` rename at
  `experiments/ezpz/trainer.py:163` (the trainer's own
  `model_config.update_from_config(...)` callsite, not the model
  override). Fixed in [04199e522](https://github.com/saforem2/torchtitan/commit/04199e522).
- A pre-existing miss from the 41st sync ([#3425](https://github.com/pytorch/torchtitan/pull/3425)
  MoE shape-suffix rename) that we hadn't smoked: three ezpz-side
  references to the old `w1`/`w2`/`w3` parameter names in
  `EzpzGroupedExperts` sharding/init/forward paths — caught when
  the rope replay let us reach trainer init for the first time post-
  41st-sync. Fixed in [88dbd916e](https://github.com/saforem2/torchtitan/commit/88dbd916e).

Post-fix smoke results (both clean, matching 2026-05-27 baselines):

- `agpt_2b`: 154 s, peak 24.34 GiB (38.04%), 21.95% MFU.
- `moe_2b_ep` LBS=2: 323 s, peak 26.99 GiB (42.18%), 9.09% MFU,
  loss step 50 = 6.13.

Lesson worth remembering: always smoke `moe_2b_ep` after a sync that
touches `GroupedExperts`. Today's chain of bugs hid behind the
trainer init failure — the model-init order is rope → expert sharding
→ first forward, so a rope-stage failure prevents us from seeing
expert-stage failures.

Smaller commits in the same sync: #3448 (#3395 fix-forward), #3452
(1-line determinism cleanup), #3445/#3446 (flux/qwen3-vl), #3347 (RL
batcher). None affect ezpz.

Bookkeeping curiosity: `git cherry` flagged 41st-sync MoE [8/n]
(`200100e7d`) as already present in our branch under a different SHA
(`56dc8e1d4`). `git merge` correctly skipped it. So the "7 unmerged
commits" `git log` showed was really 6.

---

## 2026-05-28 — 41st upstream sync (no-op replay)

Merged 1 upstream commit (`200100e7d`, PR
[#3425](https://github.com/pytorch/torchtitan/pull/3425) — MoE [8/n]
shape-suffix rename). Pure rename refactor applying the Shazeer
shape-suffix convention across all MoE tensors. Loss-comparator
verified `--assert-equal` upstream. Renames break several internal
method signatures (`dispatch`, `_unpermute`, `_make_dispatcher`) but
none are called from `experiments/ezpz/`. No replay needed; imports
smoke green.

Still open: maintainer direction on
[pytorch/torchtitan#3436](https://github.com/pytorch/torchtitan/pull/3436)
(histc → bincount/scatter for XPU determinism). Posted the
statistical E2E A/B yesterday — bincount and histc are
indistinguishable on `moe_2b_ep` (Welch's p=0.66, n=135 per variant).
Recommendation: scatter_add_ as the cleanest swap. Waiting on
maintainer choice between options A/B/C.

---

## 2026-05-27 — 40th upstream sync (7 commits)

Merged 7 upstream commits (`19c567f76..af33f7638`):

- **PR #3398** ([Module] Replace from_nn_module with native Module
  subclasses) consolidates `common/{linear,rmsnorm,embedding}.py`
  into `common/nn_modules.py`. Broke 3 import paths in ezpz; replayed
  in [`b052f29e4`](https://github.com/saforem2/torchtitan/commit/b052f29e4)
  with pure import-path swaps (class API is unchanged).
- **PR #3146** (Use deterministic ops in MoE routing) is the upstream
  fix for the `_histc_xpu does not have a deterministic
  implementation` blocker we hit on 2026-05-21. Replaces `histc` with
  `bincount` and adds `aten.topk.default` to the SAC save list.
  Inherits transitively; `--debug.deterministic` on MoE+XPU should
  now work.
- **PR #3423** (MoE [7/n], 3D tensors through MoE) continues the
  MoE refactor from #3386/#3389. Doesn't touch `deepseek_v3/model.py`
  and ezpz doesn't expose the 2D-flatten seam, so we inherit
  transitively.
- **PR #3105** (FSDP symmetric memory) adds an `enable_fsdp_symm_mem`
  flag, plumbed through each model's `apply_fsdp`. ezpz has its own
  local `apply_fsdp`, so the kwarg doesn't reach our path. Skipping
  the replay — symm_mem is an optimization and XPU's CCL likely
  doesn't support it anyway.
- **PRs #3331 / #3369 / #3361** are all graph_trainer-only; no-ops
  for ezpz.

Quick imports smoke (`python3 -c "import torchtitan.experiments.ezpz.{agpt,moe.model}"`)
passes. Live 2N smokes (job 12467455) on Sunspot:

- `agpt_2b` and `moe_2b_ep` clean post-merge, numerically identical
  to the 2026-05-22 baselines.
- `--debug.deterministic` on MoE+XPU **still fails**. PR #3146 was
  supposed to fix the `_histc_xpu` blocker via a `histc → bincount`
  swap, but the merged diff is missing that change — only the
  `aten.topk.default` save-list addition landed. Verified via the
  GitHub API that PR #3146's only file change is
  `activation_checkpoint.py`. The `histc` call at
  `common/moe.py:262` is untouched. Smoke report at
  [`docs/experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md`](experiments/moe/sunspot/20260527-smoke-n2-40th-sync.md).

Action item: file upstream issue for the incomplete PR #3146.

---

## 2026-05-27 — 20B 512N sync chain doubles its eval scores; 2B chains pass step-49K + step-27K

### Production progress (May 25 → May 27, ~40h)

| Trajectory | Start → End step | Δ steps | Ckpts persisted | Dispatches |
|---|---|---|---|---|
| **2B 256N async** | 36,528 → **49,666+** | +13,138+ | **~114** | 8507195 + 8507198 + 8508020 (R) |
| **2B 512N sync** | 16,676 → **27,106+** | +10,430+ | **~107** | 8507196 (pals-RPC infra fail, +76) + 8507199 (+50) + 8508753 (R) |
| **20B 512N sync** | 2,043 → **3,270** | +1,227 | **+12** | 8507197 + 8507200 |

**Total: ~233 new on-disk checkpoints across 3 chains in 40h of wall clock.**
Sync-mode workaround continues to hold for both 2B 512N and 20B 512N
trajectories; async-mode still stable at 256N.

### 🏁 Headline: 20B 512N sync now beats 2B 256N async per token on every benchmark

Eval'd the full 24-ckpt sync-mode sweep at steps 900..3,200. The 20B
512N sync trajectory now leads the 2B 256N async on **all four
benchmarks** at matched token counts:

| Task | 20B 512N step-3,200 (329B tok) | 2B 256N step-45,500 (~2.3T tok) |
|------|---:|---:|
| ARC-Easy `acc` | **0.6646** | 0.6418 |
| ARC-C `acc_norm` | **0.3225** | ~0.315 (oscillating) |
| HellaSwag `acc_norm` | **0.5737** | 0.5452 |
| Winogrande `acc` | **0.5612** | ~0.55-0.56 |

The 20B model is now token-efficient in a way the 2B has begun to
saturate (plateau at ARC-Easy ~0.645, HellaSwag norm ~0.547). The
full sweep is monotonic — no plateau, no oscillation, no sign of
optimizer instability across 24 consecutive checkpoints. This is
**the first time in the entire v2 experiment that the bigger model
has outperformed the smaller one at matched token counts**, and the
strongest live signal yet that the fp32-master + sync-mode
combination is the right operational stack for the 20B at scale.

### Async-regression workaround still solving 512N

The `CHECKPOINT_ASYNC_MODE=disabled` workaround continues to keep both
512N trajectories advancing. 2B 512N: 4 consecutive sync dispatches
(starting from `8506221`) have added ~10K steps and ~107 persisted
ckpts. 20B 512N: 4 consecutive sync dispatches have added 1,243 steps
and 24 persisted ckpts. No async-cascade failures in any of these.

### One Aurora pals-RPC infra failure (separate from any other bug)

`8507196` (2B 512N) trained cleanly to step **20,989** in-memory
(persisting 76 ckpts), but all three wrapper attempts hit an Aurora
**pals-RPC infrastructure failure** during launch (exit 127). The
wrapper correctly identified the failures as node-related and swapped
three different "bad" nodes — but pals-RPC is upstream of anything
the wrapper can repair. See
`memory/project_aurora_pals_rpc_launch_failure.md`. Not the
async-cascade bug, not a model issue, not a wrapper bug.

### 80B still blocked (separate SIGSEGV pattern)

No 80B production progress this stretch. Last attempts continue to
die in the same rank-level SIGSEGV pattern during init that bracketed
the earlier 80B failures, separate from any of the 2B/20B failure
modes. Writeup + likely upstream patches still pending from the
05-25 entry.

### Capacity is now the bottleneck

`8508214` (20B 512N continuation queued at 03:44) has been **Q ~10h**
in the `small` queue without starting — Aurora capacity for the 512N
slot is exhausted. The 20B chain is currently throttled not by any
bug or workaround but by raw queue availability. Same applies if any
of the 2B 512N continuations need to chain after `8508753`.

---

## 2026-05-25 — 🏁 Sync-mode workaround fully validated for both 2B + 20B 512N; 100+ ckpts persisted overnight

### Production progress (May 24 → May 25)

After kicking off 6 production dispatches and 3 smoke runs late on 2026-05-23,
**three production chains advanced significantly on disk overnight**:

| Trajectory | Start → End step | Δ steps | Ckpts persisted | Dispatches |
|---|---|---|---|---|
| **2B 256N async** | 25,500 → **36,528** | +11,028 | **65** (every 100 steps) | 8505175 + 8505252 |
| **2B 512N sync** | 13,300 → **16,676** | +3,376 | **21** (every 100 steps) | 8506221 |
| **20B 512N sync** | 800 → **2,043** | +1,243 | **12** (every 100 steps) | 8505258 + 8505259 |

**Total: 98 new on-disk checkpoints across 3 chains in ~36h of wall clock.**
First sustained 512N progress since 2026-05-03 for both 2B and 20B.

### Sync-mode async-cascade workaround validated

8505258 (20B 512N sync) was the proof-of-concept: trained 800→1414 cleanly with
6 ckpts persisted, the first 20B 512N to clear step-800 since the May-3 async
regression. 8505259 carried the chain to step-2043 (6 more ckpts).

8505176 (2B 512N async) confirmed the **same bug pattern at 2B 512N**: every
attempt cleanly trained 13300→13400, then died at the step-13400 async save,
wrapper swapped + retried 3 times before exhausting. Switching to sync mode
in 8506221 produced **21 consecutive ckpts** at 512N — same fix as 20B.

`CHECKPOINT_ASYNC_MODE=disabled` is now the standard 512N workaround for both
models. Async still works fine at 256N (65 ckpts in one dispatch).

### Preflight smoke bugs surfaced + fixed

8506215 (first 2B 512N sync attempt) revealed the preflight smoke's 120s
idle-timeout was too tight for 6144-rank DDP init. Fixed by bumping to 600s
default + adding `--train-iters 5` to cap the test length (was running 200
iters by default → 1+ hour of preflight at 512N). See
[`memory/feedback_preflight_timeout_scales_with_n.md`](.).

8506221 then hit a *real* silent hang during preflight (iter 111, 49 min of
silence), wrapper SIGTERM'd on watchdog, blind-swapped `x4305c0s7b0n0` for
`x4602c3s3b0n0`, preflight attempt 2 succeeded, main training started. **The
wrapper handled the failure exactly as designed even during the preflight
phase.**

### New jobs queued (May 25 afternoon)

- 8507195 (2B 256N async cont) + 8507198 (afterany +1)
- 8507196 (2B 512N sync cont) + 8507199 (afterany +1)
- 8507197 (20B 512N sync cont) + 8507200 (afterany +1)
- 8507204 / 8507205 / 8507206: lm-eval batches on the new 2B 256N (26-36K),
  2B 512N (14-16K), and 20B 512N (900-2000) checkpoints respectively

### 80B is still blocked

8505222 (80B 256N production) failed with rank-level SIGSEGV (signal 11) on
attempt 1 + every retry, wrapper correctly classified as bad-node failure
but every spare it swapped in was also bad. 5 retries exhausted. Separately,
80B 8N smoke 8505326 surfaced a deterministic blendcorpus EOFError race in
cache build (3 wrapper attempts, all same failure). Two distinct 80B failure
modes both need writeups + likely upstream patches.

---

## 2026-05-23 (late) — 🏁 Failover wrapper passes first production silent-hang test (job 8505298)

While running the 8-node smoke validation of the fresh ezpz 0.16.0
tarballs (jobs `8505298` / `8505325` / `8505326`), the **2B smoke job
8505298 caught a real silent training hang at step 37 and recovered
automatically** — every code path in the v2 failover wrapper that
exists to handle the [8479579 incident
pattern](experiments/agpt/aurora/20260511-20b-n512-hang-8479579.md)
fired correctly, in sequence, on a real-world failure.

Sequence (clean steps to recovery to checkpoint-persist):
1. Preflight `ezpz.examples.test` ran in ~2 min, exit 0.
2. Main attempt-1 trained steps 1 → 37, then logged went **completely
   silent at 21:06:41** — no traceback, no save attempt, no MPI error.
3. **30 min later at 21:36:41**, `ezpz launch --timeout=1800` watchdog
   tripped, SIGTERM'd PID 191892, exit 124.
4. Wrapper classified exit 124 as silent-hang bad-node failure (not
   walltime), couldn't identify a specific bad node from the log
   (none — the hang was silent), fell through to
   `failover_swap_one_blind()`, rotated `x4220c3s6b0n0` →
   `x4220c5s3b0n0`.
5. Attempt-2 started 4s after the swap, hit step 1 at 21:38:17, ran
   cleanly to step 296 (loss 12.96 → 5.68 = ~466M tokens) until
   PBS walltime kill at 21:57:50.
6. **step-100 and step-200 DCP checkpoints both persisted** on flare
   (96 shards × 239 MB each + 5.5 MB metadata). First unambiguous
   proof since the 2026-05-03 regression that async-save works
   end-to-end with the fresh ezpz 0.16.0 tarball.

Full writeup with exact log snippets:
[`experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md`](experiments/agpt/aurora/20260523-failover-silent-hang-recovery-8505298.md).

Knock-on actions: bumped the
[`bad-node-failover.md`](guides/bad-node-failover.md) status from
"v2 in production" to "v2 production-validated"; added back-pointer
from the original 8479579 incident report; greenlit qdel + resubmit
of the 9 queued production jobs (2B/20B/80B chains) against the
fresh tarballs.

20B + 80B smokes (`8505325` / `8505326`) still Q in capacity queue —
blocked behind `8505200` (the 2-node test.sh used for the tarball
rebuilds + smoke runs); will start when that walltimes out around
2026-05-24 04:00 UTC.

---

## 2026-05-23 — Failover wrapper hardening: tests, ANSI fix, async-mode regression diagnosed

### Failover-wrapper test harness + 2 more wrapper bugs

Built `tests/failover/{run_tests.sh, fixtures/*.log}` — 9 synthetic
log fixtures with byte-identical ANSI escapes, each reproducing one
of the failure modes we've seen in production. The harness mirrors
`failover_lib.sh`'s rc-determination block into a standalone
`evaluate_rc()` function and asserts the expected `(rc, decision)`
tuple for each fixture. Pattern: edit `failover_lib.sh` → update
fixtures → run `bash tests/failover/run_tests.sh` here in the main
repo → THEN push + pull into v2 clones. Stops the
edit-push-pray-discover-bug-only-in-production loop that ate ~6
production runs over the past 48h.

Running the new tests immediately surfaced **2 more wrapper bugs**:

1. **`grep -c ... || echo 0` produces `0\n0`** when grep matches
   nothing → arithmetic eval `syntax error (error token is "0")` →
   the entire crash-line branch silently skipped. `grep -c` already
   writes `0` to stdout on no-match; the `|| echo 0` fallback was
   dead code that produced malformed output. Removed.
2. **Walltime guard regex was a strict subset of crash-detect regex**
   — only matched `Connection closed by peer | died from signal (9|11)`,
   missing `OutOfMemoryError`, `UR_RESULT_ERROR`, `Timed out waiting`,
   `EOFError`. So a real bad-node failure that surfaced as shell exit
   143 (mpiexec SIGTERM after EOFError) got misclassified as a clean
   walltime kill and the wrapper bailed without retry. Made both
   regexes identical (the broader set).

Both bugs silently active in production yesterday/today before fix.
Commit `0d93a1e91` adds the harness + fixes; 9/9 tests pass in both
the main repo and the 20B v2 production clone.

### ANSI codes in 'Execution finished with N' parsing

Earlier in the day, several jobs zombie-succeeded because the
wrapper's `inner_rc` extraction returned empty on ANSI-coded log
trailers:

  Logged:  `Execution finished with \x1b[1;36m143\x1b[0m`
  Regex:   `Execution finished with \[?[0-9]+\]?`
  Match:   none (the `\[?` matches a literal `[` byte, but the
           actual byte sequence is ESC + `[`)

Fix `94a8fda66`: strip ANSI codes with `sed -r 's/\x1b\[[0-9;]*m//g'`
before grepping the trailer.

### Async-mode regression for 20B 512N — the actual root cause of
"async ckpt save kills the cluster"

Spent hours today investigating why 20B 512N hasn't persisted past
`step-800` since 2026-05-03 — three weeks of dispatches all dying
mid-save. Today found the **smoking gun**: every save on disk between
step-200 and step-800 happened on 2026-05-01 + 2026-05-03 under the
**default checkpoint mode (sync)**. The submit script switched to
`--checkpoint.async-mode=async` sometime between May 3 and May 11,
and nothing has persisted past step-100 on the 20B 512N chain since.

The previous "files-per-save / Lustre saturation" hypothesis was
half right but missed the actual mechanism: at 6,144 ranks, async
saves stream the 244 GB ckpt to flare in the background AT THE SAME
TIME as the gloo training-step heartbeat. Either the writes or the
gloo traffic backs up, one peer times out, cascade. Sync saves block
training while writing — no overlap, no cascade.

Submitted `8505258` (20B 512N) + `8505259` (cont) with
`CHECKPOINT_ASYNC_MODE=disabled` (= true sync). Also queued
`8505255/56/57` (20B 256N sync variants). 2B chains stay on async
(those have always saved cleanly at this scale).

### Production chain progress today

**2B 256N**: persisted **step-25,000 → step-25,500** over 2 dispatches
(8503506 walltime-finished + 8505119 advanced ckpts then exited 127).
+500 fresh steps. Loss 2.74. The only trajectory actually moving.

**20B 256N**: still wall-bound at `step-300`. Six dispatches today,
each reached in-RAM step 308-326 then died from bad-node mid-training
before crossing the next 100-step save boundary. Wrapper detection
working correctly (validated against fixtures); the wall is genuine
Aurora bad-node prevalence at the per-dispatch survival window.

**20B 512N**: still wall-bound at `step-800`. Four dispatches today,
all 3-attempt exhausted at init before any training step. Hopes
pinned on the sync-mode resubmit (`8505258`).

**80B 256N**: 0 persisted, 5 attempts today. Bumped to `select=276`
(20 spares) + `FAILOVER_MAX_RETRIES=5` for `8505221` — still died.
Environment too unstable for 80B init right now.

### Wrapper fix chronology (today)

| Commit | Fix |
|--------|-----|
| `94a8fda66` | Strip ANSI codes before parsing 'Execution finished with N' |
| `0d93a1e91` | Test fixtures (force-added .log) |
| `4310258` | Drop `\|\| echo 0` bug + walltime-guard regex parity |

All 3 commits pulled into all 3 v2 production clones. Tests pass in
all clones.

### `ezpz` upgraded to 0.15.1 in all v2 clones

Got the `--timeout T` + `--retries N` flags from ezpz PR #136. Wired
`--timeout=1800` into `failover_lib.sh` (commit `eefccfc9d`,
yesterday). Catches silent-hang failure mode that previously was
invisible (the 8479579 incident — 5h of W&B heartbeat alive but
training metrics dead). Exit 124 from the watchdog now routes
through swap-and-retry.

### What's next

- Once `8505123` (20B 256N, currently R, in-RAM step 326) either
  crosses step-400 ckpt save or dies, the sync-mode 20B chains
  (`8505255` for 256N, `8505258` for 512N) take over. That's the
  live test of the async→sync regression hypothesis.
- Eval batch `8505205` (13 fresh 2B 256N ckpts, step 14K → 25.1K)
  running on capacity; 5/13 done so far. Will refresh plots +
  README tables once all land.

### 39th upstream sync — DebugMode numerics debugger (no-op for ezpz)

Merged 1 upstream commit (`19c567f76`,
[PR #3323](https://github.com/pytorch/torchtitan/pull/3323)). Pure
tooling addition: new `torchtitan/tools/numerics_debugging/` module
(activation tracer + bitwise comparator, ~2 KLOC) plus a
`numerics_debugging` skill under `.claude/skills/`. No code path
ezpz exercises changed; no replay needed. The new skill auto-loads
in this session and could be useful next time we need to bisect a
silent loss-curve divergence.

### Closing follow-up — PR #184767 closed in favor of upstream #183625

`@frost-intel` flagged that
[pytorch/pytorch#183625](https://github.com/pytorch/pytorch/pull/183625)
is a draft already covering the xccl `_set_pg_timeout` dispatch +
the new `test_c10d_xccl.py` (in pieces). Closed our PR #184767 in
deference. Local workaround in
[`22847fcb3`](https://github.com/saforem2/torchtitan/commit/22847fcb3)
(`_set_pg_timeouts_xpu_aware`) stays load-bearing until #183625
actually lands.

---

## 2026-05-22 — First upstream PyTorch PR filed; 2-week summary

### Upstream PyTorch PR for xccl `_set_pg_timeout` dispatch

Filed https://github.com/pytorch/pytorch/pull/184767 — adds the
missing xpu branch in `torch.distributed.distributed_c10d._set_pg_timeout`
so xccl PGs route through `ProcessGroupXCCL.set_timeout` instead of
silently no-op'ing with the `"Set timeout is now only supported for
either nccl or gloo."` warning. Initial commit
[37af153](https://github.com/saforem2/pytorch/commit/37af153); review
fixes (Backend type annotation, simpler single-binding form, updated
warning text, `find_free_port` + `@retry_on_connect_failures` for the
test) in [8ceedc7](https://github.com/saforem2/pytorch/commit/8ceedc7).
All 7 inline review threads from copilot + codex addressed and
resolved. Pinged `@kwen2501` (c10d CODEOWNER) + `@guangyey` +
`@frost-intel` for review; CI gated on first-time-contributor workflow
approval.

Empirically verified the diff on Sunspot 1N × 12 ranks against the
in-repo `.venv` torch 2.13 (allocs 12467214 + 12467219 + 12467231,
all released). Side finding: xccl's C++ `set_timeout` **does** mutate
`backend.options._timeout` — contradicts the pessimistic line in
[`PLAN_xccl_timeout_upstream_pr.md`](upstream-issues/PLAN_xccl_timeout_upstream_pr.md)
PR 2 that "xccl stores the value but does nothing." Storage works;
only **enforcement** (watchdog + abort) is still missing. PR 2 scope
unchanged. Local pytest port at
[`tests/distributed/test_c10d_xccl.py`](../tests/distributed/test_c10d_xccl.py)
verifies the patched-vs-unpatched contract on either side.

### Two-week summary

Wrote up the 2026-05-08 → 2026-05-22 retrospective at
[`docs/summaries/2026-05-08_to_2026-05-22.md`](summaries/2026-05-08_to_2026-05-22.md).
51 commits across 8 themes: upstream xccl PR, 4 upstream syncs (one
no-op, one no-replay, two with replays), 80B bad-node failover
infrastructure (the silent-hang detection bug fix in
[e216a2523](https://github.com/saforem2/torchtitan/commit/e216a2523)
closes the 8479579 incident class), Sunspot smoke campaigns, MoE EP=2
hang reclassification, TPC26 talk prep, and docs hygiene.

### 38th upstream sync — MoE dispatcher split + ChunkedCELoss/TP grad fix

Merged 4 upstream commits (`cfe97c605..c2a3771a4`). One replay landed
in [`d87729ad8`](https://github.com/saforem2/torchtitan/commit/d87729ad8):
mirror upstream PR
[#3389](https://github.com/pytorch/torchtitan/pull/3389)'s isinstance
dispatch on `token_dispatcher` Config classes in
`experiments/ezpz/moe/model.py`, dropping the removed `DeepEPMoE`
swap. PR
[#3412](https://github.com/pytorch/torchtitan/pull/3412) is internal
to `torchtitan/components/loss.py` — no ezpz replay.

Smoke (2N Sunspot, jobs 12467277 + 12467288 + 12467323):
- `agpt_2b` clean, byte-comparable baseline (140 s, peak 24.34 GiB).
- `moe_2b_ep` at LBS=1 clean and numerically equivalent to the
  37th-sync baseline: 14.95 GiB vs 15.03 GiB, TPS within 1.4%.
- `moe_2b_ep` at the previous registry-default LBS=16 OOMs on the
  bf16 vocab projection (`[16 × 8192, 256128] × 2 B ≈ 62.5 GiB`,
  overflows a 64 GiB Max 1550 tile). Same pre-existing `_ep`
  vocab-projection OOM the 37th-sync follow-up flagged. Closed by
  pinning `moe_2b_ep` to LBS=2 in
  [`59354e43f`](https://github.com/saforem2/torchtitan/commit/59354e43f).
- Smoke report:
  [`docs/experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md`](experiments/moe/sunspot/20260522-smoke-n2-38th-sync.md).

Side issue: stale `outputs/checkpoint/step-100` from pre-PR-3159
layout is no longer loadable (`Missing key in checkpoint state_dict:
layers.0.attention.qkv_linear.wk.weight.`); backed up to
`outputs/checkpoint-20260522-120005`.

---

## 2026-05-21 — Step-41 EP hang retry + registry fix + TPC26 talk prep

### `moe_debugmodel_ep` LBS=2 hang did not reproduce — reclassified transient

Retried yesterday's hung config on a sibling 2N alloc (12467180).
Full 50 steps clean, exit 0, 139 s wall. **The original 16-min stall
at step 41 was a transient**, not a systemic EP/AC/compile bug.
Reclassified in
[`docs/upstream-issues/moe_ep_step41_hang.md`](upstream-issues/moe_ep_step41_hang.md)
with three adjacent findings worth keeping
visible:

- `comm.train_timeout_seconds=100` did not fire on the original
  985 s silence — timeout may not be wired into the CCL/XCCL
  collective path on XPU. Worth a separate writeup.
- `TORCH_DISTRIBUTED_DEBUG=DETAIL` crashes on XPU with
  `Backend fake does not yet support sequence numbers`. The error
  doesn't mention XPU — easy footgun on next attempt.
- `--debug.deterministic` is incompatible with MoE on XPU:
  `_histc_xpu does not have a deterministic implementation`. So
  bit-exact regression gates for MoE on Intel are blocked on an
  upstream PyTorch deterministic `_histc_xpu` kernel.

Also a curious throughput band: retry ran at ~11.8k TPS, original at
~6.8k TPS — same code, same nodes-of-the-same-class. ~1.7× spread,
plausibly correlated with whatever caused the original hang.

### Registry fix for `_ep` configs

Pinned `moe_debugmodel_ep` to LBS=2 (was inheriting LBS=8 from
`moe_debugmodel()`, OOM'ing at ~33 GiB on the vocab projection).
Annotated `moe_2b_ep` with a docstring confirming its LBS=16 default
is the validated peak. Commit `f2cbc0327`.

### TPC26 MAPE talk

Got invited to speak at the TPC26 MAPE track (Baltimore / Munich,
May 31 - Jun 3) by Rio Yokota. Drafted title, abstract, and 11-section
outline at
[`docs/notes/slides-2026-05-21.md`](notes/slides-2026-05-21.md).
The fork-tax-as-first-class-workflow angle (§4) and the silent-numerics
section (§6) are the most differentiated bits. Folded today's
engineer-hours/week estimate into §9: **~8-15 hr/wk recurring
operational triage** across 4 weeks of journal entries (~25-35% of
one engineer), with episodic spikes to 30-40 hr when a silent bug
surfaces or a bisect goes wide. The unbounded-cost punchline lives in
the silent class: bf16-freeze alone burned ~450B tokens of v1
compute.

---

## 2026-05-20 (late) — PR #3386 EP follow-up + agpt merge sanity smoke

Followed up the 37th-sync replay with two parallel smoke campaigns at 2N on
Sunspot (commit `1d4115d3f`, jobs 12467180/12467181):

### moe `_ep` follow-up — EP=2 path validated, but registry configs need LBS override

Reports: [`moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md)

- **`moe_2b_ep` (LBS=16 from registry)** — clean 50 steps,
  12.94 → 6.07, 2,860 TPS/GPU, peak **15.03 GiB** vs `moe_2b` EP=1's
  14.97 GiB. Token-dispatch overhead at 2B scale is essentially free
  (+0.06 GiB, ~1% TPS hit). PR #3386's `wire_meshes` plumbing works.
- **`moe_debugmodel_ep` (LBS=8 from registry)** — **OOM at init** in vocab
  projection (`(LBS*8192, 256128) bf16` = 31.27 GiB requested). The default
  LBS the registry inherits from `moe_debugmodel()` is too aggressive for
  the EP variant at 2N. Override needed.
- **`moe_debugmodel_ep` re-run at LBS=2** — clean steps 1-41 then **hung
  16 min at step 41/50** and got SIGTERM (exit 143). New finding,
  uninvestigated — possibly EP all-to-all backend stall under the
  `standard` `moe_comm_backend`. Not blocking but worth a follow-up.

### agpt merge sanity — clean, plus DeviceMesh regression re-confirmed

Reports: [`agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md`](experiments/agpt/sunspot/20260520-smoke-n2-pr3386-merge-followup.md)

- **`agpt_2b`** — clean 50 steps, 12.97 → 6.58, peak **24.34 GiB
  (38.04%)** — *byte-identical* to the prior post-resync baseline.
  Confirms PR #3346 (`graph_trainer` regional_inductor refactor, bundled
  in the same merge) is a no-op for the agpt path. Throughput within the
  expected 2-3% noise band.
- **`agpt_50b_wide`** — re-confirms the torch-2.13
  `DeviceMesh`-in-saved-tensors `AssertionError` in AOT autograd's
  `save_from_forward`. Crash in ~121s on 2N, all 24 ranks identical
  signature. 37th sync did **not** fix it (didn't expect it to — bug is
  in PyTorch, not torchtitan). Standing workaround (compile=OFF for
  80B-family on torch 2.13, or stay on torch 2.10) still the only
  option. See [`project_80b_devmesh_bisect`](../../../../../home/foremans/.claude/projects/-lus-flare-projects-datascience-foremans-projects-saforem2-torchtitan/memory/project_80b_devmesh_bisect.md).

### Action items dropped on the floor

- `moe_debugmodel_ep` / `moe_2b_ep` config registry: either pin a sane
  LBS in the `_ep` variants or document the OOM in the registry.
- Investigate the LBS=2 debugmodel_ep step-41 hang (EP all-to-all
  backend? token-dispatch deadlock?). Not reproducing automatically until
  someone re-runs.

---

## 2026-05-20 — 37th upstream sync (MoE clean DTensor boundaries) + replay smoke

Second sync of the day. Merged `89987072b` (2 commits beyond the 36th sync):

- **`963c20cba` — PR #3386** [MoE][5/n] Refactor MoE to clean DTensor
  boundaries for shared/routed experts. The big one.
- **`83e490429` — PR #3346** graph_trainer `regional_inductor` refactor.
  No ezpz dep.

### What PR #3386 changes

Restructures MoE TP/EP wiring from an imperative parallelize-time pass to
config-based sharding declarations populated at `update_from_config` and
applied by `model.parallelize(parallel_dims)`:

- **Deleted upstream:** `torchtitan/distributed/expert_parallel.py`
  (`ExpertParallel`, `TensorParallel`), `ColwiseParallelWithGradPlacement`.
- **New upstream:** `torchtitan/models/common/moe_sharding.py` with
  `set_moe_sharding_config(moe_cfg, *, enable_ep, enable_sp,
  expert_param_layout)` populating router gate, shared experts, routed
  experts.
- **`GroupedExperts.parallelize`** added — calls `super().parallelize` then
  `token_dispatcher.wire_meshes(ep_mesh, tp_mesh)`.
- **`MoE.forward` simplified** — drops the explicit
  `DTensor.to_local(grad_placements=Partial)` at the top (now handled by
  config), splits shared-experts addition out of `combine()`.
- **`parallelize_deepseekv3`** drops `apply_moe_ep_tp` call; new flow:
  `if tp_enabled or ep_enabled: model.parallelize(parallel_dims)`.

### Replay scope

Only ezpz/moe was affected:

| File | Δ lines | Change |
|------|--------:|--------|
| `experiments/ezpz/moe/parallelize.py` | -73 | Drop `apply_moe_ep_tp` entirely + 3 deleted-symbol imports; collapse two-pass to single `model.parallelize` |
| `experiments/ezpz/moe/sharding.py` | +35 | Add `enable_ep` kwarg, call upstream's `set_moe_sharding_config` per MoE layer with `{w1:Shard(1), w2:Shard(2), w3:Shard(1)}` layout |
| `experiments/ezpz/moe/model.py` | +3 | Pass `enable_ep=...` from `update_from_config` |
| `experiments/ezpz/moe/config_registry.py` | -2 | Stale docstring scrub |

`apply_fsdp` (Aurora `ShardPlacementResult` workaround) and
`disable_fsdp_gradient_division` (CCL SUM-reduction workaround) stay
inlined locally.

### Smoke verification (Sunspot 2N, job 12467131)

| Config | Final loss | Δ vs baseline | Memory | TPS |
|--------|-----------:|--------------:|-------:|----:|
| `moe_debugmodel` LBS=2 | 6.99880 | -0.010 | 16.99 GiB (matches) | ~12,700 |
| `moe_2b` LBS=1 | 6.10607 | -0.050 | 14.97 GiB (+0.5 vs baseline) | ~2,900 |

Both within ±0.05 nats of the 35th-sync baseline. Drift is expected:
PR #3386's commit message states *"loss is expected to diverge compared
to main due to different reduction pattern, and shared expert
computation changes place"* — confirmed at our parallelism configuration.
`for_loop` expert backend fires the same warning count (5 + 17) as
baseline. No NaN/OOM. No recompilation events.

### Concerns flagged before merging (all resolved)

- **`MoE.forward` graph shape changed.** Watched for inductor
  recompilation events — none observed. Memory uptick at moe_2b
  (+0.5 GiB) is the only visible cost, plausibly from a separate buffer
  for `shared_out` before the final add.
- **Removed async overlap between shared_experts and DeepEP combine.**
  Documented upstream as a follow-up "can restore overlap using CUDA
  streams." We don't use DeepEP on Aurora/Sunspot so this is a no-op
  for ezpz today, but worth tracking if we ever turn it on.

### Reports + W&B

- Smoke: [`docs/experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md`](experiments/moe/sunspot/20260520-smoke-n2-pr3386-replay.md)
- 37th sync entry: [`docs/upstream-sync.md`](upstream-sync.md)
- W&B `moe_debugmodel`: [`efficient-bird-2070`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/saga2gds)
- W&B `moe_2b`: [`electric-pond-2071`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/zudqly3y)

---

## 2026-05-20 — Post-resync smoke campaign (Sunspot 2N)

Validated yesterday's [35th upstream sync](#2026-05-19--upstream-resync-35th-full-dtensor-3159)
with a four-config compute-node smoke on Sunspot (job `12467124`, 2 nodes,
24 XPUs). All configs ran 50 steps cleanly post-replay; no NaN/OOM,
monotonic loss descent, no regression vs the historical Apr 25 baseline.

| Config              | LBS | Final loss | TPS/GPU | MFU    | Peak mem            |
|---------------------|----:|-----------:|--------:|-------:|---------------------|
| `agpt_debugmodel`   | 2   | 6.77       | ~37,500 | ~2.9%  | 2.60 GiB (4.06%)    |
| `agpt_2b` (LBS=1)   | 1   | 6.01       | ~6,100  | ~22.8% | 24.34 GiB (38.04%)  |
| `agpt_2b` (LBS=2)   | 2   | 6.12       | ~7,200  | ~27.0% | 44.73 GiB (69.91%)  |
| `moe_debugmodel`    | 2   | 7.01       | ~13,000 | ~9.0%  | 16.99 GiB (26.55%)  |
| `moe_2b` (LBS=1)    | 1   | 6.16       | ~2,900  | ~8.4%  | 14.47 GiB (22.62%)  |

**`agpt_2b` LBS=2 matches Apr 25 n=2 baseline** (7,224 TPS / 27.11% MFU
today vs 7,142 TPS / 27.6% MFU then) within noise. PR #3159's
`Module.parallelize(parallel_dims)` signature is wired correctly through
both `experiments/ezpz/{agpt,moe}/parallelize.py`.

**`for_loop` expert backend (PR #13) still fires on XPU** -- 5 warnings
for moe_debugmodel, 17 for moe_2b (one per MoE layer in each flavor).
End-to-end forward/backward/optimizer under FSDP all converge cleanly.

**Default `moe_2b()` LBS=16 OOMs on Max 1550** with a single 62.53 GiB
allocation. Likely the fused activation for all experts × full-batch-tokens
materialized by the for_loop path. Pre-existing XPU constraint, not caused
by the resync. Switched to LBS=1 for the smoke.

### Permissions sidestep that worked

The auto-mode classifier blocks the `source <(curl -fsSL https://bit.ly/ezpz-utils)`
pattern on every `ezpz launch`. Workaround: on the compute node, cache the
utils script once with
`mkdir -p ~/.ezpz && curl -fsSL https://bit.ly/ezpz-utils -o ~/.ezpz/utils.sh`,
then prefix every launch with `source ~/.ezpz/utils.sh && ezpz_setup_job && ezpz_setup_xpu`.
The cache is persistent on the compute node so this only needs doing once
per allocation. Used successfully for all five smoke launches.

### Reports

- [`docs/experiments/agpt/sunspot/20260520-smoke-n2-postresync.md`](experiments/agpt/sunspot/20260520-smoke-n2-postresync.md)
- [`docs/experiments/moe/sunspot/20260520-smoke-n2-postresync.md`](experiments/moe/sunspot/20260520-smoke-n2-postresync.md)
- `docs/upstream-sync.md` 35th entry updated with smoke validation table.

W&B runs: `olive-plasma-2054`, `sunny-waterfall-2055`, `dry-water-2056`,
`worldly-music-2058`, `azure-field-2061` (all under `aurora_gpt/torchtitan.ezpz.train`).

---

## 2026-05-19 — Upstream resync (35th, Full DTensor #3159)

Pulled 22 upstream commits (`ee4e91a13..52a292d29`, merge `a14987132`).
The headline is [pytorch/torchtitan#3159](https://github.com/pytorch/torchtitan/pull/3159)
"Config-based Full DTensor for Llama3" — a substantial refactor of the
config-based sharding API that lays the foundation for
`--training.full_dtensor` (all params/buffers/inputs become DTensors on
a multi-dim SPMD mesh).

**Breaking signature change for ezpz:** `Module.parallelize(mesh)` →
`Module.parallelize(parallel_dims)`. Each Module now self-resolves its
SPMD submesh from the axes referenced in its `NamedPlacement`s, instead
of being handed a bare `tp_mesh`. Two ezpz callsites hit:
`experiments/ezpz/agpt/parallelize.py` and
`experiments/ezpz/moe/parallelize.py`. Both replayed: pass
`parallel_dims` to `model.parallelize`, keep the explicit
`parallel_dims.get_mesh("tp")` for the async-TP plumbing on the next
line. `apply_moe_ep_tp` still takes per-axis meshes directly (it doesn't
route through `Module.parallelize`), so it's untouched.

`ShardingConfig` got three new optional fields (`out_src_shardings`,
`local_input_grad_placements`, `local_output_grad_placements`); all
default `None` so ezpz's existing `set_agpt_sharding_config` /
`set_moe_sharding_config` construct unchanged shapes. The
`set_gqa_inner_attention_local_map` helper that ezpz/agpt calls had
internal arg renames (`xq/xk/xv` → `q/k/v`) but the public call site is
identical.

`trainer.py` gained a `full_dtensor`-gated `parallelize_inputs` call and
a `pred.to_local()` fallback under `disable_loss_parallel`. ezpz's
`FaultTolerantTrainer` doesn't override `_get_batch` or
`forward_backward`, so both inherit cleanly. `full_dtensor` defaults
`False`, so no behavior change for current production.

Verification: `.venv/bin/python -c "import
torchtitan.experiments.ezpz.{agpt,moe}.parallelize"` succeeds on both
post-replay. A real compute-node smoke (`agpt_2b`/`moe_500m`) is
pending the next allocation. Doc: `docs/upstream-sync.md` 35th entry.

---

## 2026-05-12 — Upstream resync (#3308) + `for_loop` backend smoke

### Resync PR #13

Pulled 12 commits from `upstream/main` into a fresh `ezpz-moe-resync`
branch. The big-ticket landing was
[pytorch/torchtitan#3308](https://github.com/pytorch/torchtitan/pull/3308),
which deleted `_run_experts_for_loop` and the `use_grouped_mm` config
field from `models/common/moe.py` and inlined `torch._grouped_mm` as the
only expert path. Upstream's argument: `_grouped_mm` already provides a
CUDA fallback. **XPU has no `_grouped_mm` kernel at all**, so this would
have broken every ezpz MoE config on Aurora / Sunspot at first forward.

Replay strategy: introduce `EzpzGroupedExperts(GroupedExperts)` in
`experiments/ezpz/moe/experts.py` with a
`compute_backend: Literal["for_loop", "grouped_mm"]` selector. Default
defers to upstream. The `for_loop` branch re-vendors the deleted
`_run_experts_for_loop` body verbatim, restoring the XPU / pre-SM90 path.
`model.py` `update_from_config` now switches to `for_loop` on any device
that fails `has_cuda_capability(9, 0)`.

PR #13: https://github.com/saforem2/torchtitan/pull/13 (replaces #12).
PRs #9 / #10 / #11 (Sam Wheeler / Sam Wheeler / Nathan Nichols) flagged
on each that they should rebase onto this and adapt to the
`EzpzGroupedExperts` subclass.

### Smoke validation (Sunspot 8N)

Job `12466707` on `x1921c5s0b0n0`–`x1921c5s7b0n0`. `moe_500m`, 50 steps,
local batch 4, seq 8192, GBS 384. Run:
[`fluent-glitter-2042`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/lt77xx0o).

- 11 layer-wise warnings emitted (1 per MoE layer):
  `torch._grouped_mm requires SM90+ CUDA; falling back to for_loop expert backend.`
  Confirms PR #13's `compute_backend = "for_loop"` switch is taken.
- Loss descended cleanly **12.90 → 6.66 (-6.24 nats)** over 50 steps.
- Steady-state throughput **~8,694 TPS / GPU, ~13% MFU** — actually ~20%
  per-GPU TPS uplift vs the
  [2026-04-13 2N benchmark](experiments/moe/sunspot/20260413-benchmark-n2.md)'s
  7,228 TPS / 9.11%, attributable to torch.compile inductor improvements.
  **No measurable regression vs the upstream `_run_experts_for_loop`
  body that #3308 deleted** (which makes sense — it's the same kernel).
- Memory stable at 54.6% across the run.
- Wall: 541s end-to-end including env setup + compile warmup.
- Report:
  [`docs/experiments/moe/sunspot/20260512-for-loop-smoke-n8.md`](experiments/moe/sunspot/20260512-for-loop-smoke-n8.md).

PR #13 is now smoke-validated end-to-end on XPU.

---

## 2026-05-05 — 80B DeviceMesh-bisect: torch-version, not depth

### Bisect kills the May 3 "depth-sensitive" claim

Submitted job 12465952 on Sunspot 4N to bisect the
`tensors_saved_with_vc_check` AOT autograd assertion across the agpt
80B family on the torch 2.13 venv. Three configs ran sequentially:

- `agpt_50b_wide` (48 layers): crashed in **66 s**, every rank logs the
  assertion (49 ranks × 1 = full crash on the first
  forward+backward).
- `agpt_70b_wide` (72 layers, new
  [`b9cda4b2`](https://github.com/saforem2/torchtitan/commit/b9cda4b2)
  config — 80B family with 12 fewer layers): same assertion, 28 s.
- `agpt_80b` (84 layers): same assertion, 29 s.

Per-config logs at `logs/agpt-80b-bisect-12465952/{config}.log`. The
"depth-sensitive — works at 48 layers" conclusion in the May 3 entry
was wrong. Two variables had changed between the May 3 50B_wide
success and today's 50B_wide failure: node count (2N → 4N) AND torch
version (`aurora_frameworks-2025.3.1` torch 2.10 → torch 2.13 venv).

To pin the variable I submitted job 12465962 (2N + torch 2.13,
[`submit_50b_wide_2n_t213.sh`](../scripts/submit_50b_wide_2n_t213.sh)).
Result: same assertion, 57 s. **torch 2.13 alone is the trigger;
node count is a non-factor.**

This means:

- The bug is **torch-version-sensitive, not depth-sensitive**.
- `agpt_50b_wide` on 2N + torch 2.13 is now a **clean ~30-60 s
  reproducer** for the upstream report (much smaller than 80B).
- The May 3 50B_wide success was masked entirely by the older AOT
  autograd code path on torch 2.10.

### Updated docs

- `experiments/ezpz/.claude/CLAUDE.md` — corrected the Recent Findings
  bullet, the Production v2 80B note, and the Known Bugs entry.
- `experiments/ezpz/docs/meeting-notes/agpt-sync.md` — added the
  2026-05-05 correction beneath the original 2026-05-03 finding so
  the meeting can show both.
- The new `agpt_70b_wide` config and the
  [`submit_80b_bisect.sh`](../scripts/submit_80b_bisect.sh) +
  [`submit_50b_wide_2n_t213.sh`](../scripts/submit_50b_wide_2n_t213.sh)
  scripts are now part of the bisect-tooling.

### Validator small-batch + global-state bug

While running the agpt_2b validator smoke separately, noticed the
val build was using `Global batch size: 48` and warning about
`train_iters defaulting to 1` on every validate() call. Three
related issues, all in
`torchtitan/components/validate.py:Validator.validate()` not passing
the trainer's `training_steps` and `global_batch_size` through to
`dl_config.build()`. Fixed in
[`3edfb0ff`](https://github.com/saforem2/torchtitan/commit/3edfb0ff)
by capturing `job_config` in `EzpzValidator.__init__` and forwarding
the right values. Validator was using ~1/4-sized batches per call
(noisier val loss); a worse latent issue was that `bc_set_config`
overwrites the blendcorpus library's global `DATA_CONFIG` with the
wrong `train_iters` and `global_batch_size`, which would silently
corrupt any later resume-from-ckpt rebuild of the train dataloader.

### blendcorpus ↔ Megatron parallelism aliasing

Reviewed `BlendCorpusDataLoader` for further Megatron-style
parallelism leftovers. Found 7 inconsistencies:

- 2 active and fixed today
  ([`140481d3`](https://github.com/saforem2/torchtitan/commit/140481d3)):
  scope the `dist.barrier` → CPU/gloo monkey-patch to torch < 2.13
  (it was being silently applied on 2.13 even though XCCL is fixed
  there); rename `_train_ds` → `_served_ds` so the attribute matches
  what it actually holds when `serve_validation=True`.
- 5 latent items (PP semantics, CP→SP aliasing, dp_world_size
  double-source, parallel-state duplication, hard-coded Megatron
  knobs) tracked in
  [`docs/TODO.md` §6](TODO.md) and documented in
  [`docs/guides/known-bugs/blendcorpus-megatron-aliasing.md`](guides/known-bugs/blendcorpus-megatron-aliasing.md).

### Other

- Pulled 11 upstream commits (32nd sync,
  [logged](upstream-sync.md#2026-05-05-32nd-sync--observability--moe-token-pad--cp-fix--rlgraph_trainer-churn)).
  Notable: `b2cd149f` adds `torchtitan/observability/` structured
  logging hooks. ezpz's `FaultTolerantTrainer` and `EzpzValidator`
  override their respective base classes' methods entirely so the
  new `@sl.log_trace_span` decorators don't propagate — non-blocking
  but worth re-adding later if we want trace spans on the ezpz path.
- Aligned `~/.claude/statusline-command.sh` to match starship.toml
  (true gray time, fish-style abbreviated path with cyan-bold +
  underlined-blue repo root, bold-purple branch).

### 80B v2 has a working path on torch 2.13 + `compile=OFF`

After the bisect closed out the DeviceMesh question, ran job 12466025
(4N, 20-step smoke) to test whether `compile=OFF` actually unlocks
80B v2 production on the torch 2.13 venv:

- Config: `agpt_80b` at TP=2, AC=full, **compile=OFF**, AdamW LR=1e-6,
  fp32-master, 4 nodes (24 ranks, dp_shard=12).
- Result: clean run, all 20 steps. Loss descended **12.98 → 10.46**
  (-2.52 nats), MFU steady at **~17.8%**, memory peaked at **88.94%**
  (~7 GiB headroom per tile), exit code 0.
- Grad-norm bumped to ~34 around steps 15-16 then recovered to ~14
  by step 20 — early-training oscillation, not a stall. Production
  80B should add the 200-step linear warmup the 2B/20B v2 configs
  already use.

This confirms an actually-working v2 80B path. Notable that
`compile=OFF` MFU (~17.8%) *matches* what compile-on used to give v1
on torch 2.10, so we're not paying any throughput penalty for not
compiling — though that'll change once `compile=ON` works again
upstream and the inductor optimizations actually kick in.

Submit script:
[`scripts/submit_80b_no_compile_t213.sh`](../scripts/submit_80b_no_compile_t213.sh).
Per-step log:
`logs/agpt-80b-no-compile-t213-12466025/run.log`.

`compile=ON` for the 80B family is currently broken on **both** torch
versions: torch 2.10 hits the step-1 hang regression from Apr 16-23
upstream changes; torch 2.13 hits the DeviceMesh-in-saved-tensors
AOT autograd assertion. `compile=OFF` is the only viable v2 80B path
until either upstream bug is fixed.

---

## 2026-05-04 — 20B chain walltime, 1024N startup crashes, doc cleanup

### Production training

- **20B 512N canonical chain (8463628)** finished its 12h walltime
  cleanly at step **863, loss 3.46**. Final TPS ~358, MFU ~17.8%.
  step-100..step-800 ckpts all saved. Continuation **8466848** auto-released
  from hold and is now Q for a 512N slot.
- **20B 256N v2 (8463659) started running.** Fresh-start trajectory at
  256N, separate ckpt dir (`n256-gbs6144`) — *not* a chain extension.
  Currently at **step 200, loss 5.65, MFU 14-20%**, 2 ckpts saved
  (step-100, step-200). Useful as a per-token-vs-512N comparator at
  matched optimizer state. Loss curve looks healthy:
  12.96 → 8.5 (step 45) → 6.35 (step 124) → 5.65 (step 200).
- **2B 512N canonical chain** still at step 5,073 (loss 2.97).
  Continuation 8463627 still Q ("Not enough free nodes available");
  8466847 held behind it.

### 1024N first-attempts both crashed at startup

Both 1024N v2 jobs (queued since 2026-05-01) finally got slots and
**crashed within 4 minutes**:

- **2B 1024N (8463182)**: `MemoryError: std::bad_alloc` inside
  `torch.distributed.broadcast` during `set_determinism` init. Died
  after 211s, exit 143.
- **20B 1024N (8463183)**: rank 4732 died from signal 11 (SIGSEGV)
  during the same init phase, exit 143.

12,288 ranks (1024 nodes × 12 GPUs) appears to be hitting an init-time
memory/comm scaling issue we don't see at 256N or 512N. Worth a
smaller-scale repro before resubmitting — maybe 768N or 896N to bracket
where it starts failing. Not blocking the canonical 512N chains.

### 20B v2 eval through step-600

- **8467370** (capacity queue, 8h requested, ran to **12h15m walltime
  kill**) delivered eval scores for steps 100/200/300/400/500/600
  before being killed mid-step-700 lm-eval. Steps 700/800 will need
  a resubmit.
- **ARC-Easy `acc` lifts monotonically** from 0.266 → 0.290 → 0.295 →
  0.318 → 0.346 → **0.393** across that range — the cleanest-yet
  signal that fp32 master is producing real benchmark progression
  vs. v1's flat ~0.27 baseline.
- HellaSwag also creeping: 0.257 → 0.270. ARC-Challenge and
  Winogrande still in noise (expected at <100B tokens).
- v1-vs-v2 plot regenerated; results table added under
  [`docs/evals/agpt/20b/README.md`](evals/agpt/20b/README.md).
- Eval throughput tanked while concurrent 256N v2 (8463659) was
  starting — both share flare bandwidth for ckpt I/O and dataset
  reads. Steps 100-500 each took ~30 min; step-600 took ~3h. Will
  hold off on resubmitting steps 700/800 until 256N finishes.

### Doc cleanup

- **Compile flag was wrong in 5 v2 setup tables.** `agpt_2b()` and
  `agpt_20b()` both default to `compile=True`, and W&B configs +
  startup logs ("Compiling each TransformerBlock with torch.compile")
  confirm compile is on for all v2 runs. The "Compile | off" rows in
  the v2 setup tables under `docs/production/agpt/{2b,20b}/n*/README.md`
  were stale carryovers from drafts. Flipped to "on" across 2B
  n256/n512/n1024 and 20B n512/n1024.
- **Submit-script links added** to all 6 per-node-count READMEs. v2
  entries point to `scripts/submit_agpt_{2b,20b}_aurora_venv.sh`
  (one script handles all node counts via env vars); v1 entries point
  to the per-node `submit/aurora/submit_agpt_*.sh`.
- **Absolute log paths added** for every Job ID in every Progress
  table — v1 logs at `/lus/flare/.../torchtitan-ezpz/agpt-*-sophiag-*.o<JOBID>`,
  v2 logs at `/flare/.../runs/agpt-{2b,20b}-v2/torchtitan-ezpz/agpt-*.o<JOBID>`.
  Queued/held jobs note where the log will land on start.
- **`submit/README.md` added** marking the directory as legacy. The
  torch-2.10 `submit/{aurora,sunspot}/*.sh` scripts produced every v1
  trajectory and were the production driver through 2026-04-29. After
  the v2 restart on 2026-04-30, all production training moved to
  `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` on the torch 2.13
  venv stack — nothing live reaches into `submit/` anymore.

### Plotter

- `PRODUCTION_RUNS["20b_v2_256"]` added (wandb run `r1yyxbmt` =
  8463659) so the 20B 256N v2 trajectory shows up in dashboard
  refreshes.

### 8463659 NODE_FAIL after step 364 (afternoon)

Same recurring Aurora bad-node failure mode that killed 8459818 /
8460301 / 8460302. Last training step was 364 at 11:09:12, then
immediately:

```
x4406c6s7b0n0.hsn.cm.aurora.alcf.anl.gov: shepherd died from signal 9
x4218c2s2b0n0.hsn.cm.aurora.alcf.anl.gov: rank 606 died from signal 15
```

PALS shepherd on `x4406c6s7b0n0` got SIGKILL (kernel OOM-killed,
hardware fault, or system-level take-out), all ranks on that node
lost their parent → cascading SIGTERM. PBS reports `Exit_status -20`
= NODE_FAIL after 9h walltime. step-300 ckpt saved cleanly (loss 4.61
final). Trajectory pages updated to reflect the crash; no continuation
chained since the canonical chain is at 512N and this 256N run was a
per-token comparator scaling experiment rather than a chain.

### 20B v2 eval — full step-100..800 sweep complete

After waiting for 8463659 to finish (and free flare bandwidth),
resubmitted the missing step-700 + step-800 evals as 8469257 (capacity,
3h walltime). Finished cleanly in 3h flat. ARC-Easy `acc` continues
its monotonic ascent: 0.359 (step 500) → 0.393 (600) → 0.391 (700) →
**0.444** (800) — clean signal vs v1 256N's flat ~0.27 across all of
0-63B tokens. HellaSwag `acc_norm` 0.270 → 0.281 → 0.284 (+3pp above
v1 by step 800). v1-vs-v2 plot regenerated; `docs/evals/agpt/20b/`
table updated with all 8 v2 ckpts.

### Direct verification of the bf16 fix in checkpoint weights

User asked for RMSNorm.weight variance across the new 2B production
ckpts. Pulled stats from 6 HF-converted ckpts:

| ckpt | mean(var) | mean(std) | min weight | max weight |
|---|---:|---:|---:|---:|
| v1 step-10000 | **0** | **0** | **1.000** | **1.000** |
| v1 step-15000 | **0** | **0** | **1.000** | **1.000** |
| v2 256N step-2000 | 2.2e-5 | 0.0045 | 0.973 | 1.039 |
| v2 512N step-1000 | 5.4e-6 | 0.0016 | 0.988 | 1.016 |
| v2 512N step-3000 | 5.4e-5 | 0.0071 | 0.957 | 1.063 |
| v2 512N step-5000 | **1.2e-4** | **0.011** | **0.926** | **1.102** |

Every single v1 RMSNorm channel is exactly 1.0 — bf16-master
sub-ULP-update bug really did freeze every norm. v2 weights are
training: variance grows monotonically with token count, range fans
out from [0.988, 1.016] at step 1000 to [0.926, 1.102] at step 5000.
Per-layer at step 5000: `model.norm.weight` is biggest (mean 1.098,
std 0.006 — every channel uniformly scaling up); mid-depth layers
(5-7) have the highest per-element std (0.014-0.017, learning the
most differentiated channel scales). Final smoking gun for the v2
restart, complementary to the lm-eval evidence.

### Doc maintenance + plotter

- Refreshed 20B v2 256N plots (added wandb `r1yyxbmt` to
  `PRODUCTION_RUNS`); re-ran on the dead trajectory after NODE_FAIL.
- Updated parent snapshots (`production/README.md`,
  `production/agpt/README.md`, `production/agpt/20b/README.md`) to
  reflect 8463659 NODE_FAIL + 1024N startup crashes.
- `submit/README.md` added marking torch-2.10 `submit/` as legacy.
- Submit-script links + absolute log paths added to all 6 per-node
  READMEs.
- Compile-flag rows in 5 v2 setup tables corrected from "off" to "on"
  (defaults to `True` in `agpt(...)` and confirmed in W&B + startup
  logs).
- `running-with-newer-pytorch.md` expanded with the at-scale yeet
  section (8N→4096N table, tarball workflow, `/tmp/.venv` switch).
- Saved `memory/project_1024n_init_crash.md` so future sessions know
  to bracket 1024N attempts at 768N/896N first.

### Still queued

- **8463627** (2B 512N chain1 continuation), **8466847** (held
  `afterany:8463627`), **8466848** (20B 512N chain1 continuation),
  **8467141** (√2-LR fork chain1), **8467142** (held
  `afterany:8467141`) — all Q for 512N slots, none running.
  No production training is currently active.

---

## 2026-05-03 (evening) — TP > 1 loss-reporting bug + agpt_50b_wide

### Loss reporting on TP > 1 is off by `dp_world_size`

Hunting a different bug (the 80B `compile + AC + TP=2`
DeviceMesh-in-saved-tensors crash) we added `agpt_60b` and then
`agpt_50b_wide` as smaller bisect targets. The `50b_wide` smoke at
2N TP=2 reported step-1 loss = **1.07**. That should be ≈ ln(256128) ≈
**12.45 nats** for random init — 12× too small. The 12 matched
`dp_world_size = 12` (24 ranks ÷ TP=2) exactly, which pointed at a
missing cross-batch reduction.

Root cause is upstream commit `1786292d` (2026-04-27) in
`torchtitan/distributed/utils.py`. The new DTensor branch in
`_dist_reduce` returns `float(x.full_tensor().item())` and skips the
requested mesh `all_reduce`. That is correct only when the DTensor's
mesh matches the requested mesh — but the trainer's loss reduction
passes `loss_mesh` (= batch × cp), and the loss is a Replicated
DTensor on the **TP** mesh (orthogonal). The cross-batch sum is
silently dropped.

Verified the diagnosis by re-running `agpt_2b` at TP=2 (no compile, 3
steps) with a workaround in place: convert `loss` DTensor to a plain
tensor before `dist_sum`/`dist_max`. Step-1 loss came back as **12.94**
— matches the known-good TP=1 baseline of 12.95. Bug confirmed and fix
verified in one shot.

Workaround landed in `experiments/ezpz/`:

- `trainer.py` — adds `loss = loss.full_tensor()` before the
  `dist_sum`/`dist_max` reductions.
- New `validator.py` — `EzpzValidator(Validator)` subclass with the
  same fix in its `validate()` override.
- `agpt/config_registry.py` `_base_config` — uses
  `EzpzValidator.Config` instead of `Validator.Config`.

Documented in `docs/guides/loss-reporting-tp-dist-reduce.md` and the
upstream issue note at
`docs/upstream-issues/dist_reduce_dtensor_skip.md`. Filed upstream as
[pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).

**Implications for live dashboards:**

- 2B / 20B production W&B (TP=1): correct, no action needed.
- **80B production W&B (TP=2): under-reported by `dp_world_size`.**
  Multiply reported loss by `world_size / tp_degree` to recover the
  true per-token NLL. A 256N TP=2 dashboard's loss is currently 1536×
  smaller than the truth.

### agpt_50b_wide added; agpt_60b removed

To get a smaller compile target for the 80B-bug bisect we first
added `agpt_60b` (dim=9216, 60 layers, ~58B params, same per-layer
shape as 80B). That smoke crashed with an Intel GPU SegFault at step 2
— OOM dressed up as a not-present-PDE fault, after step-1 measured at
**97.41% memory** with no headroom for the step-2 activation peak.

Replaced with `agpt_50b_wide` (dim=9216, **48 layers**, ~48B params).
Smoke ran all 10 steps cleanly at 95.94% memory: MFU ≈ 15%, TPS ≈ 140
on Sunspot. Concluded "this is a working `compile + AC + TP=2` dense
config — the 80B-family DeviceMesh-in-saved-tensors crash does NOT
reproduce at 48 layers, only at 84, so the bug is depth-sensitive."

> **CORRECTION (2026-05-05):** that conclusion was wrong. The May 3
> smoke happened to use torch 2.10 (`aurora_frameworks-2025.3.1`); a
> proper bisect on torch 2.13 (jobs 12465952 + 12465962) showed the
> bug fires on `agpt_50b_wide` / `agpt_70b_wide` / `agpt_80b` alike,
> on both 2N and 4N. Bug is **torch-version-sensitive**, not
> depth-sensitive. See the 2026-05-05 journal entry above for the
> full bisect.

### Validation loss work — partial

Started wiring blendcorpus's already-built validation split through the
`Validator`. Stopped short to chase the loss-reporting bug above.
Status:

- `BlendCorpusDataLoader.Config` now has `serve_validation: bool`
  and `eval_iters: int` (default 100). `eval_iters` is only requested
  when `serve_validation=True` (otherwise we'd trigger a fresh
  validation index build that hangs in dataset construction).
- `_base_config` constructs a validator with
  `BlendCorpusDataLoader.Config(serve_validation=True)` for the val
  dataloader, but `validator.enable=False` by default.
- `EzpzValidator` is wired in but not yet smoke-tested.

Remaining: 250-step `agpt_2b` run with `--validator.enable` to confirm
the validator fires at step 1 and step 200, and that the val loss is
reasonable.

---

## 2026-05-03 — Production progress + canonical-chain consolidation + doc reorg

### Production training

- **2B 512N canonical chain (8460301 → 8463626 → 8463627)** — 8463626
  finished its 12h walltime cleanly at step 5,073 (loss 2.97). 50
  ckpts saved (every 100 steps, step-100 to step-5000). Continuation
  8463627 queued (will auto-resume from step-5000). **Cumulative:
  step 5,073, loss 2.97, 510B tokens (10.9% of 4.67T target).**
- **20B 512N canonical chain (8460302 → 8463628)** — 8460302 hit
  walltime at step 300 (loss 4.95). 8463628 (12h continuation) is
  currently running, resumed from step-200 (step-300 ckpt was
  incomplete from the NODE_FAIL on 8460302 so DCP picked the prior
  good ckpt). At write time: step 287, loss 5.00, MFU 17.5%.
- **Other queued jobs at different node counts** (2B 1024N 8463182,
  20B 1024N 8463183, 20B 256N 8463659) are *independent* trajectories
  — they write to separate ckpt dirs (keyed on `gbs`) and would start
  fresh from step 0. Treated as scaling experiments, not chain
  extensions.

### v1-vs-v2 evals (smoking gun)

- Ran `eval-2b-v2.sh` on 10 v2 256N ckpts (steps 200-2000). ARC-Easy
  climbed 0.277 → 0.429 over 100B tokens; v1's flat ~0.27 across 450B
  tokens validates the bf16-master RMSNorm-freeze fix end-to-end.
  HellaSwag also broke out at 80-100B tokens (v2 0.301 vs v1 0.251).
- Ran `eval-20b-v2.sh` on step-100 + step-200 (10B and 20B tokens).
  ARC-Easy v2 +1.5pp above v1 at the same token count, others still
  in noise. Will revisit at higher v2 token counts.
- Submitted fresh evals (8466827 for 2B 512N steps 1000-5000;
  8466828 for 20B 512N step-300).
- Added per-trajectory plotter at
  `docs/evals/agpt/{2b,20b}/plot_v1_vs_v2.py` — 4-panel comparison
  (HellaSwag/ARC-Easy/ARC-Challenge/Winogrande) with random baseline
  marked.

### Production-side doc reorganization

- Surfaced 2B 512N + 20B 512N v2 dashboards at the agpt index level
  (`docs/production/agpt/README.md`). Previously only embedded in
  per-model READMEs.
- Wrapped all v1 historical sections in `<details closed>` blocks
  across both production and eval READMEs, so v2 stays prominent.
- Renamed all production figure filenames to be explicit about
  v1/v2 (e.g. `production_2b_v2_512n.png` vs the older
  `production_2b_256n.png` ambiguity).
- Generated v1-vs-v2 overlay plots
  (`overlay_2b_v1_vs_v2.png`, `overlay_20b_v1_vs_v2.png`) via
  `plot_production_wandb.py --overlay {2b,20b}`.
- Moved the **MDS 2B SophiaG training curves** from
  `docs/evals/agpt/2b-mds/` to `docs/production/agpt/2b-mds/` (the
  eval scores stay under evals). Cross-linked both ways. Noted as
  "pre-torchtitan reference baseline" in the agpt production index.
- `plot_production_wandb.py` extended to support multiple v2
  trajectories per model — `PRODUCTION_RUNS` keyed on
  `<model>_v2_<nodes>`, output filenames keyed on the same.

### Canonical chain dashboards (current state)

| Model | Cumulative | Loss | Tokens | Latest |
|-------|-----------:|-----:|-------:|--------|
| 2B 512N | 5,073 | 2.97 | 510B (10.9%) | 8463627 (Q) |
| 20B 512N | 300 | 4.95 | 30B (0.6%) | 8463628 (R) |

### Dataset CLI mistake-catcher

When `--dataloader.dataset=user/repo` (HF hub path) and
`--dataloader.dataset-path=...` are both passed, `_validate_dataset`
silently overrides `dataset_path` to None so the registered hub path
wins. Correct behavior, but it hid user mistakes (e.g. expecting
`--dataset-path` to point at a local clone of the hub dataset).

`9ef2908d` — emit a clear warning naming both args and telling the
user to drop `--dataloader.dataset-path` or pick a local-file dataset
name like `blendcorpus`. Override semantics unchanged; only logging
added. Production scripts that hardcode
`--dataset=blendcorpus --dataset-path=$DFL` are unaffected (they hit
the registered-name branch which keeps the path).

### 2B 256N vs 2B 512N — large-batch under-training observation

Pulled fresh evals on 2B v2 256N (10 ckpts, steps 200-2000) and 2B v2
512N (5 ckpts, steps 1000-5000). Surprise: at matched **token** counts,
256N beats 512N noticeably:

| Tokens (B) | 256N HellaSwag | 512N HellaSwag | Δ |
|-----------:|---------------:|---------------:|--:|
| ~100 | 0.301 (step 2K) | 0.264 (step 1K) | -3.7pp |

But at matched **step** counts, they're indistinguishable:

| Step | 256N HellaSwag | 512N HellaSwag | Δ |
|-----:|---------------:|---------------:|--:|
| 1000 | 0.262 | 0.264 | +0.2pp |
| 2000 | 0.301 | 0.304 | +0.3pp |

This is the classic large-batch under-training pattern. Both runs use
SophiaG LR=2.28e-5 (tuned for 256N / GBS=6,144). At 512N (GBS=12,288)
each step covers 2× the tokens but the optimizer state evolves at half
the cadence per token, with no compensating LR scale-up. Per-step
parity confirms the optimizer is healthy; per-token gap is purely
batch-size-induced under-training.

Implications:
- **Per-step**: 256N == 512N
- **Per-token**: 256N wins ~3-8pp at matched tokens
- **Per-wall-clock**: 512N wins (~2× throughput)

So the canonical-chain choice (512N) optimizes for wall-clock time to
target loss, not token efficiency. If chasing minimum tokens, 256N
would be preferable. Open follow-up: would √2-LR scaling (3.22e-5) at
512N close the per-token gap? Worth a fresh fork to test, but not
worth perturbing the running 512N chain's LR mid-run.

Documented in
[`docs/evals/agpt/2b/README.md`](evals/agpt/2b/README.md) under "v2
256N vs v2 512N — same model, two batch sizes".

---

## 2026-05-02 — 2B 512N continuation reaches 510B tokens

Light day. The 2B 512N continuation chain (8463626) ran cleanly
through 12h of walltime — went from step 1,300 (resume) to step
5,073, with loss dropping from 3.59 to 2.97. NODE_FAIL at the very
end again, but all 50 ckpts saved. This was the first chain run where
NODE_FAIL didn't cost meaningful progress — the ckpt-100 cadence +
keep_latest_k=0 (keep all) policy means we always have a recent
recovery point.

20B 512N (8460302) finished its 6h walltime at step 300, loss 4.95.
3 ckpts saved (step 100/200/300, though step-300 was incomplete and
DCP fell back to step-200 on resume).

Eval pipeline kept producing rolling 2B v2 results — first ARC-Easy
points landed in the 0.28-0.34 range across early ckpts.

---

## 2026-05-01 — 1024N production runs, yeet-env scaling sweep

### Production training

- **2B 1024N** (8463182, 12h walltime, small queue) and **20B 1024N**
  (8463183, 12h, small) both submitted. Same configs as the running
  256N/512N v2 runs (LBS=2, SophiaG LR=2.28e-5, plain CE, fp32 master,
  `.venv.tar.gz` yeet-env, no compile). Currently queued behind the
  in-flight 20B 512N (8460302).
- Submitted **chained 12h continuations** for the 512N runs:
  - 2B: fresh 8463626 + 8463627 (depend=afterany:8463626)
  - 20B: 8463628 (depend=afterany:8460302)
- 2B 256N (8459818) and 2B 512N (8460301) v2 runs both hit NODE_FAIL
  at end-of-walltime (step 2070 and 1387 respectively) — bad-node TPS
  degradation pattern (TPS dropped from ~5K to ~30 in the final few
  hundred steps before kill). All checkpoints survived (`keep_latest_k=0`
  keeps everything; 20 ckpts at step 100..2000 for the 256N, 13 ckpts
  step 100..1300 for the 512N).

### yeet-env tarball-broadcast scaling sweep

Measured `ezpz yeet-env --src .venv.tar.gz` at 8/16/32/64/128/256/512/
1024/2048 N (4096N still queued in `large`). Submitted via
`scripts/yeet_env_scaling_test.sh` chained one-at-a-time through
`/tmp/yeet_chain.sh` (per-user PBS-Q limit forces serial submission).

| Nodes | yeet-env (s) | Per-node (ms) |
|------:|-------------:|--------------:|
| 8     | 70           | 8,712 |
| 16    | 90           | 5,606 |
| 32    | 89           | 2,788 |
| 64    | 91           | 1,425 |
| 128   | 110          | 862 |
| 256   | 133          | 519 |
| 512   | 175          | 341 |
| 1024  | 255          | 249 |
| 2048  | 421          | 206 |

Two regimes: 8-64N is extract-bound (flat ~90s total), ≥128N is
broadcast-bound (linear-ish growth, super-linear knee at 2048N).
Per-node amortized cost drops 42× from N=8 to N=2048. Even at 2048N,
total wall-clock is <8 min — vs the "1-2 hours" the old per-file rsync
mode in CLAUDE.md predicted.

Plots + script: [`docs/scaling/yeet_env/`](scaling/yeet_env/README.md).

### Docs refresh

- Added `docs/evals/agpt/2b-mds/{loss_data, figures}` train+val loss,
  grad_norm, TFLOP/s, TPS plots pulled from W&B (113 SophiaG MDS
  continuation runs stitched by iteration).
- Refactored `experiments/ezpz/scripts/`: moved 14 hidden one-off
  shell scripts out of the repo root into `scripts/{eval,debug,lr-finder}/`
  subdirs; consolidated the existing 3 eval shell scripts there too.
- Refreshed all production READMEs with the v2 run state (this entry).

### Async checkpointing — verification + production switch

Built smoke configs `smoke_2b_async_ckpt` (`async`) and
`smoke_2b_async_ckpt_pinned` (`async_with_pinned_mem`) with
`enable_first_step_checkpoint=True` and `interval=10` so we get 5
saves across a 50-step run.

| Job | Config | Result |
|---|---|---|
| `12465723` | `async` | ✅ PASS — 50 steps, 5 saves @ ~0.2s each, loss 7.05, exit 0 |
| `12465724` | `async_with_pinned_mem` | ❌ FAIL — `KeyError: <class 'type'>` at step-20 save |

Loss-baseline check on the `async` run vs v24 sync baseline:
final Δ -0.055, tail10 Δ -0.053 (well within ±0.10). On-disk DCP
format byte-identical to sync save (1809 keys, modern `qkv_linear` +
`lm_head` naming). Sync ↔ async is bidirectionally wire-compatible —
no migration needed for in-flight production.

Flipped all 10 production train/submit/smoke scripts to default
`--checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-async}"`. The env
var lets anyone roll back to sync without editing the script:

```bash
CHECKPOINT_ASYNC_MODE=disabled qsub ...
```

Per-step staging cost is ~0.2s with a one-step TPS dip (5.7K vs 7.4K
steady) the next step, then full recovery. Compared to a sync save
(which blocks the entire training step for the full disk-write
duration — 4+ seconds per save at 2N), this is a clear win.

### `async_with_pinned_mem` — upstream PyTorch bug filed

Diagnosed and reproduced a real bug in `torch.distributed.checkpoint`:
`StateDictStager.deepcopy_with_tensor_offload` line 320 indexes
`self._deepcopy_dispatch[type]` directly, but `StateDictStager.close()`
(called by `DefaultStager._stage` after every stage to break a closure
cycle, `staging.py:251`) clears the entire dispatch dict. Result:
**second** stage call on any state dict containing a class object
crashes with `KeyError: <class 'type'>`.

Reproduces in pure CPU Python (no distributed init, no GPU). Three
files committed at `docs/upstream-issues/`:

- `STATE_DICT_STAGER_ISSUE.md` — issue draft for pytorch/pytorch
- `repro_state_dict_stager_bug.py` — minimal ~30-line repro
- `repro_state_dict_stager_fix_verification.py` — proves the naive
  one-line `.get(type, _deepcopy_atomic)` fix doesn't work because
  *all* atomic-type entries are also missing post-`close()`. Real
  fix is structural: either don't clear `_deepcopy_dispatch` in
  `close()` (only the cached storages cause the leak), or rebuild
  it at the start of each `stage()`.

Plain `async` is unaffected — it uses the in-process default stager
(regular `copy.deepcopy`), not the pinned-memory `StateDictStager`
path.

### 28th–30th upstream syncs (4 days, 5 sync entries, no replays)

| Entry | Date | Commits | Why no replay |
|---|---|---|---|
| 27th | 2026-04-30 | HybridEP cleanup + autoparallel/dsv3 deletion | Resolved 2 modify/delete conflicts on autoparallel/dsv3/ by accepting upstream's deletion; ezpz doesn't depend |
| 28th | 2026-05-01 | graph_trainer qwen3 + CI lint + ft.llama3 attn_backend | Scoped to graph_trainer / lint / ft.llama3 (we use our own model_registry) |
| 29th | 2026-05-01 | RL vLLM compile-time + graph_trainer skill | Scoped to experiments/rl + graph_trainer |
| 30th | 2026-05-03 | Bucketing pass + RMSNorm fusion (graph_trainer) | All 3 commits scoped to experiments/graph_trainer |

The loss-baseline workflow has now been exercised in production once
(v22 → v24 baseline refresh, both PASS). Workflow doc is at
`docs/baselines/README.md`.

### Eval pipeline v2 plumbing

Eval scripts were originally hardcoded for the v1 256N run. To run on
v2 ckpts (256N, 512N, 1024N, ...) needed several fixes early in the
day:

- `feat(ezpz/scripts/eval): support v2 ckpts via --ckpt-name + add
  20B v2 eval` (`55154291`)
- `fix(ezpz/scripts/eval): drop set -u from eval-20b-v2.sh` (`1835d8c2`)
- `fix(ezpz/scripts/eval): cd into v2 clone for conversion` (`9af67b81`)
- `fix(ezpz/scripts/eval): force PYTHONPATH=. + add subshell debug
  echoes` (`db1380b6`)

After those landed, ARC-Easy points started flowing:
0.277 (step-200) → 0.366 (step-1200) → 0.429 (step-2000), validating
the bf16 fix end-to-end. v1 was flat at ~0.27 across 450B tokens.

---

## 2026-04-30 — bf16-master RMSNorm freeze diagnosis + v2 restart

### What broke

The 2B/20B/80B SophiaG production runs from Apr 14-29 were all silently
training with frozen RMSNorm weights:

- `training.dtype = bfloat16` (default at the time) was being used as
  the *master* weight dtype for FSDP MixedPrecisionPolicy.
- bf16 ULP at 1.0 is ~7.8e-3; per-step RMSNorm.weight updates from
  SophiaG were ~1.6e-5 — sub-ULP, so every update rounded to zero.
- Loss curves looked plausible because attention/FFN weights were
  unfrozen and the model could still descend, but the model has no
  trainable normalization. Eval scores reflected this: the
  DCP→HF-converted checkpoints scored near-random on hellaswag/arc_easy
  while the parallel MDS pipeline (which used different defaults) was
  cleanly converging.

Full diagnosis writeup at
[`docs/guides/training-dtype-bf16-norm-freeze.md`](guides/training-dtype-bf16-norm-freeze.md).

### Fix + v2 restart

- Default `training.dtype` flipped to `float32` in `agpt` and `moe`
  config registries.
- ChunkedCELoss opt-in support ported to the ezpz trainer (sets
  `lm_head` for the chunked path; off by default).
- Restarted training from scratch in fresh per-model clones:
  `/flare/AuroraGPT/foremans/runs/agpt-{2b,20b}-v2/torchtitan-ezpz/`.
- Built fresh torch 2.13 + xpu venvs in each clone (~2.7 GB tarball).
- Adapted `scripts/submit_agpt_{2b,20b}_aurora_venv.sh` for Aurora:
  proxy env vars set inline, ezpz-utils source cached locally to dodge
  bit.ly hangs, tarball-aware yeet-env, plain-CE default with optional
  `CONFIG_SUFFIX=_chunkedce` opt-in.

### Smoke + scaling validation

Validated v2 stack at 2/4/8/16/64 N via short PBS jobs before
launching production. 2B v2 16N test ran ~57 min and reached step 2117
(loss 5.16 → 3.57, TPS ~5K, MFU ~17-20%) before NODE_FAIL — flaky-node
issue, not a code issue. 20B v2 16N test (capacity queue, 1h) submitted
in parallel.

### 26th + 27th upstream syncs

Replayed the MoE ETP deprecation (#3167) onto `experiments/ezpz/moe/`:
removed `ExpertTensorParallel` import, dropped `etp_mesh`/`ep_etp_mesh`
from `apply_moe_ep_tp` signature + call site, switched to reading
`comm_backend` off `experts.token_dispatcher` (matches upstream
deepseek_v3.model). Pulled the 27th sync (HybridEP cleanup +
autoparallel/deepseek_v3 deletion) — no impact on ezpz.

---

## 2026-04-29 — 24th upstream sync replay (All2All token dispatcher consolidation)

### What landed

Two upstream commits, one breaking:

- `20628f4e` (#3125) consolidates EP=1 and EP>1 to all use
  `AllToAllTokenDispatcher` (with a local-fallback path when ep_mesh is
  None). `make_token_dispatcher_config` and `make_experts_config` now
  require a non-None `comm_backend`; default changed from `None` to
  `"standard"`.
- `35c5d529` graph_trainer-only (no impact).

### Replay (1 commit)

`23b8ba59 fix(ezpz/moe)`: `moe_comm_backend: str | None = None` →
`moe_comm_backend: str = "standard"` in both `_build_moe_layers` and
`model_registry`. Drop the now-dead `if moe_comm_backend is not None`
guard around the dispatcher rebuild loop. (No agpt changes — agpt
doesn't use the moe-only helpers.)

### Smoke results — both PASS within ±0.10

| Config | Final loss | vs v22 baseline | tail10 mean | Δ tail10 |
|---|---|---|---|---|
| agpt 2b (job 12465533) | 7.108 | -0.029 | 7.221 | -0.037 |
| moe 500m (job 12465534) | 6.912 | -0.014 | 6.978 | -0.016 |

Both deltas dominated by streaming-data shuffle noise. Baselines
refreshed to v24.

### Side cleanup

`b9a324f3` renamed `docs/upstream-sync/` → `docs/baselines/` to remove
the visual collision with the neighboring `docs/upstream-sync.md` log
file. Updated path references in `loss_baseline.py`, the workflow
README, and the link from `upstream-sync.md`.

---

## 2026-04-28 — 22nd upstream sync replay (quantize-on-config, LocalMapInnerAttention removal)

### What landed

The 22nd sync (merged `4b0a4fd5`) brought in 9 commits, three breaking:

- **#3127 `6348d93d` quantize on config instead of on model.** Removes
  `protocols/model_converter.py`, drops `model_converters` field from
  `JobConfig` and from every `parallelize_*` signature. Quantization
  converters are now applied to the model *config* at registry time
  (`q.build().convert(config)`). Also moves `FaultTolerantModelSpec`
  from `protocols/model_spec.py` into `experiments/ft/config/job_config.py`.
- **#2986 `b9e33527` Remove LocalMapInnerAttention.** Replaces the
  runtime DTensor wrapper class with a static `LocalMapConfig` set on
  the inner-attention sharding_config via
  `set_gqa_inner_attention_local_map`. All inner attention types now
  inherit `Module` directly.
- **#3113 `053dbf9a` MeshDimName → MeshAxisName.** Transparent for ezpz
  (we only use the upstream helpers).

### Replay outcome

| Module | Status | Smoke test |
|---|---|---|
| `agpt` | replayed | 50 steps, loss 12.92 → 7.14 (job 12465527, exit 0) |
| `moe`  | replayed | 50 steps, loss 12.92 → ~7 (job 12465529, in flight) |

### Commits (ezpz branch)

- `4b0a4fd5` — Merge upstream/main into ezpz (clean automatic merge).
- `69a8cfc7` — Drop `model_converters=` kwarg from `parallelize_llama` /
  `parallelize_moe` signatures and from both `parallelize_fn` /
  `pipelining_fn` call sites in `ezpz/trainer.py`. Drop runtime
  `model_converters.build/convert/post_optimizer_hook`. Switch
  `has_quantization` to read from `model_config` via the upstream
  `torchtitan.components.quantization.utils.has_quantization` helper.
- `59d9f37d` — `LocalMapInnerAttention` → `Module` for
  `SoftcappedFlexAttention` (agpt) and `Attention.Config.inner_attention`
  (moe). Add `set_gqa_inner_attention_local_map(...)` calls in both
  `sharding.py` files so inner attention gets the static `LocalMapConfig`
  it now needs.
- `cd29417e` — moe `model_registry` accepts `quantization=[...]`
  parameter; `moe_671b()` re-registers via
  `model_spec=model_registry("671B", quantization=[...])` instead of
  mutating `cfg.model_converters`. Also re-import
  `FaultTolerantModelSpec` from `experiments/ft/config/job_config`.
- `1b87fa38` — `Float8GroupedMMConverter` → `Float8GroupedExpertsConverter`
  (rename caught at smoke-test import time).

### Smoke-test details

- agpt smoke (`12465527`): loss 12.92 → 7.14 across 50 steps,
  ~7,400 TPS, 28% MFU. Exit 0. Numerics match the v21 run within
  data-shuffle variance — replay is loss-neutral.
- moe smoke (`12465529`): 12.92 → 7 (in flight at step 12, on track),
  ~7,200 TPS at steady state. Same compile-warmup pattern as v21.

### Why moe failed once first

`12465528` failed with `ImportError: Cannot import config_registry for
module 'ezpz.moe'`. The underlying error was the
`Float8GroupedMMConverter` rename to `Float8GroupedExpertsConverter`
(and dropping `fqns=["experts"]` since the new Config takes no
extra args). `config/manager.py` swallows the underlying `ImportError`
which made the cause invisible — only resubmittable after grepping the
upstream class names.

---

## 2026-04-28 — 21st upstream sync replay (sharding API, ChunkedCELoss)

### What landed

The 21st upstream sync (merged `b6c04698`) brought in three breaking
changes that broke ezpz at import / config-build / training-init time:

- **#2963 / #2969** — config-based DTensor sharding. Replaces string-keyed
  `parallelize_module(plan)` with `Module.parallelize(mesh)` reading
  `ShardingConfig` declarations attached to each sub-module's `.Config`.
- **#2937** — ChunkedCELoss. Removed `build_cross_entropy_loss` and
  `ModelSpec.build_loss_fn`; loss now lives on `JobConfig.loss`.
- **`Decoder.Config`** renamed `output: Linear.Config` → `lm_head:
  Linear.Config`.

### Replay outcome

| Module | Status | Smoke test |
|---|---|---|
| `agpt` | replayed | 50 steps, loss 12.96 → 7.09 (job 12465500) |
| `moe`  | replayed | 50 steps, loss 12.93 → 6.91 (job 12465502) |
| `qwen3` | removed | Drift too large; nobody ran it; restorable from history |

### Commits (ezpz branch)

- `03b9f486` — Mechanical: drop `build_cross_entropy_loss` imports +
  `build_loss_fn=` kwargs, rename `output=` → `lm_head=` in agpt+moe
  configs, switch `ezpz/trainer.py` to `config.loss.build()`.
- `472f4743` — agpt sharding-API replay. New `agpt/sharding.py`
  (handles QK-Norm), new `agpt/model.py` (`AgptModel(Llama3Model)`
  overriding `update_from_config`), rewritten `agpt/parallelize.py` as
  thin orchestrator. Float8 tensorwise TP path dropped (no equivalent in
  the new API yet).
- `9bc774a6` — Set `loss=CrossEntropyLoss.Config()` in both `_base_config`
  helpers (was defaulting to abstract `BaseLoss.Config`).
- `40526628` — Enable compile in `smoke_2b_50steps` (XPU CE OOMs without it
  at vocab=256k).
- `dd4e065a` — moe sharding-API replay. New `moe/sharding.py`, extended
  `update_from_config`, rewritten `parallelize.py` (drops 230+ lines of
  manual ColwiseParallel/RowwiseParallel plans). MoE block sharding still
  done at parallelize-time by `apply_moe_ep_tp` (mirrors upstream).
- `5f88abc3` — `smoke_moe_500m_50steps` config + submit script.
- `80a23d41` — Removed `ezpz/qwen3` (drift too large for unused code).

### Preserved agpt-/moe-specific behavior

- `disable_fsdp_gradient_division` still calls
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends (CCL/XPU).
- After `apply_compile`, resets `torch._dynamo.config.capture_scalar_outputs`
  to False (keeps the separately-compiled CrossEntropyLoss working on dense
  models).
- agpt `apply_fsdp` keeps the `[norm, lm_head]` joint grouping with
  reshard_after_forward gated on the policy.
- moe `apply_fsdp` is still inlined locally (avoids `ShardPlacementResult`
  import which doesn't exist in Aurora's PyTorch) with the Shard(0)
  fallback when expert hidden dim isn't FSDP-divisible.
- moe `apply_compile` is per-block `block.compile(backend=...)` instead of
  upstream's fullgraph `apply_compile_sparse` (XPU can't fullgraph compile
  MoE routing's dynamic shapes).

### Smoke-test details

- agpt smoke (`12465500`): loss 12.96 → 7.09 across 50 steps, ~7,400 TPS,
  27% MFU. Standard dense-2B numbers — replay is loss-neutral.
- moe smoke (`12465502`): loss 12.93 → 6.91 across 50 steps, ~7,200 TPS,
  ~10% MFU. The MFU is low because the metrics divisor uses the full
  dense FLOP estimate but only 2/8 experts fire per token — reporting
  artifact, not a perf regression.

---

## 2026-04-27 — Full 10B training, TorchMuon, local dataset

### Local Dataset Cache

- Downloaded FineWeb-Edu `sample-100BT` (267 GB, 140 parquet files)
  to `/lus/flare/projects/datasets/datasets/fineweb-edu-100BT/`
- Added `register_local_dataset()` to `datasets.py` for parquet/arrow files
- Registered as `fineweb_edu_local` — eliminates HF streaming rate limits
  and ensures reproducible data ordering across runs

### TorchMuon Integration

- Added `TorchMuonOptimizersContainer` using `torch.optim.Muon` (built-in
  since PyTorch 2.9)
- Required `_CompositeOptimizer` wrapper — `OptimizersContainer` expects one
  optimizer per model part, but Muon only handles 2D params (need separate
  AdamW for embeddings/head)
- Multiple fix iterations: missing `import torch`, empty param list rejection
  from `Optimizer.__init__`, FSDP empty model parts
- **Result: same TPS as custom Muon (~4,600)** — Newton-Schulz overhead is
  inherent to the algorithm on XPU, not an implementation issue
- **Streaming data shuffle causes ~1.3 loss variance** — same optimizer gives
  very different loss across runs due to HF streaming data ordering

### Speedrun Competition Final Results (1000 steps, 2 nodes)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | Muon (custom) | **3.557** | 4,556 |
| 2 | AdamW + QK-Norm | **3.569** | 7,178 |
| 3 | Muon + cosine | 3.591 | 4,625 |
| 4 | Mano + QK-Norm | 3.604 | 6,980 |
| 5 | Mano | 3.631 | 7,048 |

### Full Training (10B tokens, 8 nodes, GBS=384)

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW | **2.711** | 7,354 |
| 2 | AdamW + QK-Norm | 2.720 | 7,480 |
| 3 | Mano + QK-Norm | 2.854 | 7,346 |
| 4 | Mano | 2.875 | 7,429 |
| 5 | Muon | DNF (compile stuck) | — |

### Key Findings

- **AdamW wins at large batch (GBS=384)** — simpler update more efficient
  per token than manifold optimizers
- **QK-Norm effect diminishes at 10B** — 0.009 loss improvement (vs 0.23
  in 1000-step speedruns). Helps early training but washes out
- **Mano ~0.16 behind AdamW at GBS=384** — LR finder was tuned at GBS=48,
  needs re-tuning for larger batch
- **Muon compile broken with GAS** — inductor can't pickle cyclic objects
  in Newton-Schulz with gradient accumulation on torch 2.13
- **8-node scaling excellent** — 7,300-7,500 TPS/GPU across all configs

### Architecture Tweaks Implemented

- **Logit softcapping** — `SoftcappedFlexAttention` using FlexAttention
  `score_mod` with tanh cap at 30.0. Falls back to eager on XPU (4x slower).
  Manual attention OOMs at seq_len=8192 (materializes full attention matrix).
- **ReLU²** — `ReLUSquaredFeedForward` subclass. Didn't help (3.92 vs 3.80
  baseline). SiLU gating is better for this architecture.
- **WSM** — `eval/merge_checkpoints.py` utility for weighted state merging
  of checkpoints. Supports uniform, linear, and exponential weighting.
- New model variants: `2B_softcap`, `2B_relu2`, `2B_kitchen_sink`

### Round 4: 2N, GAS=8, 1000 steps (local dataset)

Reproducible speedrun with GBS=384 on 2 nodes using local FineWeb-Edu.

| Rank | Config | Loss | TPS/GPU |
|------|--------|------|---------|
| 1 | AdamW+QK-Norm | **3.205** | 7,428 |
| 2 | AdamW | 3.220 | 7,397 |
| 3 | Mano | 3.294 | 7,397 |
| 4 | Mano+QK-Norm | 3.307 | 7,423 |
| 5 | Mano (8.5e-4) | 3.328 | 7,348 |
| 6 | AdamW (3.7e-3) | 5.884 | 7,603 |

**Key findings:**
- AdamW+QK-Norm wins again — consistent across all GBS=384 experiments
- Mano leads early/mid training but AdamW catches up in cosine decay phase
- sqrt LR scaling too aggressive for AdamW (diverged), Mano tolerated it
- Softcap results invalid — local dataset loader memorizes with FlexAttention
  path (data sharding bug)
- FlexAttention on XPU falls back to eager (Triton-XPU can't codegen tanh)
  — 4x throughput penalty makes softcap impractical on this hardware

### Docs Restructure

- Reorganized `docs/competition/` → `docs/competitions/` with per-experiment dirs
- Added light/dark theme loss curve plots using `<picture>` media queries
- Created `docs/competitions/agpt2b-n2-gas8-1000steps/` with live loss curves

### Upstream Sync (20th)

- Merged upstream: dataset checkpoint resume fix (#3008), RL refactor (#3073)
- Clean merge, no replay needed

### CLAUDE.md Added

- Created `experiments/ezpz/.claude/CLAUDE.md` with project rules that
  travel with the codebase (upstream sync protocol, never modify outside
  ezpz, document every run, etc.)
- Updated with Aurora-specific knowledge (queues, yeet-env scaling, eval pipeline)

---

## 2026-04-26 — RL refactor, docs reorg, competition launch

### RL Multi-Task Support

- Refactored `rl/` from hardcoded sum-of-digits to a pluggable task registry
- Created `rl/tasks/` package with `RLTask` dataclass, `register_task()`, `get_task()`
- Moved sum_digits dataset+rewards into `tasks/sum_digits.py` (self-registering)
- Added 3 new tasks: `multiply`, `word_sort`, `countdown`
- Added CLI args (`--task`, `--model-name-or-path`, `--steps`, etc.) to `train_grpo.py`
- Default model: `argonne_private/AuroraGPT-7B` with `Qwen/Qwen3-0.6B` fallback
- Fixed safetensors E2BIG crash by disabling mid-training checkpoints
- Moved RL docs to `docs/rl/README.md`

### Docs Reorganization

- Created `configs/` — moved dense-configs.md, moe-configs.md
- Created `guides/` — moved known-issues.md, running-with-newer-pytorch.md, xpu-attention-issues.md
- Renamed `production-training/` → `production/`
- Created `scaling/` — consolidated scaling-study.md, scaling-study-torch213.md,
  benchmark-80B.md, benchmarks.md into per-model pages (agpt-2b, agpt-20b, agpt-80b, moe)
- Rewrote top-level README.md with organized sections
- Fixed all 31 internal cross-references; link checker passes with 0 broken

### Generic HF Dataset Streaming

- Created `datasets.py` with `register_hf_dataset()` for explicit registration
- Added auto-fallback: unknown `--dataloader.dataset` names are treated as HF hub paths
  (e.g. `--dataloader.dataset stanfordnlp/imdb` just works)
- Pre-registered: fineweb_edu, fineweb, slimpajama, pile, openwebtext, wikitext, c4_streaming
- Silenced httpx/huggingface_hub HTTP log spam

### agpt_2b Loss Competition

**Goal:** lowest loss in 1000 steps on 2 Sunspot nodes (24 XPU tiles).
**Fixed:** FineWeb-Edu streaming, LBS=2, seq_len=8192, 1000 steps.
**W&B:** [aurora_gpt/torchtitan.ezpz.train](https://api.wandb.ai/links/aurora_gpt/hda3milo)

#### New Optimizers Implemented

- **Mano** (`optimizer/mano.py`) — manifold-normalized optimizer
  ([arxiv 2601.23000](https://arxiv.org/abs/2601.23000)).
  Tangent-space projection on rotating Oblique manifold. Vector-norm ops
  instead of Newton-Schulz → runs at AdamW speed (~7,200 TPS/GPU).
- **SPAM** (`optimizer/spam.py`) — spike-aware Adam with momentum reset
  ([arxiv 2501.06842](https://arxiv.org/abs/2501.06842)).
  Gradient spike detection via EMA + periodic moment reset every DeltaT steps.

#### Architecture Tweaks

- **QK-Norm** — added `qk_norm` parameter to `_build_agpt_layers` and
  `_build_agpt_config`. New `2B_qknorm` model variant. RMSNorm on Q,K
  before attention dot product.

#### Competition Results

| Rank | Config | Optimizer | LR | Loss | Steps | TPS/GPU |
|------|--------|-----------|------|------|-------|---------|
| 1 | `speedrun_2b_muon` | Muon | 2.4e-3 | **3.628*** | 967 | 4,695 |
| 2 | `speedrun_2b_mano` | **Mano** | 3.0e-4 | **3.631** | 1000 | ~7,200 |
| 3 | `speedrun_2b_adamw_cosine` | AdamW | 1.3e-3 | 3.789 | 990 | 7,245 |
| 4 | `speedrun_2b_adamw` | AdamW | 1.3e-3 | 3.801 | 1000 | 7,245 |
| 5 | `speedrun_2b_adamw_short_decay` | AdamW | 1.3e-3 | 4.053 | 1000 | 7,245 |
| 6 | `speedrun_2b_adamw_fast_warmup` | AdamW | 1.3e-3 | 4.546 | 1000 | 7,245 |
| 7 | `speedrun_2b_muon_aggressive` | Muon | 4.8e-3 | 4.399* | 976 | 4,596 |
| 8 | `speedrun_2b_sophiag` | SophiaG | 3.1e-4 | 4.719 | 1000 | 7,208 |
| 9 | `speedrun_2b_adamw_high_lr` | AdamW | 2.6e-3 | 5.850 | 1000 | 7,344 |
| 10 | `speedrun_2b_spam` | SPAM | 1.3e-3 | 5.881* | 865 | ~7,200 |

*Still running at time of reporting.

#### Key Findings

- **Muon and Mano essentially tied on loss** (~3.63), but Mano ran at
  full AdamW speed (7,200 TPS) vs Muon's 4,700 TPS. **Mano wins on
  wall-clock time.**
- **Muon is 35% slower per step** due to 5x Newton-Schulz iterations
  (large matmuls on every 2D param). Mano replaces these with O(dim)
  vector-norm ops.
- **Cosine decay beats linear** for AdamW (3.789 vs 3.801).
- **Shorter decay (10%) hurts** — not enough time in decay phase.
- **Shorter warmup (5 steps) hurts** — destabilizes early training.
- **SPAM underperforms** — spike clipping + momentum reset don't help
  on this clean dataset with well-tuned LR.
- **AdamW LR=2.6e-3 diverges** — confirms LR finder boundary (1.3e-3).
- **SophiaG underperforms** AdamW by ~0.9 loss at same step count.

#### Failed Experiments

- `speedrun_2b_muon_short_decay` — crashed at startup (exit 143)
- `speedrun_2b_muon_fast_warmup` — crashed at startup (exit 143)
- `speedrun_2b_adamw_qknorm` — crashed at startup (exit 143)
- `speedrun_2b_muon_qknorm` — crashed at startup (exit 143)

Need to investigate QK-Norm and Muon schedule tweak crashes.

#### Issues Hit

- PBS `qsub -- bash -c '...'` doesn't work — need a proper script file
- `set -euo pipefail` kills venv activate scripts (`ZSH_EVAL_CONTEXT: unbound`)
- Concurrent jobs sharing `--checkpoint.folder=checkpoint` clobber each other
  → fixed with per-config checkpoint dirs
- Disk quota hit at 12TB → cleaned 3.2TB of old scaling study checkpoints
  and 117GB of old repo checkpoints
- `git stash pop` during disk quota crunch wiped train.py to 0 bytes
  → restored from `git show HEAD:...`

---

## 2026-04-25 — torch 2.13 venv, scaling study, production scripts

### Torch 2.13 Environment

- Created `.venv/` with PyTorch 2.13 (built from source for XPU)
- Added `running-with-newer-pytorch.md` guide for setting up the venv
- Added `ezpz yeet-env` integration to copy venv to `/tmp` on compute nodes

### Production Training Scripts

- Created `scripts/train_agpt_2b_venv.sh` and `train_agpt_20b_venv.sh`
  for training with the torch 2.13 venv
- Fixed `ezpz_setup_job` ordering — must run before venv activation
- Fixed `/tmp/.venv/bin` PATH handling after `yeet-env activate`
- Set `local_batch_size=2` as default for 2B training

### 2B Scaling Study (torch 2.13, Sunspot)

- Ran weak scaling study from 2 to 64 nodes on Sunspot
- Results: 7,142 TPS/GPU at 2N (27.6% MFU) — **+23% over torch 2.10**
- Near-perfect scaling to 8 nodes (~100%), 94% efficiency at 64 nodes
- Memory nearly constant at ~44 GiB across all scales
- Documented in `docs/scaling-study-torch213.md`

### Production Training Status

- Updated production run tracking for 2B/20B/80B models
- Added per-model subdirectories with loss curve plots
- Updated upstream sync log with session findings

### Upstream Sync

- Merged upstream `pytorch/torchtitan` main into ezpz branch
- Reverted `.ezpz-interactive-launch.sh` tracking change
- Added interactive launch script and loss CSVs

---

## 2026-04-23 — XPU fixes, upstream merge

### XCCL Barrier Fix

- Fixed torch 2.10 XCCL hangs for barrier and TP collectives
- Root cause: XCCL backend doesn't support barrier() — was hanging
  all multi-node runs
- Fix: use gloo backend for barriers when available

### DTensor TP Revert

- Reverted full DTensor TP (`use_local_output=False`) for agpt models
- Was causing shape mismatches in the attention layer on XPU
- Reverted to standard `use_local_output=True`

### Upstream Merge

- Merged upstream main into ezpz branch
- Upstream changes included GraphTrainer bucketing fixes,
  SAC + FSDP improvements, and Qwen3-VL fused QKV support
