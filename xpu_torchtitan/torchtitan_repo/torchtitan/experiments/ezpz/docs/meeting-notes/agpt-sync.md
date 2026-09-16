# AuroraGPT Sync — Meeting Notes

> Sam Foreman. Most recent first.

---

## 2026-06-29

### Headline: 2B v2 256N pre-training is COMPLETE (4.674T tokens, 100% of target)

The 2B 256N v2 chain reached its full **4.67T-token budget**: final
checkpoint **step-92,859 = 4.674T tokens (100.0%)**, final loss **2.652**,
grad_norm ~0.056, ~14% MFU. Chain head `8558531` (cont12) finished a clean
exit-0 ~10.2h run on 2026-06-29 03:03 UTC. This is the first agpt model to
finish the full v2 (fp32-master) base pre-training run end to end.

**Discussion / decisions for the team:**
- **Eval the final 2B checkpoint.** Convert step-92,859 DCP -> HF and run the
  full lm-eval suite (ARC-E/C, HellaSwag, Winogrande, etc.) so we have the
  end-of-pretraining scorecard. Blocked until Aurora returns from PM (lm-eval
  needs compute). Queue it first thing.
- **What's next for 2B?** Options: (a) start CPT (continued pre-training) on
  additional tokens with the constant-LR config we just built, (b) SFT, (c)
  freeze it as the reference base. The constant-LR (decay_ratio=0) plumbing is
  ready -- see the 80B launch item below.

### 80B production launched at SophiaG / constant-LR / scale brackets

Submitted the first real 80B v2 production runs (queued, will start post-PM):
**512N + 1024N + 2048N simultaneously**, SophiaG @ LR=1e-6, **constant LR after
warmup (no decay)** for the planned CPT regime, validator ON (95/5 split).

- **Optimizer/LR is the discussion point.** Per the 2026-06-27 production-batch
  LR-finder, AdamW @ GBS~6144 is on a NaN cliff (LR=1e-6 past it); the finder's
  safest pick is **mano @ ~3e-6**, with SophiaG @ ~1e-6 as the lowest-loss but
  narrow-band alternative. We launched **SophiaG @ 1e-6**. Worth a team
  decision: stick with SophiaG, or switch the 80B base to mano for the wider
  stability margin on an unattended multi-T-token run?
- **Scale is unvalidated above ~512N.** 1024N (12,288 ranks) is documented to
  crash at init (`set_determinism`); 2048N (24,576 ranks, dp_degree~6138) is 2x
  that and untested. The 512N bracket is the safety net; submitting all three
  is itself the scaling experiment. Expect possible init crashes at 1024/2048N.
- A 4N pre-check confirmed the config wiring + clean SophiaG descent before
  committing the big allocations.

### Infra fixes this week (all landed + pushed)

- **blendcorpus cold-cache index race FIXED** (was blocking any fresh-CKPT_DIR
  80B run). Root cause: at TP>1 the non-rank-0 readers raced rank-0's index
  write. Fixed at source in `saforem2/blendcorpus` -- atomic writes + poll
  (`041d015f`) + a TOCTOU follow-up (`1f7e9c0`). Confirmed end-to-end: cold 4N
  TP=4 80B now builds the index and trains with 0 EOFError. **80B no longer
  needs a manual cache prewarm.**
- **Validator at 80B TP=4: the "CCL deadlock" was a phantom** -- 3 unrelated
  bugs (cold-cache mmap race, a training-side barrier stall, a `loss_fn`
  tuple-unpack crash), all fixed. Validation CONFIRMED working at TP=4
  (job 12469784). `VALIDATOR_ENABLE=1` is now safe; 95/5 split is the default.
- **Checkpoint-resume incident (recovered).** A clone `git pull` advanced the
  prod clones past an optimizer state-dict format migration (#3623/#3269,
  nested->flat), breaking DCP resume. Recovered by pinning clones pre-#3623;
  resume verified. The clones stay pinned -- a nested->flat migration shim is
  hard/risky and not worth it (chains resume fine pinned).
- **walltime-aware checkpointing** added to the ezpz trainer (force a final
  ckpt before walltime) + a job-absolute-deadline fix so it survives failover
  retries.

### 20B 256N status

Running through the PM boundary; finished clean at **step-2,100 = 105.7B tokens
(2.3%)**, loss **2.85**, ~21.8% MFU. Resumes post-PM via `8558549`.

### Going into the PM maintenance (2026-06-29 06:00 -> 07-01 03:30 UTC)

Both 256N chains exited with valid checkpoints (2B step-92,859, 20B step-2,100);
all continuations + the 6 80B jobs are queued to resume/start post-maintenance.
Nothing at risk.

---

## 2026-05-04

### bf16-master RMSNorm-freeze fix is producing real downstream gains

Quick recap for context. All v1 production runs (2B / 20B / 80B) had
`training.dtype = bfloat16`, which kept the master parameter copy in
bf16. RMSNorm.weight initializes to 1.0; the bf16 ULP at 1.0 is
~7.8e-3 and per-step optimizer updates for those parameters are ~1.6e-5,
so every update rounded to zero and **norm weights never moved from
1.0 for the entire run**. Other parameters (linears, embeddings)
initialize at much smaller scales and updated fine, so training loss
curves looked plausible — the bug only became visible at eval time.

Default flipped to `float32` on 2026-04-30 and **v2 production was
restarted from scratch** rather than continuing from the bf16-tainted
checkpoints. The v1-vs-v2 lm-eval comparison is the smoking gun:

- 2B ARC-Easy climbed **0.277 → 0.429** over 100B tokens on v2,
  vs v1's flat ~0.27 across **450B** tokens.
- **+19.8pp ARC-Easy / +15.4pp HellaSwag at 503B tokens** vs v1's
  flat baseline.
- v1's flat trajectory across 450B+ tokens is the qualitative
  signature of the bug — a model with frozen normalization cannot
  improve on what lm-eval measures, no matter how much data it sees.
- 2B eval comparison:
  [`docs/evals/agpt/2b/`](../evals/agpt/2b/README.md);
  20B eval comparison:
  [`docs/evals/agpt/20b/`](../evals/agpt/20b/README.md);
  full diagnosis + cross-linked evidence:
  [`docs/guides/training-dtype-bf16-norm-freeze.md`](../guides/training-dtype-bf16-norm-freeze.md).

### Production status

- **2B 512N canonical chain** (`8460301 → 8463626 → 8463627`):
  step **5,073**, loss **2.97**, **510B tokens / 10.9% of 4.67T
  target**. Continuation 8463627 queued, follow-up 8466847 held on
  `afterany:8463627`. Trajectory page:
  [`docs/production/agpt/2b/n512/`](../production/agpt/2b/n512/README.md).
- **20B 512N canonical chain** (`8460302 → 8463628`): step **~862**,
  loss **~3.47**, MFU ~17.8%. 8463628 currently running near
  walltime; 8466848 held on `afterany:8463628`. Trajectory page:
  [`docs/production/agpt/20b/n512/`](../production/agpt/20b/n512/README.md).
- **20B 256N (8463659):** ran 9h walltime then **NODE_FAIL after
  step 364** (loss 4.61, 18.3B tokens). `shepherd died from signal 9`
  on `x4406c6s7b0n0`, PBS exit -20 — same recurring Aurora bad-node
  failure mode as 8459818 / 8460301. **step-300 ckpt saved cleanly,
  resumable.** Throughput on this run was bouncing 21-410 TPS
  depending on flare contention (1-20% MFU). Trajectory page:
  [`docs/production/agpt/20b/n256/`](../production/agpt/20b/n256/README.md).
  Are these recurring `signal 9` crashes being tracked anywhere?
  They've now killed three long-walltime jobs across three different
  nodes — worth raising with ALCF support if not.

### 80B blocker

- 80B `compile + AC + TP=2` still hits the
  `tensors_saved_with_vc_check` AOT autograd assertion (`DeviceMesh`
  leaks into saved-for-backward tensors). Toy minimal repro doesn't
  fire — bug needs the real `Module.parallelize` + `LocalMapConfig`
  path that torchtitan uses.
- **2026-05-03:** added `agpt_50b_wide` (dim=9216, 48 layers, ~48B
  params,
  [`98a02d04`](https://github.com/saforem2/torchtitan/commit/98a02d04))
  as a smaller bisect target. 2N + torch 2.10 smoke ran 10/10 steps
  cleanly. Concluded "bug is depth-sensitive — does NOT reproduce at
  48 layers." That conclusion turned out to be wrong (see below).
- **2026-05-05 correction:** ran a proper three-config bisect on torch
  2.13 (job 12465952 4N + job 12465962 2N). All three configs
  (`agpt_50b_wide` 48L, `agpt_70b_wide` 72L, `agpt_80b` 84L) **crash
  identically** with the same assertion. Smallest tested:
  `agpt_50b_wide` on 2N takes ~30s to crash. **Bug is
  torch-version-sensitive, not depth-sensitive.** The May 3 result was
  a torch-2.10 artifact (the failing assertion in
  `_AutogradSavedState.save_from_forward` likely doesn't exist or
  isn't reached on the older AOT-autograd code path). Working repro
  bracket: torch 2.10 (any depth) ✓ → torch 2.13 (every depth tested) ✗.
- Workaround in the meantime: `compile=OFF` for any 80B-family config
  on torch 2.13, OR pin to torch 2.10 for those configs.
- **Working v2 80B path validated 2026-05-05** (job 12466025, 4N
  smoke): `agpt_80b @ TP=2, AC=full, compile=OFF, AdamW LR=1e-6,
  fp32-master` on torch 2.13. Loss descended **12.98 → 10.46** over
  20 steps, MFU steady at **~17.8%** (matches v1 compile-on baseline),
  memory peak **88.94%** with ~7 GiB headroom. Production setup just
  needs to add the 200-step linear warmup the 2B/20B v2 configs use,
  then it's ready to launch.
- Initial toy repro (legacy `parallelize_module` — does NOT fire,
  needs the new sharding API):
  [`docs/upstream-issues/repro_devicemesh_in_saved_tensors.py`](../upstream-issues/repro_devicemesh_in_saved_tensors.py).

### Open work I'm holding

- **Validation loss wiring:** blendcorpus's existing val split (5%
  slice) is now plumbed through `EzpzValidator` (subclass that also
  fixes the TP loss-reporting bug — see "Other notes" below). Default
  `enable=False` so production isn't disturbed. Smoke test not yet
  done. Do we want held-out NLL on production runs, or are downstream
  lm-eval scores at checkpoint cadence the right signal?
- **80B production** — still has open issues from prior sessions
  (LR=1e-6 stable but bad-node Gloo timeout crash at step 51).
  Worth flagging if 80B production is on the agenda.

### Other notes

- **TP loss-reporting bug found, fix filed upstream + locally
  workaround.** Upstream `_dist_reduce` change
  ([`pytorch/torchtitan@1786292d`](https://github.com/pytorch/torchtitan/commit/1786292d),
  2026-04-27) skips the cross-batch `all_reduce` when `loss` is a
  DTensor on a mesh orthogonal to `loss_mesh`, so reported loss on
  any TP > 1 run is `true / dp_world_size`. **No current production
  runs are TP > 1**, so no live dashboards are affected — but if/when
  we restart 80B production at TP=2, the historical 80B v1 W&B traces
  show `loss / 1536`. Gradients/optimizer steps were unaffected; only
  the printed value was wrong. Fix filed at
  [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204)
  (mergeable, awaiting maintainer review). Local ezpz workaround in
  [`a24ed2e1`](https://github.com/saforem2/torchtitan/commit/a24ed2e1)
  + [`a0b9b13d`](https://github.com/saforem2/torchtitan/commit/a0b9b13d).
  Full diagnosis:
  [`docs/guides/loss-reporting-tp-dist-reduce.md`](../guides/loss-reporting-tp-dist-reduce.md).

### Action items

- [ ] (Sam) Smoke-test `EzpzValidator` end-to-end on `agpt_2b` once
      consensus on whether to enable val.
- [ ] (Sam, blocked on review) Push for review on
      [pytorch/torchtitan#3204](https://github.com/pytorch/torchtitan/pull/3204).
- [ ] (?) Volunteer to build LocalMapConfig-based minimal repro for
      the 80B compile + AC + TP=2 crash.
- [ ] (?) Decide cadence for held-out validation loss on production.
- [ ] (?) Decide whether to file an ALCF support ticket for the
      recurring `shepherd died from signal 9` NODE_FAIL pattern
      (jobs 8459818, 8460301, 8463659 — three crashes, three
      different nodes). Resume 8463659 from step-300 in the meantime?
