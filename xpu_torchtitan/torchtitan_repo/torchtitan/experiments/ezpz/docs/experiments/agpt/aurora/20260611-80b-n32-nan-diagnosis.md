# 80B NaN diagnosis — bf16 forward overflow at GBS≥384, only fp32-activations confirmed-clean

**Date**: 2026-06-11 → 2026-06-12 (3 revisions)
**Status**: Open. Two previous "fixes" refuted. The only thing that has
trained 20 steps clean at GBS≥192 is `--training.mixed-precision-param=float32`
at TP=4.

> **UPDATE 2026-06-24 — the "fp32-acts is the only clean path at GBS≥192"
> conclusion is INCOMPLETE.** A config this factorial never tested,
> **TP=4 + LBS=1 + bf16 + GBS=372 (reached via GAS=2, not LBS)**, trained
> 20 and 30 clean steps in pure bf16 (Sunspot jobs 12469494, 12469509).
> The factorial only tried TP=4 at GBS=96 (LBS=1, NaN step 18) and TP=4
> GBS=192 via **LBS=2** (NaN step 2) — it never ran TP=4 LBS=1 with GAS to
> push GBS up. Reframed trigger: the NaN tracks **LBS>1** and **large
> dp_degree** (= NGPUS/TP), not raw GBS. Keeping LBS=1 and dp_degree
> low (TP=4 halves it: 744/4=186 vs 744/2=372) stays in the safe regime
> in pure bf16. Full matrix + perf numbers:
> `docs/production/agpt/80b/README.md` ("grad_norm NaN: two independent
> triggers"). **Stability CONFIRMED 4/4 clean** (jobs 12469494/509/510/511,
> 20-30 steps each, 0 NaN, three landing at identical loss 9.69-9.70) —
> not the nondeterministic knife-edge. New production recommendation:
> **TP=4, LBS=1, bf16, GAS-to-GBS** — cheaper than fp32-acts (~3-5x) and
> determinism (~50%, doesn't scale past n=32). The underlying TP=2 /
> LBS>1 grad-path overflow remains an open upstream-worthy bug.

## 🚨🚨 Counter-evidence (2026-06-12 evening, 8540102)

**`--debug.deterministic` does NOT scale to n=64.** The fix that was
declared validated at n=32 fails at n=64 / GBS=384:

> Job 8540102, n=64, GBS=384, `--debug.seed=42 --debug.deterministic`,
> overprovisioned (select=68, FAILOVER_MAX_RETRIES=2), clean attempt.
> Training: loss 12.92 → 12.41 across steps 1–8 (clean), then
> **grad_norm = nan at step 9** → **loss = nan at step 10**, NaN
> persisted through step 20. No node failure (failover wrapper
> reported "attempt 1 succeeded (exit 0)"). MFU ~9% (matches n=32
> deterministic).

Earlier intermediate signal in the same run: grad_norm = `inf` at step 3,
the clip kicked in, loss spiked 12.86 → 13.28, and the model recovered
for 5 more steps before the second overflow at step 9 finished it.

This **refutes the "XPU nondeterminism is the root cause" hypothesis**
that was the headline of the 2026-06-12 afternoon revision. Determinism
mode is a per-step coin flip that bought us one clean run at n=32 GBS=192;
it does not fix the actual failure mode. The actual mechanism is closer
to the *original* (2026-06-11) finding: a bf16 forward overflow that
crosses some weight-state threshold and produces inf activations that
NaN the loss.

### Revised production-fix ranking (after 8540102)

| Fix | Status at n=32 | Status at n=64 | Throughput cost |
|-----|----------------|----------------|-----------------|
| `--debug.deterministic` | ✓ clean (8539896, 8539982) | **❌ NaN at step 10 (8540102)** | ~50% |
| `--training.mixed-precision-param=float32` at TP=4 | ✓ clean (8537349) | **Untested at GBS=384** | ~75% |

The fp32-activations-at-TP=4 fix is now the only candidate with any
clean-training evidence, and it has **only been tested at GBS=96**. The
next test is to confirm it survives GBS=384 — running fp32-activations
at TP=4, LBS=2 (or LBS=1 at n=64 → GBS=192) and seeing whether the
GBS=96 success was about the dtype or about the batch size.

### What this rules out

1. **"Just enable determinism"** is not a production fix.
2. **n=32 success was lucky, not causal.** The 20-step clean run at n=32
   GBS=192 deterministic was sampling a region where the bf16 forward
   happened not to overflow. Doubling GBS to 384 reaches a region where
   it does.
3. The grad_norm = inf at step 3 followed by recovery is the same
   pattern the n=16 baseline shows — the bf16 forward is producing
   overflow regularly across the chain; the model just absorbs it as
   long as the post-clip update is in the safe regime. At n=64 it's no
   longer in the safe regime.

### Open questions

- Does **fp32-activations at TP=4, GBS=384** train clean? (Next test.)
- Does **lowering LR to 1e-8** (untested in original factorial) buy
  any margin? Possibly small enough that the bf16 forward doesn't get
  pushed into overflow territory.
- Is this a torchtitan-stack issue or an Aurora-XPU bf16-matmul issue?
  Would need a Polaris (A100) comparison, but we don't have an 80B
  config that fits on A100.

### What the n=128/n=256 deterministic scripts would have shown

Nothing useful — both would NaN earlier than n=64 (GBS scales with N).
Not submitting them. (Per [[feedback_never_kill_running_jobs]], they
weren't actually queued — only n=64-det was.)

---

## 🚨 Earlier "Revised TL;DR" (2026-06-12 afternoon, partially superseded)

The original conclusion below — that this is a bf16 forward overflow —
**is wrong about the mechanism**. The actual finding:

> **Adding `--debug.deterministic` makes the failing n=32 GBS=192
> baseline train cleanly through 20 steps.** Loss 12.93 → 10.50 with
> the same config that without `--debug.deterministic` NaNs at step 6.

So the bug is **XPU nondeterministic op output** that accumulates into
overflow at GBS=192. Determinism mode bypasses it. fp32-activations also
bypasses it but at higher throughput cost. The grad-norm "spikes to
79K" we measured in the fp32 run are the *true* gradient magnitudes
that the nondeterministic bf16 path was silently overflowing.

**SUPERSEDED 2026-06-12 evening**: this was wrong too. n=64 with
determinism also NaNs (8540102). See "Counter-evidence" section above.
The factorial below correctly maps LR / data / clip-norm as non-causes,
but the production-fix recommendation in this section is wrong.

The mid-day production recommendation was: "use `--debug.deterministic`
for the 256N production chain restart". **Do not do this** — the
8540102 evidence is clear that this fix doesn't scale past n=32.

---

## Original TL;DR (2026-06-11, partially superseded)

The 80B training stack NaNs deterministically at small node counts on
Aurora. **n=16 (GBS=96) is the largest configuration that trains clean
for 20 steps without any extra flags.** At n=32 (GBS=192) loss goes
NaN at step 6; at n=256 (8530891) loss went NaN at step 2.

A 7-job factorial at n=32 isolated the cause as a **bf16 forward-path
overflow** (later revised to "nondeterministic op output that triggers
bf16 forward overflow"):

- Not LR (LR=1e-7 NaN'd at the same step as LR=1e-6)
- Not data (different seed NaN'd at the same step)
- Not the master dtype (`training.dtype=float32` doesn't change activations)
- Not the optimizer/grad clip (post-clip grads ≤1.0 by default; NaN
  happens upstream in the forward pass)
- **bf16 activations are part of the chain**: with
  `training.mixed_precision_param=float32` the model survives 20
  steps including grad_norm spikes of 21K–79K. The fp32 path absorbs
  the spikes; the bf16 path overflows them into nan.
- **NEW (2026-06-12)**: nondeterminism is the upstream cause —
  with `--debug.deterministic` enabled, the same bf16 baseline
  trains cleanly.

The grad_norm spikes themselves are present at every config including
the clean n=16 baseline — they happen around steps 15–17 with the
specific weight state Adam reaches. At GBS=96 bf16's dynamic range
just barely absorbs the resulting activations and the model recovers;
at GBS=192 the nondeterministic execution path pushes them over the
edge.

## What we know after the factorial

| Test | n | TP | LBS | GBS | dtype/clip | Result | NaN onset |
|------|--:|---:|---:|----:|------------|--------|-----------|
| n=16 baseline | 16 |  2 | 1 |  96 | bf16, clip 1.0 | ✓ 20 clean steps, loss 12.93 → 10.41 | — |
| n=32 baseline | 32 |  2 | 1 | 192 | bf16, clip 1.0 | NaN step 6 (grad_norm `inf` step 2, recovered, then nan) | step 6 |
| n=32 LR=1e-7  | 32 |  2 | 1 | 192 | bf16, clip 1.0, LR=1e-7 | NaN step 6 (same shape) | step 6 |
| n=32 seed=12345 | 32 |  2 | 1 | 192 | bf16, clip 1.0, seed=12345 | NaN step 6 (different per-step losses, same onset) | step 6 |
| n=32 dtype=fp32 (wrong flag) | 32 |  2 | 1 | 192 | bf16 acts, fp32 master, clip 1.0 | NaN step 2 (`training.dtype` only sets master) | step 2 |
| n=32 fp32 acts | 32 |  2 | 1 | 192 | **fp32**, clip 1.0 | **OOM at model init** (TP=2 + fp32 acts ≈ 80 GiB > 64 GiB tile cap) | — |
| n=32 TP=4 GBS=96 | 32 |  4 | 1 |  96 | bf16, clip 1.0 | NaN step 18 (grad_norm explosion 5→50, didn't recover) | step 18 |
| n=32 TP=4 GBS=192 | 32 |  4 | 2 | 192 | bf16, clip 1.0 | NaN step 2 (`loss = -inf`) | step 2 |
| **n=32 TP=4 fp32 acts GBS=96** | 32 |  4 | 1 |  96 | **fp32**, clip 1.0 | **✓ 20 clean steps**, loss 12.96 → 10.93, grad_norm spikes 21K, 79K, 55K, 15K, 6K | — |
| n=32 tight clip 0.1 | 32 |  2 | 1 | 192 | bf16, clip 0.1 | NaN step 4 (refuted hypothesis: clip is post-backward, can't prevent fwd overflow) | step 4 |
| **n=32 + deterministic** | 32 |  2 | 1 | 192 | bf16, clip 1.0, `--debug.seed=42 --debug.deterministic` | **✓ 20 clean steps**, loss 12.93 → 10.50, mem 55.27 GiB (86.4%), MFU 9.3% | **none** |

n=16 baseline and the TP=4 + fp32-activations run are the only two
configs that survive 20 steps. They share GBS=96, and the fp32 run
proves the bf16 forward path is the failure point: under fp32 the
model absorbs grad_norm spikes of up to 79K and keeps training; under
bf16 those same spikes drive activations into inf.

## Per-step grad_norm — the smoking gun

Side-by-side at GBS=96 (n=16 baseline vs n=32 TP=4 same GBS, both bf16):

| Step | n=16 TP=2 grad_norm | n=32 TP=4 grad_norm | n=32 TP=4 fp32-acts grad_norm |
|-----:|--------------------:|--------------------:|------------------------------:|
| 1 |   4.79 |   4.86 |   4.87 |
| 2 |   4.87 |   4.94 |   4.95 |
| 3 |   4.92 |   5.00 |   5.00 |
| 4 |   4.90 |   4.97 | **21,309** |
| 5 |   4.89 |   4.97 | **79,284** |
| 6 |   5.09 |   5.16 |   5.10 |
| 7 |   5.18 |   5.27 |   5.14 |
| 8 |   5.37 |   5.47 |   5.27 |
| 9 |   5.62 |   5.72 | **54,873** |
| 10 |   6.14 |   6.26 |   5.84 |
| 11 |   6.53 |   6.83 |   6.26 |
| 12 |   7.12 |   7.43 | **15,028** |
| 13 |   7.92 |   8.00 | **6,452** |
| 14 |   7.42 |   7.45 |   8.20 |
| 15 |  16.80 |  17.23 |   8.23 |
| 16 |  40.50 |  42.13 |   8.37 |
| 17 |  49.46 |  51.83 |  71.07 |
| 18 |  45.90 | **nan** |  91.35 |
| 19 |  27.05 |   nan  |  60.99 |
| 20 |  23.07 |   nan  |  64.05 |

Two things stand out:

1. **The bf16 runs at GBS=96 produce nearly identical grad_norms** (n=16
   TP=2 and n=32 TP=4). Both spike to ~50 at steps 15–17. n=16 survives
   the spike, n=32 doesn't. The difference is presumably small numerical
   drift in the loss-reduction path or the TP collectives.

2. **The fp32-activations run produces TOTALLY DIFFERENT grad_norms** —
   massive spikes (21K, 79K, 55K, 15K, 6K) at steps 4–13 that the bf16
   versions never see. Those spikes are the *real* gradient magnitudes;
   bf16 silently clips them via overflow, producing the "tame" 5–7
   range we see in bf16 logs. By the time `max_norm` clipping runs,
   the gradient buffer is already partial garbage.

The implication: bf16 has been masking enormous gradient activity
for the first ~13 steps even in the working configs. The model only
NaNs once that masking flips from "lucky" to "inf-producing" — which
in turn depends on the exact weight state. GBS=192 puts you over the
edge fast; GBS=96 stays just under it.

## Why `training.dtype=float32` isn't the fix

torchtitan has two dtype configs:

- `training.dtype` — master weight dtype (default `float32` in v2)
- `training.mixed_precision_param` — FSDP MixedPrecisionPolicy `param_dtype`,
  i.e. the dtype activations and parameter copies use during forward/backward
  (default `bfloat16`)

The first attempt at the bf16-overflow test (`n32-fp32-dtype.sh`,
8536657) only set `training.dtype=float32`, which is already the v2
default. Memory stayed at 41.68 GiB confirming activations were still
bf16. The fix is `training.mixed_precision_param=float32`, which doubles
activation memory.

## Why fp32 isn't free

`training.mixed_precision_param=float32` doubles activation memory:

- TP=2 LBS=1: ~80 GiB peak (exceeds the 64 GiB per-tile cap → OOM at
  model init)
- TP=4 LBS=1: ~42 GiB peak (fits; this is the validated working config)

It also halves throughput on Aurora's XPU (no native bf16 matmul speedup
when activations are fp32). The TP=4 fp32-act run at 8.3 TFLOPs/GPU vs
the bf16 baseline's 26 TFLOPs/GPU — **3.1× slower**. At the same TP
the gap is closer to 5×.

## Production-fix candidates

Ranked by cost-if-it-works:

| Candidate | Hypothesis | Throughput cost | Status |
|-----------|------------|-----------------|--------|
| Tighter `max_norm` (1.0 → 0.1) | Smaller weight updates → weights stay near init → bf16 forward doesn't overflow | ~0% | **❌ Refuted by 8539593** (NaN'd at step 4 — earlier than baseline, since clipping is post-backward and can't prevent forward-pass overflow) |
| Lower LR (1e-6 → 1e-8) | Same mechanism as tighter clip | ~0% (slower convergence in nat/token but not in throughput/sec) | Untested |
| Logit softcap (Gemma-2 style) | Caps logits pre-softmax → fewer paths to inf | ~0% via FlexAttention | **Not available on XPU** (FlexAttention unsupported, would need custom impl) |
| `mixed_precision_param=float32` at TP=4 | Forces all-fp32 forward/backward | **~3-5× slower** | **Validated** (8537349 cleanly trained 20 steps) |

**Update (2026-06-12, mid-day)**: tighter `max_norm` was refuted by
8539593 — NaN at step 4 instead of step 6. This makes sense in
hindsight: gradient clipping runs AFTER backward, so by the time
`max_norm` would constrain the gradient, the bf16 forward has already
overflowed and the gradient tensor already contains nan. Clipping
nan→nan doesn't help.

**Update (2026-06-12, afternoon)**: `--debug.seed=42
--debug.deterministic` was tested on a hunch about XPU nondeterminism
and **it works**. 8539896 trained 20 clean steps with the exact
n=32 GBS=192 config that without determinism NaN'd at step 6:

- Loss 12.93 → 10.50 (compare to n=16 baseline 12.93 → 10.41)
- grad_norm 5 → 52 across the run (same shape as n=16, the model
  absorbed the spikes around step 15-17)
- Memory bumped from 41.68 GiB → **55.27 GiB (86.4%)** — deterministic
  mode forces extra allocations
- Throughput dropped from 18% MFU → 9.3% MFU — ~50% cost

So `--debug.deterministic` is the **cheap production fix**. The bug
is some nondeterministic XPU op whose output sometimes lands in the
overflow regime at GBS≥192; determinism mode forces a stable
execution path that happens not to overflow.

This also explains why the tight-clip test NaN'd at step 4 instead
of step 6: the clip-norm change shifted FSDP scheduling enough that
the nondeterministic ops produced different (worse) numerics. The
clip itself didn't matter; just changing one thing in the run
re-rolled the nondeterminism dice.

**Production-fix ranking (final):**

1. **`--debug.deterministic`** — cheap (~50% throughput), bf16 path
   stays, no TP change. **Recommended for production restart.**
2. `--training.mixed-precision-param=float32` at TP=4 — heavier
   (~75% throughput), requires TP change. Backup if determinism
   doesn't scale to 256N.
3. Tighter clip / lower LR / softcap — **all refuted or unavailable**.

## Implications for production

- **Test `--debug.deterministic` at n=64 / 128 / 256 first.** Validated
  at n=32; needs to be confirmed at the production scale. If it works
  at 256N, that's the production fix.
- If determinism doesn't scale (e.g. some other nondeterministic op
  surfaces at higher N), fall back to `mixed_precision_param=float32`
  at TP=4.
- **Both fixes leave bf16 throughout the master path** — no need to
  reconvert checkpoints or change the architecture.
- The 4N validation run is unaffected (GBS=24, well below the
  failure threshold). 8N and 16N also fine (GBS=48 / 96).

## Test jobs

| Job ID | Script | Wandb |
|--------|--------|-------|
| 8536194 | n16-smoke.sh | https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/nivubyxr |
| 8536249 | n32-smoke.sh (baseline, NaN'd) | (job 8536249) |
| 8536417 | n32-smoke-lr1e-7.sh | (job 8536417) |
| 8536916 | n32-seed-shift.sh | (job 8536916) |
| 8536657 | n32-fp32-dtype.sh (wrong flag) | (job 8536657) |
| 8536967 | n32-fp32-mixed-precision.sh (OOM) | (job 8536967) |
| 8537029 | n32-tp4-lbs1.sh | (job 8537029) |
| 8537168 | n32-tp4-lbs2.sh | (job 8537168) |
| 8537349 | n32-tp4-fp32-mp.sh (✓ 20 steps) | (job 8537349) |
| 8539568 | n32-tight-clip.sh (PBS protocol flake — preflight failed) | — |
| 8539593 | n32-tight-clip.sh (resubmit) — NaN'd step 4, refuting tight-clip hypothesis | (job 8539593) |
| 8539896 | n32-det.sh — `--debug.seed=42 --debug.deterministic` → ✓ clean 20 steps (but see 8540102 below — does not generalize past n=32) | (job 8539896) |
| 8539982 | n32-det-tight-clip.sh — det + max_norm=0.1 → ✓ clean (n=32 only, same caveat) | (job 8539982) |
| **8540102** | **n64-det.sh** — `--debug.seed=42 --debug.deterministic` at n=64 GBS=384, overprovisioned (select=68, FAILOVER_MAX_RETRIES=2) → **❌ NaN: grad_norm=inf step 3, recovered, grad_norm=nan step 9, loss=nan step 10**. Clean exit (failover wrapper reported attempt 1/2 succeeded, exit 0) — this is a training failure not an infra failure. **Refutes the determinism hypothesis.** | https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/wz9tr023 |

All logs in `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/80b-*.o*`.

## Related

- The `--training.mixed-precision-param=float32` workaround mirrors
  what was needed for 70B+ models in the [Llama 3 405B paper](https://arxiv.org/abs/2407.21783)
  (their fp32-residual fix solved a similar bf16-overflow problem at
  scale).
- See also `docs/guides/training-dtype-bf16-norm-freeze.md` for the
  prior bf16-master regression (different problem: master weights stuck
  at init due to bf16 ULP > optimizer step). That fix doesn't help here
  because the failure is in the forward path, not the master copy.
