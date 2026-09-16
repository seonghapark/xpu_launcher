# `moe_debugmodel_ep` LBS=2 step-41 stall — transient, not reproducible

## TL;DR

The 2026-05-20 step-41 stall in `moe_debugmodel_ep` at LBS=2 on 2N
Sunspot (job 12467181, log `moe_debugmodel_ep_lbs2-20260520-230219.log`)
did **not reproduce** when re-run on 2026-05-21 with the same config
and stack. The retry completed all 50 steps cleanly in 139 s (exit 0)
on the same machine, same .venv (torch 2.13), same `moe_comm_backend="standard"`,
same EP=2. Reclassifying as **transient** (CCL state, node-local stall,
or scheduler hiccup), not a systemic EP/AC/compile bug.

## Original incident

- Log: `logs/smoke-pr3386-followup/moe_debugmodel_ep_lbs2-20260520-230219.log`
- Job: 12467181, 2N Sunspot, alloc `x1921c1s4b0n0` + `x1921c1s5b0n0`
- Symptom: clean steps 1-41 (loss 12.95 → 7.56, ~6.8k TPS), then
  **16:25 of complete silence across all 24 ranks**, then
  `rank 1 died from signal 15` (launcher SIGTERM at the 1146 s wall
  cap), exit 143.
- W&B: [`gtutjvg5`](https://wandb.ai/aurora_gpt/torchtitan.ezpz.train/runs/gtutjvg5)

Initial hypothesis: hard collective deadlock in EP token-dispatch
all-to-all under the `standard` `moe_comm_backend`.

## Retry (2026-05-21)

- Log: `logs/hang-retry-step41/moe_debugmodel_ep_lbs2_retry4-20260521-080239.log`
- Same config (`moe_debugmodel_ep --training.local_batch_size 2`),
  same .venv (torch 2.13), same alloc class (2N Sunspot,
  `x1921c1s1b0n0` + `x1921c1s2b0n0` — alloc 12467180, sibling of
  the original 12467181).
- Added `CCL_LOG_LEVEL=info` + `TORCH_SHOW_CPP_STACKTRACES=1` to
  capture more signal on any failure.
- **Result**: clean exit 0, all 50 steps, 139 s total wall time. Loss
  12.88 → 6.997. Step 41 completed in 1 s like every other step.
  Throughput ~11.8k TPS (vs original's ~6.8k TPS, but that's a
  separate variance band — see below).

## Reclassification

This is now a **transient** incident, not a systemic EP/AC/compile
bug. Plausible underlying causes (not investigated; would need an
in-flight repro):

- Single-rank CCL or driver stall that didn't propagate as a normal
  error (most likely)
- Node-local resource contention (memory, swap, or sibling process)
- Cluster-side network event during the original run

No code change recommended.

## Adjacent findings from the retry

These came out of getting the retry stack right and are worth keeping
visible:

1. **`comm.train_timeout_seconds=100` did NOT fire** on the original
   985 s silence. Either the timeout isn't wired into the CCL/XCCL
   collective path on XPU, or it covers something else. Separate
   investigation — not blocking but worth a follow-up writeup if it
   matters for any production scenario where a hang would otherwise
   waste wallclock.
2. **`TORCH_DISTRIBUTED_DEBUG=DETAIL` crashes on XPU**: triggers
   `RuntimeError: Backend fake does not yet support sequence numbers`
   from a c10d barrier sequence-number check at init. Removing the
   env var unblocks. Knowable XPU limitation, but the error message
   doesn't say "XPU" — worth noting for next time someone reaches for
   it.
3. **`--debug.deterministic` is incompatible with the MoE path on
   XPU**: `_histc_xpu does not have a deterministic implementation`
   from the routing-token-count `histc`. So `--debug.deterministic`
   + `--debug.seed=42` is not actually available for MoE bit-exact
   regression gates on Intel XPU until upstream PyTorch ships a
   deterministic `_histc_xpu` kernel.
4. **Throughput variance was wide**: retry ran at ~11.8k TPS,
   original at ~6.8k TPS — same code, same nodes-of-the-same-class,
   same stack. ~1.7× spread. Plausibly correlated with whatever
   caused the original hang (concurrent load on the alloc).

## Updates to other docs

- Smoke report `docs/experiments/moe/sunspot/20260520-smoke-n2-pr3386-ep-followup.md`
  Finding #2 still reads "uninvestigated"; should be updated to point
  at this note and reclassify as transient.
