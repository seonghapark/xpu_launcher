# Production Training — agpt 80B @ 512 nodes

> **First v2 attempt, 2026-05-11.** Uses the `compile=OFF` working
> path identified in the 4N smoke (job 12466025) on 2026-05-05, plus
> the bad-node failover wrapper. Production clone is at
> `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/`.

## v2 — 80B @ 512N — AdamW LR=1e-6 (fp32 master, compile=OFF, TP=2)

> **Working-config provenance:** 4N smoke (12466025) on 2026-05-05
> showed clean training: loss 12.98 → 10.46 over 20 steps, MFU
> ~17.8%, peak memory 88.94%, no NaN, no compile crash. User
> re-validated the same config on Sunspot 8N on 2026-05-11. See
> the **80B v2 working path** section in
> [`../../../../.claude/CLAUDE.md`](../../../../../.claude/CLAUDE.md)
> for the full diagnosis and config provenance.

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/` |
| Submit script | [`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../../../../scripts/submit_agpt_80b_aurora_venv_failover.sh) (failover wrapper, requests N+spare nodes) |
| Stack | torch 2.13 venv (yeet-env tarball mode, **shared with `agpt-2b-v2`** via symlinks) |
| Optimizer | **AdamW**, LR=**1e-6** (SophiaG/Muon broken at dim=9216 — bf16 overflow in Hessian/Newton-Schulz) |
| Compile | **off** (compile + AC + TP=2 crashes on torch 2.13 with `tensors_saved_with_vc_check` AOT autograd assertion) |
| Activation checkpoint | full (required to fit in tile memory at 80B) |
| Parallelism | TP=2, FSDP=3072 (DPS=-1 derives, DPR=1) |
| GBS | 3,072 (LBS=1 × 6,144 GPUs ÷ TP=2) |
| Total tokens | 4.67T target |
| Checkpoint dir | `outputs/checkpoints/agpt-80b-adamw-olmo-mix-1124-n512-gbs3072` (will create on first save) |
| Failover | 522 nodes requested (512 active + 10 spare); FAILOVER_MAX_RETRIES=2 |

### Progress

| Job ID | Date | Walltime | Steps | Loss | TPS/GPU | MFU | Status |
|--------|------|---------:|------:|-----:|--------:|----:|--------|
| [`8480361`](#log-8480361) | 2026-05-11 | 12h | — | — | — | — | **Queued** (first v2 80B attempt; failover wrapper, 522 nodes) |
| [`8480362`](#log-8480362) | 2026-05-11 | 12h | (cont.) | — | — | — | Held (`afterany:8480361`) |

### Logs

| Job ID | Path |
|--------|------|
| <a id="log-8480361"></a>`8480361` | `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/agpt-80b-n512-v2-failover-chain1.o8480361` (DOA — `failover_lib.sh` path bug, fixed in `e8379fd6f`) |
| <a id="log-8481301"></a>`8481301` | `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/agpt-80b-failover.o8481301` (DOA — flag-spelling bug, fixed in `603eee961`) |
| <a id="log-8481319"></a>`8481319` | `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/agpt-80b-failover.o8481319` (8 min, walltime — pre-`e216a2523` zombie-success wrapper) |
| <a id="log-8485512"></a>`8485512` | `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/agpt-80b-n512-v2-failover-chain1.o8485512` (7 min, walltime — pre-fix wrapper) |
| <a id="log-8503077"></a>`8503077` | queued — 2058N (2048 active + 10 spare) stress test of fully-fixed wrapper; log will land in `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/` |

## Why no v1 here

v1 80B trajectories all NaN'd or hung. See
[`../../historical/v1-bf16/`](../../historical/v1-bf16/README.md) for
the v1 history (256N + 512N AdamW LR=1.1e-5 NaN, SophiaG bf16
overflow at dim=9216). v2 is a fresh start with the working config.

## Risks / things to watch

- **Compile-time substitute is none**: `compile=OFF` on torch 2.13
  matches the throughput v1 got with `compile=ON` on torch 2.10
  (~17.8% MFU at 4N), so we're not paying a throughput penalty —
  but if MFU at 512N drops significantly below that, investigate.
- **Memory headroom**: 4N smoke peaked at 88.94%; FSDP at 6,144
  GPUs (vs 24 in the smoke) gives much more headroom per shard so
  this should be fine, but watch for OOM on first training step.
- **Grad-norm warmup**: 4N smoke had grad-norm spike to ~34 around
  steps 15-16 then recover to ~14 by step 20. Production should
  benefit from the 200-step linear warmup (the 2B/20B v2 configs
  already include this — verify in the printed config that 80B does
  too).
- **Set_determinism `std::bad_alloc`**: this intermittent init crash
  hit at 6,144 ranks in the 20B 512N (8466848). The failover
  wrapper handles this via blind-rotation retry.
- **Silent hang at step ~800** (8479579 incident): no_progress
  watchdog NOT yet implemented in the failover wrapper. If 8480361
  hangs we'll need to qdel manually.
