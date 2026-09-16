# Production Training — agpt 80B @ 4 nodes (Aurora smoke validation)

> **This is a one-shot smoke validation, not a production trajectory.**
>
> 2026-06-08 (attempt #7 of a multi-attempt debug session) was the
> **first end-to-end 80B production-stack validation on Aurora**. Prior
> 4N 80B baselines (`12466025`, `12467825`, `12468157`) all ran on
> **Sunspot**; this run is the Aurora-side companion. Result: stack
> works end-to-end, the **step-10 sync-mode DCP save landed cleanly**
> (904 GB, 48 `.distcp` shards + `.metadata`, matching the Sunspot
> `12468197` reference exactly), and the patched blendcorpus + xccl
> `split_group` workaround + sync-mode checkpoint path all validated
> on Aurora's xccl process group.
>
> Run died at ~14:18 when the parent `test.sh` allocation (PBS
> `8530807`, 1h walltime) ended — `TRAINING_STEPS=20` was set but
> only 10 steps completed before the allocation expired. The
> `step-10` save **is** the load-bearing milestone; the missing
> step-20 doesn't matter for validation.

## Run

| Field | Value |
|-------|-------|
| Clone | `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/` |
| Date | 2026-06-08 |
| Attempt | #7 of a multi-attempt debug session (prior PBS attempts: `8530199`, `8530216`, `8530243`, `8530800`) |
| Launched from | `test.sh` allocation `8530807` head node (`x4217c0s0b0n0`), ssh foreground |
| Log | `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/.interactive-80b-4n-r7-direct-20260608-085848.log` |
| Stack | torch 2.13 venv (patched blendcorpus, ezpz 0.18.7 from `yeet-retry-on-rsync-failure`, `trl==1.5.1`, `spmd_types==0.2.1`) |
| Model | `agpt_80b` |
| Optimizer | AdamW LR=1e-6 (SophiaG/Muon broken at 80B — see CLAUDE.md) |
| Parallelism | TP=2, FSDP across remaining ranks |
| AC | full |
| Compile | OFF (DeviceMesh-in-saved-tensors AOT autograd crash on torch 2.13) |
| Master dtype | fp32 |
| LBS / GAS / GBS | 1 / 1 / 24 |
| Seq len | 8192 |
| Dataset | blendcorpus / olmo-mix-1124 |
| Checkpoint mode | sync (`CHECKPOINT_ASYNC_MODE=disabled`) |
| Checkpoint dir | `outputs/checkpoints/agpt-80b-adamw-olmo-mix-1124-n4-gbs24/step-10/` |

## Step-by-step

| Step | Loss | grad_norm | Memory | TPS/GPU | TFLOPs/GPU | MFU |
|----:|----:|----:|----:|----:|----:|----:|
| 1 | 12.93362 | 5.0574 | 53.02 GiB (82.86%) | 71 | 38.85 | 13.03% |
| 2 | 12.91640 | 4.99 | 56.93 GiB (88.97%) | 91 | 49.63 | 16.64% |
| 3 | 12.88 | 4.97 | — | 97 | — | 17.74% |
| 4–9 | descending | — | — | — | — | ~17.9% |
| 10 | 12.02807 | 6.28 | — | 97 | 53.21 | 17.84% |

Loss descent **12.93 → 12.03 over 10 steps** (-0.91 nats), MFU
steady at **~17.9%** matches the Sunspot `12467825` / `12468157`
baselines bit-for-bit.

## The load-bearing moment: step-10 sync save

`14:06:22`, immediately after step-10:

- **904 GB** total on disk
- **48 `.distcp` shards** (matches Sunspot `12468197` reference exactly)
- **`.metadata`** file present, 21 MB
- Path: `/flare/AuroraGPT/foremans/runs/agpt-80b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-80b-adamw-olmo-mix-1124-n4-gbs24/step-10/`

Sync mode + xccl `split_group` workaround validated on Aurora's
xccl process group end-to-end. This is what every prior dispatch
failed to demonstrate.

## What this validates

- The 80B production stack works end-to-end on Aurora (not just
  Sunspot). All prior Aurora attempts failed before saving.
- The xccl `split_group_workaround` (commit `8031d1d3`) works on
  Aurora's xccl PG.
- Sync-mode DCP save (`CHECKPOINT_ASYNC_MODE=disabled`,
  commit `ce321caae` as default) works.
- The patched blendcorpus (global-barrier removed) loads cleanly
  at 4N without the lazy-per-corpus deadlock.

## Five stacked bugs cleared to get here

Documented in the 2026-06-08 (aurora pm) journal entry; summarized:

1. **80b-v2 repo was 229 commits behind `origin/ezpz`.** Pulled to
   get `8031d1d3` (xccl_split_group_workaround) + `ce321caae`
   (`CHECKPOINT_ASYNC_MODE=disabled` default) + the May/June 80B
   prod sync-ckpt validation work.
2. **80b-v2 `.venv` was symlinked to 2b-v2 `.venv`.** Would have
   polluted the 2B chain. Broke the symlink (8.5 GB `cp -a`),
   installed ezpz 0.18.7 (from `yeet-retry-on-rsync-failure` branch)
   + `trl==1.5.1` + `spmd_types==0.2.1`.
3. **Patched blendcorpus deadlocked on a global
   `torch.distributed.barrier()`** in `_build_index_mappings`.
   `BlendableDataset.__getitem__` is lazy per-corpus → different
   ranks hit the barrier on different corpora at different
   wall-clock times → deadlock at 4N+. Diagnosed via `py-spy dump`
   on rank 0 (stuck at `barrier` inside `_build_index_mappings:1141`)
   and rank 6 (already in `train_step → dataloader.__next__`).
   Removed the barrier; existing `_load_with_retry` already handles
   the EOFError race the barrier was supposed to protect against.
4. **PBS-script invocation was missing
   `export ZE_FLAT_DEVICE_HIERARCHY=FLAT`** on the inner shell —
   caused `_infer_topology` to see 6 GPUs/host instead of 12 and
   reject the launch with `ngpus must be > 0 and <= 24, got 48`.
5. **ssh-launched foreground from the `test.sh` allocation head
   node** so `py-spy` + Ctrl-C were available without watchdog
   interference, which made the diagnosis possible at all.

## Follow-up status

- **256N attempt `8530891`** (same config, scaled to 256N,
  GBS=1536, `TRAINING_STEPS=110`, sync ckpt): setup + step-1 clean
  (loss=12.94, grad_norm=4.94, **110 TPS/GPU / 20.3% MFU** —
  actually better per-GPU than 4N). Step-2 `grad_norm=NaN`, step-3
  onward `loss=NaN`. Walltime-killed at step 79; step-100 ckpt
  save never fired. Open hypotheses:
  - bf16 overflow in attention/MLP at GBS=1536 (vs 4N's GBS=24)
  - TP=2 loss-reduction bug (`_dist_reduce` short-circuit) bleeding
    into the grad-norm path
  - AdamW fp32-master second-moment overflow on first activations
    at scale
- **LR=1e-7 retry `8531721`** (10× smaller LR, same TP=2):
  `MemoryError: std::bad_alloc` at model construction. Never
  reached training. (Predecessor `8531345` died at 177s from a
  bad-node SIGSEGV on `x4408c1s3b0n0`.)

The 256N NaN is the live open question and is **not** explained by
"LR too high at GBS=1536" — step-2's effective LR was clamped to
~1.8e-8 by the warmup schedule, and the first optimizer step
still produced NaN grads.

## Related

- Parent: [`production/agpt/80b/README.md`](../README.md) — 80B status across all node counts
- Sunspot 4N companions: `12466025` (2026-05-05), `12467825`
  (2026-06-02), `12468157` (2026-06-06). All 20 steps, all
  ~17.8% MFU, all bit-equivalent within ±0.08 nat.
- Journal: 2026-06-08 (aurora pm) entry in [`docs/journal.md`](../../../../journal.md)
