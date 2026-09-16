# `CheckpointManager` async-staging `new_group(backend="gloo")` fails on xccl-only PGs

## TL;DR

Upstream `torchtitan/components/checkpoint.py:CheckpointManager.__init__`
unconditionally calls `dist.new_group(backend="gloo")` when
`async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM)`. On
XPU (xccl-only default PG), this raises a C++ RuntimeError from
`libtorch_cpu` at trainer init:

```
RuntimeError: No backend type associated with device type xpu
```

(error string is in `libtorch_cpu.so`; no Python-side site greps for
it.) The traceback bottoms out at:

```
File "torchtitan/components/checkpoint.py", line 468, in __init__
    self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))
```

The crash is fatal — trainer can't even build the `CheckpointManager`,
so training never starts.

**Workaround:** pass `--checkpoint.async-mode=disabled` (or set
`CHECKPOINT_ASYNC_MODE=disabled` if launching through
[`scripts/submit_agpt_80b_aurora_venv_failover.sh`](../../scripts/submit_agpt_80b_aurora_venv_failover.sh)).
Sync checkpoint mode skips the gloo subgroup creation entirely.

## Repro

```bash
qsub -A datascience -q workq -l filesystems=flare:home \
     -l walltime=2:00:00 -l select=4 \
     -v NHOSTS_TRAIN=4,FAILOVER_MAX_RETRIES=0,\
        DATASET=eliplutchok/fineweb-small-sample,\
        CHECKPOINT_ASYNC_MODE=async \
     torchtitan/experiments/ezpz/scripts/submit_agpt_80b_aurora_venv_failover.sh
```

`CHECKPOINT_ASYNC_MODE=async` is the script's default; the bug bites on
the default. Confirmed reproduction on Sunspot 4N, job `12468189`
(2026-06-07, post-47th-sync). Resubmitting `12468190` with
`CHECKPOINT_ASYNC_MODE=disabled` got past the crash and reached training.

## Why it fails

`torchtitan/components/checkpoint.py:466-468` (post upstream PR #3393
"components.checkpoint: Improve readability"):

```python
self.pg: dist.ProcessGroup | None = None
if self.async_mode in (AsyncMode.ASYNC, AsyncMode.ASYNC_WITH_PINNED_MEM):
    self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))
```

The intent is to create a CPU-side ProcessGroup so async-staging
checkpoint work (tensor → CPU pin → host file) can use a gloo
collective without blocking the GPU/XPU compute stream's collectives.

But `dist.new_group(backend="gloo")` ultimately routes through
`torch._C._distributed_c10d.Backend.backend_type_map["gloo"]`. When the
default ProcessGroup was initialized with `backend="xccl"` only (no
multi-backend `cpu:gloo,xpu:xccl` spec), the gloo backend isn't
registered for any device type, and the C++ side raises:

```
RuntimeError: No backend type associated with device type xpu
```

(The error mentions `xpu` not `gloo` because the lookup goes through
the default group's accelerator backend first; xpu has no gloo entry.)

NCCL/CUDA users don't hit this because torchtitan's default
distributed init configures cuda groups with a default gloo PG, so
the subgroup creation lands on the cpu-side gloo backend cleanly.
xccl on XPU init doesn't get the cpu gloo backend bundled in.

## What needs to land upstream

Two paths upstream could take, either resolves it:

1. **`CheckpointManager.__init__` should fall back to `disabled`
   silently** (or warn) when `new_group(backend="gloo")` fails — async
   staging is a perf optimization, not a correctness requirement.
2. **`torchtitan.distributed.utils.init_distributed`** should bind a
   gloo cpu backend alongside xccl on XPU, like it does for cuda. This
   would also fix the sibling `dist.new_group(backend="gloo")` call in
   `parallel_dims.py` (already worked around in
   [`xccl_split_group_workaround.py`](../../xccl_split_group_workaround.py),
   same shape of "upstream code path assumes gloo is available"). The
   xccl_split_group fix and this CheckpointManager fix would both
   evaporate if gloo were universally available on XPU.

Filing the issue against `pytorch/torchtitan` is the cleaner path —
this is torchtitan-side code, not PyTorch core. The PyTorch core
behavior (xccl groups not carrying a gloo subgroup by default) is
arguably intentional.

## Where this could live in ezpz

The cleanest local workaround would mirror `xccl_split_group_workaround.py`:
a module that monkey-patches `CheckpointManager.__init__` to skip the
`new_group(backend="gloo")` call when the default PG is xpu-only.
Lazily installed from `FaultTolerantTrainer.init_distributed`, no-op on
cuda/cpu builds, idempotent.

Not done yet — for now `CHECKPOINT_ASYNC_MODE=disabled` (sync mode) is
the operational workaround. Filed under future work; revisit if async
staging perf matters enough to justify monkey-patching upstream
checkpoint code.

## Removal criteria

Either:
- Upstream `CheckpointManager.__init__` makes the gloo-subgroup creation
  defensive (wrap in try/except, fall back to sync mode), OR
- `init_distributed` registers a gloo backend alongside xccl on XPU.

## Related

- [`xccl_split_group_unsupported.md`](xccl_split_group_unsupported.md)
  — sibling xpu/gloo-related bug we worked around in
  `xccl_split_group_workaround.py`. Same shape: upstream assumes a
  capability that xpu's default PG doesn't provide.
- [`train_timeout_xpu_silent_noop.md`](train_timeout_xpu_silent_noop.md)
  — sibling xpu/Python-dispatch gap (no `xpu` branch in
  `_set_pg_timeout`).
