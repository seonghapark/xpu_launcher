# `train_timeout_seconds` silently no-ops on XPU (xccl)

## TL;DR

`CommConfig.train_timeout_seconds` (default 100 s) is silently ignored
on Intel XPU / xccl. Every torchtitan run on XPU emits the warning
`Set timeout is now only supported for either nccl or gloo.` — and
we've been treating it as noise. **Every production XPU run has no
per-collective training-time timeout.** If a collective hangs, the
job consumes its full PBS walltime instead of aborting at 100 s.

This was the diagnostic gap behind the 2026-05-20
`moe_debugmodel_ep` step-41 hang: 100 s configured, 985 s silence
observed, no abort. The hang itself was transient
(see [`moe_ep_step41_hang.md`](moe_ep_step41_hang.md)) — but
discovering that took 16 minutes of wallclock instead of 100 seconds.

## The plumbing gap

`torchtitan/trainer.py:869` and `torchtitan/experiments/ezpz/trainer.py:624`
call `set_pg_timeouts(timeout=train_timeout_seconds, ...)` after the
first training step. `set_pg_timeouts` in
`torchtitan/distributed/utils.py:467` calls
`torch.distributed.distributed_c10d._set_pg_timeout(timeout, group)`
for each mesh process group.

PyTorch's `_set_pg_timeout` (current torch 2.13 in our `.venv`) has
device-conditional branches for **cpu** (gloo) and **cuda** (nccl /
gloo / torchcomms) — **no `torch.device("xpu")` branch**:

```python
devices = group._device_types
backends = set()
if torch.device("cpu") in devices and is_gloo_available():
    ...
if torch.device("cuda") in devices:
    ...  # nccl / gloo / torchcomms
if len(backends) == 0:
    warnings.warn(
        "Set timeout is now only supported for either nccl or gloo.",
        stacklevel=2,
    )
for backend in backends:
    backend._set_default_timeout(timeout)
```

For an xccl PG, `backends` stays empty, the warning fires, and
`_set_default_timeout` is never called.

This is **not** because `ProcessGroupXCCL` lacks the capability:
```
>>> from torch._C._distributed_c10d import ProcessGroupXCCL
>>> 'set_timeout' in dir(ProcessGroupXCCL)
True
>>> 'abort'       in dir(ProcessGroupXCCL)
True
```

The method is there; the dispatch in `_set_pg_timeout` simply doesn't
route to it.

## What still works

- **`init_timeout_seconds`** (default 300 s) does reach
  `init_process_group(timeout=...)` at
  `torchtitan/distributed/utils.py:431`. The call is backend-uniform,
  so an init-time deadlock should still respect the configured
  timeout. (Untested.)
- **Async error handling** (`TORCH_NCCL_ASYNC_ERROR_HANDLING=3`,
  `TORCH_NCCL_DUMP_ON_TIMEOUT=1`) is set unconditionally in
  `torchtitan/distributed/utils.py:397-414` — but those env vars
  govern NCCL, not XCCL, so they're also silently inert on XPU.
- **gloo PGs** (the ones our distributed code creates for barriers
  after the XCCL barrier-missing workaround) DO honor
  `_set_default_timeout` via the cpu+gloo branch.

## What this means in practice

For any XPU torchtitan run with a hung collective (e.g. EP all-to-all
desync, single-rank stall, fabric flap):

- The job will not abort at `train_timeout_seconds`. It will sit
  silent until PBS walltime expires.
- For production 80B-style 12 h walltime: a single hang wastes ~12 h
  of node-hours per job instead of ~2 min.
- The warning at startup is the only signal that this is happening.

## Empirical confirmation (2026-05-21)

We wrote a minimal 2-rank repro
([`repro_xccl_timeout_abort.py`](repro_xccl_timeout_abort.py)) that:

1. Calls `backend.set_timeout(timedelta(seconds=30))` directly on
   `ProcessGroupXCCL` (i.e. bypassing the broken Python dispatch).
2. Has rank 0 enter `dist.all_reduce` while rank 1 sleeps 120 s and
   never participates. With a working timeout-honoring backend rank
   0 should abort at t≈30 s.

Result:

```
[rank0] set_timeout(0:00:30) called on ProcessGroupXCCL
[rank1] set_timeout(0:00:30) called on ProcessGroupXCCL
[rank0] entering allreduce with no rank-1 peer at t=0.0; timeout=30.0s
[rank1] not entering collective; sleeping 120.0s (rank 0 is supposed to abort first)
[rank1] woke from sleep — rank 0 hung, no abort fired
rank 1 exited with code 3
rank 0 died from signal 15        # outer wall-clock kill
```

**Rank 0 never aborted.** It stayed in `all_reduce` until killed
externally. The `set_timeout` call was accepted (no error), but
xccl's C++ runtime did not act on it.

This means the bug has **two layers**:

1. **Python dispatch gap** (`_set_pg_timeout` has no xpu branch) —
   ezpz workaround in `experiments/ezpz/trainer.py:_set_pg_timeouts_xpu_aware`
   addresses this; the user-visible warning goes away and
   `set_timeout` is actually called on the xccl PG.
2. **C++ runtime gap** (`ProcessGroupXCCL` stores the timeout but
   doesn't enforce it per collective) — needs a watchdog thread +
   abort plumbing, analogous to NCCL's `WorkNCCL::checkAndSetException`
   chain. This is the bigger upstream fix.

So the ezpz workaround is necessary but not sufficient: even with
it, a hung xccl collective will not abort at `train_timeout_seconds`.
The workaround still has value (it lets users / tooling see a
deterministic value when they query the PG, and it's the right
shape for the day the C++ side gets wired up), but operationally
the only thing that will save a hung XPU training job today is the
PBS walltime ceiling or an out-of-band watchdog (a separate process
that monitors training-step progress and SIGTERMs the launcher).

## Suggested follow-ups

1. **Emit our own warning at config-build time.** When
   `cfg.training.dtype` or compute device is XPU and
   `train_timeout_seconds` is non-default, log a clear ezpz-side
   warning that the timeout won't fire. Cheap; user-visible.
2. **Add an XPU branch to `_set_pg_timeout` upstream.** The dispatch
   is the one-line fix:
   ```python
   if torch.device("xpu") in devices:
       backend = group._get_backend(torch.device("xpu"))
       if is_xccl_available() and isinstance(backend, ProcessGroupXCCL):
           backends.add(backend)
   ```
   Worth filing against `pytorch/pytorch`.
3. **Provide an ezpz-side watchdog** as a workaround: a separate
   thread that monitors training-step progress (e.g. via the metrics
   logger) and SIGTERMs the launcher if no step is logged for
   `train_timeout_seconds × N`. Coarser than per-collective abort
   but covers the production case.

## Evidence in run logs

The warning appears in **every** torchtitan-ezpz XPU run we have
logs for. Some examples (line numbers, file paths under
`logs/`):

- `smoke-pr3386-followup/moe_debugmodel_ep_lbs2-20260520-230219.log:384`
  — the original hang. Warning emitted at startup, 16 min of silence
  later, killed by walltime.
- `smoke-pr3386-followup/moe_2b_ep-20260520-232257.log:396` — the
  EP=2 clean run. Same warning, no consequence (no hang).
- `smoke-pr3386-followup/agpt_2b-20260520-230009.log:359` — agpt
  side. Same warning.
- `hang-retry-step41/registry_fix_verify-20260521-081457.log:447`
  — today's verify run. Same warning, no consequence.

So 100% of XPU runs land in this state.
