# Upstream PR plan: xccl per-collective timeout enforcement

## Context

The empirical work in
[`train_timeout_xpu_silent_noop.md`](train_timeout_xpu_silent_noop.md)
and [`repro_xccl_timeout_abort.py`](repro_xccl_timeout_abort.py)
established that PyTorch's XCCL backend has **two distinct gaps**
that together cause `train_timeout_seconds` to be silently inert on
Intel XPU:

1. **Python dispatch**: `torch.distributed.distributed_c10d._set_pg_timeout`
   has no `torch.device("xpu")` branch, so the call never reaches
   `ProcessGroupXCCL.set_timeout`. Warning emitted, no action taken.
2. **C++ runtime**: even when `ProcessGroupXCCL.set_timeout` *is*
   called directly (verified by the repro), xccl stores the value
   but does not enforce it — there's no watchdog thread firing
   `ccl_abort()`-equivalent on collectives that exceed the timeout.

This document plans a two-PR upstream sequence.

## PR 1 (small): Python dispatch fix

**Repo**: `pytorch/pytorch`
**Files**: `torch/distributed/distributed_c10d.py`

The change: add an `xpu` branch to `_set_pg_timeout` analogous to the
existing `cuda` branch, calling the public Backend API method
(`set_timeout`) since `ProcessGroupXCCL` doesn't expose
`_set_default_timeout` (the private name `ProcessGroupNCCL` uses).

Proposed diff (sketch):

```python
# torch/distributed/distributed_c10d.py around the existing
# device-dispatch block in _set_pg_timeout:

if torch.device("xpu") in devices:
    if is_xccl_available():
        backend = group._get_backend(torch.device("xpu"))
        from torch._C._distributed_c10d import ProcessGroupXCCL
        if isinstance(backend, ProcessGroupXCCL):
            # XCCL exposes the public Backend API method, not the
            # _set_default_timeout private alias NCCL uses. Call it
            # directly here instead of going through the trailing
            # `backend._set_default_timeout(...)` loop.
            backend.set_timeout(timeout)
            # mark "handled" so the empty-backends warning doesn't fire
            backends.add(backend)
```

(Final form will need to thread through the existing
`backends`-collection / final-loop structure rather than calling
`set_timeout` inline — but the call shape is the point.)

**Tests to add**:

- Unit test in `test/distributed/test_c10d_xccl.py` that creates an
  xccl PG, calls `_set_pg_timeout`, and asserts no warning is
  emitted and that the backend's timeout state has been updated.
  (Whether the backend *enforces* it is PR 2; PR 1 only verifies
  the dispatch reaches the backend.)

**Risk**: very low. Mirrors the existing cuda/nccl pattern. The only
behavior change for non-xpu users is the empty-backends warning no
longer fires for xccl PGs.

**Status**: ready to draft once we have CLA + branch.

## PR 2 (larger): xccl C++ per-collective timeout enforcement

**Repo**: `pytorch/pytorch`
**Files**: `torch/csrc/distributed/c10d/ProcessGroupXCCL.{hpp,cpp}`,
likely also `torch/csrc/distributed/c10d/WorkXCCL.cpp` (or however
xccl wraps its async work).

The change: implement the equivalent of NCCL's watchdog +
`checkAndSetException` chain. Each `WorkXCCL` object needs to know
its enqueue time and the PG timeout, and a watchdog thread (or a
poll loop on completion) needs to fire an abort when the deadline
passes.

This is **much** less mechanical than PR 1. Key open questions:

- **What does oneCCL expose for cancel/abort?** NCCL has
  `ncclCommAbort`. oneCCL has `ccl::comm::~comm()` for destroying
  the communicator, but a per-op cancel is unclear. May need to
  destroy and recreate the communicator on abort, which is much
  heavier than NCCL's path.
- **What does Intel's xccl backend reviewer want?** Worth an
  RFC issue before code: "ProcessGroupXCCL doesn't enforce
  per-PG timeouts — here's the repro — what's the intended
  enforcement story?" The Intel team is the primary owner of
  ProcessGroupXCCL.cpp and they may have a roadmap already.
- **Does TORCH_NCCL_ASYNC_ERROR_HANDLING have an xccl equivalent?**
  The env var name suggests the abort mechanism is NCCL-specific.
  An xccl equivalent (e.g. `TORCH_XCCL_ASYNC_ERROR_HANDLING`) is
  probably the right knob to add.

**Tests to add**: a port of our `repro_xccl_timeout_abort.py` as a
proper c10d distributed test that asserts the abort fires within
`timeout × 2`. Should sit next to the existing nccl timeout tests.

**Risk**: medium to high. Watchdog code touches threading and the
collective lifecycle; getting it wrong can deadlock the watchdog
itself, mask real errors, or cause spurious aborts. Wants Intel
reviewer eyes.

**Status**: blocked on RFC issue + Intel reviewer.

## Sequencing

1. **File RFC issue** on `pytorch/pytorch` describing both gaps.
   Link our repro and writeup. Tag the Intel XPU + c10d folks. Ask
   specifically: "Is PR 1 (Python dispatch) acceptable as a
   standalone fix even before PR 2 (C++ enforcement)?" Answer is
   probably yes — it makes the dispatch correct and removes a
   misleading warning, even if the underlying enforcement is still
   missing.
2. **Land PR 1** (Python dispatch). Cheap, contained, defensible.
3. **Wait for guidance on PR 2** from Intel folks. If they have a
   work-in-progress timeout/abort branch, contribute the repro as
   a regression test there. If they don't, the RFC discussion
   determines whether we drive the PR ourselves or hand it off.

## Out of scope (for now)

- Adding our own ezpz-side step-progress watchdog as a workaround.
  Worth doing if PR 2 stays stuck for weeks; not the most urgent
  thing right now because no current XPU production run has been
  observed hanging long enough for it to matter (the May 20
  step-41 incident was 16 min and transient).

## Artifacts to attach to the RFC

- `train_timeout_xpu_silent_noop.md` (this dir)
- `repro_xccl_timeout_abort.py` (this dir)
- The retry verification log
  `logs/hang-retry-step41/xpu_timeout_workaround_verify-*.log`
  showing the ezpz Python-side workaround does emit the right
  "Applied train timeout to N xccl ProcessGroup(s)" line but
  the underlying enforcement is still missing.
- The repro output:
  `logs/hang-retry-step41/xccl_deadlock_repro5-*.log` showing
  rank 1 wakes from a 120 s sleep with rank 0 still hung.
