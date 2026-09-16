"""Verify PR 1 (xccl Python dispatch fix) for ``_set_pg_timeout``.

This script exercises ``torch.distributed.distributed_c10d._set_pg_timeout``
on an xccl-backed default process group and checks whether the dispatch
path actually reaches the backend.

It is designed to run in **two passes**:

1. Against unpatched torch: expect the warning
   ``Set timeout is now only supported for either nccl or gloo.`` to
   fire and the backend's stored timeout to remain unchanged.
2. Against torch with PR 1 applied: expect no warning, and the
   backend's stored timeout to update to the new value.

Exit code:

- ``0`` on PASS (matches the expectation for the current torch state,
  inferred from whether the warning fired)
- ``1`` on unexpected behavior

Run with exactly 2 ranks on an XPU node, e.g.::

    ezpz launch python3 \
        torchtitan/experiments/ezpz/docs/upstream-issues/verify_pr1_dispatch.py
"""

from __future__ import annotations

import sys
import warnings
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _set_pg_timeout

# ezpz.setup_torch handles MASTER_ADDR/PORT discovery + init_process_group
# across the various ALCF launchers (mpiexec, PALS, torchrun).
import ezpz  # noqa: E402

INITIAL_TIMEOUT_S = 120
NEW_TIMEOUT = timedelta(seconds=37)
TARGET_WARNING_SUBSTR = "Set timeout is now only supported"


def main() -> int:
    if not torch.xpu.is_available():
        print("XPU not available; this verifier only makes sense on Intel XPU.")
        return 0

    # ezpz.setup_torch picks backend (xccl on XPU), sets MASTER_ADDR/PORT,
    # initialises the default process group, and returns the rank.
    rank = ezpz.setup_torch(timeout=INITIAL_TIMEOUT_S)

    if not dist.is_initialized():
        print(f"[rank{rank}] ezpz.setup_torch did not init dist", flush=True)
        return 1

    pg = dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(torch.device("xpu"))
    backend_kind = type(backend).__name__

    # Two independent ways to detect that dispatch reached the backend:
    #
    # 1) Monkey-patch ProcessGroupXCCL.set_timeout to record every call.
    #    This is the most direct evidence that _set_pg_timeout routed
    #    through the new xpu branch.
    # 2) Inspect backend.options._timeout (the C++ Backend base exposes
    #    a _timeout property on its Options object). PR 1 only routes
    #    the call via backend.set_timeout(timeout); whether that mutates
    #    options._timeout is xccl's business — we just report what we see.
    calls_recorded: list[timedelta] = []
    original_set_timeout = type(backend).set_timeout

    def _spy_set_timeout(self, timeout):  # type: ignore[no-untyped-def]
        calls_recorded.append(timeout)
        return original_set_timeout(self, timeout)

    type(backend).set_timeout = _spy_set_timeout

    pre_options_timeout = getattr(getattr(backend, "options", None), "_timeout", None)
    print(
        f"[rank{rank}] backend={backend_kind} "
        f"pre_options_timeout={pre_options_timeout}",
        flush=True,
    )

    # Invoke the upstream dispatch path with a fresh warnings context so
    # we can determine whether the "only supported for nccl or gloo"
    # warning fires (= unpatched) or not (= patched).
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _set_pg_timeout(NEW_TIMEOUT, pg)

    # Restore the original method.
    type(backend).set_timeout = original_set_timeout

    warning_fired = any(
        TARGET_WARNING_SUBSTR in str(w.message) for w in caught
    )
    post_options_timeout = getattr(getattr(backend, "options", None), "_timeout", None)
    dispatch_reached_backend = any(t == NEW_TIMEOUT for t in calls_recorded)

    print(
        f"[rank{rank}] backend={backend_kind} "
        f"post_options_timeout={post_options_timeout} "
        f"warning_fired={warning_fired} "
        f"dispatch_reached_backend={dispatch_reached_backend} "
        f"calls_recorded={calls_recorded} "
        f"caught_warnings={[str(w.message) for w in caught]}",
        flush=True,
    )

    # The contract:
    #   - unpatched: warning fires AND set_timeout never called on backend
    #   - patched:   warning does NOT fire AND set_timeout WAS called
    if warning_fired:
        verdict = (
            "UNPATCHED: warning fired; dispatch did not reach xccl backend."
        )
        ok = not dispatch_reached_backend
    else:
        verdict = (
            "PATCHED: warning suppressed; verifying dispatch reached backend."
        )
        ok = dispatch_reached_backend

    print(
        f"[rank{rank}] verdict={verdict} ok={ok} "
        f"(expected_new_timeout={NEW_TIMEOUT})",
        flush=True,
    )

    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
