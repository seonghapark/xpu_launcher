"""Minimal repro: does xccl actually abort a deadlocked collective on timeout?

Setup: 2 ranks on the same node (1 XPU each). Both call all_to_all
with mismatched recv counts so the collective can never satisfy
itself. A 30-second timeout is set on the process group via
``ProcessGroupXCCL.set_timeout``.

Expected outcomes
-----------------

If xccl's runtime honors the per-PG timeout (the optimistic case
behind ezpz's xpu-aware set_pg_timeouts workaround), the program
should abort or raise within ~30 s of the deadlock starting.

If xccl's runtime ignores the timeout (the pessimistic case — the
real bug lives in C++, not Python dispatch), the program hangs
indefinitely and only dies when killed externally.

Run
---

    ssh <node> "bash --login -c 'cd <repo> && source ~/.ezpz/utils.sh \\
        && ezpz_setup_env && source .venv/bin/activate \\
        && ezpz launch python3 \\
            torchtitan/experiments/ezpz/docs/upstream-issues/repro_xccl_timeout_abort.py'"

Expects exactly 2 ranks (the launch harness usually gives you 12 per
node — override with ``MPICH_PROCESSES`` or run via a smaller mpiexec
invocation if needed).
"""

from __future__ import annotations

import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist


TIMEOUT = timedelta(seconds=30)


def main() -> int:
    if not torch.xpu.is_available():
        print("XPU not available; this repro only makes sense on Intel XPU.")
        return 0

    def _env_int(*names: str, default: int = 0) -> int:
        for n in names:
            v = os.environ.get(n, "")
            if v.strip():
                return int(v)
        return default

    # Aurora PALS exports ALPS_APP_PE for rank; world size has to come
    # from us. The smoke tests rely on ezpz.launch to populate these;
    # here we run via raw mpiexec so we fall through PMI_* / OMPI_* /
    # ALPS_APP_PE in order, then use --np to know world size via the
    # WORLD_SIZE env var we set ourselves on the outer shell.
    rank = _env_int(
        "RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "ALPS_APP_PE"
    )
    world = _env_int(
        "WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", default=2
    )
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world)
    if world != 2:
        print(
            f"[rank{rank}] WORLD_SIZE={world}; this repro is shaped for "
            "exactly 2 ranks. Skipping mismatch — set up a 2-rank launch.",
            flush=True,
        )

    torch.xpu.set_device(rank % torch.xpu.device_count())
    dist.init_process_group(
        backend="xccl",
        init_method="env://",
        timeout=timedelta(minutes=2),
    )

    pg = dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(torch.device("xpu"))

    # Apply the tight per-PG timeout via the Backend API directly
    # (this is what the ezpz workaround does in trainer.py).
    backend.set_timeout(TIMEOUT)
    print(
        f"[rank{rank}] set_timeout({TIMEOUT}) called on "
        f"{type(backend).__name__}",
        flush=True,
    )

    # NOTE: no dist.barrier() here — the whole point of the test is
    # that rank 1 will NOT participate in the next collective. A
    # barrier would deadlock the test itself.

    # Force an unambiguous deadlock: rank 0 enters allreduce, rank 1
    # sleeps and never enters. With a working timeout-honoring backend,
    # rank 0 should abort at ~TIMEOUT seconds. With a no-op timeout,
    # rank 0 hangs until the outer wall-clock kill.
    tensor = torch.full((1024,), float(rank), dtype=torch.float32, device="xpu")
    t_start = time.monotonic()

    if rank == 1:
        sleep_s = TIMEOUT.total_seconds() * 4  # outlive any honest timeout
        print(
            f"[rank{rank}] not entering collective; sleeping {sleep_s}s "
            "(rank 0 is supposed to abort first)",
            flush=True,
        )
        time.sleep(sleep_s)
        print(f"[rank{rank}] woke from sleep — rank 0 hung, no abort fired",
              flush=True)
        return 3

    # rank 0
    print(
        f"[rank{rank}] entering allreduce with no rank-1 peer at "
        f"t=0.0; timeout={TIMEOUT.total_seconds()}s",
        flush=True,
    )
    try:
        dist.all_reduce(tensor)
        torch.xpu.synchronize()
        elapsed = time.monotonic() - t_start
        print(
            f"[rank{rank}] UNEXPECTED: allreduce returned at t={elapsed:.1f}s "
            f"without aborting. tensor[0]={tensor[0].item()}",
            flush=True,
        )
        return 2
    except Exception as exc:
        elapsed = time.monotonic() - t_start
        print(
            f"[rank{rank}] EXPECTED-FOR-TIMEOUT-HONORING-BACKEND: raised "
            f"after t={elapsed:.1f}s: {type(exc).__name__}: {exc}",
            flush=True,
        )
        # 0 if the abort happened within ~2x the configured timeout
        if elapsed < TIMEOUT.total_seconds() * 2:
            print(
                f"[rank{rank}] PASS: abort fired within tolerance "
                f"({TIMEOUT.total_seconds() * 2}s)",
                flush=True,
            )
            return 0
        print(
            f"[rank{rank}] FAIL: abort fired but late "
            f"({elapsed:.1f}s vs {TIMEOUT.total_seconds()}s configured)",
            flush=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
