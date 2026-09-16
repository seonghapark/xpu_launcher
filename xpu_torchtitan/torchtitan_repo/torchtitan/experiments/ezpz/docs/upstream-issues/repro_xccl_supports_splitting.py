# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Minimal repro for the ``ProcessGroupXCCL`` missing-``supportsSplitting`` bug.

Background:
    ``torch/csrc/distributed/c10d/Backend.hpp`` declares
    ``virtual bool supportsSplitting() const { return false; }``.
    ``ProcessGroupNCCL`` overrides this to ``return true;``;
    ``ProcessGroupXCCL`` does NOT, so it inherits the base ``false``.

    Combined with ``DeviceMesh._unflatten`` routing nested mesh PG creation
    through ``split_group`` whenever the default group has
    ``bound_device_id`` set (which xccl's eager-init pathway always does),
    this kills every nested mesh build on XPU.

This script demonstrates both layers in a self-contained way that does
NOT depend on torchtitan / ezpz / mpiexec — just torch's distributed
runtime + xccl + standard env-driven init.

Usage (single-node, single-rank is enough — bug is purely Python-level
on the gate, no inter-rank communication needed to trigger it)::

    MASTER_ADDR=127.0.0.1 MASTER_PORT=29501 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \\
        python repro_xccl_supports_splitting.py

The script is structured as two independent checks::

    [Layer 1] Direct backend query.
        Init xccl PG with ``device_id=rank``, query
        ``pg._get_backend(xpu_device).supports_splitting``,
        assert it is ``False``. This pinpoints the missing C++ override.

    [Layer 2] User-facing symptom.
        Init the same PG, then call ``DeviceMesh.__init__`` with a 1D
        mesh (or ``_unflatten`` if multi-rank), which internally calls
        ``split_group``. Catch the ``RuntimeError`` to confirm the
        gate fires the way ``ParallelDims.build_mesh`` hits in practice.

Exit code is 0 if **both** layers behave as described (i.e. the bug
reproduces). Non-zero indicates the bug is fixed or one of the layers
unexpectedly behaved differently.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def _layer1_query_supports_splitting() -> bool:
    """Returns True if the bug reproduces (supports_splitting is False)."""
    print("\n=== Layer 1: ProcessGroupXCCL.supports_splitting query ===")

    pg = dist.distributed_c10d._get_default_group()
    xpu = torch.device("xpu")

    try:
        backend = pg._get_backend(xpu)
    except Exception as e:
        print(f"  FAIL: could not get xpu backend from default PG: {type(e).__name__}: {e}")
        return False

    name = backend.name()
    supports = backend.supports_splitting
    print(f"  default PG xpu backend: {type(backend).__name__} (name='{name}')")
    print(f"  backend.supports_splitting = {supports}")

    if supports is False:
        print("  BUG REPRODUCED: xccl backend inherits Backend::supportsSplitting() == false")
        print("    (NCCL overrides this in ProcessGroupNCCL.hpp; XCCL does not.)")
        return True
    elif supports is True:
        print("  BUG FIXED: supports_splitting now returns True for xccl")
        return False
    else:
        print(f"  UNEXPECTED: supports_splitting returned {supports!r}")
        return False


def _layer2_device_mesh_unflatten() -> bool:
    """Returns True if the bug reproduces (RuntimeError on nested mesh init)."""
    print("\n=== Layer 2: DeviceMesh._unflatten triggers split_group on xccl ===")

    # Build a 1D world mesh first (this works — it's the *nested* mesh
    # creation that fails).
    world_size = dist.get_world_size()
    try:
        world_mesh = init_device_mesh("xpu", (world_size,), mesh_dim_names=("world",))
        print(f"  init_device_mesh OK ({world_size}-rank world mesh)")
    except Exception as e:
        print(f"  init_device_mesh failed early: {type(e).__name__}: {e}")
        return False

    # Now try to unflatten into a nested mesh — this is what
    # `ParallelDims.build_mesh` does and where the bug actually bites.
    # Even at world_size=1, the gate is triggered: DeviceMesh's PG
    # creation goes through split_group whenever bound_device_id is set.
    if world_size < 2:
        # _unflatten needs world_size > 1 to meaningfully reshape; use a
        # trivial split that still exercises the gate by creating a
        # sub-PG via dist.new_group with the split-aware code path.
        print("  (world_size=1; using direct dist.new_group split probe instead)")
        default_pg = dist.distributed_c10d._get_default_group()
        try:
            sub = dist.split_group(parent_pg=default_pg, split_ranks=[[0]])
            print(f"  split_group returned: {sub}")
            print("  BUG NOT REPRODUCED at Layer 2: split_group succeeded")
            return False
        except RuntimeError as e:
            msg = str(e)
            print(f"  split_group raised: RuntimeError: {msg}")
            if "does not support splitting" in msg:
                print("  BUG REPRODUCED: split_group hit the supports_splitting gate")
                return True
            print(f"  UNEXPECTED RuntimeError shape: {msg}")
            return False

    try:
        # 2D unflatten (e.g. (1, world_size)) — exercises the path
        # ParallelDims.build_mesh uses for the EP sparse mesh.
        sub_mesh = world_mesh._unflatten(
            0,
            (1, world_size),
            ("outer", "inner"),
            backend_override={},
        )
        print(f"  _unflatten OK: {sub_mesh}")
        print("  BUG NOT REPRODUCED at Layer 2: nested mesh built cleanly")
        return False
    except RuntimeError as e:
        msg = str(e)
        print(f"  _unflatten raised: RuntimeError: {msg}")
        if "does not support splitting" in msg:
            print("  BUG REPRODUCED: nested mesh creation hit the supports_splitting gate")
            return True
        print(f"  UNEXPECTED RuntimeError shape: {msg}")
        return False


def main() -> int:
    # Standard env-driven init. Works under `torchrun`, plain
    # MASTER_ADDR/PORT export, or any MPI launcher that sets RANK +
    # WORLD_SIZE + LOCAL_RANK.
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29501")

    if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
        print("ERROR: this repro requires torch.xpu.is_available() == True", file=sys.stderr)
        return 2
    if not dist.is_xccl_available():
        print("ERROR: this repro requires dist.is_xccl_available() == True", file=sys.stderr)
        return 2

    torch.xpu.set_device(local_rank)
    dist.init_process_group(
        backend="xccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        # device_id is what gives the default PG its bound_device_id,
        # which is what makes DeviceMesh prefer split_group over
        # new_group for nested mesh creation.
        device_id=torch.device(f"xpu:{local_rank}"),
    )

    bug1 = _layer1_query_supports_splitting()
    bug2 = _layer2_device_mesh_unflatten()

    print("\n=== Summary ===")
    print(f"  Layer 1 (supports_splitting query)         : {'REPRODUCED' if bug1 else 'NOT REPRODUCED'}")
    print(f"  Layer 2 (DeviceMesh nested unflatten)      : {'REPRODUCED' if bug2 else 'NOT REPRODUCED'}")

    dist.destroy_process_group()
    # Exit 0 = bug confirmed (both layers behave as the issue describes).
    # Exit 1 = bug may be fixed, or one of the layers behaves unexpectedly.
    return 0 if (bug1 and bug2) else 1


if __name__ == "__main__":
    sys.exit(main())
