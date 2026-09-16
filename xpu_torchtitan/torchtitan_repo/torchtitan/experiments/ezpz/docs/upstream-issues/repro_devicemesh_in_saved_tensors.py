"""Minimal repro: AOT autograd raises AssertionError when
torch.compile + non-reentrant activation checkpointing + TP are
combined.

A DeviceMesh reference reaches _AutogradSavedState.save_from_forward
in the tensors_saved_for_backwards_with_vc_check_slice, which is
asserted (runtime_wrappers.py:2827-2831) to contain only Tensors:

    AssertionError: expected all tensors_saved_with_vc_check to be
    Tensors, got types: [..., <class 'torch.distributed.device_mesh.DeviceMesh'>]

All three of {TP, AC use_reentrant=False, torch.compile} are required.
Drop any one and the bug goes away.

Tested: torch 2.13.0.dev20260429+xpu and torch 2.13.0.dev20260503+xpu
(also reproduced on torch 2.12.0.dev20260415+xpu). torchtitan trips
this on **every dense agpt config we have tested at TP > 1**, with the
bisect on 2026-05-05 (jobs 12465952 + 12465962) confirming the
assertion fires for `agpt_50b_wide` (48 layers, ~48B params),
`agpt_70b_wide` (72 layers, ~70B), and `agpt_80b` (84 layers, ~80B)
alike on both 2N and 4N. Smallest reliable repro is
`agpt_50b_wide @ 2N + torch 2.13`, which crashes ~30-60s into compile.
**The previously documented "depth-sensitive — works at 48 layers"
finding was a torch-2.10-only artifact; on torch 2.13 every depth
crashes.** Only workaround today is `--compile.no-enable` (or stay on
torch 2.10 for these configs).

Run on a single CPU host with 2 ranks (gloo):

    torchrun --standalone --nproc-per-node 2 \\
        repro_devicemesh_in_saved_tensors.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.utils.checkpoint import checkpoint


class TwoLinear(nn.Module):
    """Toy block: w1 (colwise) -> relu -> w2 (rowwise)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, dim, bias=False)
        self.w2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.relu(self.w1(x)))


class Stack(nn.Module):
    """N TwoLinear blocks; each block forward goes through
    torch.utils.checkpoint(use_reentrant=False)."""

    def __init__(self, num_layers: int, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(TwoLinear(dim) for _ in range(num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = checkpoint(layer, x, use_reentrant=False)
        return x


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "use --nproc-per-node 2"

    dist.init_process_group(backend="gloo")
    tp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("tp",))

    dim, batch, seq = 32, 4, 8
    model = Stack(num_layers=2, dim=dim)

    # Apply colwise->rowwise TP to each block.
    for layer in model.layers:
        parallelize_module(
            layer,
            tp_mesh,
            {"w1": ColwiseParallel(), "w2": RowwiseParallel()},
        )

    # torch.compile the whole model. Together with checkpoint(use_reentrant=False)
    # inside Stack.forward and the TP DeviceMesh closed over by w1/w2,
    # this is the failing combination.
    model = torch.compile(model)

    x = torch.randn(batch, seq, dim, requires_grad=True)
    y = model(x)
    loss = y.sum()
    print(f"[rank {rank}] forward OK, loss={loss.item():.4f}")
    loss.backward()  # <-- AssertionError fires inside save_from_forward
    print(f"[rank {rank}] backward OK — bug NOT reproduced")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()