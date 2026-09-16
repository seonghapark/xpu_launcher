# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz expert compute backends for MoE.

Subclasses upstream `GroupedExperts` to add a `compute_backend` selector
without modifying core. Two backends are supported here:

- ``"grouped_mm"`` (default): defer to upstream's ``torch._grouped_mm``
  path. Requires SM90+ on CUDA; on XPU there is no grouped-mm fallback.
- ``"for_loop"``: per-expert ``matmul`` loop. Slower but works on every
  device. Re-vendored from upstream's ``_run_experts_for_loop`` which
  was deleted in pytorch/torchtitan#3308.
"""

from dataclasses import dataclass
import os
from typing import Literal

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.utils.checkpoint import checkpoint

# GroupedExperts comes straight from upstream — the local `.moe` copy
# was deleted because it was byte-identical to the upstream module. See
# moe/__init__.py for the same import-re-route.
from torchtitan.models.common.moe import GroupedExperts


ExpertComputeBackend = Literal["for_loop", "grouped_mm"]


def _env_flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}


def _empty_expert_output(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Zero-token expert call. Preserve a zero-gradient path through w1/w2/w3."""
    out = x.new_empty((0, w2.shape[1]))
    return out + (w1.sum() + w2.sum() + w3.sum() + x.sum()) * 0


@torch.compiler.disable
def _run_experts_for_loop(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if num_tokens_per_expert.numel() == 0:
        return _empty_expert_output(w1, w2, w3, x)

    # NOTE: this incurs a device-host sync.
    num_tokens_per_expert_list = num_tokens_per_expert.tolist()

    if _env_flag_enabled("TT_MOE_EXPERT_PREALLOC_OUTPUT"):
        out = x.new_empty((x.shape[0], w2.shape[1]))
        offset = 0
        for expert_idx, num_tokens in enumerate(num_tokens_per_expert_list):
            if num_tokens == 0:
                continue
            x_expert = x[offset : offset + num_tokens]
            x_expert_bf16 = x_expert.bfloat16()
            h = F.silu(
                torch.matmul(
                    x_expert_bf16,
                    w1[expert_idx].bfloat16().transpose(-2, -1),
                )
            )
            gate = torch.matmul(
                x_expert_bf16,
                w3[expert_idx].bfloat16().transpose(-2, -1),
            )
            if _env_flag_enabled("TT_MOE_EXPERT_INPLACE_GATE_MUL"):
                h.mul_(gate)
            else:
                h = h * gate
            h = torch.matmul(h, w2[expert_idx].bfloat16().transpose(-2, -1))
            out[offset : offset + num_tokens].copy_(h.type_as(x))
            offset += num_tokens
        return out

    out_experts_splits = []
    offset = 0
    for expert_idx, num_tokens in enumerate(num_tokens_per_expert_list):
        if num_tokens == 0:
            continue
        x_expert = x[offset : offset + num_tokens]
        x_expert_bf16 = x_expert.bfloat16()
        h = F.silu(
            torch.matmul(
                x_expert_bf16,
                w1[expert_idx].bfloat16().transpose(-2, -1),
            )
        )
        gate = torch.matmul(
            x_expert_bf16,
            w3[expert_idx].bfloat16().transpose(-2, -1),
        )
        if _env_flag_enabled("TT_MOE_EXPERT_INPLACE_GATE_MUL"):
            h.mul_(gate)
        else:
            h = h * gate
        h = torch.matmul(h, w2[expert_idx].bfloat16().transpose(-2, -1))
        out_experts_splits.append(h.type_as(x))
        offset += num_tokens
    if len(out_experts_splits) == 0:
        return _empty_expert_output(w1, w2, w3, x)
    return torch.cat(out_experts_splits, dim=0)


class EzpzGroupedExperts(GroupedExperts):
    """GroupedExperts variant that selects between expert compute backends.

    Defers to upstream's grouped-mm path by default. Set ``compute_backend``
    to ``"for_loop"`` on devices without grouped-mm support (e.g. XPU,
    pre-SM90 CUDA).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(GroupedExperts.Config):
        compute_backend: ExpertComputeBackend = "grouped_mm"

    def __init__(self, config: Config):
        super().__init__(config)
        self.compute_backend: ExpertComputeBackend = config.compute_backend

    def _experts_forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: this method is intentionally NOT marked with
        # @torch.compiler.disable. The grouped_mm path delegates straight to
        # the upstream `super()._experts_forward(...)` which is
        # compile-friendly, and we want torch.compile to see it. The
        # for-loop path is opted out of compile via the module-level
        # @torch.compiler.disable decorator on `_run_experts_for_loop`.
        if self.compute_backend == "grouped_mm":
            return super()._experts_forward(x, num_tokens_per_expert)

        # Param names use Shazeer shape-suffix style post upstream PR #3425
        # (41st sync): w1_EFD, w2_EDF, w3_EFD.
        if isinstance(self.w1_EFD, DTensor):
            w1 = self.w1_EFD.to_local()
            # pyrefly: ignore [missing-attribute]
            w2 = self.w2_EDF.to_local()
            # pyrefly: ignore [missing-attribute]
            w3 = self.w3_EFD.to_local()
        else:
            w1 = self.w1_EFD
            w2 = self.w2_EDF
            w3 = self.w3_EFD

        if self.compute_backend == "for_loop":
            if _env_flag_enabled("TT_MOE_CHECKPOINT_EXPERTS"):
                return checkpoint(
                    _run_experts_for_loop,
                    w1,
                    w2,
                    w3,
                    x,
                    num_tokens_per_expert,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            return _run_experts_for_loop(w1, w2, w3, x, num_tokens_per_expert)
        raise ValueError(f"Unknown expert compute backend: {self.compute_backend!r}")
