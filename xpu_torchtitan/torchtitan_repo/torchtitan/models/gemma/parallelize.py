# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 wrapping for the HF Gemma model used in SFT.

We wrap each ``GemmaDecoderLayer`` and the top-level model. Mixed precision is
applied via ``MixedPrecisionPolicy`` so parameters are computed in ``bf16``
while gradients are reduced in ``fp32`` (a common SFT setting).
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from .model import resolve_dtype


def parallelize_gemma(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype | str = torch.bfloat16,
    reduce_dtype: torch.dtype | str = torch.float32,
) -> nn.Module:
    """Apply FSDP2 to a ``GemmaModel`` wrapper (see ``model.py``).

    Args:
        model: The ``GemmaModel`` wrapper whose ``model`` attribute is the HF
            ``GemmaForCausalLM``.
        mesh: 1D device mesh used as the FSDP shard group.
        param_dtype: Compute dtype for FSDP-managed parameters.
        reduce_dtype: Dtype for the gradient all-reduce.
    """
    if isinstance(param_dtype, str):
        param_dtype = resolve_dtype(param_dtype)
    if isinstance(reduce_dtype, str):
        reduce_dtype = resolve_dtype(reduce_dtype)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_kwargs = {"mesh": mesh, "mp_policy": mp_policy}

    hf_model = model.model  # GemmaForCausalLM
    # Shard each decoder layer individually to overlap comm with compute.
    for layer in hf_model.model.layers:
        fully_shard(layer, **fsdp_kwargs)

    # Then shard the outer HF model, and finally the wrapper.
    fully_shard(hf_model, **fsdp_kwargs)
    fully_shard(model, **fsdp_kwargs)

    return model
