# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Apply PT-D parallelisms + AC + compile + FSDP to the agpt model.

This is the agpt mirror of `torchtitan.models.llama3.parallelize`. It uses
the new config-based DTensor sharding API: TP is applied via
`model.parallelize(parallel_dims)`, which reads `sharding_config` declarations
that were filled in by `AgptModel.Config.update_from_config`.

Differences vs upstream `parallelize_llama`:

- `disable_fsdp_gradient_division` additionally enables
  `set_force_sum_reduction_for_comms(True)` for non-NCCL backends (CCL on
  XPU). Upstream's version only sets the divide factor.
- After `apply_compile`, resets `torch._dynamo.config.capture_scalar_outputs`
  to False. apply_compile sets it True for MoE; that breaks the
  separately-compiled CrossEntropyLoss for dense models.
- Names the FSDP grouping `[norm, lm_head]` together with
  `reshard_after_forward=False` (upstream uses
  `reshard_after_forward=reshard_after_forward_policy == "always"`).
"""

import ezpz
import ezpz.distributed
import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: Llama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """Apply TP, AC, compile, and FSDP to an agpt model.

    The passed-in model preferably should be on meta device. Otherwise
    the model must fit on GPU or CPU memory.
    """
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    # TP via the config-based sharding API. The model's sharding_config
    # declarations were filled in by update_from_config (see model.py).
    # Upstream #3159 changed Module.parallelize to take ParallelDims (not a
    # bare tp_mesh) so each Module can resolve its own SPMD submesh.
    if parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)
        maybe_enable_async_tp(
            parallelism, compile_config, parallel_dims.get_mesh("tp")
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    # 57th sync: PR #3674 refactored AC into a Configurable policy
    # hierarchy. ac_config is now an ActivationCheckpointing.Config
    # subclass (FullAC.Config / SelectiveAC.Config / MemoryBudgetAC.Config)
    # or None. The old `mode="none"` sentinel is replaced by `None`.
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    if model_compile_enabled:
        apply_compile(model, compile_config)
        # apply_compile unconditionally sets capture_scalar_outputs=True
        # (needed for MoE dynamic shapes). For dense models this breaks
        # the separately-compiled loss_fn when loss_parallel + ignore_index
        # produce unbacked symbols in cross_entropy.
        torch._dynamo.config.capture_scalar_outputs = False

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh = parallel_dims.get_mesh(names)
    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
    )

    if parallel_dims.dp_replicate_enabled:
        logger.info("Applied HSDP to the model")
    else:
        logger.info("Applied FSDP to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Disable FSDP's automatic gradient division and (on XPU/CCL) force
    sum reduction for cross-rank gradient comms.

    On NCCL the default reduce-mean works correctly. On CCL (XPU) we need
    SUM and divide ourselves to avoid losing precision.
    """
    force_sum_reduction = False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        backend = ezpz.distributed.get_torch_backend() or str(
            torch.distributed.get_backend()
        )
        if backend and "nccl" not in str(backend).lower():
            force_sum_reduction = True

    fsdp_modules_updated = 0
    for module in model.modules():
        # Be resilient to FSDPModule class location changes across PyTorch
        # releases by going through the public method.
        set_divide_factor = getattr(module, "set_gradient_divide_factor", None)
        if callable(set_divide_factor):
            set_divide_factor(1.0)
            fsdp_modules_updated += 1
            if force_sum_reduction:
                set_force_sum = getattr(
                    module, "set_force_sum_reduction_for_comms", None
                )
                if callable(set_force_sum):
                    set_force_sum(True)

    logger.info(
        "Configured FSDP gradient division for %d modules (force_sum_reduction=%s)",
        fsdp_modules_updated,
        force_sum_reduction,
    )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
):
    """FSDP2 with the same per-block grouping as upstream llama3.

    Note: matches upstream's `[norm, lm_head]` joint grouping with
    `reshard_after_forward=reshard_after_forward_policy == "always"`
    (last layers don't reshard after forward by default — FSDP would
    prefetch them immediately).
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in model.layers.values():
        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)
    disable_fsdp_gradient_division(model)
