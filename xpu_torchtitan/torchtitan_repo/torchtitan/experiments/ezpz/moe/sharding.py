# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config-based DTensor sharding for the ezpz/moe (DeepSeek-V3-derived) model.

Mirrors `torchtitan.models.deepseek_v3.sharding` but binds against
`torchtitan.experiments.ezpz.moe.model.Attention` (our MLA Attention is a
separate class from upstream's, even though the structure is identical).

Post upstream PR #3386 (37th sync), MoE sub-configs (router gate, shared
experts, routed experts) are populated unconditionally via upstream's
``set_moe_sharding_config`` helper from ``models/common/moe_sharding``.
``resolve_mesh`` filters disabled axes at runtime, so this matches the
new ``model.parallelize(parallel_dims)`` flow.
"""

from typing import TYPE_CHECKING

import spmd_types as spmd
from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.experiments.ezpz.moe.model import Attention
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.models.common.moe_sharding import (
    set_moe_sharding_config as _set_moe_block_sharding_config,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from torchtitan.experiments.ezpz.moe.model import (
        moeModel,
        moeTransformerBlock,
    )


# Routed-expert layout for the shared ``GroupedExperts`` / ``EzpzGroupedExperts``.
# After upstream PR #3425 (41st sync, MoE [8/n] shape-suffix rename), the
# parameters are named w{1,2,3}_E{F,D}D using Shazeer shape-suffix style.
# Matches upstream ``deepseek_v3.sharding._GROUPED_EXPERTS_PARAM_LAYOUT``.
_GROUPED_EXPERTS_PARAM_LAYOUT: dict[str, Placement] = {
    "w1_EFD": Shard(1),
    "w2_EDF": Shard(2),
    "w3_EFD": Shard(1),
}


def set_moe_sharding_config(
    config: "moeModel.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Fill ``sharding_config`` on all moe sub-configs (dense + MoE).

    Dense sub-configs (attention, norms, dense FFN) are populated
    unconditionally -- ``Module.parallelize`` filters disabled axes at
    runtime.

    MoE sub-configs (router, shared experts, routed experts) are
    populated unconditionally via upstream's ``set_moe_sharding_config``
    helper -- ``resolve_mesh`` filters disabled axes at runtime.
    """
    # 57th sync: upstream PR #3694 removed the loss_parallel kwarg.
    # TP-on now always implies LP-on via tp_gather_logits=False; logits
    # come out vocab-sharded and cross_entropy_loss handles the
    # all-reduce via _LossParallelCrossEntropy autograd.
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_moe_layer_sharding(
            layer_cfg, enable_sp=enable_sp, enable_ep=enable_ep
        )


def _set_moe_layer_sharding(
    layer_cfg: "moeTransformerBlock.Config",
    *,
    enable_sp: bool,
    enable_ep: bool,
) -> None:
    """Set sharding on one moe transformer layer.

    MLA attention: low-rank projections (wkv_a, wq_a, kv_norm, q_norm)
    stay replicated. Up-projections (wkv_b, wq_b, wq) are colwise.
    MoE FFN is routed through upstream's ``set_moe_sharding_config``.
    """
    attention = layer_cfg.attention
    assert isinstance(attention, Attention.Config)

    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    # Upstream PR #3501 (SpmdLayout for NamedPlacement) renamed
    # set_dense_ffn_sharding's `attn_x_placement: Placement` arg to
    # `attn_x_layout: SpmdLayout`. Build via the dense_*_placement
    # helpers (same pattern as deepseek_v3/sharding.py).
    attn_x_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.R)
    )

    # MLA attention input: x is gathered to Replicate; freqs_cis always Replicate.
    attention.sharding_config = ShardingConfig(
        in_src_shardings={
            "x": attn_x_layout,
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
        in_dst_shardings={
            "x": dense_activation_placement(tp=Replicate()),
            "freqs_cis": dense_param_placement(tp=Replicate()),
        },
    )
    # Low-rank projections and norms keep Replicate weights on TP. We still
    # distribute them (Replicate DTensor) so DTensor activations flow through
    # without mixing plain Tensor + DTensor in the matmul.
    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    attention.wkv_a.sharding_config = replicate_weight
    attention.kv_norm.sharding_config = replicate_weight

    attention.wkv_b.sharding_config = colwise_config()
    attention.wo.sharding_config = rowwise_config(output_sp=enable_sp)

    # Static LocalMapConfig on the inner-attention config (upstream #2986
    # replaced runtime DTensor detection in `LocalMapInnerAttention` with
    # this config-driven approach).
    set_gqa_inner_attention_local_map(attention.inner_attention)

    # Query projection: depends on q_lora_rank
    if attention.q_lora_rank == 0:
        assert attention.wq is not None
        attention.wq.sharding_config = colwise_config()
    else:
        # Low-rank: wq_a + q_norm stay Replicate DTensors; wq_b is Colwise.
        assert attention.wq_a is not None
        assert attention.wq_b is not None
        attention.wq_a.sharding_config = replicate_weight
        attention.q_norm.sharding_config = replicate_weight
        attention.wq_b.sharding_config = colwise_config()

    # Dense FFN (non-MoE layers only).
    if layer_cfg.feed_forward is not None:
        set_dense_ffn_sharding(
            layer_cfg.feed_forward,
            attn_x_layout=attn_x_layout,
            enable_sp=enable_sp,
        )

    # MoE FFN (MoE-enabled layers only). Routes through upstream's helper
    # which populates router gate / shared experts / routed experts
    # ``sharding_config`` declarations.
    if layer_cfg.moe is not None:
        _set_moe_block_sharding_config(
            layer_cfg.moe,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=_GROUPED_EXPERTS_PARAM_LAYOUT,
        )
