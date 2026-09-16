# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.models.common import (
    ComplexRoPE,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.config_utils import (
    get_attention_config,
    make_ffn_config,
)
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.protocols.module import Module
from torchtitan.protocols.model_spec import ModelSpec

# MoE and TokenChoiceTopKRouter come straight from upstream — we have no
# ezpz-specific override for them. Earlier this re-imported from a local
# `.moe` copy that was a byte-for-byte fork of `torchtitan/models/common/moe.py`;
# that fork has been deleted to avoid silent skew on upstream MoE/router
# fixes (e.g. the CP-friendly 3-D experts output added in upstream PR #3447).
from torchtitan.models.common.moe import MoE, TokenChoiceTopKRouter

from .experts import ExpertComputeBackend, EzpzGroupedExperts
from .model import Attention, moeModel, moeTransformerBlock
from .token_dispatcher import (
    AllToAllTokenDispatcher,
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
)

from .parallelize import parallelize_moe
from .state_dict_adapter import moeStateDictAdapter


class EzpzScaledDotProductAttention(ScaledDotProductAttention):
    """SDPA variant that avoids set_priority=True in sdpa_kernel."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(ScaledDotProductAttention.Config):
        pass

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        q_BLNH: torch.Tensor,
        k_BLNH: torch.Tensor,
        v_BLNH: torch.Tensor,
        *,
        scale: float | None = None,
        enable_gqa: bool = False,
        is_causal: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # 57th sync: positional arg names MUST be the shape-suffixed
        # q_BLNH/k_BLNH/v_BLNH to match set_gqa_inner_attention_local_map's
        # in_dst_shardings keys; the local_map contract check matches by
        # positional-arg name and asserts under TP>1 otherwise.
        q, k, v = (
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
        )
        with sdpa_kernel(self.sdpa_backends):
            out = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
            )
        return out.transpose(1, 2)


class XPUScaledDotProductAttention(EzpzScaledDotProductAttention):
    """SDPA with OVERRIDEABLE first for XPU fused attention."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(EzpzScaledDotProductAttention.Config):
        pass

    sdpa_backends = [
        SDPBackend.OVERRIDEABLE,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]


def _default_inner_attention() -> ScaledDotProductAttention.Config:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return XPUScaledDotProductAttention.Config()
    return EzpzScaledDotProductAttention.Config()


def _ezpz_get_attention_config(backend: str) -> Module.Config:
    """XPU-aware attention config selection.

    Mirrors the agpt sibling. Upstream PR #3571 (replayed at ezpz
    `db3b916a8`) dropped the `(config, mask_type)` tuple return in
    favor of returning just the config; the caller now supplies
    mask_type separately (see `_build_moe_layers` which sets
    `_mask = "causal"` next to the call site). Returning a tuple
    here would break the downstream `inner_attention.sharding_config`
    setattr in `moe/sharding.py:set_gqa_inner_attention_local_map`.
    """
    if backend == "sdpa":
        return _default_inner_attention()
    return get_attention_config(backend)


def make_ezpz_router_config(
    *,
    dim: int,
    num_experts: int,
    gate_param_init: dict[str, Callable],
    top_k: int = 1,
    score_func: Literal["sigmoid", "softmax"] = "sigmoid",
    route_norm: bool = False,
    route_scale: float = 1.0,
    num_expert_groups: int | None = None,
    num_limited_groups: int | None = None,
    bias: bool = False,
) -> TokenChoiceTopKRouter.Config:
    return TokenChoiceTopKRouter.Config(
        num_experts=num_experts,
        gate=Linear.Config(
            in_features=dim,
            out_features=num_experts,
            bias=bias,
            param_init=gate_param_init,
        ),
        top_k=top_k,
        score_func=score_func,
        route_norm=route_norm,
        route_scale=route_scale,
        num_expert_groups=num_expert_groups,
        num_limited_groups=num_limited_groups,
    )


def make_ezpz_token_dispatcher_config(
    *,
    num_experts: int,
    top_k: int,
    score_before_experts: bool = True,
    comm_backend: str,
    non_blocking_capacity_factor: float | None = None,
):
    if comm_backend == "deepep":
        return DeepEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
        )
    if comm_backend == "hybridep":
        return HybridEPTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        )
    if comm_backend == "standard":
        return AllToAllTokenDispatcher.Config(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
        )
    raise ValueError(
        f"Unknown comm_backend: {comm_backend!r}. "
        "Must be one of 'standard', 'deepep', 'hybridep'."
    )


def make_ezpz_experts_config(
    *,
    dim: int,
    hidden_dim: int,
    num_experts: int,
    top_k: int,
    param_init: dict[str, Callable],
    score_before_experts: bool = True,
    comm_backend: str = "standard",
    non_blocking_capacity_factor: float | None = None,
    compute_backend: ExpertComputeBackend = "grouped_mm",
) -> EzpzGroupedExperts.Config:
    """Build an EzpzGroupedExperts.Config from the same args as upstream
    `make_experts_config`, plus a `compute_backend` selector.
    """
    return EzpzGroupedExperts.Config(
        dim=dim,
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        param_init=param_init,
        token_dispatcher=make_ezpz_token_dispatcher_config(
            num_experts=num_experts,
            top_k=top_k,
            score_before_experts=score_before_experts,
            comm_backend=comm_backend,
            non_blocking_capacity_factor=non_blocking_capacity_factor,
        ),
        compute_backend=compute_backend,
    )


def make_ezpz_moe_config(
    *,
    num_experts: int = 8,
    router: TokenChoiceTopKRouter.Config,
    experts: EzpzGroupedExperts.Config,
    shared_experts=None,
    load_balance_coeff: float | None = 1e-3,
) -> MoE.Config:
    return MoE.Config(
        num_experts=num_experts,
        load_balance_coeff=load_balance_coeff,
        router=router,
        experts=experts,
        shared_experts=shared_experts,
    )


__all__ = [
    "parallelize_moe",
    "moeModel",
    "moe_configs",
]


_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }


def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    # Param names use Shazeer shape-suffix style post upstream PR #3425
    # (41st sync): w1_EFD, w2_EDF, w3_EFD.
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }


def _make_moe_attn_config(
    *,
    layer_id: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    rope: RoPE.Config,
    mscale: float = 1.0,
    attn_backend: str = "sdpa",
) -> Attention.Config:
    """Build a fully-specified MoE MLA Attention.Config.

    All Linear and RMSNorm sub-configs have their dimensional fields set.
    When q_lora_rank == 0, sets wq (not wq_a/wq_b).
    When q_lora_rank > 0, sets wq_a/wq_b (not wq).
    """
    # Upstream PR #3571 dropped the (config, mask_type) tuple — see
    # _ezpz_get_attention_config docstring. ezpz/moe's Attention.Config
    # still has its own mask_type field (see moe/model.py), so we
    # preserve the previous default of "causal" here.
    _inner = _ezpz_get_attention_config(attn_backend)
    _mask = "causal"
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

    if q_lora_rank == 0:
        wq = Linear.Config(
            in_features=dim,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        wq_a = None
        wq_b = None
        # q_norm is unused when q_lora_rank == 0 (never built), but the field is
        # required on Attention.Config so we supply a placeholder.
        q_norm = RMSNorm.Config(normalized_shape=1, param_init=_NORM_INIT)
    else:
        wq = None
        wq_a = Linear.Config(
            in_features=dim,
            out_features=q_lora_rank,
            param_init=_LINEAR_INIT,
        )
        wq_b = Linear.Config(
            in_features=q_lora_rank,
            out_features=n_heads * qk_head_dim,
            param_init=_LINEAR_INIT,
        )
        q_norm = RMSNorm.Config(normalized_shape=q_lora_rank, param_init=_NORM_INIT)

    return Attention.Config(
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        mscale=mscale,
        wq=wq,
        wq_a=wq_a,
        wq_b=wq_b,
        q_norm=q_norm,
        wkv_a=Linear.Config(
            in_features=dim,
            out_features=kv_lora_rank + qk_rope_head_dim,
            param_init=_LINEAR_INIT,
        ),
        kv_norm=RMSNorm.Config(normalized_shape=kv_lora_rank, param_init=_NORM_INIT),
        wkv_b=Linear.Config(
            in_features=kv_lora_rank,
            out_features=n_heads * (qk_nope_head_dim + v_head_dim),
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * v_head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        inner_attention=_inner,
        mask_type=_mask,
        rope=dataclasses.replace(rope),
    )


def _build_moe_layers(
    *,
    n_layers: int,
    n_dense_layers: int,
    dim: int,
    n_heads: int,
    q_lora_rank: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    mscale: float,
    dense_hidden_dim: int,
    moe_hidden_dim: int,
    num_experts: int,
    num_shared_experts: int,
    router_top_k: int,
    router_score_func: Literal["sigmoid", "softmax"],
    router_num_expert_groups: int | None = None,
    router_num_limited_groups: int | None = None,
    router_route_scale: float = 1.0,
    router_route_norm: bool = False,
    score_before_experts: bool = False,
    attn_backend: str = "sdpa",
    moe_comm_backend: str = "standard",
    compute_backend: ExpertComputeBackend = "grouped_mm",
    rope: RoPE.Config,
) -> list[TransformerBlock.Config]:
    """Build the list of per-layer TransformerBlock configs.

    Layers with layer_id < n_dense_layers get a dense FeedForward and no MoE.
    Layers with layer_id >= n_dense_layers get a MoE and no FeedForward.

    Router and expert inits are constructed per-layer so depth-scaled
    initializers are correct for each layer's position.
    """
    layers = []
    for layer_id in range(n_layers):
        attn_cfg = _make_moe_attn_config(
            layer_id=layer_id,
            dim=dim,
            n_heads=n_heads,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            mscale=mscale,
            attn_backend=attn_backend,
            rope=rope,
        )

        if layer_id < n_dense_layers:
            ffn_cfg = make_ffn_config(
                dim=dim,
                hidden_dim=dense_hidden_dim,
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(layer_id),
            )
            moe_cfg = None
        else:
            ffn_cfg = None
            moe_cfg = make_ezpz_moe_config(
                num_experts=num_experts,
                router=make_ezpz_router_config(
                    dim=dim,
                    num_experts=num_experts,
                    gate_param_init=_depth_init(layer_id),
                    top_k=router_top_k,
                    score_func=router_score_func,
                    num_expert_groups=router_num_expert_groups,
                    num_limited_groups=router_num_limited_groups,
                    route_scale=router_route_scale,
                    route_norm=router_route_norm,
                ),
                experts=make_ezpz_experts_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim,
                    num_experts=num_experts,
                    top_k=router_top_k,
                    score_before_experts=score_before_experts,
                    comm_backend=moe_comm_backend,
                    param_init=_depth_experts_init(layer_id),
                    compute_backend=compute_backend,
                ),
                shared_experts=make_ffn_config(
                    dim=dim,
                    hidden_dim=moe_hidden_dim * num_shared_experts,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
            )

        layers.append(
            moeTransformerBlock.Config(
                attention=attn_cfg,
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                feed_forward=ffn_cfg,
                moe=moe_cfg,
            )
        )
    return layers


def _debugmodel() -> moeModel.Config:
    dim = 256
    n_layers = 6
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _debugmodel_flex_attn() -> moeModel.Config:
    dim = 256
    n_layers = 6
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 256
    num_shared_experts = 2
    dense_hidden_dim = 1024
    rope_dim = 64
    num_experts = 8
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="flex",
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _small() -> moeModel.Config:
    dim = 2048
    n_layers = 24
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 512
    num_shared_experts = 2
    dense_hidden_dim = 4096
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        router_route_norm=True,
        router_route_scale=1.0,
        score_before_experts=False,
        attn_backend="flex",
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=256128,
            theta=50000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _16b() -> moeModel.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 64
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="flex",
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _236b() -> moeModel.Config:
    dim = 5120
    n_layers = 60
    vocab_size = 256128
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 1536
    num_shared_experts = 2
    dense_hidden_dim = 12288
    rope_dim = 64
    num_experts = 160
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=6,
        router_score_func="softmax",
        router_num_expert_groups=8,
        router_num_limited_groups=3,
        router_route_scale=16.0,
        score_before_experts=False,
        attn_backend="flex",
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _671b() -> moeModel.Config:
    dim = 7168
    n_layers = 61
    vocab_size = 256128
    n_heads = 128
    q_lora_rank = 1536
    moe_hidden_dim = 2048
    num_shared_experts = 1
    dense_hidden_dim = 18432
    rope_dim = 64
    num_experts = 256
    n_dense_layers = 3

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=1.0,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=8,
        router_score_func="sigmoid",
        router_num_expert_groups=8,
        router_num_limited_groups=4,
        router_route_scale=2.5,
        router_route_norm=True,
        score_before_experts=False,
        attn_backend="flex",
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _500m() -> moeModel.Config:
    """~500M active params. Halfway between debugmodel (48M) and 10B_2B."""
    dim = 512
    n_layers = 12
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 512
    num_shared_experts = 2
    dense_hidden_dim = 2048
    rope_dim = 64
    num_experts = 16
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _2b() -> moeModel.Config:
    """~2B active params. Between small and 10B_2B."""
    dim = 1024
    n_layers = 18
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 1024
    num_shared_experts = 2
    dense_hidden_dim = 4096
    rope_dim = 64
    num_experts = 24
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _4b() -> moeModel.Config:
    """~4B total / ~1B active. Optimized for 12 XPU tiles per node.

    n_heads=12 divides evenly across 12 tiles for TP.
    """
    dim = 1536
    n_layers = 22
    vocab_size = 256128
    n_heads = 12
    moe_hidden_dim = 1024
    num_shared_experts = 2
    dense_hidden_dim = 6144
    rope_dim = 64
    num_experts = 24
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _7b() -> moeModel.Config:
    """~7B total / ~1.5B active. Optimized for 12 XPU tiles per node.

    n_heads=24 divides by 2,3,4,6,12 for flexible TP.
    num_experts=36 matches 10B_2B routing complexity.
    """
    dim = 2048
    n_layers = 24
    vocab_size = 256128
    n_heads = 24
    moe_hidden_dim = 1280
    num_shared_experts = 2
    dense_hidden_dim = 8192
    rope_dim = 64
    num_experts = 36
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _10b_2b() -> moeModel.Config:
    dim = 2048
    n_layers = 27
    vocab_size = 256128
    n_heads = 16
    moe_hidden_dim = 1408
    num_shared_experts = 2
    dense_hidden_dim = 10944
    rope_dim = 64
    num_experts = 36
    n_dense_layers = 1

    layers = _build_moe_layers(
        n_layers=n_layers,
        n_dense_layers=n_dense_layers,
        dim=dim,
        n_heads=n_heads,
        q_lora_rank=0,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=rope_dim,
        v_head_dim=128,
        mscale=0.70,
        dense_hidden_dim=dense_hidden_dim,
        moe_hidden_dim=moe_hidden_dim,
        num_experts=num_experts,
        num_shared_experts=num_shared_experts,
        router_top_k=3,
        router_score_func="softmax",
        score_before_experts=False,
        attn_backend="flex",
        rope=ComplexRoPE.Config(
            dim=rope_dim,
            max_seq_len=4096 * 4,
            theta=10000.0,
            scaling="yarn",
            rope_factor=40.0,
            beta_fast=32.0,
            beta_slow=1.0,
            original_seq_len=4096,
        ),
    )
    return moeModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def _10b_2b_sdpa() -> moeModel.Config:
    """10B_2B with SDPA instead of FlexAttention.

    Avoids FlexAttention Triton compilation and the fp32 autocast
    issue on XPU (torch.autocast doesn't support fp32 on XPU).
    """
    cfg = _10b_2b()
    sdpa_cfg = _default_inner_attention()
    for layer_cfg in cfg.layers:
        layer_cfg.attention.inner_attention = sdpa_cfg
        layer_cfg.attention.mask_type = "causal"
    return cfg


moe_configs = {
    "debugmodel": _debugmodel,
    "debugmodel_flex_attn": _debugmodel_flex_attn,
    "500M": _500m,
    "2B": _2b,
    "4B": _4b,
    "7B": _7b,
    "small": _small,
    "16B": _16b,
    "236B": _236b,
    "671B": _671b,
    "10B_2B": _10b_2b,
    "10B_2B_sdpa": _10b_2b_sdpa,
}

moe_configs["debugmodel_hf"] = moe_configs["debugmodel"]
moe_configs["debugmodel_flex_attn_hf"] = moe_configs["debugmodel_flex_attn"]


def model_registry(
    flavor: str,
    moe_comm_backend: str = "standard",
    quantization: list | None = None,
) -> ModelSpec:
    from torchtitan.components.quantization import QuantizationConverter
    from torchtitan.distributed.pipeline_parallel import pipeline_llm

    config = moe_configs[flavor]()

    # Rebuild token dispatchers per #3125 — comm_backend is now always set
    # (default "standard"), and AllToAllTokenDispatcher falls back to local
    # dispatch when EP=1.
    for layer_cfg in config.layers:
        if layer_cfg.moe is not None:
            experts_cfg = layer_cfg.moe.experts
            experts_cfg.token_dispatcher = make_ezpz_token_dispatcher_config(
                num_experts=experts_cfg.num_experts,
                top_k=experts_cfg.token_dispatcher.top_k,
                score_before_experts=experts_cfg.token_dispatcher.score_before_experts,
                comm_backend=moe_comm_backend,
            )

    # Quantization is now applied to the config at model_registry time
    # rather than to the runtime model (#3127).
    if quantization is not None:
        for q in quantization:
            assert isinstance(q, QuantizationConverter.Config)
            q.build().convert(config)

    return ModelSpec(
        name="moe",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_moe,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=moeStateDictAdapter,
    )
