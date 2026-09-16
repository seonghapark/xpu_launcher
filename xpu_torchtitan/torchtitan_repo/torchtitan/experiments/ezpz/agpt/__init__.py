# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.experiments.ezpz.agpt.parallelize import parallelize_llama
from torchtitan.models.common import (
    ComplexRoPE,
    compute_ffn_hidden_dim,
    CosSinRoPE,
    Embedding,
    Linear,
    RMSNorm,
    RoPE,
    TransformerBlock,
)
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.models.common.config_utils import get_attention_config
from torchtitan.protocols.module import Module


class EzpzScaledDotProductAttention(ScaledDotProductAttention):
    """SDPA that avoids ``set_priority=True`` in the ``sdpa_kernel`` context.

    Works around a torch._dynamo bug in PyTorch 2.11 where
    ``sdpa_kernel(..., set_priority=True)`` passes FX proxy nodes to
    ``int()`` during fake tensor tracing, causing
    ``RuntimeError('Invalid backend')``.
    """

    @dataclass(kw_only=True, slots=True)
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
        # q_BLNH/k_BLNH/v_BLNH to match the keys in
        # set_gqa_inner_attention_local_map's in_dst_shardings -- the
        # local_map contract check (protocols/module.py:_maybe_wrap_local_map)
        # matches by positional-arg name, and asserts under TP>1 if a
        # mapped input name is missing from in_dst_shardings.
        q, k, v = (
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
        )
        # Avoid set_priority=True — triggers a torch._dynamo bug in
        # PyTorch 2.11 where FX proxy nodes are incorrectly passed to
        # int() during fake tensor tracing.
        with sdpa_kernel(self.sdpa_backends):
            out = F.scaled_dot_product_attention(
                q, k, v, scale=scale, is_causal=is_causal, enable_gqa=enable_gqa
            )
        return out.transpose(1, 2)


class XPUScaledDotProductAttention(EzpzScaledDotProductAttention):
    """SDPA with OVERRIDEABLE backend for XPU-optimized fused attention.

    Adds OVERRIDEABLE to the backend priority list for the XPU fused
    attention kernel. On single-device the OVERRIDEABLE backend is 23x
    faster than MATH and avoids materializing the N×N attention matrix.

    Note: On XPU, the sdpa_kernel context manager and
    torch.backends.cuda.enable_math_sdp are not respected inside
    FSDP-wrapped modules (PyTorch XPU bug). The MATH backend is always
    used inside FSDP regardless of this setting. TP is required for
    models where the MATH attention matrix exceeds device memory
    (e.g., 80B with 72 heads at seq_len=8192 = 9 GiB per tile).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(EzpzScaledDotProductAttention.Config):
        pass

    sdpa_backends = [
        SDPBackend.OVERRIDEABLE,
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.MATH,
    ]


from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config


# ---------------------------------------------------------------------------
# Architecture tweaks for competition
# ---------------------------------------------------------------------------


class SoftcappedFlexAttention(Module):
    """FlexAttention with logit softcapping (Gemma 2 style).

    Uses FlexAttention's score_mod to apply tanh softcapping inside the
    fused kernel — no O(seq_len²) materialization. Requires torch.compile.

    TP: relies on `set_gqa_inner_attention_local_map` setting a static
    `LocalMapConfig` on the inner-attention sharding_config (upstream
    #2986 replaced runtime DTensor detection in `LocalMapInnerAttention`
    with config-driven local_map).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        logit_cap: float = 30.0

    def __init__(self, config: Config):
        super().__init__()
        self.logit_cap = config.logit_cap
        from torch.nn.attention.flex_attention import flex_attention
        self._flex_attention = torch.compile(flex_attention)

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
        # 57th sync: shape-suffixed positional names required to match
        # set_gqa_inner_attention_local_map's in_dst_shardings under TP>1.
        q, k, v = (
            q_BLNH.transpose(1, 2),
            k_BLNH.transpose(1, 2),
            v_BLNH.transpose(1, 2),
        )

        cap = self.logit_cap

        def softcap_mod(score, b, h, q_idx, kv_idx):
            return cap * torch.tanh(score / cap)

        out = self._flex_attention(
            q, k, v,
            score_mod=softcap_mod,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        return out.transpose(1, 2)


class ReLUSquaredFeedForward(FeedForward):
    """FFN with ReLU-squared activation instead of SiLU.

    ReLU²(x) = max(0, x)². Used in NanoGPT speedrun entries for faster
    convergence. The squared activation creates sharper sparsity patterns.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.w1(x))
        return self.w2(h * h * self.w3(x))
from torchtitan.experiments.ezpz.agpt.model import AgptModel
from torchtitan.models.common.param_init import depth_scaled_std
from torchtitan.models.llama3.model import Llama3TransformerBlock
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter
from torchtitan.experiments.torchft.config.job_config import FaultTolerantModelSpec

__all__ = [
    "EzpzScaledDotProductAttention",
    "XPUScaledDotProductAttention",
    "_default_inner_attention",
    "model_registry",
    "parallelize_llama",
]


_NORM_INIT = {"weight": nn.init.ones_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}


def _linear_init(dim: int) -> dict[str, Callable]:
    """Weight init with std = sqrt(2/(5*d)), following Megatron-DeepSpeed.

    Reference: https://arxiv.org/pdf/2312.16903
    """
    s = (2.0 / (5 * dim)) ** 0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s),
        "bias": nn.init.zeros_,
    }


def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }


def _depth_init(dim: int, layer_id: int) -> dict[str, Callable]:
    base_std = (2.0 / (5 * dim)) ** 0.5
    return {
        "weight": partial(
            nn.init.trunc_normal_, std=depth_scaled_std(base_std, layer_id)
        ),
        "bias": nn.init.zeros_,
    }


def _default_inner_attention() -> ScaledDotProductAttention.Config:
    """Return the right SDPA config for the current device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return XPUScaledDotProductAttention.Config()
    return EzpzScaledDotProductAttention.Config()


def _ezpz_get_attention_config(
    backend: str,
) -> Module.Config:
    """XPU-aware attention config selection.

    For the "sdpa" backend, uses the XPU-optimized SDPA classes instead
    of upstream's ScaledDotProductAttention. Other backends delegate
    to upstream get_attention_config().

    Upstream PR #3571 (2026-06-09) removed SDPA + mask_type from the
    language-model attention path entirely; ezpz keeps the "sdpa"
    branch alive because XPU lacks a working FlexAttention backend.
    Returns just the config now (upstream dropped the (config, mask_type)
    tuple too).
    """
    if backend == "sdpa":
        return _default_inner_attention()
    return get_attention_config(backend)


def _build_agpt_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    hidden_dim: int,
    rope: RoPE.Config,
    n_kv_heads: int | None = None,
    fuse_qkv: bool = False,
    attn_backend: str = "sdpa",
    qk_norm: bool = False,
    logit_softcap: float | None = None,
    relu_squared: bool = False,
) -> list[TransformerBlock.Config]:
    """Build a list of per-layer TransformerBlock configs with depth-scaled inits."""
    if logit_softcap is not None:
        inner_attention = SoftcappedFlexAttention.Config(
            logit_cap=logit_softcap,
        )
    else:
        inner_attention = _ezpz_get_attention_config(attn_backend)
    linear_init = _linear_init(dim)
    head_dim = dim // n_heads
    qk_norm_config = RMSNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT) if qk_norm else None
    layers = []
    for layer_id in range(n_layers):
        if relu_squared:
            ffn_config = ReLUSquaredFeedForward.Config(
                w1=Linear.Config(
                    in_features=dim, out_features=hidden_dim,
                    param_init=linear_init,
                ),
                w2=Linear.Config(
                    in_features=hidden_dim, out_features=dim,
                    param_init=_depth_init(dim, layer_id),
                ),
                w3=Linear.Config(
                    in_features=dim, out_features=hidden_dim,
                    param_init=_depth_init(dim, layer_id),
                ),
            )
        else:
            ffn_config = make_ffn_config(
                dim=dim,
                hidden_dim=hidden_dim,
                w1_param_init=linear_init,
                w2w3_param_init=_depth_init(dim, layer_id),
            )
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=dim, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=dim,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=linear_init,
                    wo_param_init=_depth_init(dim, layer_id),
                    inner_attention=inner_attention,
                    fuse_qkv=fuse_qkv,
                    rope=rope,
                    qk_norm=qk_norm_config,
                ),
                feed_forward=ffn_config,
            )
        )
    return layers


def _build_agpt_config(
    *,
    dim: int,
    n_layers: int,
    n_heads: int,
    n_kv_heads: int | None,
    rope_theta: int,
    vocab_size: int,
    hidden_dim: int,
    fuse_qkv: bool = False,
    attn_backend: str = "sdpa",
    rope_backend: Literal["complex", "cos_sin"] = "complex",
    scaling: Literal["none", "llama", "yarn"] = "none",
    max_seq_len: int = 131072,
    qk_norm: bool = False,
    logit_softcap: float | None = None,
    relu_squared: bool = False,
) -> AgptModel.Config:
    # PR #3458 (RoPE refactor): RoPE.Config split into ComplexRoPE.Config /
    # CosSinRoPE.Config; the backend= field is gone (backend is encoded in
    # the type). Top-level Model.Config.rope is gone too — each layer's
    # Attention.Config owns its own rope. ``rope_backend`` keyword on this
    # builder is kept for back-compat with existing config callers, but
    # internally it now selects a concrete subclass.
    rope_cls: type[RoPE.Config] = (
        ComplexRoPE.Config if rope_backend == "complex" else CosSinRoPE.Config
    )
    rope_cfg = rope_cls(
        dim=dim // n_heads,
        max_seq_len=max_seq_len,
        theta=rope_theta,
        scaling=scaling,
    )
    return AgptModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        norm=RMSNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=_build_agpt_layers(
            n_layers=n_layers,
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            hidden_dim=hidden_dim,
            fuse_qkv=fuse_qkv,
            attn_backend=attn_backend,
            rope=rope_cfg,
            qk_norm=qk_norm,
            logit_softcap=logit_softcap,
            relu_squared=relu_squared,
        ),
    )


agpt_configs = {
    "debugmodel": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
    ),
    "debugmodel_flex_attn": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
        attn_backend="flex",
    ),
    "debugmodel_varlen_attn": _build_agpt_config(
        dim=256,
        n_layers=6,
        n_heads=16,
        n_kv_heads=None,
        rope_theta=500000,
        vocab_size=32000,
        hidden_dim=compute_ffn_hidden_dim(256, multiple_of=256),
        attn_backend="varlen",
    ),
    "2B": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
    ),
    "2B_qknorm": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        qk_norm=True,
    ),
    "2B_softcap": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        logit_softcap=30.0,
    ),
    "2B_relu2": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        relu_squared=True,
    ),
    "2B_kitchen_sink": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        qk_norm=True,
        logit_softcap=30.0,
        relu_squared=True,
    ),
    "2B_flex_attn": _build_agpt_config(
        dim=2048,
        n_layers=12,
        n_heads=16,
        n_kv_heads=4,
        rope_theta=50000,
        vocab_size=256128,
        hidden_dim=11008,
        attn_backend="flex",
    ),
    "7B": _build_agpt_config(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=10000,
        vocab_size=32000,
        hidden_dim=11008,
    ),
    "8B": _build_agpt_config(
        dim=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=128256,
        hidden_dim=compute_ffn_hidden_dim(
            4096, multiple_of=1024, ffn_dim_multiplier=1.3
        ),
    ),
    "20B": _build_agpt_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
    ),
    "20B_flex_attn": _build_agpt_config(
        dim=5120,
        n_layers=64,
        n_heads=40,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(5120, multiple_of=1024),
        attn_backend="flex",
    ),
    "50B": _build_agpt_config(
        dim=8192,
        n_layers=56,
        n_heads=64,
        n_kv_heads=8,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=compute_ffn_hidden_dim(
            8192, multiple_of=1024, ffn_dim_multiplier=1.3
        ),
    ),
    # Same per-layer shape as 80B (dim=9216, 72 heads, 12 kv heads,
    # hidden_dim=25600) but only 48 layers. ~50B params total.
    # Distinct from "50B" (dim=8192) — this one keeps the 80B-family
    # head pattern (n_kv_heads=12) so it shares the TP=2 sharding plan.
    # Use as a smaller compile target that still exercises the
    # compile+AC+TP=2 path that the dense 80B configs depend on.
    "50B_wide": _build_agpt_config(
        dim=9216,
        n_layers=48,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
    ),
    # Same per-layer shape as 80B with 72 layers (~70B params total).
    # Bisect midpoint between the working 50B_wide (48L, no crash) and
    # the broken 80B (84L, DeviceMesh-in-saved-tensors AOT autograd
    # crash) — used to pin down whether the depth-sensitivity threshold
    # is at 72L or somewhere else in [48, 84).
    "70B_wide": _build_agpt_config(
        dim=9216,
        n_layers=72,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
    ),
    # Aurora-native ~80B configs: n_kv_heads=12, n_heads divisible by 12
    # so TP can be any factor of 12 (2, 3, 4, 6, 12).
    #
    # ~80.8B: Balanced width/depth. PP divides 84: {1,2,3,4,6,12}.
    "80B": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25600,
    ),
    # ~80.0B: Wider (dim=10752), shallower (48 layers).
    # PP divides 48: {1,2,3,4,6,8,12,16,24}.
    "80B_wide": _build_agpt_config(
        dim=10752,
        n_layers=48,
        n_heads=84,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=39936,
    ),
    # ~80.9B: Narrower (dim=7680), deeper (96 layers).
    # PP divides 96: {1,2,3,4,6,8,12,16,24}.
    "80B_deep": _build_agpt_config(
        dim=7680,
        n_layers=96,
        n_heads=60,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=28672,
    ),
    # Variants with hidden_dim divisible by 12, for clean TP sharding
    # across all factors of 12 (2, 3, 4, 6, 12).
    "80B_alt": _build_agpt_config(
        dim=9216,
        n_layers=84,
        n_heads=72,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=25596,  # 25600 -> 25596 (multiple of 12)
    ),
    "80B_deep_alt": _build_agpt_config(
        dim=7680,
        n_layers=96,
        n_heads=60,
        n_kv_heads=12,
        rope_theta=500000,
        vocab_size=256128,
        hidden_dim=28668,  # 28672 -> 28668 (multiple of 12)
    ),
}


# Case-insensitive aliases
agpt_configs["2b"] = agpt_configs["2B"]
agpt_configs["2b_flex_attn"] = agpt_configs["2B_flex_attn"]
agpt_configs["7b"] = agpt_configs["7B"]
agpt_configs["8b"] = agpt_configs["8B"]
agpt_configs["20b"] = agpt_configs["20B"]
agpt_configs["20b_flex_attn"] = agpt_configs["20B_flex_attn"]
agpt_configs["50b"] = agpt_configs["50B"]
agpt_configs["50b_wide"] = agpt_configs["50B_wide"]
agpt_configs["70b_wide"] = agpt_configs["70B_wide"]
agpt_configs["80b"] = agpt_configs["80B"]
agpt_configs["80b_wide"] = agpt_configs["80B_wide"]
agpt_configs["80b_deep"] = agpt_configs["80B_deep"]
agpt_configs["80b_alt"] = agpt_configs["80B_alt"]
agpt_configs["80b_deep_alt"] = agpt_configs["80B_deep_alt"]
agpt_configs["2b_qknorm"] = agpt_configs["2B_qknorm"]
agpt_configs["2b_softcap"] = agpt_configs["2B_softcap"]
agpt_configs["2b_relu2"] = agpt_configs["2B_relu2"]
agpt_configs["2b_kitchen_sink"] = agpt_configs["2B_kitchen_sink"]


def model_registry(
    flavor: str,
    attn_backend: str = "sdpa",
) -> FaultTolerantModelSpec:
    from torchtitan.distributed.pipeline_parallel import pipeline_llm
    from torchtitan.experiments.torchft.diloco import fragment_llm

    config = agpt_configs[flavor]

    return FaultTolerantModelSpec(
        name="ezpz.agpt",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_llama,
        pipelining_fn=pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=Llama3StateDictAdapter,
        fragment_fn=fragment_llm,
    )
