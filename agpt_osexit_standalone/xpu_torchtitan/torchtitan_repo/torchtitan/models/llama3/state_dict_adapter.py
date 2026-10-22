# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger()

from torchtitan.models.common.rope import ComplexRoPE, CosSinRoPE
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from .model import Llama3Model


class Llama3StateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_config: Llama3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def _rope_cls(self) -> type:
        """Return the RoPE config class this model's layers use.

        All layers share one RoPE flavor, so layer 0 is representative.
        """
        # pyrefly: ignore [missing-attribute]
        return type(self.model_config.layers[0].attention.rope)

    def _needs_qk_permute(self) -> bool:
        """Whether Q/K need the HF<->Meta rotary row permutation.

        The permutation converts between HF's half-split rotate_half layout
        (rows i and i + head_dim/2 form a rotary pair) and Meta's interleaved
        layout (rows 2i and 2i+1 form a pair) that ``ComplexRoPE`` consumes via
        ``view_as_complex``. ``CosSinRoPE`` already implements HF's rotate_half
        convention, so for those models the HF weights are used as-is and
        permuting would corrupt them.
        """
        return issubclass(self._rope_cls(), ComplexRoPE.Config)

    # HuggingFace permutation function (exact copy from their conversion script)
    # Modified to handle FSDP-sharded tensors during reshape operations
    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        """
        Permute tensor weights for RoPE compatibility.
        Handles both replicated and FSDP-sharded tensor inputs.

        Args:
            w: Tensor to permute (may be distributed)
            n_heads_arg: Number of attention heads
            dim1: First dimension (defaults to w.shape[0])
            dim2: Second dimension (defaults to w.shape[1])

        Returns:
            Permuted tensor
        """
        # Convert FSDP-sharded tensor to local to avoid sharding issues during reshape
        if isinstance(w, DTensor):
            w = w.to_local()

        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        """
        Reverse the RoPE permutation operation.
        Handles both replicated and FSDP-sharded tensor inputs.
        """
        # Convert FSDP-sharded tensor to local to avoid sharding issues during reshape
        if isinstance(w, DTensor):
            w = w.to_local()

        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:

        # pyrefly: ignore [missing-attribute]
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        dim = self.model_config.dim  # pyrefly: ignore [missing-attribute]
        head_dim = attn.head_dim or dim // n_heads
        hf_state_dict = {}

        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        permute_qk = self._needs_qk_permute()

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue

                # Apply HF permutation for Q/K weights (ComplexRoPE only --
                # CosSinRoPE already uses HF's rotate_half layout).
                if permute_qk:
                    if abstract_key == "layers.{}.attention.qkv_linear.wq.weight":
                        value = self._permute(value, n_heads)
                    if abstract_key == "layers.{}.attention.qkv_linear.wk.weight":
                        key_value_dim = head_dim * n_kv_heads
                        value = self._permute(value, n_kv_heads, key_value_dim, dim)

                new_key = new_key.format(layer_num)
            else:
                if (
                    self.model_config.enable_weight_tying  # pyrefly: ignore [missing-attribute]
                    and key == "lm_head.weight"
                ):
                    if self.fqn_to_index_mapping:
                        self.fqn_to_index_mapping.pop("lm_head.weight", None)
                    continue
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        self._validate_hf_rope_config((ComplexRoPE.Config, CosSinRoPE.Config))
        if (
            self.model_config.enable_weight_tying  # pyrefly: ignore [missing-attribute]
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        # pyrefly: ignore [missing-attribute]
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        dim = self.model_config.dim  # pyrefly: ignore [missing-attribute]
        head_dim = attn.head_dim or dim // n_heads
        state_dict = {}
        permute_qk = self._needs_qk_permute()

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                # Reverse-permute Q and K for RoPE compatibility (ComplexRoPE
                # only -- CosSinRoPE consumes HF's layout directly).
                if permute_qk:
                    if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                        value = self._reverse_permute(value, n_heads)
                    if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                        key_value_dim = head_dim * n_kv_heads
                        value = self._reverse_permute(
                            value, n_kv_heads, key_value_dim, dim
                        )

                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue

            state_dict[new_key] = value

        return state_dict
