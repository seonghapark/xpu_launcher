# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Thin wrapper around a HuggingFace causal LM for SFT.

Uses ``AutoModelForCausalLM`` so any HF architecture that follows the standard
``model.model.layers`` layout works transparently (gemma / gemma-2 / gemma-3
have all been tested; llama / qwen-style models should also work).
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}


def resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{name}'. Valid: {sorted(_DTYPE_MAP.keys())}"
        )
    return _DTYPE_MAP[name]


class GemmaModel(nn.Module):
    """Instruction-tuning wrapper around a HF causal LM.

    Loads pretrained weights via ``AutoModelForCausalLM.from_pretrained`` and
    exposes a plain ``forward(input_ids, labels, attention_mask) -> loss``
    interface so the training loop stays framework-agnostic.

    The concrete architecture is picked from the HF config's ``architectures``
    field, so the same wrapper handles gemma, gemma-2, gemma-3, etc.
    """

    def __init__(self, args):
        super().__init__()
        dtype = resolve_dtype(args.dtype)

        # `attn_implementation` default is "sdpa". Gemma-2 requires
        # transformers >= 4.42 for sdpa + soft-capping; older versions must use
        # "eager". Users can override via `args.attn_implementation` if needed.
        attn_impl = getattr(args, "attn_implementation", "sdpa")

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=getattr(args, "model_revision", "main"),
            torch_dtype=dtype,
            attn_implementation=attn_impl,
        )
        # Cache is incompatible with training / gradient checkpointing.
        self.model.config.use_cache = False

        if getattr(args, "activation_checkpoint", False):
            # Non-reentrant checkpointing is required for FSDP2 compatibility.
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss

    def save_pretrained(self, save_directory: str) -> None:
        """Save the underlying HF model to `save_directory` (rank-0 only)."""
        self.model.save_pretrained(save_directory)
