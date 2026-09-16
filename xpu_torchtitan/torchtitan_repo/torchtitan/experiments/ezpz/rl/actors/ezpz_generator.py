# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz subclass of upstream `VLLMGenerator` for the XPU stack.

WIP — substantial integration ahead. The skeleton lives here so the
import boundary is settled and Phase 2 (Monarch framework smoke,
2026-06-13) can be followed by Phase 3 (this file) without churning
on file placement.

What needs overriding from upstream `VLLMGenerator`:

  1. **`has_cuda_capability(9, 0)` check** at the engine-args
     construction site (generator.py:465). XPU has no equivalent
     SM-version concept; the check controls a `block_size` override
     for non-H100. The simplest port is "always set block_size=256
     on XPU" (the non-CUDA-9.0 branch).

  2. **`cudagraph.enable` config**. vLLM-XPU's compilation pipeline
     does not support CUDA-graph capture. Force `enforce_eager=True`
     unconditionally. Could surface this as an `EzpzVLLMCudagraphConfig`
     subclass that ignores the `enable` field, or override the
     `engine_kwargs` construction directly.

  3. **`AttentionBackendEnum`** — upstream uses
     `FLEX_ATTENTION`/`varlen`/etc. The vLLM-XPU backend exposes its
     own set of attention impls (`spda_v1`, `vllm_xpu_attention`).
     Need to map ezpz's `Decoder.first_attention.inner_attention`
     to the right XPU backend.

  4. **`set_batch_invariance`** (called from upstream `__init__`).
     Lives in `torchtitan.distributed.utils`; uses
     `torch._inductor.config.fallback_random` which is CUDA-flavored.
     Probably fine on XPU since fallback_random is just a config
     flag, but verify before relying on it for bitwise repro.

  5. **`torchstore` strategy** — must force `TransportType.Gloo`
     (or `MonarchRPC`) instead of the default which picks
     `MonarchRDMA` (CUDA-only). Override in `pull_model_state_dict`.

Open question: do we override by subclass or by reimplementation?
Upstream's `VLLMGenerator` is ~990 LOC with the LLMEngine
constructor woven into `setup_async`. Subclassing means overriding
the parts of `setup_async` that build `EngineArgs`; in practice
that's mostly the constructor since most other endpoints are
about pulling state dicts + serving generation requests.

For now this file defines the EzpzVLLMGenerator class signature
and stubs out the engine-args override. The Monarch framework smoke
in 2026-06-13 confirmed the actor framework itself works on XPU;
this class is what plugs vLLM-XPU into that framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

# Upstream imports — note these only resolve in the rl-vllm venv
# (py3.12 + torch 2.12 + monarch + vllm-xpu). Don't import this module
# from a context that doesn't have all of those.
try:
    from torchtitan.experiments.rl.actors.generator import (
        VLLMCudagraphConfig,
        VLLMGenerator,
    )
except ImportError as e:  # noqa: BLE001
    raise ImportError(
        "EzpzVLLMGenerator requires upstream `torchtitan.experiments.rl.actors`"
        " which only resolves in the rl-vllm venv (py3.12 + torch 2.12 +"
        " monarch + vllm-xpu). See docs/rl/vllm-xpu-wiring-plan.md."
    ) from e


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class EzpzVLLMCudagraphConfig(VLLMCudagraphConfig):
    """vLLM cudagraph config that always disables capture on XPU.

    vLLM-XPU's compilation pipeline does not currently support
    CUDA-graph capture. Setting `enable=False` here propagates
    `enforce_eager=True` into the engine args via upstream
    `VLLMGenerator`'s existing logic.
    """

    enable: bool = False


class EzpzVLLMGenerator(VLLMGenerator):
    """Subclass of upstream `VLLMGenerator` for the ezpz XPU stack.

    WIP — see `docs/rl/vllm-xpu-wiring-plan.md` for the full
    integration plan. Currently a placeholder.
    """

    # TODO Phase 3a: override engine-args construction
    #   - skip `has_cuda_capability(9, 0)` check (always set
    #     block_size=256, like the non-H100 branch upstream)
    #   - force `enforce_eager=True` via EzpzVLLMCudagraphConfig
    #     default
    #   - map our attention configs to XPU-supported vLLM backends

    # TODO Phase 3b: override `pull_model_state_dict` to use
    #   `TransportType.Gloo` strategy (RDMA path is CUDA-only)

    # TODO Phase 3c: drop or replace `set_batch_invariance` if it
    #   touches CUDA-only torch._inductor config
