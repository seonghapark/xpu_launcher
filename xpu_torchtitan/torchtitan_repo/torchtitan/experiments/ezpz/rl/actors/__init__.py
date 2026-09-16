# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ezpz-side Monarch actors for the RL pipeline.

Mirrors the upstream `torchtitan.experiments.rl.actors.{generator,trainer}`
module structure, but specialized for the XPU stack:

  - `VLLMGenerator` is subclassed (or where impractical, re-implemented)
    to skip CUDA-only code paths and default to vLLM-XPU's
    `enforce_eager=True`.
  - `PolicyTrainer` overrides for our ezpz training loop.

Lives in its own namespace (`ezpz.rl.actors`) so the import resolution
is unambiguous when both the upstream and ezpz actor modules are on
sys.path. Use this from a venv that has both monarch + vllm-xpu
installed (see `venvs/rl-vllm/`).
"""

from .ezpz_generator import EzpzVLLMGenerator  # noqa: F401

# from .ezpz_trainer import EzpzPolicyTrainer  # TODO
