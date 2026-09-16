#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone vLLM-XPU sanity smoke. No TRL wrapper.

Loads an HF checkpoint into vLLM and generates a single completion.
Used to isolate "is the vLLM-XPU stack itself working" vs failures
that surface only when wrapped (e.g. TRL `vllm_serve`'s
`multiprocessing.spawn` of an llm_worker).

Run from `venvs/rl-actors/` on a Sunspot compute node:

    venvs/rl-actors/bin/python torchtitan/experiments/ezpz/rl/scripts/vllm_xpu_bare_smoke.py

Or via the PBS wrapper `vllm_xpu_bare_smoke.sh`.

NOTE: must be a real .py file (not a heredoc piped to `python -`)
because vLLM uses `multiprocessing.spawn` to launch its engine
worker, and the spawned worker tries to re-execute the main script
via `runpy.run_path()`. A stdin-pipe heredoc breaks that with
`FileNotFoundError: '<stdin>'`.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    model = os.environ.get(
        "MODEL",
        "outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf",
    )

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"torch={torch.__version__} vllm={vllm.__version__}")
    print(f"xpu_count={torch.xpu.device_count()}")
    print(f"MODEL={model}")
    sys.stdout.flush()

    llm = LLM(
        model=model,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )

    out = llm.generate(
        ["What is 3 + 7 + 2?"],
        SamplingParams(max_tokens=32, temperature=0),
    )
    print("GEN:", out[0].outputs[0].text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
