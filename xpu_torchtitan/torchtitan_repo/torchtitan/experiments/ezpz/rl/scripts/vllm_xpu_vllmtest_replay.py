#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replay of the 2026-06-10 vLLM-XPU end-to-end verification.

Discriminator test for the open question in
`docs/rl/vllm-xpu-current-status.md`:

    Why did vLLM 0.22 + torch 2.12+xpu work on 2026-06-10 from
    `venvs/vllm-test/` (py3.14) and not today from `venvs/rl-actors/`
    (py3.13)? System hasn't drifted, so something about the
    invocation differs.

This script uses the **exact** original recipe from
`docs/rl/vllm-xpu-investigation.md` (lines 207-218):

  - dtype="bfloat16"
  - enforce_eager=True
  - gpu_memory_utilization=0.85
  - max_model_len=2048
  - both prompts ("What is 3 + 7 + 2?" + haiku)

Differences from the failing `vllm_xpu_bare_smoke.py`:

  - That used dtype=auto (no override) and
    gpu_memory_utilization=0.5, no max_model_len override.

If THIS script works from venvs/vllm-test/ and the bare smoke
fails from venvs/rl-actors/, the breakage is the rl-actors venv
(py3.13 ABI binding).

If THIS script also fails, the breakage is the invocation context
(e.g. 2026-06-10 was an interactive run; today is PBS-direct).

Run via the PBS wrapper `vllm_xpu_vllmtest_replay.sh`.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    print(f"python={sys.version.split()[0]}")
    print(f"torch={torch.__version__} vllm={vllm.__version__}")
    print(f"xpu_count={torch.xpu.device_count()}")
    sys.stdout.flush()

    llm = LLM(
        model=os.environ.get(
            "MODEL",
            "outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf",
        ),
        dtype="bfloat16",
        enforce_eager=True,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
    )

    sp = SamplingParams(temperature=0.7, max_tokens=32, top_p=0.95)
    outputs = llm.generate(
        [
            "What is 3 + 7 + 2?",
            "Write a haiku about Aurora supercomputer.",
        ],
        sp,
    )

    for i, out in enumerate(outputs):
        print(f"GEN[{i}]: {out.outputs[0].text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
