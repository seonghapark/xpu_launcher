# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Every entry must resolve to a `<name>/config_registry.py` under
# torchtitan.models or torchtitan.experiments — ConfigManager imports that
# module when `--module <name>` is passed, and a name listed without one
# fails at import time rather than with a usable "unsupported module" error.
_supported_experiments = frozenset(
    [
        "graph_trainer.llama3",
        "graph_trainer.deepseek_v3",
        "graph_trainer.qwen3",
        "transformers_modeling_backend",
        "torchft.llama3",
        "agpt",
    ]
)
