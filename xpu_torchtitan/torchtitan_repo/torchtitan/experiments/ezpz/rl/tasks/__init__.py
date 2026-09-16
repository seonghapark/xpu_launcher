# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Task registry for GRPO training.
#
# Each task is a self-contained module that registers itself on import.
# Add new tasks by creating a module in this package and importing it below.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from datasets import Dataset

# Reward func signature: (completions, **dataset_columns) -> list[float]
RewardFunc = Callable[..., list[float]]


@dataclass
class RLTask:
    """A GRPO training task: dataset builder + reward functions."""

    name: str
    build_dataset: Callable[..., Dataset]
    reward_funcs: list[RewardFunc] = field(default_factory=list)
    description: str = ""


TASK_REGISTRY: dict[str, RLTask] = {}


def register_task(task: RLTask) -> RLTask:
    """Register a task in the global registry."""
    TASK_REGISTRY[task.name] = task
    return task


def get_task(name: str) -> RLTask:
    """Look up a registered task by name."""
    if name not in TASK_REGISTRY:
        available = ", ".join(sorted(TASK_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown task {name!r}. Available: {available}")
    return TASK_REGISTRY[name]


# Import task modules so they self-register.
from torchtitan.experiments.ezpz.rl.tasks import arithmetic as _  # noqa: F401, E402
from torchtitan.experiments.ezpz.rl.tasks import countdown as _  # noqa: F401, E402
from torchtitan.experiments.ezpz.rl.tasks import multiply as _  # noqa: F401, E402
from torchtitan.experiments.ezpz.rl.tasks import sum_digits as _  # noqa: F401, E402
from torchtitan.experiments.ezpz.rl.tasks import word_sort as _  # noqa: F401, E402
