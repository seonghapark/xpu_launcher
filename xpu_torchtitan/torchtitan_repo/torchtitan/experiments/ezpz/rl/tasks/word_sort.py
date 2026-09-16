# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Word sorting task for GRPO training.
#
# "Sort alphabetically: banana, apple, cherry" → "apple, banana, cherry"
# Tests instruction following beyond pure arithmetic.

import random
import re

from datasets import Dataset

from torchtitan.experiments.ezpz.rl.tasks import RLTask, register_task
from torchtitan.experiments.ezpz.rl.tasks.common import (
    build_streaming_or_finite,
    get_completion_text,
)

# Common English words (short, unambiguous, easy to tokenize)
WORD_POOL = [
    "apple", "banana", "cherry", "date", "elder", "fig", "grape", "honey",
    "iris", "jam", "kale", "lemon", "mango", "nest", "olive", "peach",
    "quilt", "rice", "salt", "tulip", "umbrella", "vine", "walnut", "yarn",
    "zebra", "bread", "cake", "drum", "eagle", "flame", "gold", "harp",
    "ice", "jade", "kite", "lamp", "moon", "note", "oak", "pine",
    "rain", "snow", "tree", "wave", "bell", "coin", "door", "fern",
]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _sample_one(rng, min_words: int, max_words: int) -> dict:
    """Generate one word-sort prompt+answer pair."""
    n = rng.randint(min_words, max_words)
    words = rng.sample(WORD_POOL, n)
    # Ensure the sample isn't already sorted
    shuffled = words[:]
    while shuffled == sorted(shuffled):
        rng.shuffle(shuffled)

    word_list = ", ".join(shuffled)
    sorted_list = ", ".join(sorted(shuffled))

    return {
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"Sort these words alphabetically: {word_list}\n"
                    f"Reply with just the sorted list, separated by commas."
                ),
            }
        ],
        "answer": sorted_list,
    }


def build_dataset(
    num_samples: int = 0,
    min_words: int = 3,
    max_words: int = 6,
    seed: int = 42,
) -> Dataset:
    """Generate word-sorting prompts with ground truth answers.

    Args:
        num_samples: Number of samples to materialize. ``0`` (default)
            uses a large pool (~100k via
            ``build_streaming_or_finite``) so a typical run never
            reuses the same prompt. Any positive integer materializes
            that exact count up front.
        min_words: Minimum words per problem.
        max_words: Maximum words per problem.
        seed: Random seed for reproducibility.

    Returns:
        HuggingFace Dataset.
    """
    rng = random.Random(seed)
    return build_streaming_or_finite(
        lambda: _sample_one(rng, min_words, max_words),
        num_samples,
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def _normalize(text: str) -> list[str]:
    """Extract comma-separated words from text, lowercased and stripped."""
    words = re.findall(r"[a-zA-Z]+", text)
    return [w.lower() for w in words]


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if the sorted word list matches ground truth exactly."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = get_completion_text(completion)
        extracted = _normalize(text)
        expected_words = _normalize(expected)
        if extracted == expected_words:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def partial_reward(completions, answer, **kwargs) -> list[float]:
    """Partial credit: fraction of words in the correct sorted position."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = get_completion_text(completion)
        extracted = _normalize(text)
        expected_words = _normalize(expected)
        if not extracted or not expected_words:
            rewards.append(0.0)
            continue
        # Count positional matches (truncate to shorter list)
        matches = sum(
            a == b for a, b in zip(extracted, expected_words)
        )
        rewards.append(matches / len(expected_words))
    return rewards


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_task(
    RLTask(
        name="word_sort",
        build_dataset=build_dataset,
        reward_funcs=[accuracy_reward, partial_reward],
        description="Sort words alphabetically",
    )
)
