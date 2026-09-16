# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# General arithmetic task for GRPO training.
#
# Mixes addition, subtraction, multiplication, and division on small
# integers. Generalization of `sum_digits` + `multiply` so the model
# doesn't overfit to a single operation. All answers are integers
# (division uses divisor + quotient sampling so the operation always
# divides evenly).

import random
import re

from datasets import Dataset

from torchtitan.experiments.ezpz.rl.tasks import RLTask, register_task
from torchtitan.experiments.ezpz.rl.tasks.common import (
    build_streaming_or_finite,
    extract_answer,
    get_completion_text,
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


# Operation symbols used in prompts (Unicode for ×/÷ so the model sees
# the same notation it likely saw in pretraining text).
_OP_SYMBOLS = {
    "add": "+",
    "sub": "-",
    "mul": "×",
    "div": "÷",
}


def _sample_one(
    rng,
    operations: tuple[str, ...],
    min_operands: int,
    max_operands: int,
    max_value: int,
) -> dict:
    """Generate one arithmetic prompt+answer pair.

    Strategy per operation:
      add: random integers in [0, max_value]
      sub: a >= b so the answer is non-negative (clearer for grading)
      mul: random integers in [2, max_value] (avoid trivial *0/*1)
      div: pick divisor + quotient first, then dividend = divisor*quotient
           — guarantees integer answer
    """
    op = rng.choice(operations)
    n = rng.randint(min_operands, max_operands)

    if op == "add":
        nums = [rng.randint(0, max_value) for _ in range(n)]
        answer = sum(nums)
    elif op == "sub":
        # Generate left-to-right so the partial sum stays >= 0.
        # Start with a value large enough that subtractions can land.
        nums = [rng.randint(0, max_value)]
        running = nums[0]
        for _ in range(n - 1):
            b = rng.randint(0, running)
            nums.append(b)
            running -= b
        answer = running
    elif op == "mul":
        nums = [rng.randint(2, max_value) for _ in range(n)]
        answer = 1
        for x in nums:
            answer *= x
    elif op == "div":
        # Always pairwise (n is forced to 2) — multi-operand division is
        # ambiguous without parentheses, and we want clean integer answers.
        quotient = rng.randint(1, max_value)
        divisor = rng.randint(2, max_value)
        dividend = quotient * divisor
        nums = [dividend, divisor]
        answer = quotient
    else:
        raise ValueError(f"Unknown operation {op!r}")

    symbol = _OP_SYMBOLS[op]
    expression = f" {symbol} ".join(str(x) for x in nums)
    return {
        "prompt": [
            {
                "role": "user",
                "content": f"What is {expression}? Reply with just the number.",
            }
        ],
        "answer": str(answer),
        "op": op,  # extra column so per-op rewards / dashboards are possible
    }


def build_dataset(
    num_samples: int = 0,
    operations: tuple[str, ...] = ("add", "sub", "mul", "div"),
    min_operands: int = 2,
    max_operands: int = 5,
    max_value: int = 12,
    seed: int = 42,
) -> Dataset:
    """Generate mixed-arithmetic prompts with ground truth answers.

    Args:
        num_samples: Number of samples to materialize. ``0`` (default)
            uses a large pool (~100k via
            ``build_streaming_or_finite``) so a typical run never
            reuses the same prompt. Any positive integer materializes
            that exact count up front.
        operations: Operations to sample from. Valid values: 'add', 'sub',
            'mul', 'div'. Default is all four.
        min_operands, max_operands: Operand-count range. (Division always
            uses 2 operands regardless, since multi-operand division is
            ambiguous without parentheses.)
        max_value: Cap on individual operand values (and the per-op output
            range for division). Keep small (~12) so the answer space is
            tractable for a small model.
        seed: RNG seed.

    Returns:
        HuggingFace Dataset (finite) or IterableDataset (streaming).
    """
    valid_ops = set(_OP_SYMBOLS)
    bad = set(operations) - valid_ops
    if bad:
        raise ValueError(
            f"unknown operations {sorted(bad)}; valid: {sorted(valid_ops)}"
        )
    if not operations:
        raise ValueError("operations must be non-empty")

    rng = random.Random(seed)
    return build_streaming_or_finite(
        lambda: _sample_one(
            rng, tuple(operations), min_operands, max_operands, max_value
        ),
        num_samples,
    )


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if extracted answer matches ground truth, 0.0 otherwise."""
    rewards = []
    for completion, expected in zip(completions, answer):
        text = get_completion_text(completion)
        extracted = extract_answer(text)
        if extracted is not None and extracted == str(expected):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """0.5 bonus if the completion shows ANY arithmetic expression.

    Looser than the per-op format checks in sum_digits/multiply since
    this task mixes ops. Matches a digit followed by any of + - × x * ÷ /
    followed by another digit.
    """
    pattern = re.compile(r"\d+\s*[+\-×x*÷/]\s*\d+")
    rewards = []
    for completion in completions:
        text = get_completion_text(completion)
        rewards.append(0.5 if pattern.search(text) else 0.0)
    return rewards


# Length-penalty tuning constants. The arithmetic prompts ask for "just the
# number," so a well-formatted answer is ~1-3 tokens. Allow some slack
# (TARGET) before any penalty kicks in (lets the model show brief work),
# then ramp linearly to a hard cost at the clip ceiling.
_LENGTH_PENALTY_TARGET = 8   # whitespace tokens with zero cost
_LENGTH_PENALTY_HARD = 64    # matches default max_completion_length


def length_penalty(completions, **kwargs) -> list[float]:
    """Soft penalty on completion length (in whitespace-tokens).

    Reward shape:
      n <= TARGET  →   0.0   (no penalty for short answers)
      TARGET < n < HARD →  linear from 0 to -1.0
      n >= HARD    →  -1.0   (full penalty)

    GRPO sums reward functions, so this directly biases the policy
    toward terse responses without forbidding work-shown answers
    outright. The shape is intentionally additive (not multiplicative)
    with accuracy_reward so a correct-but-long answer still nets
    positive reward (1.0 - up_to_1.0 >= 0).

    Whitespace-tokenization is an approximation of the model's actual
    token count but is good enough for a smooth gradient signal and
    avoids dragging the tokenizer into the reward path.
    """
    rewards = []
    span = _LENGTH_PENALTY_HARD - _LENGTH_PENALTY_TARGET
    for completion in completions:
        text = get_completion_text(completion)
        n = max(len(text.split()), 1)
        if n <= _LENGTH_PENALTY_TARGET:
            r = 0.0
        elif n >= _LENGTH_PENALTY_HARD:
            r = -1.0
        else:
            r = -(n - _LENGTH_PENALTY_TARGET) / span
        rewards.append(r)
    return rewards


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_task(
    RLTask(
        name="arithmetic",
        build_dataset=build_dataset,
        reward_funcs=[accuracy_reward, format_reward, length_penalty],
        description="Mixed arithmetic ({+, -, ×, ÷}) on small integers",
    )
)