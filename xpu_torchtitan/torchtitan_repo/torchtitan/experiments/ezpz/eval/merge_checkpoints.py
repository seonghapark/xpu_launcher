# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# WSM: Weighted State Merging for LLM checkpoints.
#
# Instead of online LR decay during training, train at constant LR and
# merge checkpoints offline with theoretically-derived weights. Shown to
# improve over WSD in "WSM: Decay-Free Learning Rate Schedule via
# Checkpoint Merging for LLM Pre-training" (arxiv 2507.17634).
#
# Usage:
#   python3 -m torchtitan.experiments.ezpz.eval.merge_checkpoints \
#       --checkpoints step-500 step-1000 step-1500 step-2000 \
#       --checkpoint-dir outputs/checkpoints/my_run \
#       --output merged_checkpoint \
#       --method uniform
#
# Methods:
#   uniform  — equal weight for all checkpoints (simple average)
#   linear   — linearly increasing weights (later = more important)
#   exp      — exponentially increasing weights

import argparse
import logging
import math
import os
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def compute_weights(num_checkpoints: int, method: str = "uniform") -> list[float]:
    """Compute merge weights for checkpoints.

    Args:
        num_checkpoints: Number of checkpoints to merge.
        method: Weighting method — "uniform", "linear", or "exp".

    Returns:
        Normalized weight list (sums to 1.0).
    """
    if method == "uniform":
        weights = [1.0] * num_checkpoints
    elif method == "linear":
        # Linearly increasing: later checkpoints get more weight
        weights = [float(i + 1) for i in range(num_checkpoints)]
    elif method == "exp":
        # Exponentially increasing
        weights = [math.exp(i) for i in range(num_checkpoints)]
    else:
        raise ValueError(f"Unknown method {method!r}. Use: uniform, linear, exp")

    total = sum(weights)
    return [w / total for w in weights]


def merge_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]],
    weights: list[float],
) -> dict[str, torch.Tensor]:
    """Weighted merge of multiple state dicts.

    Args:
        state_dicts: List of model state dicts.
        weights: Merge weight per state dict (must sum to 1.0).

    Returns:
        Merged state dict.
    """
    merged = {}
    keys = state_dicts[0].keys()
    for key in keys:
        tensors = [sd[key] for sd in state_dicts]
        if tensors[0].is_floating_point():
            merged[key] = sum(w * t.float() for w, t in zip(weights, tensors)).to(
                tensors[0].dtype
            )
        else:
            # Non-float tensors (e.g., step counters) — take from last checkpoint
            merged[key] = tensors[-1]
    return merged


def load_checkpoint(path: str) -> dict[str, torch.Tensor]:
    """Load a checkpoint state dict from a DCP or safetensors directory."""
    path = Path(path)

    # Try safetensors first
    safetensor_files = list(path.glob("*.safetensors"))
    if safetensor_files:
        from safetensors.torch import load_file

        state_dict = {}
        for f in sorted(safetensor_files):
            state_dict.update(load_file(str(f)))
        return state_dict

    # Try PyTorch DCP
    pt_files = list(path.glob("*.pt")) + list(path.glob("*.bin"))
    if pt_files:
        state_dict = {}
        for f in sorted(pt_files):
            state_dict.update(torch.load(f, map_location="cpu", weights_only=True))
        return state_dict

    raise FileNotFoundError(
        f"No checkpoint files found in {path}. "
        f"Expected .safetensors, .pt, or .bin files."
    )


def save_checkpoint(state_dict: dict[str, torch.Tensor], path: str) -> None:
    """Save a merged state dict as safetensors."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    try:
        from safetensors.torch import save_file

        save_file(state_dict, str(path / "model.safetensors"))
    except ImportError:
        torch.save(state_dict, str(path / "model.pt"))

    log.info(f"Saved merged checkpoint to {path}")


def main():
    parser = argparse.ArgumentParser(description="WSM: Weighted State Merging")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint subdirectory names (e.g., step-500 step-1000)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Parent directory containing checkpoint subdirs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for merged checkpoint",
    )
    parser.add_argument(
        "--method",
        default="linear",
        choices=["uniform", "linear", "exp"],
        help="Weighting method (default: linear)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Custom weights (overrides --method). Must match number of checkpoints.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Build checkpoint paths
    ckpt_paths = [
        os.path.join(args.checkpoint_dir, name) for name in args.checkpoints
    ]

    # Compute or use custom weights
    if args.weights:
        if len(args.weights) != len(ckpt_paths):
            raise ValueError(
                f"Got {len(args.weights)} weights for {len(ckpt_paths)} checkpoints"
            )
        total = sum(args.weights)
        weights = [w / total for w in args.weights]
    else:
        weights = compute_weights(len(ckpt_paths), args.method)

    log.info(f"Merging {len(ckpt_paths)} checkpoints with method={args.method}")
    for path, w in zip(ckpt_paths, weights):
        log.info(f"  {path}: weight={w:.4f}")

    # Load all checkpoints
    state_dicts = []
    for path in ckpt_paths:
        log.info(f"Loading {path}...")
        state_dicts.append(load_checkpoint(path))

    # Merge
    log.info("Merging state dicts...")
    merged = merge_state_dicts(state_dicts, weights)

    # Save
    save_checkpoint(merged, args.output)
    log.info("Done.")


if __name__ == "__main__":
    main()
