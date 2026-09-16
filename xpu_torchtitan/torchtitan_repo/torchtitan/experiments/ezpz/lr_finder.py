# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Learning rate finder for torchtitan-ezpz.

Sweeps learning rates exponentially from init_lr to max_lr, recording
EMA-smoothed loss at each step. The resulting curve identifies the optimal
learning rate (steepest descent point or blow-up point / 10).

Reference:
  - Smith 2015: https://arxiv.org/abs/1506.01186
  - Gugger: https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
  - Megatron-DeepSpeed: argonne-lcf/Megatron-DeepSpeed
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from torchtitan.tools.logging import logger

if TYPE_CHECKING:
    from torchtitan.experiments.ezpz.trainer import FaultTolerantTrainer


@dataclass
class LRFinderConfig:
    """Configuration for the learning rate finder."""

    enable: bool = False
    """Run LR finder sweep instead of normal training."""

    init_lr: float = 1e-6
    """Starting learning rate for the sweep."""

    max_lr: float = 1.0
    """Maximum learning rate for the sweep."""

    fraction: float = 0.1
    """Fraction of training.steps to use for the sweep."""

    beta: float = 0.98
    """EMA smoothing factor for loss."""

    warmup_fraction: float = 0.0
    """Fraction of finder steps to hold at init_lr before sweeping.
    Lets the model settle before measuring LR sensitivity."""

    smooth_frac: float = 0.05
    """Moving-average window fraction for derivative-based analysis.
    Increase for noisier curves or fewer steps (e.g. 0.1 for <50 steps)."""


def find_optimal_lr(
    lrs: list[float],
    losses: list[float],
    smooth_frac: float = 0.05,
) -> list[float]:
    """Find optimal learning rates using derivative analysis.

    Smooths the loss curve, computes derivative vs log10(LR), and finds
    zero-crossings from negative to positive (local minima = blow-up points).

    Returns sorted list of candidate LRs (blow-up points). Divide by 10
    for suggested training LR.

    Ported from argonne-lcf/Megatron-DeepSpeed.
    """
    import numpy as np

    lr_arr = np.array(lrs)
    loss_arr = np.array(losses)
    n = len(lr_arr)

    if n < 5:
        return []

    # Moving-average smoothing
    window = max(1, int(n * smooth_frac))
    if window > 1:
        kernel = np.ones(window) / window
        smoothed = np.convolve(loss_arr, kernel, mode="same")
        # Fix boundary effects
        for i in range(window // 2):
            smoothed[i] = loss_arr[: i + window // 2 + 1].mean()
            smoothed[-(i + 1)] = loss_arr[-(i + window // 2 + 1) :].mean()
    else:
        smoothed = loss_arr.copy()

    # Derivative of smoothed loss vs log10(LR)
    log_lr = np.log10(lr_arr)
    dloss = np.gradient(smoothed, log_lr)

    # Find zero-crossings: negative -> positive (local minima)
    minima_lrs = []
    for i in range(1, len(dloss)):
        if dloss[i - 1] < 0 and dloss[i] >= 0:
            # Interpolate the crossing point
            frac = -dloss[i - 1] / (dloss[i] - dloss[i - 1] + 1e-12)
            crossing_log_lr = log_lr[i - 1] + frac * (log_lr[i] - log_lr[i - 1])
            minima_lrs.append(10**crossing_log_lr)

    return sorted(minima_lrs)


def run_lr_finder(trainer: FaultTolerantTrainer) -> None:
    """Run an exponential LR sweep and save results.

    Sweeps LR from init_lr to max_lr over (fraction * training.steps) steps,
    recording EMA-smoothed loss at each LR. Saves CSV, NPZ, and a plot.
    Does not continue to normal training.
    """
    config = trainer.config.lr_finder
    training_steps = trainer.config.training.steps
    total_iters = max(1, int(training_steps * config.fraction))
    warmup_steps = int(total_iters * config.warmup_fraction)
    sweep_steps = total_iters - warmup_steps
    mult = (config.max_lr / config.init_lr) ** (1.0 / max(1, sweep_steps))

    logger.info(
        f"LR Finder: sweeping from {config.init_lr:.2e} to {config.max_lr:.2e} "
        f"over {sweep_steps} steps (mult={mult:.6f})"
    )
    if warmup_steps > 0:
        logger.info(
            f"LR Finder: warmup {warmup_steps} steps at lr={config.init_lr:.2e} "
            f"before sweep"
        )

    # Set initial LR on all optimizer param groups
    curr_lr = config.init_lr
    for optimizer in trainer.optimizers.optimizers:
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr

    # EMA tracking
    avg_loss = 0.0
    best_loss = float("inf")
    batch_num = 0

    lrs: list[float] = []
    losses: list[float] = []

    data_iterator = trainer.batch_generator(trainer.dataloader)

    for i in range(total_iters):
        trainer.step += 1
        in_warmup = i < warmup_steps

        # Run one training step; returns global_avg_loss
        loss_val = trainer.train_step(data_iterator)
        if loss_val is None:
            continue

        if isinstance(loss_val, torch.Tensor):
            loss_val = float(loss_val.item())

        batch_num += 1

        # EMA-smoothed loss with bias correction
        avg_loss = config.beta * avg_loss + (1.0 - config.beta) * loss_val
        smoothed_loss = avg_loss / (1.0 - config.beta**batch_num)

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Only record data points during the sweep phase
        if not in_warmup:
            lrs.append(curr_lr)
            losses.append(smoothed_loss)

        # Log progress
        log_freq = trainer.config.metrics.log_freq
        if (i + 1) % log_freq == 0:
            phase = "warmup" if in_warmup else "sweep"
            logger.info(
                f"LR Finder [{phase}]: step {i + 1}/{total_iters}, "
                f"lr={curr_lr:.8f}, smoothed_loss={smoothed_loss:.4f}"
            )

        # Advance LR exponentially only during sweep
        if not in_warmup:
            curr_lr *= mult
            for optimizer in trainer.optimizers.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr

    # Save results on rank 0
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        # Group outputs: lr_finder/ezpz/<name>/<flavor>/<optimizer>/
        model_spec = trainer.config.model_spec
        # Derive optimizer name from container class
        opt_cls = type(trainer.optimizers).__name__
        opt_name = (
            opt_cls.removesuffix("OptimizersContainer")
            .removesuffix("Container")
            .lower()
            or "adamw"
        )
        if model_spec is not None:
            sub_path = os.path.join(
                "ezpz", model_spec.name, model_spec.flavor, opt_name
            )
        else:
            sub_path = os.path.join("unknown", opt_name)
        out_dir = os.path.join(trainer.config.dump_folder, "lr_finder", sub_path)
        os.makedirs(out_dir, exist_ok=True)

        # Metadata for this run
        from datetime import datetime

        run_timestamp = datetime.now().isoformat()
        job_id = os.environ.get(
            "PBS_JOBID",
            os.environ.get("SLURM_JOB_ID", "local"),
        )
        hostname = os.environ.get("HOSTNAME", "unknown")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_batch_size = trainer.config.training.global_batch_size
        seq_len = trainer.config.training.seq_len

        # CSV — append mode so runs accumulate across experiments.
        # If file exists with outdated header, rename as backup.
        csv_path = os.path.join(out_dir, "lr_finder_data.csv")
        new_header = [
            "learning_rate", "loss",
            "timestamp", "job_id", "hostname", "world_size",
            "global_batch_size", "seq_len",
        ]
        write_header = True
        if os.path.isfile(csv_path):
            with open(csv_path) as f:
                first_line = f.readline().strip()
            if "global_batch_size" in first_line:
                write_header = False  # already has current header
            elif "timestamp" in first_line:
                # Has old 6-column header but missing GBS — back up
                backup = csv_path + ".bak2"
                os.rename(csv_path, backup)
                logger.info(f"LR Finder: backed up old CSV to {backup}")
            else:
                # Very old 2-column format
                backup = csv_path + ".bak"
                os.rename(csv_path, backup)
                logger.info(f"LR Finder: backed up old CSV to {backup}")
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(new_header)
            for lr, loss in zip(lrs, losses):
                writer.writerow([
                    lr, loss,
                    run_timestamp, job_id, hostname, world_size,
                    global_batch_size, seq_len,
                ])
        logger.info(f"LR Finder: appended {len(lrs)} rows to {csv_path}")

        # NPZ
        try:
            import numpy as np

            npz_path = os.path.join(out_dir, "lr_finder_data.npz")
            np.savez(
                npz_path,
                learning_rates=np.array(lrs),
                losses=np.array(losses),
            )
            logger.info(f"LR Finder: saved NPZ to {npz_path}")
        except ImportError:
            logger.warning("numpy not available, skipping NPZ output")

        # Derivative-based optimal LR analysis
        try:
            blow_up_lrs = find_optimal_lr(
                lrs, losses, smooth_frac=config.smooth_frac
            )
            if blow_up_lrs:
                suggested = blow_up_lrs[0] / 10
                logger.info(
                    f"LR Finder: suggested LR = {suggested:.2e} "
                    f"(blow-up at {blow_up_lrs[0]:.2e})"
                )
                if len(blow_up_lrs) > 1:
                    logger.info(
                        f"LR Finder: all blow-up points: "
                        f"{[f'{lr:.2e}' for lr in blow_up_lrs]}"
                    )
            else:
                suggested = None
                logger.warning(
                    "LR Finder: could not detect blow-up point. "
                    "Try increasing max_lr or fraction."
                )
        except Exception as e:
            suggested = None
            logger.warning(f"LR Finder: derivative analysis failed: {e}")
            blow_up_lrs = []

        # Plot
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # House style (ambivalent + Iosevka), matching the production /
            # eval / docs charts. apply_style is import-safe in the XPU
            # training .venv (loads the stylesheet from file rather than
            # importing ambivalent, which would pull IPython).
            from torchtitan.experiments.ezpz.utils.plot_style import (
                apply_style,
            )

            apply_style()

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(lrs, losses, linewidth=1.5)
            ax.set_xscale("log")
            ax.set_xlabel("Learning Rate")
            ax.set_ylabel("Smoothed Loss")
            ax.set_title("LR Finder: Learning Rate vs. Loss")
            ax.grid(True, alpha=0.3)

            # Mark the minimum loss point
            min_idx = losses.index(min(losses))
            ax.axvline(
                x=lrs[min_idx],
                color="b",
                linestyle="--",
                alpha=0.7,
                label=f"Min loss @ lr={lrs[min_idx]:.2e}",
            )

            # Mark blow-up and suggested LR
            if blow_up_lrs:
                ax.axvline(
                    x=blow_up_lrs[0],
                    color="r",
                    linestyle="-.",
                    alpha=0.7,
                    label=f"Blow-up @ lr={blow_up_lrs[0]:.2e}",
                )
            if suggested is not None:
                ax.axvline(
                    x=suggested,
                    color="g",
                    linestyle=":",
                    linewidth=2,
                    alpha=0.8,
                    label=f"Suggested lr={suggested:.2e}",
                )

            ax.legend()

            plot_path = os.path.join(out_dir, "lr_vs_loss.png")
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"LR Finder: saved plot to {plot_path}")
        except ImportError:
            logger.warning("matplotlib not available, skipping plot output")

        logger.info(
            f"LR Finder complete. {len(lrs)} data points saved to {out_dir}/"
        )

    # Synchronize all ranks before exit
    if dist.is_initialized():
        dist.barrier()
