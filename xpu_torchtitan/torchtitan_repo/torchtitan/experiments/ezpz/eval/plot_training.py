# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Plotting utilities for training experiment analysis.
#
# Usage:
#   python3 -m torchtitan.experiments.ezpz.eval.plot_training \
#       --output figures/ \
#       --configs r4_adamw r4_adamw_qknorm r4_mano r4_mano_qknorm \
#       --labels "AdamW" "AdamW+QK-Norm" "Mano" "Mano+QK-Norm"
#
# Or as a library:
#   from torchtitan.experiments.ezpz.eval.plot_training import (
#       load_run_data, plot_loss_curves, plot_metrics_panel,
#   )

from __future__ import annotations

import argparse
import glob
import os
import re
from dataclasses import dataclass, field

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ambivalent is required — silent fallback hides style regressions.
# Install with: uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: F401

plt.style.use(ambivalent.STYLES["ambivalent"])


def compute_wsd_lr_schedule(
    total_steps: int,
    peak_lr: float,
    warmup_steps: int = 20,
    decay_ratio: float = 0.2,
    decay_type: str = "cosine",
    min_lr_factor: float = 0.0,
) -> list[float]:
    """Reconstruct the WSD learning rate schedule.

    Args:
        total_steps: Total training steps.
        peak_lr: Peak learning rate.
        warmup_steps: Number of warmup steps.
        decay_ratio: Fraction of total steps for decay phase.
        decay_type: "cosine" or "linear".
        min_lr_factor: Minimum LR as fraction of peak.

    Returns:
        List of LR values, one per step.
    """
    import math

    min_lr = peak_lr * min_lr_factor
    decay_steps = int(total_steps * decay_ratio)
    stable_steps = total_steps - warmup_steps - decay_steps

    lrs = []
    for step in range(total_steps):
        if step < warmup_steps:
            # Linear warmup
            lr = peak_lr * (step + 1) / warmup_steps
        elif step < warmup_steps + stable_steps:
            # Stable phase
            lr = peak_lr
        else:
            # Decay phase
            decay_step = step - warmup_steps - stable_steps
            progress = decay_step / max(decay_steps - 1, 1)
            if decay_type == "cosine":
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            elif decay_type == "linear":
                lr = peak_lr + (min_lr - peak_lr) * progress
            else:
                lr = peak_lr
        lrs.append(lr)
    return lrs


METRICS_PATTERNS = {
    "loss": r"loss:\s*([\d.]+)",
    "grad_norm": r"grad_norm:\s*([\d.]+)",
    "tps": r"tps:\s*([\d,]+)",
    "tflops": r"tflops:\s*([\d.]+)",
    "mfu": r"mfu:\s*([\d.]+)%",
    "memory": r"memory:\s*([\d.]+)",
}


@dataclass
class RunData:
    name: str
    steps: list[int] = field(default_factory=list)
    metrics: dict[str, list[float]] = field(default_factory=dict)
    label: str = ""
    color: str | None = None
    linestyle: str = "-"


def load_run_data(
    config_name: str,
    *,
    label: str | None = None,
    output_dir: str = ".",
    ema_alpha: float = 0.03,
    grad_norm_alpha: float = 0.05,
    peak_lr: float | None = None,
    warmup_steps: int = 20,
    decay_ratio: float = 0.2,
    decay_type: str = "cosine",
    min_lr_factor: float = 0.0,
) -> RunData | None:
    """Load training metrics from PBS output files.

    Finds the latest output file matching `{config_name}.o*` and extracts
    step, loss, grad_norm, tps, tflops, mfu, memory.

    Args:
        config_name: Config name prefix (e.g. "r4_adamw").
        label: Display label for plots.
        output_dir: Directory containing output files.
        ema_alpha: EMA smoothing factor for most metrics.
        grad_norm_alpha: EMA smoothing factor for grad_norm.

    Returns:
        RunData with EMA-smoothed metrics, or None if no data found.
    """
    pattern = os.path.join(output_dir, f"{config_name}.o*")
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not files:
        return None

    with open(files[0]) as fh:
        text = fh.read()

    steps = [int(s) for s in re.findall(r"step:\s*(\d+)", text)]
    if not steps:
        return None

    raw_metrics = {}
    for metric, pat in METRICS_PATTERNS.items():
        vals = re.findall(pat, text)
        if metric == "tps":
            vals = [float(v.replace(",", "")) for v in vals]
        else:
            vals = [float(v) for v in vals]
        raw_metrics[metric] = vals

    n = min(len(steps), *[len(v) for v in raw_metrics.values() if v])
    steps = steps[:n]

    # EMA smooth
    smoothed = {}
    for metric, vals in raw_metrics.items():
        if not vals:
            continue
        vals = vals[:n]
        alpha = grad_norm_alpha if metric == "grad_norm" else ema_alpha
        ema = vals[0]
        smooth = []
        for v in vals:
            ema = alpha * v + (1 - alpha) * ema
            smooth.append(ema)
        smoothed[metric] = smooth

    # Inject reconstructed LR schedule if peak_lr provided
    if peak_lr is not None:
        lr_schedule = compute_wsd_lr_schedule(
            total_steps=max(steps) if steps else 1000,
            peak_lr=peak_lr,
            warmup_steps=warmup_steps,
            decay_ratio=decay_ratio,
            decay_type=decay_type,
            min_lr_factor=min_lr_factor,
        )
        smoothed["lr"] = [lr_schedule[min(s, len(lr_schedule) - 1)] for s in steps]

    return RunData(
        name=config_name,
        steps=steps,
        metrics=smoothed,
        label=label or config_name,
    )


def plot_loss_curves(
    runs: list[RunData],
    *,
    title: str = "Loss Curves",
    gbs: int = 384,
    seq_len: int = 8192,
    inset: bool = True,
    inset_start_frac: float = 0.6,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot loss curves by step and by tokens with optional inset zoom."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, run in enumerate(runs):
        c = run.color or colors[i % len(colors)]
        if "loss" not in run.metrics:
            continue
        ax1.plot(run.steps, run.metrics["loss"], label=run.label,
                 color=c, linestyle=run.linestyle, linewidth=1.5)
        tokens = [s * gbs * seq_len / 1e9 for s in run.steps]
        ax2.plot(tokens, run.metrics["loss"], label=run.label,
                 color=c, linestyle=run.linestyle, linewidth=1.5)

    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss (EMA)")
    ax1.set_title(f"{title} — by Step"); ax1.legend(fontsize=7, loc="upper right")
    ax2.set_xlabel("Tokens (B)"); ax2.set_ylabel("Loss (EMA)")
    ax2.set_title(f"{title} — by Tokens"); ax2.legend(fontsize=7, loc="upper right")

    if inset and runs:
        max_step = max(r.steps[-1] for r in runs if r.steps)
        start = int(max_step * inset_start_frac)
        for ax, x_fn in [(ax1, lambda r: r.steps),
                          (ax2, lambda r: [s * gbs * seq_len / 1e9 for s in r.steps])]:
            axins = inset_axes(ax, width="45%", height="45%", loc="center right",
                              bbox_to_anchor=(0, 0.05, 1, 1), bbox_transform=ax.transAxes)
            for i, run in enumerate(runs):
                if "loss" not in run.metrics or "--" in run.linestyle:
                    continue
                c = run.color or colors[i % len(colors)]
                xs = x_fn(run)
                mask = [j for j, s in enumerate(run.steps) if s >= start]
                if mask:
                    axins.plot([xs[j] for j in mask],
                              [run.metrics["loss"][j] for j in mask],
                              color=c, linestyle=run.linestyle, linewidth=1.5)
            axins.set_title("Tail zoom", fontsize=8)
            axins.tick_params(labelsize=7)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_panel(
    runs: list[RunData],
    *,
    title: str = "Training Metrics",
    metrics: list[tuple[str, str]] | None = None,
    output_path: str | None = None,
) -> matplotlib.figure.Figure:
    """Plot a multi-panel grid of training metrics with tail-zoom insets."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if metrics is None:
        # Check if any run has LR data
        has_lr = any("lr" in r.metrics for r in runs)
        if has_lr:
            metrics = [
                ("loss", "Loss (EMA)"),
                ("lr", "Learning Rate"),
                ("grad_norm", "Gradient Norm (EMA)"),
                ("tps", "TPS / GPU"),
                ("mfu", "MFU (%)"),
            ]
        else:
            metrics = [
                ("loss", "Loss (EMA)"),
                ("grad_norm", "Gradient Norm (EMA)"),
                ("tps", "TPS / GPU"),
                ("mfu", "MFU (%)"),
            ]

    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    # Hide any unused axes
    for idx in range(len(metrics), nrows * ncols):
        axes.flat[idx].set_visible(False)

    # Metrics that benefit from a tail-zoom inset
    inset_metrics = {"loss", "grad_norm", "tps", "mfu"}

    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for i, run in enumerate(runs):
            if metric not in run.metrics:
                continue
            c = run.color or colors[i % len(colors)]
            ax.plot(run.steps, run.metrics[metric], label=run.label,
                    color=c, linestyle=run.linestyle, linewidth=1.5)
        ax.set_xlabel("Step"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel); ax.legend(fontsize=7, loc="best")

        # Add inset zoom for applicable metrics
        if metric in inset_metrics and runs:
            valid_runs = [r for r in runs if metric in r.metrics and r.steps]
            if valid_runs:
                max_step = max(r.steps[-1] for r in valid_runs)
                start = int(max_step * 0.6)
                axins = inset_axes(
                    ax, width="40%", height="40%",
                    loc="center right",
                    bbox_to_anchor=(0, 0.05, 1, 1), bbox_transform=ax.transAxes,
                )
                for i, run in enumerate(valid_runs):
                    c = run.color or colors[i % len(colors)]
                    mask = [j for j, s in enumerate(run.steps) if s >= start]
                    if mask:
                        axins.plot(
                            [run.steps[j] for j in mask],
                            [run.metrics[metric][j] for j in mask],
                            color=c, linestyle=run.linestyle, linewidth=1.2,
                        )
                axins.set_title("Tail zoom", fontsize=7)
                axins.tick_params(labelsize=6)

    fig.suptitle(title, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics")
    parser.add_argument("--configs", nargs="+", required=True,
                        help="Config names to plot")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Display labels (must match --configs count)")
    parser.add_argument("--linestyles", nargs="+", default=None,
                        help="Line styles (must match --configs count)")
    parser.add_argument("--output", default="figures",
                        help="Output directory for plots")
    parser.add_argument("--title", default="Training Metrics",
                        help="Plot title prefix")
    parser.add_argument("--gbs", type=int, default=384,
                        help="Global batch size for token calculation")
    parser.add_argument("--seq-len", type=int, default=8192,
                        help="Sequence length for token calculation")
    parser.add_argument("--input-dir", default=".",
                        help="Directory containing output files")
    args = parser.parse_args()

    labels = args.labels or args.configs
    linestyles = args.linestyles or ["-"] * len(args.configs)
    os.makedirs(args.output, exist_ok=True)

    runs = []
    for config, label, ls in zip(args.configs, labels, linestyles):
        run = load_run_data(config, label=label, output_dir=args.input_dir)
        if run:
            run.linestyle = ls
            runs.append(run)
            print(f"Loaded {config}: {len(run.steps)} steps, "
                  f"final loss={run.metrics.get('loss', [None])[-1]}")
        else:
            print(f"No data for {config}")

    if not runs:
        print("No data to plot")
        return

    plot_loss_curves(
        runs, title=args.title, gbs=args.gbs, seq_len=args.seq_len,
        output_path=os.path.join(args.output, "loss_curves.png"),
    )
    plot_metrics_panel(
        runs, title=args.title,
        output_path=os.path.join(args.output, "metrics.png"),
    )
    print(f"Plots saved to {args.output}/")


if __name__ == "__main__":
    main()
