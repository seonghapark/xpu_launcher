#!/usr/bin/env python3
"""Plot LR finder results using ambivalent style.

Reads lr_finder_data.csv files and generates publication-quality plots.

Usage:
    python3 torchtitan/experiments/ezpz/utils/plot_lr_finder.py

    # Custom data dir:
    python3 torchtitan/experiments/ezpz/utils/plot_lr_finder.py \
        --data-dir outputs/lr_finder/ezpz/ezpz.agpt

    # Custom output dir:
    python3 torchtitan/experiments/ezpz/utils/plot_lr_finder.py \
        --output-dir torchtitan/experiments/ezpz/docs/experiments/lr-finder/figures
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from torchtitan.experiments.ezpz.lr_finder import find_optimal_lr
from torchtitan.experiments.ezpz.utils.plot_style import apply_style


OPTIMIZER_COLORS = {
    "adamw": "#1E88E5",
    "muon": "#D32F2F",
    "sophiag": "#388E3C",
}

OPTIMIZER_LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "sophiag": "SophiaG",
}


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lrs, losses = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            lrs.append(float(row["learning_rate"]))
            losses.append(float(row["loss"]))
    return np.array(lrs), np.array(losses)


def find_data(
    data_dir: Path,
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Find all lr_finder_data.csv files organized by model/optimizer.

    Returns: {model: {optimizer: (lrs, losses)}}
    """
    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

    for csv_path in sorted(data_dir.rglob("lr_finder_data.csv")):
        parts = csv_path.relative_to(data_dir).parts
        if len(parts) >= 2:
            model = parts[-3] if len(parts) >= 3 else "unknown"
            optimizer = parts[-2]
        else:
            continue

        if model not in results:
            results[model] = {}
        results[model][optimizer] = load_csv(csv_path)

    return results


def plot_single_model(
    model: str,
    optimizers: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> Path:
    """Plot LR vs loss for all optimizers of a single model."""
    apply_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    for opt_name, (lrs, losses) in sorted(optimizers.items()):
        color = OPTIMIZER_COLORS.get(opt_name, "#888888")
        label = OPTIMIZER_LABELS.get(opt_name, opt_name)

        min_idx = np.argmin(losses)
        ax.plot(lrs, losses, color=color, linewidth=2, label=label, alpha=0.9)

        # Mark min loss point
        ax.scatter(
            [lrs[min_idx]],
            [losses[min_idx]],
            color=color,
            s=80,
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )

        # Derivative-based suggested LR (use blow-up closest to min loss)
        blow_ups = find_optimal_lr(lrs.tolist(), losses.tolist())
        if blow_ups:
            min_lr = lrs[min_idx]
            best_blowup = min(blow_ups, key=lambda b: abs(np.log10(b) - np.log10(min_lr)))
            suggested = best_blowup / 10
            ax.axvline(
                x=suggested,
                color=color,
                linestyle=":",
                alpha=0.6,
                linewidth=1.5,
            )
            ax.annotate(
                f"{label}: {suggested:.1e}",
                xy=(suggested, losses[min_idx]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color=color,
                fontweight="bold",
            )
        else:
            ax.axvline(
                x=lrs[min_idx],
                color=color,
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )
            ax.annotate(
                f"{label}: {lrs[min_idx]:.1e}",
                xy=(lrs[min_idx], losses[min_idx]),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Smoothed Loss (EMA)")
    ax.set_title(f"LR Finder — agpt {model}")
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3)

    # Clip y-axis to avoid blow-up dominating the plot
    all_losses = np.concatenate([v[1] for v in optimizers.values()])
    finite_losses = all_losses[np.isfinite(all_losses)]
    if len(finite_losses) > 0:
        min_loss = finite_losses.min()
        y_upper = min(finite_losses.max(), min_loss * 5)
        ax.set_ylim(min_loss * 0.95, y_upper)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / f"lr_finder_{model}.png"
    out_svg = output_dir / f"lr_finder_{model}.svg"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")
    return out_png


def plot_comparison(
    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
) -> Path:
    """Plot all models side-by-side for comparison."""
    apply_style()

    models = sorted(results.keys())
    n_models = len(models)
    if n_models == 0:
        return output_dir / "lr_finder_comparison.png"

    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        optimizers = results[model]
        for opt_name, (lrs, losses) in sorted(optimizers.items()):
            color = OPTIMIZER_COLORS.get(opt_name, "#888888")
            label = OPTIMIZER_LABELS.get(opt_name, opt_name)

            min_idx = np.argmin(losses)
            ax.plot(lrs, losses, color=color, linewidth=2, label=label, alpha=0.9)
            ax.scatter(
                [lrs[min_idx]],
                [losses[min_idx]],
                color=color,
                s=80,
                zorder=5,
                edgecolors="white",
                linewidths=1.5,
            )

            blow_ups = find_optimal_lr(lrs.tolist(), losses.tolist())
            if blow_ups:
                min_lr = lrs[min_idx]
                best_blowup = min(blow_ups, key=lambda b: abs(np.log10(b) - np.log10(min_lr)))
                suggested = best_blowup / 10
                ax.axvline(
                    x=suggested,
                    color=color,
                    linestyle=":",
                    alpha=0.6,
                    linewidth=1.5,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel("Smoothed Loss (EMA)")
        ax.set_title(f"agpt {model}")
        ax.legend(framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Clip y-axis
        all_losses = np.concatenate([v[1] for v in optimizers.values()])
        finite_losses = all_losses[np.isfinite(all_losses)]
        if len(finite_losses) > 0:
            min_loss = finite_losses.min()
            y_upper = min(finite_losses.max(), min_loss * 5)
            ax.set_ylim(min_loss * 0.95, y_upper)

    fig.suptitle("LR Finder — Optimizer Comparison", fontsize=14, fontweight="bold")
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "lr_finder_comparison.png"
    out_svg = output_dir / "lr_finder_comparison.svg"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")
    return out_png


def plot_optimal_lr_summary(
    results: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
) -> Path:
    """Bar chart of optimal LR per model x optimizer."""
    apply_style()

    models = sorted(results.keys())
    optimizers_all = sorted(
        {opt for model_data in results.values() for opt in model_data}
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(models))
    width = 0.25
    n_opts = len(optimizers_all)

    for i, opt in enumerate(optimizers_all):
        optimal_lrs = []
        for model in models:
            if opt in results[model]:
                lrs, losses = results[model][opt]
                blow_ups = find_optimal_lr(lrs.tolist(), losses.tolist())
                if blow_ups:
                    # Use the blow-up closest to the global min loss
                    min_lr = lrs[np.argmin(losses)]
                    best = min(blow_ups, key=lambda b: abs(np.log10(b) - np.log10(min_lr)))
                    optimal_lrs.append(best / 10)
                else:
                    min_idx = np.argmin(losses)
                    optimal_lrs.append(lrs[min_idx])
            else:
                optimal_lrs.append(0)

        offset = (i - n_opts / 2 + 0.5) * width
        color = OPTIMIZER_COLORS.get(opt, "#888888")
        label = OPTIMIZER_LABELS.get(opt, opt)
        bars = ax.bar(x + offset, optimal_lrs, width, color=color, label=label, alpha=0.85)

        for bar, lr in zip(bars, optimal_lrs):
            if lr > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{lr:.1e}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    ax.set_yscale("log")
    ax.set_ylabel("Suggested LR (blow-up / 10)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"agpt {m}" for m in models])
    ax.set_title("Suggested Learning Rate by Model & Optimizer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_png = output_dir / "lr_finder_optimal_lr.png"
    out_svg = output_dir / "lr_finder_optimal_lr.svg"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_svg}")
    return out_png


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LR finder results")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("outputs/lr_finder/ezpz/ezpz.agpt"),
        help="Directory containing model/optimizer/lr_finder_data.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/figures"
        ),
        help=(
            "Directory to save plots. Figures are shared per family; prefix "
            "filenames by machine (e.g. sunspot_2b.png) when committing."
        ),
    )
    args = parser.parse_args()

    results = find_data(args.data_dir)
    if not results:
        print(f"No lr_finder_data.csv files found in {args.data_dir}")
        return

    print(f"Found data for {len(results)} model(s):")
    for model, opts in sorted(results.items()):
        print(f"  {model}: {', '.join(sorted(opts.keys()))}")
    print()

    # Per-model plots
    for model, optimizers in sorted(results.items()):
        plot_single_model(model, optimizers, args.output_dir)

    # Comparison plot
    plot_comparison(results, args.output_dir)

    # Optimal LR summary
    plot_optimal_lr_summary(results, args.output_dir)


if __name__ == "__main__":
    main()
