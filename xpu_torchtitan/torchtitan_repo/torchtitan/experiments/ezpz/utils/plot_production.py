#!/usr/bin/env python3
"""Plot production training metrics from PBS output files.

DEPRECATED: prefer ``plot_production_wandb.py``, which pulls history
from W&B and therefore captures in-progress runs whose PBS log files
haven't been written yet. This script is retained for plotting old
runs that pre-date W&B logging, or as a fallback if W&B is
unavailable.

Parses PBS job output files containing training step logs, extracts
loss/throughput/MFU metrics, and generates publication-quality plots.

Usage:
    # Plot 2B and 20B production runs:
    python3 torchtitan/experiments/ezpz/utils/plot_production.py

    # Plot specific model with custom files:
    python3 torchtitan/experiments/ezpz/utils/plot_production.py \
        --model 2b \
        --files agpt-2b-sophiag-n256.o8444122 agpt-2b-sophiag-n256.o8446337 \
        --output-dir torchtitan/experiments/ezpz/docs/production/agpt/2b/figures

    # Generate combined loss plot only:
    python3 torchtitan/experiments/ezpz/utils/plot_production.py --combined-only
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ambivalent is required — silent fallback hides style regressions.
# Install with: uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: F401

plt.style.use(ambivalent.STYLES["ambivalent"])

# Regex to strip ANSI escape sequences
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Regex to parse a step log line (after ANSI stripping).
# Example: step: 16911  loss:  5.73478  grad_norm:  0.3047  memory: 47.02GiB(73.49%)  tps: 2,280  tflops: 25.51  mfu: 8.55%
STEP_RE = re.compile(
    r"step:\s*(?P<step>\d+)\s+"
    r"loss:\s*(?P<loss>[\d.]+)\s+"
    r"grad_norm:\s*[\d.]+\s+"
    r"memory:\s*[\d.]+GiB\([\d.]+%\)\s+"
    r"tps:\s*(?P<tps>[\d,]+)\s+"
    r"tflops:\s*[\d.]+\s+"
    r"mfu:\s*(?P<mfu>[\d.]+)%"
)


@dataclass
class StepRecord:
    step: int
    loss: float
    tps: int
    mfu: float


def parse_file(path: Path) -> list[StepRecord]:
    """Parse a single PBS output file, returning StepRecords."""
    records = []
    with open(path, errors="replace") as f:
        for line in f:
            clean = ANSI_RE.sub("", line)
            m = STEP_RE.search(clean)
            if m:
                records.append(
                    StepRecord(
                        step=int(m.group("step")),
                        loss=float(m.group("loss")),
                        tps=int(m.group("tps").replace(",", "")),
                        mfu=float(m.group("mfu")),
                    )
                )
    return records


def parse_files(paths: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse multiple PBS output files, deduplicate by step, sort, return arrays.

    When multiple files contain the same step number, the last occurrence
    (from the last file in the list) is kept, matching checkpoint-resume
    semantics.

    Returns: (steps, losses, tps, mfu) arrays sorted by step.
    """
    # Collect all records, later files override earlier for same step
    by_step: dict[int, StepRecord] = {}
    for path in paths:
        for rec in parse_file(path):
            by_step[rec.step] = rec

    if not by_step:
        raise ValueError(f"No step data found in {[str(p) for p in paths]}")

    records = sorted(by_step.values(), key=lambda r: r.step)
    steps = np.array([r.step for r in records])
    losses = np.array([r.loss for r in records])
    tps = np.array([r.tps for r in records])
    mfu = np.array([r.mfu for r in records])
    return steps, losses, tps, mfu


def smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    # Use 'valid' mode but pad to keep same length
    smoothed = np.convolve(values, kernel, mode="same")
    # Fix edge effects by using original values at boundaries
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


MODEL_COLORS = {
    "2b": "#1E88E5",
    "20b": "#D32F2F",
    "80b": "#388E3C",
}


def plot_model_dashboard(
    steps: np.ndarray,
    losses: np.ndarray,
    tps: np.ndarray,
    mfu: np.ndarray,
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """Generate 3-subplot dashboard (loss, tps/gpu, mfu) for a single model."""
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    num_gpus = num_nodes * 12  # Aurora: 6 tiles/node * 2 (but counted as 12 GPUs/node)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    max_step = int(steps[-1])
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} Production Training  |  "
        f"{num_nodes} nodes ({num_gpus} GPUs)  |  "
        f"Step {max_step:,}",
        fontsize=14,
        fontweight="bold",
    )

    # --- Loss ---
    ax = axes[0]
    ax.plot(steps, losses, color=color, alpha=0.25, linewidth=0.5)
    ax.plot(steps, smooth(losses, 100), color=color, linewidth=1.8, label="Loss (smoothed)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    # --- TPS / GPU ---
    ax = axes[1]
    tps_per_gpu = tps / num_gpus
    ax.plot(steps, tps_per_gpu, color="#43A047", alpha=0.25, linewidth=0.5)
    ax.plot(
        steps,
        smooth(tps_per_gpu, 100),
        color="#43A047",
        linewidth=1.8,
        label="TPS/GPU (smoothed)",
    )
    ax.set_ylabel("Tokens/sec/GPU")
    ax.set_title("Throughput per GPU")
    ax.legend()

    # --- MFU ---
    ax = axes[2]
    ax.plot(steps, mfu, color="#FF9800", alpha=0.25, linewidth=0.5)
    ax.plot(steps, smooth(mfu, 100), color="#FF9800", linewidth=1.8, label="MFU (smoothed)")
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Training Step")
    ax.set_title("Model FLOPs Utilization")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


def plot_combined_loss(
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> Path:
    """Plot loss curves for multiple models on the same axes.

    Args:
        model_data: {model_name: (steps, losses)}
        output_path: Where to save the figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    for model_name, (steps, losses) in sorted(model_data.items()):
        color = MODEL_COLORS.get(model_name, "#888888")
        label = f"AuroraGPT {model_name.upper()}"
        max_step = int(steps[-1])
        ax.plot(steps, losses, color=color, alpha=0.15, linewidth=0.5)
        ax.plot(
            steps,
            smooth(losses, 100),
            color=color,
            linewidth=2.0,
            label=f"{label} (step {max_step:,})",
        )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(
        "AuroraGPT Production Training Loss  |  256 nodes (3072 GPUs)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")
    return output_path


# --- Default configurations for known production runs ---

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DOCS_BASE = REPO_ROOT / "torchtitan" / "experiments" / "ezpz" / "docs"

PRODUCTION_RUNS: dict[str, dict] = {
    "2b": {
        "files": [
            "agpt-2b-sophiag-n256.o8444122",
            "agpt-2b-sophiag-n256.o8446337",
            "agpt-2b-sophiag-n256.o8446338",
        ],
        "num_nodes": 256,
        "output_dir": DOCS_BASE / "production" / "agpt" / "2b" / "figures",
    },
    "20b": {
        "files": [
            "agpt-20b-sophiag-n256.o8443212",
            "agpt-20b-sophiag-n256.o8444123",
            "agpt-20b-sophiag-n256.o8446340",
            "agpt-20b-sophiag-n256.o8446341",
            "agpt-20b-sophiag-n256.o8446342",
        ],
        "num_nodes": 256,
        "output_dir": DOCS_BASE / "production" / "agpt" / "20b" / "figures",
    },
}

COMBINED_OUTPUT = (
    DOCS_BASE / "experiments" / "agpt" / "aurora" / "figures" / "production_training_loss.png"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot production training metrics from PBS output files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to plot (e.g. '2b', '20b'). If omitted, plots all known models.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help="PBS output files to parse (overrides defaults for --model).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (overrides default).",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=256,
        help="Number of nodes (default: 256).",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only generate the combined loss plot.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (where PBS output files live).",
    )
    args = parser.parse_args()

    repo_root = args.repo_root if args.repo_root else REPO_ROOT

    # Determine which models to process
    if args.model:
        models_to_plot = [args.model]
    else:
        models_to_plot = list(PRODUCTION_RUNS.keys())

    # Collect parsed data for combined plot
    combined_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name in models_to_plot:
        if args.files:
            file_paths = [Path(f) if f.is_absolute() else repo_root / f for f in args.files]
            num_nodes = args.num_nodes
            output_dir = args.output_dir or PRODUCTION_RUNS.get(model_name, {}).get(
                "output_dir",
                DOCS_BASE / "production" / "agpt" / model_name / "figures",
            )
        elif model_name in PRODUCTION_RUNS:
            cfg = PRODUCTION_RUNS[model_name]
            file_paths = [repo_root / f for f in cfg["files"]]
            num_nodes = cfg["num_nodes"]
            output_dir = args.output_dir or cfg["output_dir"]
        else:
            print(f"Unknown model '{model_name}', skipping.")
            continue

        # Filter to files that exist
        existing = [f for f in file_paths if f.exists()]
        missing = [f for f in file_paths if not f.exists()]
        if missing:
            print(f"Warning: {len(missing)} file(s) not found for {model_name}:")
            for f in missing:
                print(f"  {f}")
        if not existing:
            print(f"No files found for {model_name}, skipping.")
            continue

        print(f"\nParsing {model_name.upper()} from {len(existing)} file(s)...")
        steps, losses, tps, mfu = parse_files(existing)
        print(
            f"  {len(steps)} unique steps, range [{int(steps[0])}, {int(steps[-1])}]"
        )
        print(f"  Loss: {losses[-1]:.4f} (latest), {losses.min():.4f} (min)")
        print(f"  TPS:  {int(tps[-10:].mean()):,} (recent avg)")
        print(f"  MFU:  {mfu[-10:].mean():.2f}% (recent avg)")

        combined_data[model_name] = (steps, losses)

        if not args.combined_only:
            output_path = Path(output_dir) / f"production_{model_name}_{num_nodes}n.png"
            plot_model_dashboard(steps, losses, tps, mfu, model_name, num_nodes, output_path)

    # Combined loss plot
    if len(combined_data) >= 1:
        combined_path = args.output_dir / "production_training_loss.png" if args.output_dir else COMBINED_OUTPUT
        plot_combined_loss(combined_data, combined_path)


if __name__ == "__main__":
    main()
