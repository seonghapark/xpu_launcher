#!/usr/bin/env python3
"""Plot training metrics for the MDS AuroraGPT-2B SophiaG continuation.

Reads ``train_metrics.csv`` (lm_loss, grad_norm, tflops, tps_per_gpu) and
``val_loss.csv`` (val_loss) — both produced by ``pull_wandb_loss.py`` —
and writes the following PNGs into ``../figures/``:

    train_loss.png        train loss vs iteration (raw + EMA)
    val_loss.png          val loss vs iteration
    train_val_loss.png    train + val loss on shared axes
    grad_norm.png         loss/grad_norm (raw + EMA)
    tflops.png            throughput/tflops (per global step)
    tps.png               throughput/tokens_per_gpu_per_sec

The MDS run was a single SophiaG continuation that the W&B project split
across ~113 restart-runs; the puller already collapsed those into one
sequence keyed on the global iteration counter.

To keep the plots from being dominated by the warm-up transient, all
plots drop iterations below ``--start-iter`` (default 1000). The
data-mix transitions are dotted vertical lines.

Usage:
    python3 plot_loss.py [--start-iter 1000]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ambivalent is required — silent fallback hides style regressions.
# Install with: uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: F401

plt.style.use(ambivalent.STYLES["ambivalent"])

# Force monospace across every text element in every figure.
plt.rcParams["font.family"] = "monospace"

DATA_DIR = Path(__file__).parent
FIG_DIR = DATA_DIR.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Stage transitions in the 3-stage SophiaG continuation. Iteration values
# come from the README's data-slice token counts divided by global batch
# size (6144) × seq_len (8192).
STAGE_BOUNDARIES = [
    (95_000, "ntok4673B → ntok7064B"),
    (134_000, "ntok7064B → ntok7770B"),
]

DEFAULT_START_ITER = 1000


def load_train_metrics(
    path: Path, start_iter: int,
) -> tuple[list[int], dict[str, list[float | None]]]:
    iters: list[int] = []
    cols: dict[str, list[float | None]] = {
        "lm_loss": [], "grad_norm": [], "tflops": [], "tps_per_gpu": [],
    }
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = int(row["iteration"])
            if it < start_iter:
                continue
            iters.append(it)
            for k in cols:
                v = row[k]
                cols[k].append(float(v) if v else None)
    return iters, cols


def load_val(
    path: Path, start_iter: int,
) -> tuple[list[int], list[float]]:
    iters: list[int] = []
    vals: list[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            it = int(row["iteration"])
            if it < start_iter:
                continue
            iters.append(it)
            vals.append(float(row["val_loss"]))
    return iters, vals


def ema(vals: list[float], alpha: float = 0.05) -> list[float]:
    if not vals:
        return []
    out = [vals[0]]
    for v in vals[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


def _drop_none(
    iters: list[int], vals: list[float | None],
) -> tuple[list[int], list[float]]:
    keep_iters = []
    keep_vals = []
    for it, v in zip(iters, vals):
        if v is not None:
            keep_iters.append(it)
            keep_vals.append(v)
    return keep_iters, keep_vals


def _annotate_stages(ax, max_iter: int) -> None:
    ymin, ymax = ax.get_ylim()
    for x, label in STAGE_BOUNDARIES:
        if x <= max_iter:
            ax.axvline(x, color="#888", ls=":", lw=1)
            ax.text(
                x, ymax - 0.03 * (ymax - ymin), label,
                rotation=90, fontsize=8, va="top", ha="right", color="#666",
            )


def plot_train_loss(iters: list[int], vals: list[float], out_path: Path) -> None:
    smooth = ema(vals, alpha=0.02)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(iters, vals, alpha=0.18, lw=0.6, label="raw", color="#3b82f6")
    ax.plot(iters, smooth, lw=1.6, label="EMA(α=0.02)", color="#1e3a8a")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("LM Train Loss")
    ax.set_title("AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Training Loss")
    _annotate_stages(ax, iters[-1] if iters else 0)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(iters)} points)")


def plot_val_loss(iters: list[int], vals: list[float], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(iters, vals, marker="o", ms=3, lw=1.2, color="#dc2626")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("LM Validation Loss")
    ax.set_title("AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Validation Loss")
    _annotate_stages(ax, iters[-1] if iters else 0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(iters)} points)")


def plot_train_val(
    train_iters: list[int], train_vals: list[float],
    val_iters: list[int], val_vals: list[float],
    out_path: Path,
) -> None:
    smooth = ema(train_vals, alpha=0.02)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(
        train_iters, train_vals, alpha=0.15, lw=0.6,
        label="train (raw)", color="#3b82f6",
    )
    ax.plot(
        train_iters, smooth, lw=1.4,
        label="train (EMA α=0.02)", color="#1e3a8a",
    )
    ax.plot(
        val_iters, val_vals, marker="o", ms=3, lw=1.0,
        label="validation", color="#dc2626",
    )
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("LM Loss")
    ax.set_title(
        "AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Training + Validation Loss"
    )
    last_iter = max(
        train_iters[-1] if train_iters else 0,
        val_iters[-1] if val_iters else 0,
    )
    _annotate_stages(ax, last_iter)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_grad_norm(iters: list[int], vals: list[float], out_path: Path) -> None:
    smooth = ema(vals, alpha=0.05)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(iters, vals, alpha=0.20, lw=0.5, label="raw", color="#16a34a")
    ax.plot(iters, smooth, lw=1.4, label="EMA(α=0.05)", color="#14532d")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Gradient Norm")
    ax.set_yscale("log")
    _annotate_stages(ax, iters[-1] if iters else 0)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(iters)} points)")


def _filter_below(
    iters: list[int], vals: list[float], min_val: float,
) -> tuple[list[int], list[float]]:
    """Drop (iter, val) pairs where val < min_val.

    Megatron-DeepSpeed logs `throughput/*` once per restart (not per
    iteration) plus on every step. The first iteration after a checkpoint
    load measures the *full* save/load round-trip wall time, not steady-
    state compute, so the per-restart entry is effectively a single
    huge-elapsed-time outlier on the order of 1/10 the steady-state
    throughput. Filtering below the obvious gap removes them.
    """
    out_iters: list[int] = []
    out_vals: list[float] = []
    for it, v in zip(iters, vals):
        if v >= min_val:
            out_iters.append(it)
            out_vals.append(v)
    return out_iters, out_vals


def plot_tflops(iters: list[int], vals: list[float], out_path: Path) -> None:
    # Steady-state band is 35–55 TFLOP/s; everything below 30 is the
    # restart/eval-step throughput artifact (a few thousand outliers in
    # a 154K-point series, but they dominate the visual range).
    iters, vals = _filter_below(iters, vals, min_val=30.0)
    smooth = ema(vals, alpha=0.05)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(iters, vals, alpha=0.18, lw=0.5, label="raw", color="#a855f7")
    ax.plot(iters, smooth, lw=1.4, label="EMA(α=0.05)", color="#581c87")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("TFLOP/s (per replica)")
    ax.set_title(
        "AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Throughput (TFLOP/s)"
    )
    _annotate_stages(ax, iters[-1] if iters else 0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(iters)} points after filter)")


def plot_tps(iters: list[int], vals: list[float], out_path: Path) -> None:
    # Steady-state band is ~3500–4900 TPS/GPU; bottom 5% are restart
    # artifacts (p5 ≈ 3035, p1 ≈ 643).
    iters, vals = _filter_below(iters, vals, min_val=3000.0)
    smooth = ema(vals, alpha=0.05)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(iters, vals, alpha=0.18, lw=0.5, label="raw", color="#f97316")
    ax.plot(iters, smooth, lw=1.4, label="EMA(α=0.05)", color="#7c2d12")
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Tokens / GPU / sec")
    ax.set_title(
        "AuroraGPT-2B (MDS, SophiaG, lr=2.17e-5) — Throughput (TPS / GPU)"
    )
    _annotate_stages(ax, iters[-1] if iters else 0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(iters)} points after filter)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start-iter", type=int, default=DEFAULT_START_ITER,
        help=f"Drop iterations below this value (default {DEFAULT_START_ITER})",
    )
    args = parser.parse_args()

    train_iters, train_cols = load_train_metrics(
        DATA_DIR / "train_metrics.csv", args.start_iter,
    )
    val_iters, val_vals = load_val(DATA_DIR / "val_loss.csv", args.start_iter)

    print(
        f"loaded (start_iter={args.start_iter}): "
        f"train={len(train_iters)} pts "
        f"(iter {train_iters[0] if train_iters else '?'}..{train_iters[-1] if train_iters else '?'}); "
        f"val={len(val_iters)} pts "
        f"(iter {val_iters[0] if val_iters else '?'}..{val_iters[-1] if val_iters else '?'})"
    )

    loss_iters, loss_vals = _drop_none(train_iters, train_cols["lm_loss"])
    gn_iters, gn_vals = _drop_none(train_iters, train_cols["grad_norm"])
    tf_iters, tf_vals = _drop_none(train_iters, train_cols["tflops"])
    tps_iters, tps_vals = _drop_none(train_iters, train_cols["tps_per_gpu"])

    plot_train_loss(loss_iters, loss_vals, FIG_DIR / "train_loss.png")
    plot_val_loss(val_iters, val_vals, FIG_DIR / "val_loss.png")
    plot_train_val(
        loss_iters, loss_vals, val_iters, val_vals,
        FIG_DIR / "train_val_loss.png",
    )
    plot_grad_norm(gn_iters, gn_vals, FIG_DIR / "grad_norm.png")
    plot_tflops(tf_iters, tf_vals, FIG_DIR / "tflops.png")
    plot_tps(tps_iters, tps_vals, FIG_DIR / "tps.png")


if __name__ == "__main__":
    main()
