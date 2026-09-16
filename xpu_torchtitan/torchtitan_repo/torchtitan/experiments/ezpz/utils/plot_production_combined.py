#!/usr/bin/env python3
"""Per-model and combined production overlay charts.

For each configured production trajectory we emit one or two artifacts:

    - ``all_production_training.{svg,png}`` — every trajectory overlaid
      (top-level ``docs/production/README.md`` + cross-model
      ``docs/production/agpt/README.md``).
    - ``production_2b_training.{svg,png}`` — 2B trajectories only
      (embedded on ``docs/production/agpt/2b/README.md``).
    - ``production_20b_training.{svg,png}`` — 20B only
      (embedded on ``docs/production/agpt/20b/README.md``).
    - ``production_80b_training.{svg,png}`` — 80B only
      (embedded on ``docs/production/agpt/80b/README.md``); skipped if
      no 80B trajectory has live W&B data yet.

Each chart is 3 subplot panels (Loss / TPS-per-GPU / MFU) vs tokens
consumed. The 2B-MDS reference is on the Loss + TPS panels only (MDS
doesn't log MFU). All charts pull from the same trajectory definitions
below and the same W&B fetch path as ``plot_production_wandb.py``, so
per-model and combined views can't drift from each other.

Run:
    python3 -m torchtitan.experiments.ezpz.utils.plot_production_combined
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

# Shared ambivalent + Iosevka style helper.
from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

# Reuse the W&B fetch + .o-log fallback already proven in
# plot_production_wandb.py — same data path that feeds the per-trajectory
# dashboards, so this chart can't diverge from those.
from torchtitan.experiments.ezpz.utils.plot_production_wandb import (  # noqa: E402
    PRODUCTION_RUNS,
    concat_runs,
)

import wandb  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
FIGURES_DIR = REPO_ROOT / "torchtitan/experiments/ezpz/docs/production/figures"
OUT_PATH = FIGURES_DIR / "all_production_training.svg"

# MDS data lives as a CSV pulled separately from the MDS W&B project.
MDS_CSV = (
    REPO_ROOT
    / "torchtitan/experiments/ezpz/docs/production/agpt/2b-mds/loss_data/train_metrics.csv"
)
MDS_TOKENS_PER_STEP = 7_770e9 / 140_000  # ~55.5M tokens/step at GBS=3072, seq=8192

# Canonical per-trajectory palette — keep in sync with
# eval/plot_evals_combined.py and per-model {2b,20b}/plot_eval_overview.py.
COLOR_2B_MDS      = "C0"       # matplotlib C0 (ambivalent palette first color — reads well on both light + dark bg)
COLOR_2B_TT_256N  = "#ef5350"  # salmon-red
COLOR_2B_TT_512N  = "#b71c1c"  # dark red
COLOR_20B_TT_256N = "#fb8c00"  # orange (20B 256N: one-off NODE_FAIL run; no canonical color)
COLOR_20B_TT_512N = "#1b8a3a"  # green

TRAJECTORIES: list[dict] = [
    {
        "model": "2b",
        "label": "2B-MDS (n256, SophiaG ref)",
        "source": "mds",
        "csv_path": str(MDS_CSV),
        "tokens_per_step": MDS_TOKENS_PER_STEP,
        "color": COLOR_2B_MDS,
        "linestyle": "--",
        "marker": None,
    },
    {
        "model": "2b",
        "label": "2B-TT v2 (n256, async)",
        "source": "wandb",
        "key": "2b_v2_256",
        "tokens_per_step": 6144 * 8192,
        "color": COLOR_2B_TT_256N,
        "linestyle": "-",
        "marker": None,
    },
    {
        "model": "2b",
        "label": "2B-TT v2 (n512, sync)",
        "source": "wandb",
        "key": "2b_v2_512",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_2B_TT_512N,
        "linestyle": "-",
        "marker": None,
    },
    {
        "model": "20b",
        "label": "20B-TT v2 (n256)",
        "source": "wandb",
        "key": "20b_v2_256",
        "tokens_per_step": 3072 * 8192,
        "color": COLOR_20B_TT_256N,
        "linestyle": "--",
        "marker": None,
    },
    {
        "model": "20b",
        "label": "20B-TT v2 (n512, sync)",
        "source": "wandb",
        "key": "20b_v2_512",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_20B_TT_512N,
        "linestyle": "-",
        "marker": None,
    },
]


def smooth(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Centered moving average; falls back to identity for short series."""
    values = np.asarray(values, dtype=float)
    if len(values) <= window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="same")
    half = window // 2
    smoothed[:half] = values[:half]
    smoothed[-half:] = values[-half:]
    return smoothed


def load_wandb_trajectory(
    api: wandb.Api, key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pull a trajectory from W&B via the existing concat_runs path.
    Returns (steps, loss, tps_per_gpu, mfu) arrays with NaN-rows filtered.
    """
    cfg = PRODUCTION_RUNS[key]
    data = concat_runs(api, cfg["run_ids"], cfg.get("olog_fallbacks"))
    steps = data["_step"].astype(float)
    loss = data["loss_metrics/global_avg_loss"].astype(float)
    tps = data["throughput(tps)"].astype(float)
    mfu = data["mfu(%)"].astype(float)
    valid = ~np.isnan(steps) & ~np.isnan(loss)
    return steps[valid], loss[valid], tps[valid], mfu[valid]


def load_mds_trajectory(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse train_metrics.csv (iteration, lm_loss, grad_norm, tflops,
    tps_per_gpu, run_id). Returns (iteration, loss, tps_per_gpu). MDS
    doesn't log MFU so the MFU panel just skips this trajectory.
    """
    iters, losses, tps_list = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                it = int(row["iteration"])
                lo = float(row["lm_loss"])
                tp = float(row["tps_per_gpu"])
            except (ValueError, KeyError):
                continue
            iters.append(it)
            losses.append(lo)
            tps_list.append(tp)
    return np.array(iters), np.array(losses), np.array(tps_list)


def _apply_loss_ylim(ax, series: list[dict]) -> None:
    """Crop the loss y-axis to the informative band.

    Bottom = ~0.1 below the global min loss. Top = the max loss reached
    after each series' first 5% of tokens (drops the step-0 warmup spike
    of ~12+ nats while keeping the full descent). No-op if degenerate.
    """
    mins, tops = [], []
    for s in series:
        loss = np.asarray(s["loss"], dtype=float)
        loss = loss[np.isfinite(loss)]
        if loss.size == 0:
            continue
        mins.append(float(loss.min()))
        # Drop the leading 5% (warmup spike) before taking the max.
        start = max(1, int(0.05 * loss.size))
        tail = loss[start:] if loss.size > start else loss
        tops.append(float(tail.max()))
    if not mins or not tops:
        return
    lo = min(mins) - 0.1
    hi = max(tops)
    # A touch of headroom above the post-warmup max so curves don't kiss
    # the top border.
    hi = hi + 0.05 * (hi - lo)
    if hi > lo:
        ax.set_ylim(lo, hi)


def render_figure(
    series: list[dict],
    *,
    suptitle: str,
    out_path: Path,
) -> None:
    """Render a 3-panel (Loss / TPS / MFU) figure for the given series
    list. Saves both SVG and PNG next to ``out_path``.
    """
    if not series:
        print(f"  (skipping {out_path.name}: no trajectories)")
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(suptitle, fontsize=15, fontweight="bold")

    # Panel 1: Loss vs tokens
    ax = axes[0]
    for s in series:
        ax.plot(
            s["tokens_b"], s["loss"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["loss"], window=min(100, max(2, len(s["loss"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}  (n={len(s['loss'])})",
        )
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    # Crop the y-axis so the converged region (where the trajectories
    # actually diverge) is legible instead of being squashed under the
    # step-0 spike (~12+ nats). Bottom: a hair below the global min.
    # Top: the highest loss reached AFTER each curve's first ~5% of
    # tokens, so the early warmup spike is cropped but the full descent
    # still shows. Falls back to autoscale if data is degenerate.
    _apply_loss_ylim(ax, series)

    # Panel 2: TPS/GPU vs tokens
    ax = axes[1]
    for s in series:
        ax.plot(
            s["tokens_b"], s["tps"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["tps"], window=min(100, max(2, len(s["tps"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}",
        )
    ax.set_ylabel("Tokens / sec / GPU")
    ax.set_title("Throughput per GPU")
    ax.grid(alpha=0.25)

    # Panel 3: MFU vs tokens (skip MDS — no MFU column)
    ax = axes[2]
    have_mfu = False
    for s in series:
        if s["mfu"] is None:
            continue
        have_mfu = True
        ax.plot(
            s["tokens_b"], s["mfu"],
            color=s["color"], alpha=0.18, linewidth=0.5, rasterized=True,
        )
        ax.plot(
            s["tokens_b"], smooth(s["mfu"], window=min(100, max(2, len(s["mfu"]) // 20))),
            color=s["color"], linestyle=s["linestyle"], linewidth=1.8,
            label=f"{s['label']}",
        )
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Tokens consumed (B)")
    mfu_title = (
        "Model FLOPs Utilization (TT only — MDS does not log MFU)"
        if any(s["mfu"] is None for s in series) and have_mfu
        else "Model FLOPs Utilization"
    )
    ax.set_title(mfu_title)
    ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", transparent=True)
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"  saved: {out_path}")
    print(f"  saved: {png_path}")


def main() -> None:
    api = wandb.Api()

    # Pull all trajectories once and cache results.
    series: list[dict] = []
    for traj in TRAJECTORIES:
        print(f"\n=== loading {traj['label']} ===")
        if traj["source"] == "wandb":
            steps, loss, tps, mfu = load_wandb_trajectory(api, traj["key"])
            tokens_b = steps * traj["tokens_per_step"] / 1e9
            print(f"  {len(steps)} rows, tokens [{tokens_b[0]:.1f}B, {tokens_b[-1]:.1f}B]")
            series.append({**traj, "tokens_b": tokens_b, "loss": loss, "tps": tps, "mfu": mfu})
        else:  # mds
            iters, loss, tps = load_mds_trajectory(traj["csv_path"])
            tokens_b = iters * traj["tokens_per_step"] / 1e9
            print(f"  {len(iters)} rows, tokens [{tokens_b[0]:.1f}B, {tokens_b[-1]:.1f}B]")
            series.append({**traj, "tokens_b": tokens_b, "loss": loss, "tps": tps, "mfu": None})

    # 1) Combined chart (all models)
    print("\n=== rendering all_production_training ===")
    render_figure(
        series,
        suptitle="AuroraGPT production training — all canonical chains overlaid",
        out_path=OUT_PATH,
    )

    # 2) Per-model charts (one per distinct `model` field)
    models_present = sorted({s.get("model", "?") for s in series})
    for model in models_present:
        model_series = [s for s in series if s.get("model") == model]
        suptitle = f"AuroraGPT-{model.upper()} production training"
        out_path = FIGURES_DIR / f"production_{model}_training.svg"
        print(f"\n=== rendering production_{model}_training ===")
        render_figure(model_series, suptitle=suptitle, out_path=out_path)


if __name__ == "__main__":
    main()
