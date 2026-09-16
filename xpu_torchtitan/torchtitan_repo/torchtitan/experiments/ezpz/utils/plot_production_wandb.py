#!/usr/bin/env python3
"""Plot production training metrics pulled from W&B.

W&B is the single source of truth for production training history,
since PBS log files only exist for jobs that have already exited and
the active runs span multiple resumes. Generates three figures per
model size:

  - ``production_<model>_<num_nodes>n.svg``: loss / tps-per-gpu / mfu
    vs step.
  - ``training_diagnostics_<model>_<num_nodes>n.svg``: grad_norm, lr,
    and max_loss vs step.
  - ``tokens_vs_time_<model>_<num_nodes>n.svg``: cumulative
    ``n_tokens_seen`` vs wall-clock datetime.

The dense raw per-step traces (13K+ points) use ``rasterized=True`` so
matplotlib embeds them as a small PNG inside the otherwise-vector SVG.
Axes, labels, ticks, smoothed curves, and the legend all stay vector;
``savefig(..., dpi=200)`` controls the resolution of the rasterized
region. Net file size: ~50-150 KB SVG vs ~300-500 KB PNG.

Run from the repo root:

    python3 torchtitan/experiments/ezpz/utils/plot_production_wandb.py
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Shared ambivalent + Iosevka style helper.
from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

import wandb  # noqa: E402

PROJECT = "aurora_gpt/torchtitan.ezpz.train"


def _savefig_both(fig, svg_path, dpi=200):
    """Save figure as both .svg (vector + rasterized data) and .png.

    GitHub's markdown renderer doesn't reliably display SVGs over ~500KB
    (the production charts hit 700-800KB with their dense rasterized
    per-step traces), so we emit a PNG alongside that the README can
    reference for reliable rendering. SVG stays available for anyone
    who wants higher-fidelity inspection.
    """
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, dpi=dpi, bbox_inches="tight")
    png_path = svg_path.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {svg_path}")
    print(f"Saved: {png_path}")

# Production runs identified by step ranges (cross-checked with PBS logs).
# Listed oldest first so concatenation matches resume order.
# Each key here drives the figure filename + output dir:
#   docs/production/agpt/<model>/n<num_nodes>/figures/<scope>_<key>n.svg
# So the key MUST encode model + version + node count, e.g. "2b_v1_256",
# "20b_v2_512". Don't include the trailing "n" — the template adds it.
#
# v1 = original 2026-04-{14..29} runs (torch 2.10, LBS=1) trained with
#      `--training.dtype=bfloat16`. Sub-ULP master-weight updates froze
#      every RMSNorm.weight at its 1.0 init; loss curves are real but
#      the model has no trainable normalization. Tainted, superseded
#      by v2. Kept here for the historical record. See
#      docs/guides/training-dtype-bf16-norm-freeze.md.
# v2 = fresh restarts on 2026-04-30 from /flare/AuroraGPT/foremans/runs/
#      agpt-{2b,20b}-v2/ (torch 2.13 venv, LBS=2,
#      `--training.dtype=float32`, plain CrossEntropyLoss). These are
#      the current production runs.
# PRODUCTION_RUNS now lives in trajectories.py (the single source of
# truth shared with check_stale_docs.sh, fill_trajectory_fields.py, and
# the eval scripts). It is imported here as a drop-in: same dict shape
# {key: {run_ids, num_nodes, model[, olog_fallbacks]}}, same key order.
# Add new W&B run-ids by appending to a trajectory's wandb_run_ids in
# trajectories.py — that single edit updates the charts, the stale-doc
# map, the field-filler, and the eval plots together.
from torchtitan.experiments.ezpz.utils.trajectories import (  # noqa: E402
    PRODUCTION_RUNS,
)

MODEL_COLORS = {
    "2b": "#1E88E5",
    "20b": "#D32F2F",
    "80b": "#388E3C",
}

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
DOCS_BASE = REPO_ROOT / "torchtitan" / "experiments" / "ezpz" / "docs"

METRIC_KEYS = (
    "_step",
    "_timestamp",
    "grad_norm",
    "lr",
    "loss_metrics/global_avg_loss",
    "loss_metrics/global_max_loss",
    "n_tokens_seen",
    "throughput(tps)",
    "mfu(%)",
)


def fetch_run(api: wandb.Api, run_id: str) -> dict[str, np.ndarray]:
    """Pull the full history for the given metric keys from one wandb run.

    Uses ``scan_history`` so we get every logged step, not a downsample.
    """
    run = api.run(f"{PROJECT}/{run_id}")
    columns: dict[str, list] = {k: [] for k in METRIC_KEYS}
    for row in run.scan_history(keys=list(METRIC_KEYS)):
        for k in METRIC_KEYS:
            columns[k].append(row.get(k))
    return {k: np.array(v) for k, v in columns.items()}


# Per-step metric lines in PBS .o files look like:
#   [TIMESTAMP][I][.../metrics:526:log] step: 3300  loss:  2.61866  grad_norm:  0.1672
#   memory: 44.55GiB(69.63%)  tps: 349  tflops: 51.93  mfu: 17.41%
# ANSI escape codes wrap each field — strip them first. memory shows
# "GiB(PCT%)"; we only need PCT for the dashboards. We don't have
# loss_metrics/global_max_loss, lr, or _timestamp from the .o lines, so
# those columns stay NaN — downstream callers already filter for valid
# _step + loss via `~np.isnan(steps)` so missing diagnostics degrade
# gracefully (the max-loss panel will just lack the recovered range).
_OLOG_STEP_RE = re.compile(
    r"step:\s+(?P<step>\d+)\s+"
    r"loss:\s+(?P<loss>[\d.]+)\s+"
    r"grad_norm:\s+(?P<grad_norm>[\d.]+)\s+"
    r"memory:\s+[\d.]+GiB\((?P<mem_pct>[\d.]+)%\)\s+"
    r"tps:\s+(?P<tps>[\d,]+)\s+"
    r"tflops:\s+(?P<tflops>[\d.]+)\s+"
    r"mfu:\s+(?P<mfu>[\d.]+)%"
)
_OLOG_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def fetch_from_olog(log_path: str) -> dict[str, np.ndarray]:
    """Parse per-step metric lines from a PBS .o log into the same
    shape as ``fetch_run``. Used as a fallback when a W&B run's
    ``history()`` is empty (run crashed mid-sync but the trainer was
    still printing to stdout). Missing METRIC_KEYS (loss_metrics/*, lr,
    _timestamp, n_tokens_seen) come back as NaN arrays.
    """
    p = Path(log_path)
    if not p.exists():
        print(f"    .o-log fallback: {log_path} not found")
        return {k: np.array([]) for k in METRIC_KEYS}
    columns: dict[str, list] = {k: [] for k in METRIC_KEYS}
    with p.open() as f:
        for raw in f:
            line = _OLOG_ANSI_RE.sub("", raw)
            m = _OLOG_STEP_RE.search(line)
            if not m:
                continue
            step = int(m.group("step"))
            columns["_step"].append(step)
            columns["loss_metrics/global_avg_loss"].append(float(m.group("loss")))
            columns["grad_norm"].append(float(m.group("grad_norm")))
            columns["throughput(tps)"].append(float(m.group("tps").replace(",", "")))
            columns["mfu(%)"].append(float(m.group("mfu")))
            # Unavailable from .o lines — leave as NaN
            columns["_timestamp"].append(np.nan)
            columns["lr"].append(np.nan)
            columns["loss_metrics/global_max_loss"].append(np.nan)
            columns["n_tokens_seen"].append(np.nan)
    return {k: np.array(v) for k, v in columns.items()}


def concat_runs(
    api: wandb.Api,
    run_ids: list[str],
    olog_fallbacks: dict[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Fetch and concatenate multiple wandb runs by ascending _step.

    When a run's W&B history() returns 0 rows AND that run_id appears
    in ``olog_fallbacks``, parse the corresponding PBS .o file instead.
    Filled metrics are limited (no lr/max_loss/timestamp/n_tokens_seen
    available from stdout), but the step+loss+tps+mfu trajectory
    survives — which is what the main dashboard plots use.
    """
    olog_fallbacks = olog_fallbacks or {}
    parts = []
    for rid in run_ids:
        data = fetch_run(api, rid)
        if len(data["_step"]) == 0:
            if rid in olog_fallbacks:
                print(f"  {rid}: W&B history empty — falling back to .o log {olog_fallbacks[rid]}")
                data = fetch_from_olog(olog_fallbacks[rid])
                if len(data["_step"]) == 0:
                    print(f"  {rid}: .o-log fallback also empty, skipping")
                    continue
                print(f"  {rid}: .o-log {len(data['_step'])} rows, steps [{int(data['_step'][0])}, {int(data['_step'][-1])}]")
            else:
                print(f"  {rid}: no rows, skipping")
                continue
        else:
            print(f"  {rid}: {len(data['_step'])} rows, steps [{data['_step'][0]}, {data['_step'][-1]}]")
        parts.append(data)

    # For each step, keep the record from the latest run (resume semantics).
    by_step: dict[int, dict] = {}
    for data in parts:
        for i, s in enumerate(data["_step"]):
            if s is None:
                continue
            by_step[int(s)] = {k: data[k][i] for k in METRIC_KEYS}

    sorted_steps = sorted(by_step)
    out = {k: np.array([by_step[s][k] for s in sorted_steps]) for k in METRIC_KEYS}
    return out


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


def plot_dashboard(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """3-panel figure: loss, tps/gpu, mfu vs step.

    Replaces the log-parsing version in ``plot_production.py``: pulls
    from W&B so it reflects in-progress runs whose PBS log files
    haven't been written yet.
    """
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    num_gpus = num_nodes * 12
    steps = data["_step"].astype(float)
    loss = data["loss_metrics/global_avg_loss"].astype(float)
    # `throughput(tps)` is already logged per-rank (per-GPU) by torchtitan,
    # not aggregated — values for the 2B run sit around 1k-3k, matching
    # the per-GPU numbers in the PBS-log dashboards. Don't divide.
    tps_per_gpu = data["throughput(tps)"].astype(float)
    mfu = data["mfu(%)"].astype(float)

    valid = ~np.isnan(steps)
    steps = steps[valid]
    loss = loss[valid]
    tps_per_gpu = tps_per_gpu[valid]
    mfu = mfu[valid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} Production Training  |  "
        f"{num_nodes} nodes ({num_gpus} GPUs)  |  "
        f"step {int(steps[-1]):,}",
        fontsize=14,
        fontweight="bold",
    )

    # rasterized=True on the dense raw lines keeps them as a small embedded
    # PNG inside the SVG (13K+ points would otherwise be 13K SVG path nodes).
    # Smoothed lines + axes + legend stay vector.
    ax = axes[0]
    ax.plot(steps, loss, color=color, alpha=0.25, linewidth=0.5, rasterized=True)
    ax.plot(steps, smooth(loss), color=color, linewidth=1.8, label="Loss (smoothed)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, tps_per_gpu, color="#43A047", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(
        steps,
        smooth(tps_per_gpu),
        color="#43A047",
        linewidth=1.8,
        label="TPS/GPU (smoothed)",
    )
    ax.set_ylabel("Tokens/sec/GPU")
    ax.set_title("Throughput per GPU")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, mfu, color="#FF9800", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(mfu), color="#FF9800", linewidth=1.8, label="MFU (smoothed)")
    ax.set_ylabel("MFU (%)")
    ax.set_xlabel("Training Step")
    ax.set_title("Model FLOPs Utilization")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    # dpi here sets resolution of the rasterized region in the SVG.
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_diagnostics(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """3-panel figure: grad_norm, lr, max_loss vs step."""
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    steps = data["_step"].astype(float)
    grad_norm = data["grad_norm"].astype(float)
    lr = data["lr"].astype(float)
    max_loss = data["loss_metrics/global_max_loss"].astype(float)

    # Filter out NaN-only rows that some logging configs emit
    valid = ~np.isnan(steps)
    steps = steps[valid]
    grad_norm = grad_norm[valid]
    lr = lr[valid]
    max_loss = max_loss[valid]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} Diagnostics  |  "
        f"{num_nodes} nodes ({num_nodes * 12} GPUs)  |  "
        f"step {int(steps[-1]):,}",
        fontsize=14,
        fontweight="bold",
    )

    ax = axes[0]
    ax.plot(steps, grad_norm, color=color, alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(grad_norm), color=color, linewidth=1.8, label="grad_norm (smoothed)")
    ax.set_ylabel("grad_norm")
    ax.set_title("Gradient Norm")
    ax.set_yscale("log")
    ax.legend()

    ax = axes[1]
    ax.plot(steps, lr, color="#FF9800", linewidth=1.8, label="lr")
    ax.set_ylabel("lr")
    ax.set_title("Learning Rate")
    ax.legend()

    ax = axes[2]
    ax.plot(steps, max_loss, color="#7B1FA2", alpha=0.25, linewidth=0.5,
            rasterized=True)
    ax.plot(steps, smooth(max_loss), color="#7B1FA2", linewidth=1.8, label="max_loss (smoothed)")
    ax.set_ylabel("global_max_loss")
    ax.set_xlabel("Training Step")
    ax.set_title("Per-Step Max Loss (across DP ranks)")
    ax.legend()

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_tokens_vs_time(
    data: dict[str, np.ndarray],
    model_name: str,
    num_nodes: int,
    output_path: Path,
) -> Path:
    """Cumulative tokens vs wall-clock datetime.

    Sorts by timestamp (not step) so resumed runs read as a single
    monotonic curve. When restarts re-tokenize early steps after a
    checkpoint, those rows are filtered out via a running max so the
    cumulative curve never goes backwards.
    """
    color = MODEL_COLORS.get(model_name, "#1E88E5")
    ts = data["_timestamp"].astype(float)
    tokens = data["n_tokens_seen"].astype(float)

    valid = ~np.isnan(ts) & ~np.isnan(tokens)
    ts = ts[valid]
    tokens = tokens[valid]

    # Sort by wall-clock time so the curve reflects actual execution order.
    order = np.argsort(ts)
    ts = ts[order]
    tokens = tokens[order]

    # Restarts that wandb logged before the resume's `n_tokens_seen` caught
    # up to the previous best produce dips. Take the running max so the
    # curve stays monotonic non-decreasing.
    tokens = np.maximum.accumulate(tokens)

    times = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, tokens / 1e9, color=color, linewidth=1.8)

    final_tokens_b = tokens[-1] / 1e9
    target_b = 4670  # 4.67T tokens
    pct = 100 * final_tokens_b / target_b

    ax.set_xlabel("Date")
    ax.set_ylabel("Tokens consumed (billions)")
    ax.set_title(
        f"AuroraGPT {model_name.upper()} — Tokens vs Wall Clock  |  "
        f"{num_nodes} nodes  |  "
        f"{final_tokens_b:,.1f}B tokens ({pct:.1f}% of 4.67T target)",
        fontsize=14,
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    fig.tight_layout()
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_overlay(
    series: list[dict],
    model_name: str,
    output_path: Path,
) -> Path:
    """Overlay multiple PRODUCTION_RUNS entries on a 3-panel loss/TPS/MFU
    dashboard.

    Each ``series`` entry is a dict with:
        - ``key``    : PRODUCTION_RUNS key (used in the legend)
        - ``data``   : already-fetched + concatenated W&B history
        - ``color``  : matplotlib color
        - ``alpha``  : line alpha (raw curve uses 0.4× this)

    The point is to make the v1 (bf16-tainted) vs v2 (fp32) contrast
    visually unmissable: the loss curves descend together but their
    *eval-time* behavior diverges because v1 has frozen RMSNorm
    weights. See docs/guides/training-dtype-bf16-norm-freeze.md.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(
        f"AuroraGPT {model_name.upper()} — v1 (bf16-master) vs v2 (fp32-master)",
        fontsize=14,
        fontweight="bold",
    )

    last_steps = []
    for s in series:
        data = s["data"]
        steps = data["_step"].astype(float)
        loss = data["loss_metrics/global_avg_loss"].astype(float)
        tps = data["throughput(tps)"].astype(float)
        mfu = data["mfu(%)"].astype(float)
        valid = ~np.isnan(steps)
        steps = steps[valid]
        loss = loss[valid]
        tps = tps[valid]
        mfu = mfu[valid]
        if len(steps) == 0:
            continue
        last_steps.append(int(steps[-1]))

        color = s["color"]
        alpha = s["alpha"]
        label = s["key"]

        axes[0].plot(steps, loss, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[0].plot(steps, smooth(loss), color=color, alpha=alpha, linewidth=1.8, label=label)

        axes[1].plot(steps, tps, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[1].plot(steps, smooth(tps), color=color, alpha=alpha, linewidth=1.8, label=label)

        axes[2].plot(steps, mfu, color=color, alpha=0.4 * alpha, linewidth=0.5,
                     rasterized=True)
        axes[2].plot(steps, smooth(mfu), color=color, alpha=alpha, linewidth=1.8, label=label)

    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("Tokens/sec/GPU")
    axes[1].set_title("Throughput per GPU")
    axes[1].legend(loc="lower right")
    axes[2].set_ylabel("MFU (%)")
    axes[2].set_xlabel("Training Step")
    axes[2].set_title("Model FLOPs Utilization")
    axes[2].legend(loc="lower right")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _savefig_both(fig, output_path, dpi=200)
    plt.close(fig)
    return output_path


# Color/alpha presets per PRODUCTION_RUNS key. Lower alpha for the
# v1 (bf16-tainted) entries so v2 reads as the "real" curve.
OVERLAY_STYLE: dict[str, dict] = {
    "2b_v1_256":  {"color": "#94a3b8", "alpha": 0.55},  # slate, faded
    "2b_v2_256":  {"color": "#1E88E5", "alpha": 1.00},
    "2b_v2_512":  {"color": "#0d47a1", "alpha": 1.00},
    "20b_v1_256": {"color": "#94a3b8", "alpha": 0.55},
    "20b_v2_512": {"color": "#D32F2F", "alpha": 1.00},
}


def overlay_keys_for_model(model: str) -> list[str]:
    """Return PRODUCTION_RUNS keys whose `model` field matches `model`."""
    return [k for k, v in PRODUCTION_RUNS.items() if v.get("model", k) == model]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=None,
        help=(
            "Run key from PRODUCTION_RUNS to plot (e.g. 2b_v1_256, "
            "2b_v2_256, 2b_v2_512, 20b_v1_256, 20b_v2_512), or omit "
            "to plot all."
        ),
    )
    parser.add_argument(
        "--overlay", type=str, default=None, choices=["2b", "20b"],
        help=(
            "Generate a v1-vs-v2 overlay dashboard for the given model, "
            "instead of (or in addition to) per-run dashboards. The "
            "figure goes to docs/production/agpt/<model>/figures/"
            "overlay_<model>_v1_vs_v2.svg."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: docs/production/agpt/<model>/n<num_nodes>/figures/ for per-trajectory dashboards; docs/production/agpt/<model>/figures/ for overlays)",
    )
    args = parser.parse_args()

    api = wandb.Api()

    if args.overlay is not None:
        keys_to_overlay = overlay_keys_for_model(args.overlay)
        if not keys_to_overlay:
            raise SystemExit(
                f"no PRODUCTION_RUNS entries found for model={args.overlay!r}"
            )
        # v1-vs-v2 overlay charts live in the historical archive.
        out_dir = args.output_dir or (
            DOCS_BASE / "production" / "agpt" / "historical" / "v1-bf16" / "figures"
        )
        series = []
        for key in keys_to_overlay:
            cfg = PRODUCTION_RUNS[key]
            print(f"\n=== Pulling {key} ({len(cfg['run_ids'])} runs) ===")
            data = concat_runs(api, cfg["run_ids"], cfg.get("olog_fallbacks"))
            print(f"  Concatenated: {len(data['_step'])} unique steps")
            if len(data["_step"]) == 0:
                print(f"  no data, skipping {key}")
                continue
            style = OVERLAY_STYLE.get(
                key, {"color": MODEL_COLORS.get(args.overlay, "#666"), "alpha": 1.0}
            )
            series.append({"key": key, "data": data, **style})
        if not series:
            raise SystemExit("no series with data — nothing to overlay")
        plot_overlay(
            series,
            args.overlay,
            out_dir / f"overlay_{args.overlay}_v1_vs_v2.svg",
        )
        return

    keys = [args.model] if args.model else list(PRODUCTION_RUNS)
    for key in keys:
        cfg = PRODUCTION_RUNS[key]
        # `model` (for color + figure title) defaults to the dict key.
        model_name = cfg.get("model", key)
        # Output dir is keyed on (model, node count) — per-trajectory
        # figures land at production/agpt/<model>/n<nodes>/figures/.
        # v1 (bf16-tainted) per-trajectory figures live in the
        # historical archive; the model-level overlay also goes there
        # (handled in the --overlay branch above).
        if "_v1_" in key:
            default_out = (
                DOCS_BASE
                / "production"
                / "agpt"
                / "historical"
                / "v1-bf16"
                / "figures"
            )
        else:
            default_out = (
                DOCS_BASE
                / "production"
                / "agpt"
                / model_name
                / f"n{cfg['num_nodes']}"
                / "figures"
            )
        out_dir = args.output_dir or default_out

        print(f"\n=== Pulling {key} ({len(cfg['run_ids'])} runs) ===")
        data = concat_runs(api, cfg["run_ids"])
        print(f"  Concatenated: {len(data['_step'])} unique steps")
        if len(data["_step"]) == 0:
            print(f"  no data, skipping {key}")
            continue

        plot_dashboard(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"production_{key}n.svg",
        )
        plot_diagnostics(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"training_diagnostics_{key}n.svg",
        )
        plot_tokens_vs_time(
            data,
            model_name,
            cfg["num_nodes"],
            out_dir / f"tokens_vs_time_{key}n.svg",
        )


if __name__ == "__main__":
    main()
