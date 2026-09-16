#!/usr/bin/env python3
"""Plot yeet-env tarball broadcast scaling.

Reads .yeet-env-scaling-results.csv produced by
torchtitan/experiments/ezpz/scripts/yeet_env_scaling_test.sh and writes:

    yeet_env_seconds.png      yeet-env Done-in-X seconds vs nodes
    yeet_env_per_node.png     per-node mean wall-clock (seconds / N)

Usage:
    python3 plot_yeet_env_scaling.py [--csv path/to/results.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ambivalent is required — silent fallback hides style regressions.
# Install with: uv pip install --no-deps "git+https://github.com/saforem2/ambivalent"
import ambivalent  # noqa: F401

plt.style.use(ambivalent.STYLES["ambivalent"])

plt.rcParams["font.family"] = "monospace"

DEFAULT_CSV = (
    Path(__file__).resolve().parents[6]  # ezpz/docs/scaling/yeet_env/ -> repo root
    / ".yeet-env-scaling-results.csv"
)
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["nodes"] = int(row["nodes"])
                row["yeet_seconds"] = float(row["yeet_seconds"])
            except (ValueError, KeyError):
                continue
            rows.append(row)
    rows.sort(key=lambda r: r["nodes"])
    return rows


def _decimal_log_axes(ax, x_ticks, y_ticks) -> None:
    """Use log-log scaling but show actual decimal numbers on the ticks
    instead of 2^N exponents."""
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_locator(mticker.FixedLocator(x_ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.yaxis.set_major_locator(mticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def plot_total(rows: list[dict], out_path: Path) -> None:
    nodes = [r["nodes"] for r in rows]
    secs = [r["yeet_seconds"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nodes, secs, marker="o", lw=1.6, color="#0ea5e9")
    for n, s in zip(nodes, secs):
        ax.annotate(
            f"{s:.0f}s",
            xy=(n, s), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9, color="#0c4a6e",
        )
    # Choose y ticks at meaningful round seconds covering the data range.
    y_min, y_max = min(secs), max(secs)
    y_candidates = [50, 70, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000]
    y_ticks = [t for t in y_candidates if y_min * 0.85 <= t <= y_max * 1.15]
    _decimal_log_axes(ax, x_ticks=nodes, y_ticks=y_ticks)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("yeet-env wall-clock (seconds)")
    ax.set_title(
        "Aurora yeet-env tarball broadcast — total wall-clock vs node count"
    )
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path} ({len(rows)} points)")


def plot_per_node(rows: list[dict], out_path: Path) -> None:
    nodes = [r["nodes"] for r in rows]
    per_node_ms = [1000 * r["yeet_seconds"] / r["nodes"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nodes, per_node_ms, marker="o", lw=1.6, color="#7c3aed")
    for n, m in zip(nodes, per_node_ms):
        ax.annotate(
            f"{m:.0f}ms",
            xy=(n, m), xytext=(0, 8), textcoords="offset points",
            ha="center", fontsize=9, color="#3b0764",
        )
    y_min, y_max = min(per_node_ms), max(per_node_ms)
    y_candidates = [100, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000]
    y_ticks = [t for t in y_candidates if y_min * 0.85 <= t <= y_max * 1.15]
    _decimal_log_axes(ax, x_ticks=nodes, y_ticks=y_ticks)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Per-node wall-clock (ms)")
    ax.set_title(
        "Aurora yeet-env tarball broadcast — per-node amortized cost"
    )
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"no results file at {args.csv}")
        return

    rows = load_csv(args.csv)
    print(f"loaded {len(rows)} scaling points from {args.csv}")
    if not rows:
        return

    plot_total(rows, FIG_DIR / "yeet_env_seconds.png")
    plot_per_node(rows, FIG_DIR / "yeet_env_per_node.png")

    # Print a markdown table for the README.
    print("\n| Nodes | yeet-env (s) | Per-node (ms) |")
    print("|------:|-------------:|--------------:|")
    for r in rows:
        n = r["nodes"]
        s = r["yeet_seconds"]
        print(f"| {n} | {s:.1f} | {1000*s/n:.0f} |")


if __name__ == "__main__":
    main()
