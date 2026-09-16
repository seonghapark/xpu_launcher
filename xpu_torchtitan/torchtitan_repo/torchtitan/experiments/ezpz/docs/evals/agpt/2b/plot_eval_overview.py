#!/usr/bin/env python3
"""Overlay v2 (fp32-master, current production) 2B eval scores
against the 2B-MDS reference ceiling on a single figure per task.

v1 (bf16-master, archived) results are now at
docs/production/agpt/historical/v1-bf16/ and no longer plotted here.

Usage:
    python3 plot_eval_overview.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: F401

# Shared ambivalent + Iosevka styling (identical to the production
# charts) via the one helper, so eval + production charts match.
from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

REPO_ROOT = Path(__file__).resolve().parents[7]
# Per-trajectory v2 eval roots; gbs differs per node count, so they
# can't be conflated.
V2_TRAJECTORIES = {
    # node_count -> (results dir, GBS)
    256: (REPO_ROOT / "outputs" / "evals" / "agpt-2b-v2-256n", 6_144),   # LBS=2 × 256N × 12 GPUs
    512: (REPO_ROOT / "outputs" / "evals" / "agpt-2b-v2-512n", 12_288),  # LBS=2 × 512N × 12 GPUs
}
# 2B-MDS reference baseline (pre-torchtitan SophiaG, 140K steps / 7.77T tokens).
# Three stage dirs all symlink to the same physical ckpt dir — same step
# can appear up to 3 times; we average across replicates.
MDS_RESULTS_BASE = REPO_ROOT / "outputs" / "evals" / "agpt-2b-mds"
MDS_TOKENS_PER_STEP = 7_770e9 / 140_000  # ~55.5M tokens/step
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

V2_SEQ = 8192

TASKS = ["hellaswag", "arc_easy", "arc_challenge", "winogrande"]
TASK_TITLES = {
    "hellaswag": "HellaSwag",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "winogrande": "Winogrande",
}
RANDOM_BASELINE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.50,
}


def _acc(metrics: dict) -> float | None:
    for k in ("acc_norm,none", "acc,none"):
        if k in metrics:
            return float(metrics[k])
    return None


def load_v2_trajectory(results_base: Path) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    if not results_base.exists():
        return out
    for step_dir in sorted(results_base.glob("step-*")):
        results_file = step_dir / "results" / "results.json"
        if not results_file.exists():
            continue
        try:
            step = int(step_dir.name.split("-")[1])
        except ValueError:
            continue
        with results_file.open() as f:
            payload = json.load(f)
        scores: dict[str, float] = {}
        results = payload.get("results", payload)
        for task in TASKS:
            if task in results and (acc := _acc(results[task])) is not None:
                scores[task] = acc
        if scores:
            out[step] = scores
    return out


def load_mds() -> dict[int, dict[str, float]]:
    """MDS layout: <stage>/step-<N>/results/results.json with 3 stages all
    symlinked to the same physical ckpt dir for SophiaG. Average across
    replicates (XPU lm-eval is not bitwise-deterministic)."""
    if not MDS_RESULTS_BASE.exists():
        return {}
    by_step: dict[int, dict[str, list[float]]] = {}
    for p in sorted(MDS_RESULTS_BASE.glob("*/step-*/results/results.json")):
        try:
            step = int(p.parent.parent.name.split("-")[1])
        except ValueError:
            continue
        with p.open() as f:
            payload = json.load(f)
        results = payload.get("results", payload)
        for task in TASKS:
            if task in results and (acc := _acc(results[task])) is not None:
                by_step.setdefault(step, {}).setdefault(task, []).append(acc)
    return {
        step: {task: sum(vs) / len(vs) for task, vs in d.items()}
        for step, d in sorted(by_step.items())
    }


def _tokens_for(step: int, gbs: int, seq: int) -> float:
    return step * gbs * seq / 1e9


# Per-trajectory plot styles (one entry per V2_TRAJECTORIES key).
V2_STYLE = {
    # Canonical palette shared across all production charts.
    256: {"color": "#ef5350", "marker": "s", "label_prefix": "v2 256N"},
    512: {"color": "#b71c1c", "marker": "D", "label_prefix": "v2 512N"},
}


def plot_per_task(
    v2_by_nodes: dict[int, dict[int, dict[str, float]]],
    v2_gbs: dict[int, int],
    mds: dict[int, dict[str, float]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "AuroraGPT 2B  —  v2 (fp32-master, current production) vs MDS (SophiaG reference)",
        fontsize=12,
        fontweight="bold",
    )
    for ax, task in zip(axes.flat, TASKS):
        ax.axhline(
            RANDOM_BASELINE[task], color="#808080", lw=1, ls=":", label="random",
        )

        # MDS reference (pre-torchtitan SophiaG, 7.77T tokens).
        mds_steps = sorted(s for s in mds if task in mds[s])
        if mds_steps:
            mds_tokens = [s * MDS_TOKENS_PER_STEP / 1e9 for s in mds_steps]
            mds_y = [mds[s][task] for s in mds_steps]
            ax.plot(
                mds_tokens, mds_y, marker="x", ms=5, lw=1.4,
                color="C0", alpha=0.85, linestyle="--",
                label=f"MDS SophiaG ref (n={len(mds_steps)})",
            )

        all_y_max = 0.55
        if mds_steps:
            all_y_max = max(all_y_max, max(mds[s][task] for s in mds_steps))
        for nodes, v2_traj in v2_by_nodes.items():
            if not v2_traj:
                continue
            style = V2_STYLE.get(nodes, {"color": "#666", "marker": "x", "label_prefix": f"v2 {nodes}N"})
            gbs = v2_gbs[nodes]
            v2_steps = sorted(v2_traj)
            v2_tokens = [_tokens_for(s, gbs, V2_SEQ) for s in v2_steps]
            v2_y = [v2_traj[s].get(task) for s in v2_steps]
            v2_pts = [(t, y) for t, y in zip(v2_tokens, v2_y) if y is not None]
            if not v2_pts:
                continue
            xs, ys = zip(*v2_pts)
            ax.plot(
                xs, ys,
                marker=style["marker"], ms=6, lw=1.8,
                color=style["color"], alpha=1.0,
                label=f"{style['label_prefix']} (n={len(xs)})",
            )
            all_y_max = max(all_y_max, max(ys))

        ax.set_xlabel("Tokens consumed (B)")
        ax.set_ylabel("Accuracy")
        ax.set_title(TASK_TITLES[task])
        ax.set_ylim(0.20, max(0.55, all_y_max * 1.10))
        ax.grid(alpha=0.25)

    # Single shared legend above the grid — per-axis legends were landing
    # directly over the climbing v2 trajectories.
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(len(labels), 4),
        fontsize=9,
        frameon=False,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=150, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {out_path}")


def print_table(
    v2_by_nodes: dict[int, dict[int, dict[str, float]]],
    v2_gbs: dict[int, int],
) -> None:
    print("\n## v2 — by training step\n")
    print("| Run | Step | Tokens (B) | HellaSwag | ARC-Easy | ARC-Chall | Winogrande |")
    print("|-----|-----:|-----------:|----------:|---------:|----------:|-----------:|")
    for nodes in sorted(v2_by_nodes):
        v2 = v2_by_nodes[nodes]
        gbs = v2_gbs[nodes]
        for step in sorted(v2):
            s = v2[step]
            tok = _tokens_for(step, gbs, V2_SEQ)
            cells = [f"{s[t]:.4f}" if t in s else "—" for t in TASKS]
            # Use {tok:.1f} (no width) so we don't get leading-space-in-bold
            # rendering like `** 10.1**`.
            print(
                f"| **v2 {nodes}N** | **{step:,}** | **{tok:.1f}** | "
                + " | ".join(f"**{c}**" for c in cells)
                + " |"
            )


def main() -> None:
    v2_by_nodes = {}
    v2_gbs = {}
    for nodes, (path, gbs) in V2_TRAJECTORIES.items():
        traj = load_v2_trajectory(path)
        v2_by_nodes[nodes] = traj
        v2_gbs[nodes] = gbs
        print(f"loaded v2 {nodes}N: {len(traj)} steps from {path}")
    mds = load_mds()
    print(f"loaded MDS reference: {len(mds)} steps from {MDS_RESULTS_BASE}")
    plot_per_task(v2_by_nodes, v2_gbs, mds, FIG_DIR / "eval_overview.svg")
    print_table(v2_by_nodes, v2_gbs)


if __name__ == "__main__":
    main()
