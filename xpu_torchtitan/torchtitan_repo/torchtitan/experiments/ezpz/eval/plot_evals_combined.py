#!/usr/bin/env python3
"""Combined-overlay eval chart across all 5 production trajectories.

One figure, 4 subplot panels (HellaSwag acc_norm, ARC-Easy acc, ARC-C acc_norm,
Winogrande acc), with 5 curves per panel — one per production trajectory —
plotted against **tokens consumed** so trajectories with different GBS can
be compared directly:

    - 2B-MDS (Megatron-DeepSpeed reference, ~7.77T tokens)
    - 2B 256N async (current production comparator)
    - 2B 512N sync (canonical 2B chain)
    - 20B 256N (per-token comparator)
    - 20B 512N sync (canonical 20B chain)

Writes to docs/evals/figures/all_production_evals.svg (single artifact
referenced from docs/evals/README.md as the landing-page chart).

Run:
    python3 -m torchtitan.experiments.ezpz.eval.plot_evals_combined
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

# Shared ambivalent + Iosevka styling (identical to the production
# charts). apply_style registers Iosevka when installed and falls back
# to a monospace chain otherwise; ambivalent itself is a hard dep there
# (silent fallback previously gave us months of wrong-style charts).
import matplotlib.pyplot as plt  # noqa: F401,E402

from torchtitan.experiments.ezpz.utils.plot_style import apply_style  # noqa: E402

apply_style()

REPO_ROOT = Path(__file__).resolve().parents[4]
EVALS_DIR = REPO_ROOT / "outputs" / "evals"
OUT_PATH = (
    REPO_ROOT
    / "torchtitan/experiments/ezpz/docs/evals/figures/all_production_evals.svg"
)

# Tokens-per-step for each trajectory (computed from GBS × SEQ_LEN where
# SEQ_LEN=8192 across the board).
# Canonical per-trajectory palette — shared across all production
# charts (eval + training). Keep these consistent with
# plot_production_combined.py and the per-model {2b,20b}/plot_eval_overview.py
# scripts so a given trajectory always renders the same color.
COLOR_2B_MDS      = "C0"       # matplotlib C0 (ambivalent palette first color — reads well on both light + dark bg)
COLOR_2B_TT_256N  = "#ef5350"  # salmon-red
COLOR_2B_TT_512N  = "#b71c1c"  # dark red
COLOR_20B_TT_512N = "#1b8a3a"  # green
COLOR_RANDOM      = "#808080"  # gray

TRAJECTORIES: list[dict] = [
    {
        "label": "2B-MDS (SophiaG, n256)",
        "eval_subdir": "agpt-2b-mds",
        "layout": "mds",
        "tokens_per_step": 7_770e9 / 140_000,
        "color": COLOR_2B_MDS,
        "linestyle": "--",
        "marker": "x",
    },
    {
        "label": "2B 256N async (GBS=6144)",
        "eval_subdir": "agpt-2b-v2-256n",
        "layout": "dcp",
        "tokens_per_step": 6144 * 8192,
        "color": COLOR_2B_TT_256N,
        "linestyle": "-",
        "marker": "o",
    },
    {
        "label": "2B 512N sync (GBS=12288)",
        "eval_subdir": "agpt-2b-v2-512n",
        "layout": "dcp",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_2B_TT_512N,
        "linestyle": "-",
        "marker": "s",
    },
    # 20B 256N (8463659) was a one-off 364-step NODE_FAIL run, 18.3B tokens.
    # Production is consolidated on 20B 512N — dropping the 256N from the
    # combined chart removes a noisy 3-pt cluster that crowded the legend.
    {
        "label": "20B 512N sync (GBS=12288)",
        "eval_subdir": "agpt-20b-v2-512n",
        "layout": "dcp",
        "tokens_per_step": 12288 * 8192,
        "color": COLOR_20B_TT_512N,
        "linestyle": "-",
        "marker": "D",
    },
]

PANELS: list[tuple[str, str, str]] = [
    ("hellaswag", "acc_norm,none", "HellaSwag (acc_norm)"),
    ("arc_easy", "acc,none", "ARC-Easy (acc)"),
    ("arc_challenge", "acc_norm,none", "ARC-Challenge (acc_norm)"),
    ("winogrande", "acc,none", "Winogrande (acc)"),
    ("piqa", "acc_norm,none", "PIQA (acc_norm)"),
    ("openbookqa", "acc_norm,none", "OpenBookQA (acc_norm)"),
    ("boolq", "acc,none", "BoolQ (acc)"),
]

RANDOM_BASELINE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "arc_challenge": 0.25,
    "winogrande": 0.5,
    "piqa": 0.5,         # binary choice
    "openbookqa": 0.25,  # 4-way MCQ
    "boolq": 0.5,        # yes/no
}


def _read_metric(path: Path, task: str, metric: str) -> float | None:
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        return None
    t = d.get(task)
    if not isinstance(t, dict):
        return None
    return t.get(metric)


def load_dcp(subdir: str, task: str, metric: str) -> list[tuple[int, float]]:
    base = EVALS_DIR / subdir
    out: list[tuple[int, float]] = []
    for p in sorted(base.glob("step-*/results/results.json")):
        step = int(p.parent.parent.name.split("-")[1])
        val = _read_metric(p, task, metric)
        if val is not None:
            out.append((step, val))
    return sorted(out)


def load_mds(subdir: str, task: str, metric: str) -> list[tuple[int, float]]:
    """MDS layout: 3 stage dirs containing step-{N}/results/results.json.

    The three stages all symlink to the same physical ckpt dir for the
    SophiaG sweep — same step appears up to 3 times. Average across
    replicates (XPU lm-eval isn't bit-deterministic).
    """
    base = EVALS_DIR / subdir
    by_step: dict[int, list[float]] = {}
    for p in sorted(base.glob("*/step-*/results/results.json")):
        step = int(p.parent.parent.name.split("-")[1])
        val = _read_metric(p, task, metric)
        if val is not None:
            by_step.setdefault(step, []).append(val)
    return sorted((s, sum(v) / len(v)) for s, v in by_step.items())


def main() -> None:
    # Grid: 2 columns, enough rows to fit every panel. With 7 panels
    # that's a 4x2 (one empty cell, hidden below).
    ncols = 2
    nrows = (len(PANELS) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.5 * nrows))
    axes = axes.flatten()

    for ax, (task, metric, title) in zip(axes, PANELS):
        for traj in TRAJECTORIES:
            loader = load_mds if traj["layout"] == "mds" else load_dcp
            pts = loader(traj["eval_subdir"], task, metric)
            if not pts:
                print(f"  [{title}] no data for {traj['label']}")
                continue
            steps, accs = zip(*pts)
            tokens_b = [s * traj["tokens_per_step"] / 1e9 for s in steps]
            ax.plot(
                tokens_b,
                accs,
                marker=traj["marker"],
                color=traj["color"],
                linestyle=traj["linestyle"],
                label=traj["label"],
                markersize=5,
                linewidth=1.8,
                alpha=0.9,
            )
            print(
                f"  [{title}] {traj['label']}: "
                f"{len(pts)} pts, tokens {tokens_b[0]:.1f}B → {tokens_b[-1]:.1f}B, "
                f"acc {accs[0]:.3f} → {accs[-1]:.3f}"
            )

        ax.axhline(
            y=RANDOM_BASELINE[task],
            color=COLOR_RANDOM,
            linestyle=":",
            alpha=0.6,
            linewidth=1,
            label="random",
        )
        ax.set_xlabel("Tokens consumed (B)")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    # Hide any unused cells when len(PANELS) doesn't fill the grid evenly.
    for ax in axes[len(PANELS):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "AuroraGPT v2 — Eval Benchmarks vs Training Tokens (all production trajectories)",
        y=1.06,
        fontsize=14,
    )
    plt.tight_layout()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", transparent=True)
    plt.close()
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
