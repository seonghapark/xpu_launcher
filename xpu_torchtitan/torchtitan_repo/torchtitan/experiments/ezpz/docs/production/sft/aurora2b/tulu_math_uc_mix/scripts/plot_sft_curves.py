"""Plot SFT training curves with ambivalent + Iosevka style.

Reads trainer_state.json (cumulative TRL metrics) and writes an SVG
to ../charts/sft-curves.svg.

Reproduce:
  source /home/foremans/venvs/sunspot/foremans-aurora_frameworks-2025.3.1/bin/activate
  python3 plot_sft_curves.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from torchtitan.experiments.ezpz.utils.plot_style import apply_style

apply_style()

ROOT = Path("/lus/flare/projects/datascience/foremans/projects/saforem2/torchtitan")
TRAINER_STATE = ROOT / "outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729/trainer_state.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JOB_SPANS = [
    ("12468404", 1,  10, 140, "#fde0dc"),
    ("12468404", 2,  10, 100, "#fdcec5"),
    ("12468409", 1, 110, 200, "#dde9fb"),
    ("12468409", 2, 210, 300, "#c7dcfa"),
    ("12468437", 1, 310, 400, "#d8f0d8"),
    ("12468437", 2, 410, 500, "#c5e9c5"),
    ("12468437", 3, 510, 600, "#b1e2b1"),
    ("12468437", 4, 610, 729, "#9bdb9b"),
]


def add_job_shading(ax, smin, smax):
    for job_id, attempt, lo, hi, color in JOB_SPANS:
        if hi < smin or lo > smax:
            continue
        ax.axvspan(lo, hi, alpha=0.35, color=color, zorder=0)


def main():
    state = json.load(open(TRAINER_STATE))
    h = state["log_history"]
    steps = np.array([e["step"] for e in h])
    loss = np.array([e["loss"] for e in h])
    grad_norm = np.array([e["grad_norm"] for e in h])
    lr = np.array([e["learning_rate"] for e in h])
    acc = np.array([e["mean_token_accuracy"] for e in h])
    entropy = np.array([e["entropy"] for e in h])
    tokens_b = np.array([e["num_tokens"] for e in h]) / 1e9

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    fig.suptitle(
        f"AuroraGPT-2B x tulu_math_uc_mix SFT trajectory  (step {steps[-1]}, 3 epochs, GBS=6144)",
        fontsize=13, y=0.995,
    )
    smin, smax = int(steps.min()), int(steps.max())
    panels = [
        (axes[0, 0], loss,       "loss",                 "loss"),
        (axes[0, 1], grad_norm,  "grad_norm",            "grad_norm"),
        (axes[0, 2], lr * 1e5,   "learning rate",        "lr (x 1e-5)"),
        (axes[1, 0], acc,        "mean token accuracy",  "mean_token_accuracy"),
        (axes[1, 1], entropy,    "entropy",              "entropy (nats)"),
        (axes[1, 2], tokens_b,   "tokens seen",          "tokens (B)"),
    ]
    for ax, y, title, ylabel in panels:
        add_job_shading(ax, smin, smax)
        ax.plot(steps, y, lw=1.5)
        ax.scatter(steps, y, s=10)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    for ax in axes[1]:
        ax.set_xlabel("global step")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5, label=f"{jid} attempt {a}")
        for jid, a, _, _, color in JOB_SPANS
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    out = OUT_DIR / "sft-curves.svg"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
