"""Plot GRPO training curves with ambivalent + Iosevka style.

Reads trainer_state.json from the GRPO run and writes an SVG (+ PNG)
to ../charts/. Same recipe as plot_sft_curves.py.

Reproduce:
  source /home/foremans/venvs/sunspot/foremans-aurora_frameworks-2025.3.1/bin/activate
  python3 plot_grpo_curves.py
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
TRAINER_STATE = ROOT / "outputs/grpo/aurora2b-sft-arithmetic-8n/checkpoint-1000/trainer_state.json"
OUT_DIR = Path(__file__).resolve().parent.parent / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ewma(x, alpha=0.05):
    """Exponentially weighted moving average for noisy RL traces."""
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def main():
    state = json.load(open(TRAINER_STATE))
    h = state["log_history"]
    steps = np.array([e["step"] for e in h])
    reward = np.array([e["reward"] for e in h])
    reward_std = np.array([e["reward_std"] for e in h])
    acc_reward = np.array([e["rewards/accuracy_reward/mean"] for e in h])
    fmt_reward = np.array([e["rewards/format_reward/mean"] for e in h])
    length_pen = np.array([e["rewards/length_penalty/mean"] for e in h])
    entropy = np.array([e["entropy"] for e in h])
    grad_norm = np.array([e["grad_norm"] for e in h])
    lr = np.array([e["learning_rate"] for e in h])
    mean_len = np.array([e["completions/mean_length"] for e in h])
    clipped = np.array([e["completions/clipped_ratio"] for e in h])

    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    fig.suptitle(
        f"GRPO aurora2b-sft-arithmetic-8n trajectory  (step {steps[-1]}, sum_digits, lr=1e-6, bf16)",
        fontsize=13, y=0.995,
    )
    panels = [
        (axes[0, 0], reward,      "reward (total)",          "reward"),
        (axes[0, 1], acc_reward,  "rewards/accuracy_reward",  "accuracy_reward"),
        (axes[0, 2], fmt_reward,  "rewards/format_reward",    "format_reward"),
        (axes[1, 0], length_pen,  "rewards/length_penalty",   "length_penalty"),
        (axes[1, 1], entropy,     "entropy",                  "entropy (nats)"),
        (axes[1, 2], mean_len,    "completions/mean_length",  "tokens"),
        (axes[2, 0], grad_norm,   "grad_norm",                "grad_norm"),
        (axes[2, 1], lr * 1e6,    "learning rate",            "lr (x 1e-6)"),
        (axes[2, 2], clipped,     "completions/clipped_ratio","clipped fraction"),
    ]
    for ax, y, title, ylabel in panels:
        ax.plot(steps, y, lw=0.6, alpha=0.4)
        if len(y) > 10:
            ax.plot(steps, ewma(y, alpha=0.03), lw=1.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
    for ax in axes[2]:
        ax.set_xlabel("global step")
    fig.tight_layout(rect=(0, 0.0, 1, 0.97))
    out_svg = OUT_DIR / "grpo-curves.svg"
    out_png = OUT_DIR / "grpo-curves.png"
    fig.savefig(out_svg, bbox_inches="tight")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
