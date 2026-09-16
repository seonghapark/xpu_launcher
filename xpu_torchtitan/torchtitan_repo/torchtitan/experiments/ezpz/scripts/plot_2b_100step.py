"""Plot the 2B 100-step all-optimizer LR finder (job 12469840).

Standalone so it can run on Sunspot against the real CSVs. Reuses the
canonical optimizer colors + smoothing convention from plot_lr_trend.py.
"""
import csv, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# repo root = 4 levels up: scripts/ -> ezpz/ -> experiments/ -> torchtitan/ -> <root>
REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BASE = f"{REPO}/outputs/lrfind-2b-100step/lr_finder/ezpz/ezpz.agpt/2b"
OUT = f"{REPO}/torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt/2b/figures/lr_finder_2b_100step_all_optimizers.png"
OPT_COLOR = {"adamw": "#ff7f0e", "mano": "#1f77b4", "sophiag": "#2ca02c"}
OPT_LABEL = {"adamw": "AdamW", "mano": "mano", "sophiag": "sophiag"}


def house_style():
    hits = glob.glob(f"{REPO}/.venv/**/ambivalent/stylefiles/ambivalent.mplstyle",
                     recursive=True)
    iose = os.path.expanduser("~/.local/share/fonts/Iosevka")
    if os.path.isdir(iose):
        for f in os.listdir(iose):
            if f.lower().endswith((".ttf", ".ttc", ".otf")):
                fm.fontManager.addfont(os.path.join(iose, f))
    if hits:
        plt.style.use(hits[0])
    plt.rcParams["font.family"] = ["Iosevka", "DejaVu Sans Mono", "monospace"]


def smooth(ys, frac=0.10):
    n = len(ys)
    if n < 4:
        return ys[:]
    w = max(2, int(n * frac))
    return [sum(ys[max(0, i - w // 2):min(n, i + w // 2 + 1)]) /
            (min(n, i + w // 2 + 1) - max(0, i - w // 2)) for i in range(n)]


def load(opt):
    p = f"{BASE}/{opt}/lr_finder_data.csv"
    rows = list(csv.DictReader(open(p)))
    lr, ls = [], []
    for r in rows:
        try:
            v = float(r["loss"])
            if v == v:
                lr.append(float(r["learning_rate"])); ls.append(v)
        except (ValueError, TypeError):
            pass
    return lr, ls


house_style()
fig, ax = plt.subplots(figsize=(9, 6))
for opt in ("adamw", "mano", "sophiag"):
    lr, ls_raw = load(opt)
    ls = smooth(ls_raw)
    mi = min(range(len(ls)), key=lambda k: ls[k])
    ax.plot(lr, ls, "-", lw=1.8, color=OPT_COLOR[opt],
            label=f"{OPT_LABEL[opt]} -- min {ls[mi]:.2f} @ {lr[mi]:.1e} (0 NaN)")
    ax.scatter([lr[mi]], [ls[mi]], s=150, facecolors="none",
               edgecolors=OPT_COLOR[opt], linewidths=1.8, zorder=4)
ax.set_xscale("log")
# Cap y at 13: minima cluster ~7.8-8.5; sophiag's blow-up runs to ~32 and
# otherwise crushes the basin into the bottom strip. 13 zooms into the minima
# while still showing the descent + start of each blow-up.
ax.set_ylim(top=13)
ax.set_xlabel("Learning rate")
ax.set_ylabel("LR-finder smoothed loss")
ax.set_title("agpt 2B LR finder -- 100 steps, small batch (dp=192)\n"
             "deeper minima than the 15-step sweep (cumulative training); "
             "all 0 NaN (y capped at 13)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
