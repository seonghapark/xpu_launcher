"""Plot the 2B 100-step *production-batch* LR finder ladder (jobs 12469854-868).

Companion to plot_2b_100step.py (which is the small-batch single-GBS run). This
one sweeps the production GBS ladder 1536/3072/6144/12288/24576 x
{adamw,mano,sophiag} at 100 steps, dp=192. Standalone so it runs on Sunspot
against the real CSVs. Reuses canonical optimizer colors + smoothing.

Encoding: hue = optimizer (canonical color family), and within a family the
batch size is shown by shade (light -> dark), line width (thin -> thick), and
alpha (faint -> opaque) all increasing with GBS. Only batches whose CSV exists
are drawn, so this is safe to run before the full ladder finishes.
"""
import csv, os, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# repo root = 4 levels up: scripts/ -> ezpz/ -> experiments/ -> torchtitan/ -> <root>
REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
BASE = f"{REPO}/outputs/lrfind-2b-100step-prod"
FIGDIR = (f"{REPO}/torchtitan/experiments/ezpz/docs/experiments/"
          f"lr-finder/agpt/2b/figures")

GBS_LADDER = [1536, 3072, 6144, 12288, 24576]
OPTS = ["adamw", "mano", "sophiag"]
OPT_COLOR = {"adamw": "#ff7f0e", "mano": "#1f77b4", "sophiag": "#2ca02c"}
OPT_CMAP = {"adamw": "Oranges", "mano": "Blues", "sophiag": "Greens"}
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


def load(gbs, opt):
    p = f"{BASE}/gbs{gbs}/lr_finder/ezpz/ezpz.agpt/2b/{opt}/lr_finder_data.csv"
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    lr, ls = [], []
    nan = 0
    for r in rows:
        try:
            v = float(r["loss"])
        except (ValueError, TypeError, KeyError):
            continue
        if v == v:  # not NaN
            lr.append(float(r["learning_rate"])); ls.append(v)
        else:
            nan += 1
    if not lr:
        return None
    return lr, ls, nan


def available():
    """Return the sorted batch sizes that have at least one optimizer CSV."""
    out = []
    for gbs in GBS_LADDER:
        if any(load(gbs, o) for o in OPTS):
            out.append(gbs)
    return out


def shade_width_alpha(cmap_name, idx, total):
    """Light/thin/faint for small batch -> dark/thick/opaque for large."""
    if total <= 1:
        frac = 0.75
    else:
        frac = 0.40 + 0.55 * (idx / (total - 1))
    color = plt.get_cmap(cmap_name)(frac)
    lw = 1.0 + 2.0 * (idx / max(1, total - 1))
    alpha = 0.45 + 0.55 * (idx / max(1, total - 1))
    return color, lw, alpha


def plot_all_opts_all_gbs(gbs_list):
    house_style()
    fig, ax = plt.subplots(figsize=(10, 6.5))
    n = len(gbs_list)
    for opt in OPTS:
        for i, gbs in enumerate(gbs_list):
            d = load(gbs, opt)
            if not d:
                continue
            lr, ls_raw, nan = d
            ls = smooth(ls_raw)
            color, lw, alpha = shade_width_alpha(OPT_CMAP[opt], i, n)
            mi = min(range(len(ls)), key=lambda k: ls[k])
            ax.plot(lr, ls, "-", lw=lw, color=color, alpha=alpha,
                    label=f"{OPT_LABEL[opt]} gbs{gbs} -- "
                          f"min {ls[mi]:.2f} @ {lr[mi]:.1e} ({nan} NaN)")
            ax.scatter([lr[mi]], [ls[mi]], s=70, facecolors="none",
                       edgecolors=color, linewidths=1.4, alpha=alpha, zorder=4)
    ax.set_xscale("log")
    # Minima cluster ~7.5-8.5; cap zooms into the basin (blow-ups run higher).
    ax.set_ylim(top=13)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    miss = [g for g in GBS_LADDER if g not in gbs_list]
    sub = "complete ladder" if not miss else \
        "PRELIMINARY -- pending " + ",".join(str(m) for m in miss)
    ax.set_title("agpt 2B LR finder -- 100 steps, production GBS ladder (dp=192)\n"
                 "hue=optimizer, shade/width/opacity=batch size; "
                 f"all 0 NaN (y capped at 13)\n{sub}")
    ax.legend(fontsize=7.5, ncol=3, loc="upper left")
    fig.tight_layout()
    out = f"{FIGDIR}/lr_finder_2b_100step_prod_all_opts_all_gbs.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def plot_minlr_vs_gbs(gbs_list):
    house_style()
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for opt in OPTS:
        xs, ys = [], []
        for gbs in gbs_list:
            d = load(gbs, opt)
            if not d:
                continue
            lr, ls_raw, _ = d
            ls = smooth(ls_raw)
            mi = min(range(len(ls)), key=lambda k: ls[k])
            xs.append(gbs); ys.append(lr[mi])
        if xs:
            ax.plot(xs, ys, "o-", color=OPT_COLOR[opt], lw=1.8, ms=8,
                    label=OPT_LABEL[opt])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(GBS_LADDER)
    ax.set_xticklabels([str(g) for g in GBS_LADDER])
    ax.set_xlabel("Global batch size (tokens/step ladder)")
    ax.set_ylabel("Optimal (min-loss) learning rate")
    miss = [g for g in GBS_LADDER if g not in gbs_list]
    sub = "complete ladder" if not miss else \
        "PRELIMINARY -- pending " + ",".join(str(m) for m in miss)
    ax.set_title("agpt 2B optimal LR vs batch size -- 100-step sweeps (dp=192)\n"
                 f"{sub}")
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = f"{FIGDIR}/lr_finder_2b_100step_prod_minlr_vs_gbs.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)
    plt.close(fig)


def main():
    gbs_list = available()
    if not gbs_list:
        print("no CSVs found under", BASE)
        return
    print("batches available:", gbs_list)
    plot_all_opts_all_gbs(gbs_list)
    plot_minlr_vs_gbs(gbs_list)


if __name__ == "__main__":
    main()
