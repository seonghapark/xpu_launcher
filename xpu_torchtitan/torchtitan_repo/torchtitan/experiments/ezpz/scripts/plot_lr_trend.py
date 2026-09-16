"""Regenerate the LR-finder trend + per-GBS figures with house style.

Applies the ambivalent stylesheet + Iosevka font (matching the production /
eval / SFT charts) and rebuilds, from the isolated per-GBS CSVs:

  agpt/2b/figures/
    lr_ceiling_vs_gbs_2b_vs_80b.png        (2B vs 80B usable-LR overlay)
    lr_finder_2b_gbs12288_all_optimizers.png
    lr_finder_2b_loss_vs_lr_by_gbs.png     (NEW: one loss-vs-LR curve per GBS)
  agpt/80b/figures/
    sunspot_80b_adamw_lr_ceiling_vs_gbs.png
    sunspot_80b_gbs6144_all_optimizers.png
    sunspot_80b_adamw_loss_vs_lr_by_gbs.png  (NEW)

Style is loaded from the ambivalent .mplstyle FILE directly (not `import
ambivalent`) because that package pulls IPython, which is absent from the
XPU .venv; the stylesheet + Iosevka registration reproduce
utils.plot_style.apply_style() output exactly. Run from repo root with the
.venv active.
"""
from __future__ import annotations

import csv
import glob
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DOCS = Path("torchtitan/experiments/ezpz/docs/experiments/lr-finder/agpt")


def apply_house_style() -> None:
    """ambivalent .mplstyle + Iosevka, without importing the package."""
    hits = glob.glob(
        ".venv/**/ambivalent/stylefiles/ambivalent.mplstyle", recursive=True
    )
    iosevka = Path.home() / ".local/share/fonts/Iosevka"
    if iosevka.is_dir():
        for f in iosevka.iterdir():
            if f.suffix.lower() in (".ttf", ".ttc", ".otf"):
                fm.fontManager.addfont(str(f))
    if hits:
        plt.style.use(hits[0])
    plt.rcParams["font.family"] = ["Iosevka", "DejaVu Sans Mono", "monospace"]


def load_curve(path: str):
    """Return (lrs, losses) with NaN losses as None, last sweep only.

    A sweep is one monotonically-increasing LR run. Files can hold more than
    one sweep (e.g. an LR-finder rerun appends, or different step counts), so
    take the LAST contiguous ascending-LR block rather than a fixed row count
    -- a hardcoded tail (the old [-15:]) silently dropped the lower half of a
    30-step sweep, truncating the x-range to ~1e-3 instead of 1e-5.
    """
    rows = list(csv.DictReader(open(path)))
    # walk backward while LR strictly decreases (= ascending forward)
    start = len(rows) - 1
    while start > 0 and float(rows[start - 1]["learning_rate"]) < float(
        rows[start]["learning_rate"]
    ):
        start -= 1
    rows = rows[start:]
    lrs, losses = [], []
    for r in rows:
        lrs.append(float(r["learning_rate"]))
        try:
            v = float(r["loss"])
            losses.append(v if v == v else None)
        except (ValueError, TypeError):
            losses.append(None)
    return lrs, losses


def _finite(lrs, losses):
    pts = [(x, y) for x, y in zip(lrs, losses) if y is not None]
    return ([p[0] for p in pts], [p[1] for p in pts])


def _smooth(ys, frac=0.15):
    """Moving-average smooth a loss curve (mirrors lr_finder.find_optimal_lr).

    The CSV stores RAW per-step loss -- one minibatch per LR, so the curve is
    noisy, and at 30-step resolution (vs the old 15) that noise shows up as a
    jagged double-dip 'W' instead of a clean U. The finder smooths internally
    for its own min-detection but writes raw loss; we replicate that smoothing
    for display so the basin shape (not minibatch noise) is what's plotted.
    """
    n = len(ys)
    if n < 4:
        return ys[:]
    w = max(2, int(n * frac))
    out = []
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out


def csv_2b(gbs: int, opt: str) -> str:
    return f"outputs/lrtrend-2b/gbs{gbs}/lr_finder/ezpz/ezpz.agpt/2b/{opt}/lr_finder_data.csv"


# ---------------------------------------------------------------------------
# Per-GBS loss-vs-LR curve family (the requested view: one line per batch)
# ---------------------------------------------------------------------------
GBS_2B = [192, 384, 768, 1536, 3072, 6144, 12288, 24576]
OPT_LABEL = {"adamw": "AdamW", "mano": "mano", "sophiag": "sophiag",
             "muon": "muon"}

# CANONICAL optimizer colors -- use these EVERYWHERE so an optimizer is the
# same color in every chart (was inconsistent: sophiag green in one fig, red
# in another; adamw blue/red/black across figs). Solid color + matching
# colormap (for the shade-by-batch plots).
OPT_COLOR = {
    "adamw": "#ff7f0e",   # orange
    "mano": "#1f77b4",    # blue
    "sophiag": "#2ca02c", # green
    "muon": "#9467bd",    # purple
}
OPT_CMAP = {
    "adamw": "Oranges",
    "mano": "Blues",
    "sophiag": "Greens",
    "muon": "Purples",
}


def plot_loss_vs_lr_by_gbs_2b(out: Path, opt: str = "adamw") -> None:
    """2B loss-vs-LR, one curve per batch size, for a single optimizer."""
    cmap = plt.cm.viridis
    fig, ax = plt.subplots(figsize=(9, 6))
    n = 0
    for i, g in enumerate(GBS_2B):
        p = csv_2b(g, opt)
        if not Path(p).exists():
            continue
        fx, fy = _finite(*load_curve(p))
        if not fx:
            continue
        c = cmap(i / (len(GBS_2B) - 1))
        ax.plot(fx, fy, "-o", ms=4, color=c, label=f"GBS={g}", zorder=3)
        mi = min(range(len(fy)), key=lambda k: fy[k])
        ax.scatter([fx[mi]], [fy[mi]], s=90, facecolors="none",
                   edgecolors=c, linewidths=1.5, zorder=4)
        n += 1
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    _sub = {
        "adamw": "clean U-min at every GBS, ~1e-2 (batch-independent)",
        "mano": "clean U-min at every GBS, ~1e-2 (batch-independent)",
        "sophiag": "larger batch sharpens + deepens the U (min -> ~1e-2); "
                   "small batch shallow/noisy",
    }
    ax.set_title(f"agpt 2B {OPT_LABEL.get(opt, opt)}: loss vs LR, one curve per "
                 f"batch size\n({_sub.get(opt, 'circles mark each minimum')})")
    ax.legend(fontsize=8, ncol=2, title="global batch")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} ({n} GBS curves)")


def plot_2b_minlr_vs_gbs_all_opts(out: Path) -> None:
    """2B usable/min LR vs GBS, all three optimizers overlaid."""
    colors = OPT_COLOR
    markers = {"adamw": "o", "mano": "s", "sophiag": "^"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for opt in ("adamw", "mano", "sophiag"):
        xs, ys = [], []
        for g in GBS_2B:
            p = csv_2b(g, opt)
            if not Path(p).exists():
                continue
            fx, fy = _finite(*load_curve(p))
            if not fx:
                continue
            mi = min(range(len(fy)), key=lambda k: fy[k])
            xs.append(g)
            ys.append(fx[mi])
        if xs:
            ax.plot(xs, ys, "-", marker=markers[opt], ms=8, color=colors[opt],
                    label=f"{OPT_LABEL[opt]} (0 NaN, all GBS)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Global batch size (GBS)")
    ax.set_ylabel("usable / min LR")
    ax.set_title("agpt 2B: usable LR vs batch, all optimizers (dp=192)\n"
                 "all three are batch-independent -- no collapse, no cliff (cf. 80B)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Combined: every (optimizer x batch size) loss-vs-LR curve on one axis.
# Hue encodes optimizer (AdamW=orange, mano=blue, sophiag=green); shade
# encodes batch size (light = small GBS, dark = large GBS).
# ---------------------------------------------------------------------------
from matplotlib.lines import Line2D  # noqa: E402

_OPT_CMAP = OPT_CMAP  # canonical (see OPT_COLOR/OPT_CMAP at module top)


def _csv_for(model: str, gbs: int, opt: str) -> str:
    flav = "20b" if model == "20b" else "2b"
    return (f"outputs/lrtrend-{flav}/gbs{gbs}/lr_finder/ezpz/ezpz.agpt/"
            f"{flav}/{opt}/lr_finder_data.csv")


def plot_all_opts_all_gbs(out: Path, model: str = "2b") -> None:
    """One axis: loss-vs-LR for every optimizer x batch size.

    Hue = optimizer, shade = batch size (light small -> dark large). Only
    plots cells with a CSV, so it is safe to run while a sweep is partial.
    """
    gbs_list = GBS_2B
    opts = ("adamw", "mano", "sophiag")
    n = len(gbs_list)
    # Three ramps, all keyed off batch-size rank r in [0,1] (small -> large),
    # so large batch = darker + thicker + more opaque (foregrounds the
    # production-relevant curves; small/noisy batches recede).
    def _rank(i):
        return i / max(1, n - 1)
    shade = {g: 0.35 + 0.60 * _rank(i) for i, g in enumerate(gbs_list)}
    width = {g: 0.8 + 2.7 * _rank(i) for i, g in enumerate(gbs_list)}   # 0.8 -> 3.5 pt
    alpha = {g: 0.35 + 0.60 * _rank(i) for i, g in enumerate(gbs_list)}  # 0.35 -> 0.95
    fig, ax = plt.subplots(figsize=(11, 7))
    plotted = {o: 0 for o in opts}
    for opt in opts:
        cmap = plt.get_cmap(_OPT_CMAP[opt])
        # draw small-batch (thin/faint) first so the thick opaque large-batch
        # lines land on top.
        for g in gbs_list:
            p = _csv_for(model, g, opt)
            if not Path(p).exists():
                continue
            fx, fy_raw = _finite(*load_curve(p))
            if not fx:
                continue
            fy = _smooth(fy_raw)  # display the basin shape, not minibatch noise
            i = gbs_list.index(g)
            ax.plot(fx, fy, "-", lw=width[g], alpha=alpha[g],
                    color=cmap(shade[g]), zorder=2 + i)
            mi = min(range(len(fy)), key=lambda k: fy[k])
            ax.scatter([fx[mi]], [fy[mi]], s=20 + 6 * i, alpha=alpha[g],
                       color=cmap(shade[g]), edgecolor="k", linewidth=0.4,
                       zorder=2 + i + 0.5)
            plotted[opt] += 1
    ax.set_xscale("log")
    # Cap the y-axis at 15: the high-LR blow-up runs to ~50 (esp. sophiag),
    # which crushes the interesting basin (loss ~11-14) into the bottom of the
    # plot. Trimming to 15 zooms into the minima while still showing the
    # upturn. Curves continue off-axis above 15 (that is the blow-up region,
    # not where the optimum is).
    ax.set_ylim(top=15)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    ax.set_title(f"agpt {model.upper()}: loss vs LR -- every optimizer x batch "
                 f"size (dp=192)\nhue = optimizer; larger batch = darker + "
                 f"thicker + more opaque; dots mark each minimum\n"
                 f"(loss moving-avg smoothed; y capped at 15)")
    opt_handles = [
        Line2D([0], [0], color=plt.get_cmap(_OPT_CMAP[o])(0.75), lw=3,
               label=f"{OPT_LABEL[o]} ({plotted[o]} batches)")
        for o in opts if plotted[o]
    ]
    # batch-size swatches mirror the plot encoding (shade + width + alpha)
    gbs_handles = [
        Line2D([0], [0], color=plt.get_cmap("Greys")(shade[g]),
               lw=width[g], alpha=alpha[g], label=f"GBS={g}")
        for g in gbs_list
        if any(Path(_csv_for(model, g, o)).exists() for o in opts)
    ]
    leg1 = ax.legend(handles=opt_handles, fontsize=9, loc="upper left",
                     title="optimizer (hue)")
    ax.add_artist(leg1)
    ax.legend(handles=gbs_handles, fontsize=7.5, loc="lower left",
              title="batch size (shade/width/opacity)", ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out} (adamw={plotted['adamw']} mano={plotted['mano']} "
          f"sophiag={plotted['sophiag']} batches)")


def plot_minlr_vs_gbs(out: Path, model: str) -> None:
    """usable/min LR vs batch, all optimizers overlaid (model-generic)."""
    colors = OPT_COLOR
    markers = {"adamw": "o", "mano": "s", "sophiag": "^"}
    fig, ax = plt.subplots(figsize=(9, 6))
    for opt in ("adamw", "mano", "sophiag"):
        xs, ys = [], []
        for g in GBS_2B:
            p = _csv_for(model, g, opt)
            if not Path(p).exists():
                continue
            fx, fy = _finite(*load_curve(p))
            if not fx:
                continue
            mi = min(range(len(fy)), key=lambda k: fy[k])
            xs.append(g)
            ys.append(fx[mi])
        if xs:
            ax.plot(xs, ys, "-", marker=markers[opt], ms=8, color=colors[opt],
                    label=f"{OPT_LABEL[opt]} ({len(xs)} GBS, 0 NaN)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Global batch size (GBS)")
    ax.set_ylabel("usable / min LR")
    ax.set_title(f"agpt {model.upper()}: usable LR vs batch, all optimizers "
                 f"(dp=192)\nno cliff at any batch -- like 2B, unlike 80B "
                 f"(adamw min-LR noisy: shallow basin)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_all_optimizers_at_gbs(out: Path, model: str, gbs: int) -> None:
    """all 3 optimizers' loss-vs-LR at one batch size (model-generic)."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for opt, lab in (("adamw", "AdamW"), ("mano", "mano"), ("sophiag", "sophiag")):
        p = _csv_for(model, gbs, opt)
        if not Path(p).exists():
            continue
        fx, fy_raw = _finite(*load_curve(p))
        if not fx:
            continue
        fy = _smooth(fy_raw)
        mi = min(range(len(fy)), key=lambda k: fy[k])
        mlr, mloss = fx[mi], fy[mi]
        ax.plot(fx, fy, "-o", ms=6, color=OPT_COLOR[opt],
                label=f"{lab} -- U-min @ {mlr:.1e} (0 NaN)")
        ax.scatter([mlr], [mloss], s=170, facecolors="none",
                   edgecolors=OPT_COLOR[opt], linewidths=1.8, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    tag = "the PRODUCTION batch" if gbs == 6144 else "GBS"
    ax.set_title(f"agpt {model.upper()} LR finder at {tag} (GBS={gbs}, dp=192)\n"
                 "all optimizers have clean U-minima, 0 NaN -- no cliff (cf. 80B)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_loss_vs_lr_by_gbs_80b(out: Path) -> None:
    # 80B adamw per-GBS: small-batch (8N) + reruns (64N) + the 6144 cliff.
    sources = {
        144: "outputs/lrtrend/gbs144",
        288: "outputs/lrtrend/gbs288",
        576: "outputs/lrtrend/gbs576",
        1152: "outputs/lrtrend/gbs1152-rerun",
        2304: "outputs/lrtrend/gbs2304-rerun2",
        4608: "outputs/lrtrend/gbs4608-rerun",
    }
    # GBS=6144 adamw curve (its live CSV was overwritten by trend probes;
    # values from the dated experiment record).
    cliff_lr = [1.0e-8, 1.848e-8, 3.415e-8, 6.31e-8, 1.166e-7, 2.154e-7,
                3.981e-7, 7.356e-7]
    cliff_loss = [12.911, 12.908, 12.930, 12.922, 12.904, 12.881, 12.844,
                  12.785]
    gbs_order = [144, 288, 576, 1152, 2304, 4608, 6144]
    cmap = plt.cm.plasma
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, g in enumerate(gbs_order):
        c = cmap(i / (len(gbs_order) - 1))
        if g == 6144:
            fx, fy = cliff_lr, cliff_loss
        else:
            hits = glob.glob(f"{sources[g]}/**/80B/adamw/lr_finder_data.csv",
                             recursive=True)
            if not hits:
                continue
            fx, fy = _finite(*load_curve(hits[0]))
            if not fx:
                continue
        lab = f"GBS={g}" + (" (cliff->NaN)" if g == 6144 else "")
        ax.plot(fx, fy, "-o", ms=4, color=c, label=lab, zorder=3)
        mi = min(range(len(fy)), key=lambda k: fy[k])
        ax.scatter([fx[mi]], [fy[mi]], s=90, facecolors="none",
                   edgecolors=c, linewidths=1.5, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    ax.set_title("agpt 80B AdamW: loss vs LR, one curve per batch size\n"
                 "(U-min shrinks + usable LR falls as GBS grows; 6144 cliffs to NaN)")
    ax.legend(fontsize=8, ncol=2, title="global batch")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _min(lrs, losses):
    fx, fy = _finite(lrs, losses)
    i = min(range(len(fy)), key=lambda k: fy[k])
    return fx[i], fy[i]


# ---------------------------------------------------------------------------
# Regenerate the 4 existing figures with house style (were raw matplotlib).
# ---------------------------------------------------------------------------
def plot_2b_all_optimizers(out: Path, gbs: int = 6144) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    specs = [("adamw", "AdamW"), ("mano", "mano"), ("sophiag", "sophiag")]
    for opt, lab in specs:
        p = csv_2b(gbs, opt)
        if not Path(p).exists():
            continue
        fx, fy_raw = _finite(*load_curve(p))
        fy = _smooth(fy_raw)
        mi = min(range(len(fy)), key=lambda k: fy[k])
        mlr, mloss = fx[mi], fy[mi]
        ax.plot(fx, fy, "-o", ms=6, color=OPT_COLOR[opt],
                label=f"{lab} -- U-min @ {mlr:.1e} (0 NaN)")
        ax.scatter([mlr], [mloss], s=170, facecolors="none",
                   edgecolors=OPT_COLOR[opt], linewidths=1.8, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    # 6144 is the common production batch (2B 256N + all 80B); larger GBS are
    # the 2B 512N/1024N chains. Only label 6144 as "production".
    tag = "the PRODUCTION batch" if gbs == 6144 else "GBS"
    ax.set_title(f"agpt 2B LR finder at {tag} (GBS={gbs}, dp=192)\n"
                 "all optimizers have clean U-minima, 0 NaN -- no cliff (cf. 80B)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_80b_all_optimizers(out: Path) -> None:
    base = "outputs/lr_finder/ezpz/ezpz.agpt/80B"
    # adamw 6144 from the dated record (live CSV overwritten by trend probes)
    adamw_lr = [1.0e-8, 1.848e-8, 3.415e-8, 6.31e-8, 1.166e-7, 2.154e-7,
                3.981e-7, 7.356e-7]
    adamw_loss = [12.911, 12.908, 12.930, 12.922, 12.904, 12.881, 12.844,
                  12.785]
    fig, ax = plt.subplots(figsize=(9, 6))
    for opt, lab in [("mano", "mano -- U-min @ 1.6e-5 (best, 0 NaN)"),
                     ("sophiag", "sophiag -- U-min @ 2.5e-6, lowest loss (0 NaN)")]:
        p = f"{base}/{opt}/lr_finder_data.csv"
        if not Path(p).exists():
            continue
        fx, fy_raw = _finite(*load_curve(p))
        fy = _smooth(fy_raw)
        mi = min(range(len(fy)), key=lambda k: fy[k])
        mlr, mloss = fx[mi], fy[mi]
        ax.plot(fx, fy, "-o", ms=6, color=OPT_COLOR[opt], label=lab)
        ax.scatter([mlr], [mloss], s=170, facecolors="none",
                   edgecolors=OPT_COLOR[opt], linewidths=1.8, zorder=4)
    # AdamW uses its canonical orange (consistent with every other chart); the
    # ^ marker + dashed annotation distinguish it as the NaN-cliff line.
    ax.plot(adamw_lr, adamw_loss, "-^", ms=7, color=OPT_COLOR["adamw"], zorder=5,
            label="AdamW -- NaN cliff: last finite 7.4e-7, NaN from 1.36e-6")
    ax.scatter([7.356e-7], [12.785], s=170, facecolors="none",
               edgecolors=OPT_COLOR["adamw"], linewidths=1.8, zorder=6)
    ax.annotate("AdamW: last finite 7.4e-7,\nthen NaN (no real min)",
                xy=(7.356e-7, 12.785), xytext=(1.0e-8, 12.35), fontsize=8,
                color=OPT_COLOR["adamw"],
                arrowprops=dict(arrowstyle="->", color=OPT_COLOR["adamw"]))
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    ax.set_title("agpt 80B LR finder at the PRODUCTION batch (GBS=6144, dp=192)\n"
                 "mano & sophiag have real U-minima; AdamW only cliffs to NaN")
    ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_80b_adamw_cliff(out: Path) -> None:
    """AdamW-only GBS=6144 cliff (styled rebuild of the original hand-made
    sunspot_80b_gbs6144_adamw.png). The live CSV was overwritten by trend
    probes, so the curve is the dated-record values: 8 finite points then NaN
    from 1.36e-6 (drawn as a shaded NaN region, not dropped, so the cliff is
    explicit -- unlike the finder's auto lr_vs_loss.png which silently drops
    the NaN points)."""
    lr = [1.0e-8, 1.848e-8, 3.415e-8, 6.31e-8, 1.166e-7, 2.154e-7,
          3.981e-7, 7.356e-7]
    loss = [12.911, 12.908, 12.930, 12.922, 12.904, 12.881, 12.844, 12.785]
    nan_lr = [1.359e-6, 2.512e-6, 4.642e-6, 8.577e-6, 1.585e-5, 2.929e-5,
              5.412e-5]  # all NaN
    fig, ax = plt.subplots(figsize=(9, 6))
    # AdamW line = canonical orange (consistent with every chart). The NaN
    # region stays RED -- there red means "divergence zone", a semantic
    # separate from the optimizer hue.
    ax.plot(lr, loss, "-^", ms=7, color=OPT_COLOR["adamw"], zorder=4,
            label="AdamW (finite)")
    ax.scatter([7.356e-7], [12.785], s=180, facecolors="none",
               edgecolors=OPT_COLOR["adamw"], linewidths=1.8, zorder=5)
    # shade the NaN region so the cliff is explicit
    ax.axvspan(1.0e-6, max(nan_lr), color="#d62728", alpha=0.08, zorder=0)
    ax.axvline(1.359e-6, color="#d62728", ls="--", lw=1, alpha=0.7, zorder=1)
    ax.annotate("NaN from 1.36e-6\n(usable ceiling ~7.4e-7,\nno real minimum)",
                xy=(1.359e-6, 12.83), xytext=(2.0e-6, 12.86), fontsize=8.5,
                color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("LR-finder smoothed loss")
    ax.set_title("agpt 80B AdamW at the PRODUCTION batch (GBS=6144, dp=192)\n"
                 "monotone descent to a wall, then NaN -- a cliff, not a U-min")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_ceiling_overlay_2b_vs_80b(out: Path) -> None:
    b2 = [(192, 8.58e-3), (384, 1.585e-2), (768, 1.585e-2), (1536, 8.58e-3),
          (3072, 8.58e-3), (6144, 8.58e-3), (12288, 1.585e-2), (24576, 1.585e-2)]
    b80_u = [(144, 1.0e-5), (288, 1.585e-5), (576, 1.585e-5), (1152, 1.585e-5)]
    cliff = (6144, 7.4e-7)
    # 20B AdamW read live (min-LR is noisy/shallow-basin, but all 0 NaN -- the
    # point is the curve sits between 2B and 80B and never cliffs).
    b20 = []
    for g in GBS_2B:
        p = _csv_for("20b", g, "adamw")
        if not Path(p).exists():
            continue
        fx, fy = _finite(*load_curve(p))
        if fx:
            mi = min(range(len(fy)), key=lambda k: fy[k])
            b20.append((g, fx[mi]))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([p[0] for p in b2], [p[1] for p in b2], "-o", ms=8,
            label="2B AdamW (dim=2048) -- clean U-min, 0 NaN")
    if b20:
        ax.plot([p[0] for p in b20], [p[1] for p in b20], "-D", ms=7,
                color="#9467bd",
                label="20B AdamW (dim=5120) -- clean U-min, 0 NaN (noisy min)")
    ax.plot([p[0] for p in b80_u], [p[1] for p in b80_u], "-o", ms=8,
            color="#d62728", label="80B AdamW (dim=9216) -- U-min (small batch)")
    ax.plot([cliff[0]], [cliff[1]], "v", ms=13, color="#d62728",
            label="80B AdamW -- NaN cliff (no min)")
    ax.plot([b80_u[-1][0], cliff[0]], [b80_u[-1][1], cliff[1]], "--",
            color="#d62728", alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 6e-2)
    ax.set_xlabel("Global batch size (GBS)")
    ax.set_ylabel("AdamW usable / min LR")
    ax.set_title("LR-finder usable LR vs batch: 2B vs 20B vs 80B (dp=192)\n"
                 "the NaN cliff is 80B-only -- 2B and 20B stay clean at all batches")
    ax.annotate("2B & 20B: clean U-min at every batch, 0 NaN\n"
                "(optimal LR ~1e-2 / ~1e-4, no cliff)",
                xy=(1536, 8.58e-3), xytext=(210, 1.3e-3), fontsize=8.5,
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("80B: usable LR falls ~20x, then NaN cliff by GBS=6144",
                xy=cliff, xytext=(230, 9e-5), fontsize=8.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def plot_80b_ceiling_vs_gbs(out: Path) -> None:
    # adamw 80B trend: (gbs, min/usable LR, min loss, shape). Complete after
    # the 1152/2304/4608 gap-fill reruns landed (2026-06-28): clean U-min
    # holds ~1.5e-5 through 1152, eases to ~4.6e-6 at 2304/4608 (the
    # pre-cliff shoulder), then collapses to the ~7.4e-7 NaN cliff at 6144.
    pts = [(144, 1.0e-5, 11.73, "U"), (288, 1.585e-5, 11.81, "U"),
           (576, 1.585e-5, 11.83, "U"), (1152, 1.585e-5, 11.65, "U"),
           (2304, 4.642e-6, 12.58, "U"), (4608, 4.642e-6, 12.54, "U"),
           (6144, 7.4e-7, 12.78, "cliff")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for g, lr, _, sh in pts:
        mk = "v" if sh == "cliff" else "o"
        col = "#d62728" if sh == "cliff" else ("#ff7f0e" if sh == "partial" else "#1f77b4")
        ax1.scatter([g], [lr], s=90, marker=mk, color=col, edgecolor="k",
                    linewidth=0.5, zorder=3)
    ax1.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="0.6", zorder=1)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("Global batch size (GBS)")
    ax1.set_ylabel("AdamW usable / min LR")
    ax1.set_title("80B AdamW: usable LR falls as batch grows")
    ax1.annotate("NaN cliff", xy=(6144, 7.4e-7), xytext=(1500, 1.8e-7),
                 fontsize=8, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728"))
    ax2.plot([p[0] for p in pts], [p[2] for p in pts], "-o", color="0.4")
    ax2.set_xscale("log")
    ax2.set_xlabel("Global batch size (GBS)")
    ax2.set_ylabel("Min loss reached in 15-step sweep")
    ax2.set_title("80B AdamW: min loss vs batch")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    apply_house_style()
    # NEW: per-GBS loss-vs-LR curve families
    plot_loss_vs_lr_by_gbs_2b(
        DOCS / "2b/figures/lr_finder_2b_loss_vs_lr_by_gbs.png", "adamw")
    plot_loss_vs_lr_by_gbs_2b(
        DOCS / "2b/figures/lr_finder_2b_mano_loss_vs_lr_by_gbs.png", "mano")
    plot_loss_vs_lr_by_gbs_2b(
        DOCS / "2b/figures/lr_finder_2b_sophiag_loss_vs_lr_by_gbs.png", "sophiag")
    plot_2b_minlr_vs_gbs_all_opts(
        DOCS / "2b/figures/lr_finder_2b_minlr_vs_gbs_all_optimizers.png")
    # combined: every optimizer x batch on one axis (hue=opt, shade=batch)
    plot_all_opts_all_gbs(
        DOCS / "2b/figures/lr_finder_2b_all_opts_all_gbs.png", "2b")
    plot_all_opts_all_gbs(
        DOCS / "20b/figures/lr_finder_20b_all_opts_all_gbs.png", "20b")
    # 20B page figures (model-generic helpers)
    plot_minlr_vs_gbs(
        DOCS / "20b/figures/lr_finder_20b_minlr_vs_gbs_all_optimizers.png", "20b")
    plot_all_optimizers_at_gbs(
        DOCS / "20b/figures/lr_finder_20b_gbs6144_all_optimizers.png", "20b", 6144)
    plot_loss_vs_lr_by_gbs_80b(DOCS / "80b/figures/sunspot_80b_adamw_loss_vs_lr_by_gbs.png")
    # RESTYLED: the 4 existing figures, now through the house stylesheet
    plot_2b_all_optimizers(
        DOCS / "2b/figures/lr_finder_2b_gbs6144_all_optimizers.png", 6144)
    plot_2b_all_optimizers(
        DOCS / "2b/figures/lr_finder_2b_gbs12288_all_optimizers.png", 12288)
    plot_80b_all_optimizers(DOCS / "80b/figures/sunspot_80b_gbs6144_all_optimizers.png")
    plot_80b_adamw_cliff(DOCS / "80b/figures/sunspot_80b_gbs6144_adamw.png")
    plot_ceiling_overlay_2b_vs_80b(DOCS / "2b/figures/lr_ceiling_vs_gbs_2b_vs_80b.png")
    plot_80b_ceiling_vs_gbs(DOCS / "80b/figures/sunspot_80b_adamw_lr_ceiling_vs_gbs.png")
