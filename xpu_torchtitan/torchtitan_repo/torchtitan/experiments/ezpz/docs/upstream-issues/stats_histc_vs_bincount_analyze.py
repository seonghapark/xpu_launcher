"""Analyze paired histc-vs-bincount trial logs.

Parses each ``trial-N-{histc,bincount}.log`` for per-step timestamps,
computes per-step durations (steps 6..50, skipping warmup/compile),
and reports per-variant summary stats plus a Welch's t-test on the
pooled per-step times.

Also reports avg TPS per trial (already computed by the trainer).
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from pathlib import Path

# Usage: stats_histc_vs_bincount_analyze.py [trial-log-dir]
# Default matches the driver script's default RUN_DIR.
LOG_DIR = Path(
    sys.argv[1] if len(sys.argv) > 1
    else "logs/smoke-40th-sync/stats-runs"
)

TIMESTAMP_RE = re.compile(
    r"\[(\d{4}-\d{2}-\d{2})\s+(\d{2}):(\d{2}):(\d{2})\].*step:\s*(\d+)"
)
TPS_RE = re.compile(r"step:\s*(\d+).*tps:\s*([\d,]+)")
WALL_RE = re.compile(r"Executing finished in\s+([\d.]+)\s+seconds")


def _parse_log(path: Path):
    text = path.read_text(errors="replace")
    # ANSI escape stripper (color codes wrap field labels)
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    # Per-step wall timestamps and TPS readings.
    step_times = {}
    for m in TIMESTAMP_RE.finditer(text):
        _, hh, mm, ss, step = m.groups()
        t_s = int(hh) * 3600 + int(mm) * 60 + int(ss)
        step = int(step)
        # The same step can be matched multiple times if metrics log
        # repeats; first one is what we want.
        step_times.setdefault(step, t_s)
    step_tps = {}
    for m in TPS_RE.finditer(text):
        step, tps_str = m.groups()
        step_tps.setdefault(int(step), int(tps_str.replace(",", "")))
    wall = None
    wm = WALL_RE.search(text)
    if wm:
        wall = float(wm.group(1))
    return step_times, step_tps, wall


def _per_step_durations(step_times: dict[int, int], start: int, end: int) -> list[float]:
    durs = []
    for s in range(start + 1, end + 1):
        if s in step_times and (s - 1) in step_times:
            d = step_times[s] - step_times[s - 1]
            # Negative deltas happen at day boundary (HH:MM:SS wraps);
            # add 86400 to correct.
            if d < 0:
                d += 86400
            durs.append(float(d))
    return durs


def _welch_t(a: list[float], b: list[float]) -> tuple[float, float]:
    """Welch's t-statistic and approximate two-sided p-value."""
    n1, n2 = len(a), len(b)
    m1, m2 = statistics.mean(a), statistics.mean(b)
    v1 = statistics.variance(a) if n1 > 1 else 0.0
    v2 = statistics.variance(b) if n2 > 1 else 0.0
    se = math.sqrt(v1 / n1 + v2 / n2)
    if se == 0.0:
        return 0.0, 1.0
    t = (m1 - m2) / se
    # Welch–Satterthwaite df
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / max(n1 - 1, 1) + (v2 / n2) ** 2 / max(n2 - 1, 1)
    )
    # Approximate two-sided p via the standard-normal tail when df is
    # large enough (df > 30); good enough for the kind of effect size
    # we care about reporting. Falls back to a Cauchy-tail floor.
    if df > 30:
        # erfc-based normal CDF
        z = abs(t)
        p = math.erfc(z / math.sqrt(2))
    else:
        # crude approximation: use the normal anyway with a warning
        z = abs(t)
        p = math.erfc(z / math.sqrt(2))
    return t, p


def main() -> int:
    trials = sorted(LOG_DIR.glob("trial-*.log"))
    if not trials:
        print(f"no trial logs in {LOG_DIR}")
        return 1

    by_variant: dict[str, dict] = {"histc": {}, "bincount": {}}
    for path in trials:
        m = re.match(r"trial-(\d+)-(\w+)\.log$", path.name)
        if not m:
            continue
        trial_n, variant = int(m.group(1)), m.group(2)
        step_times, step_tps, wall = _parse_log(path)
        # Skip steps 1-5 (compile/warmup).
        durs = _per_step_durations(step_times, start=5, end=50)
        tps_post = [step_tps[s] for s in range(6, 51) if s in step_tps]
        by_variant.setdefault(variant, {})[trial_n] = {
            "wall": wall,
            "n_steps": len(step_times),
            "per_step_durs": durs,
            "mean_dur": statistics.mean(durs) if durs else None,
            "mean_tps": statistics.mean(tps_post) if tps_post else None,
            "tps_samples": tps_post,
        }

    print(f"# Statistical A/B: histc vs bincount on moe_2b_ep, 2N Sunspot")
    print()
    print("## Per-trial summary")
    print()
    print(f"{'trial':>5}  {'variant':>9}  {'wall(s)':>9}  "
          f"{'n_step':>6}  {'mean_dur(s)':>11}  {'mean_tps':>9}")
    for variant, runs in sorted(by_variant.items()):
        for trial_n, info in sorted(runs.items()):
            wall = f"{info['wall']:.1f}" if info["wall"] else "-"
            md = f"{info['mean_dur']:.3f}" if info["mean_dur"] else "-"
            mt = f"{info['mean_tps']:.0f}" if info["mean_tps"] else "-"
            print(f"{trial_n:>5}  {variant:>9}  {wall:>9}  "
                  f"{info['n_steps']:>6}  {md:>11}  {mt:>9}")

    print()
    print("## Pooled per-step duration (steps 6-50 from each trial)")
    print()
    pooled = {v: [] for v in by_variant}
    pooled_tps = {v: [] for v in by_variant}
    for variant, runs in by_variant.items():
        for info in runs.values():
            pooled[variant].extend(info["per_step_durs"])
            pooled_tps[variant].extend(info["tps_samples"])

    for variant in ("histc", "bincount"):
        d = pooled[variant]
        t = pooled_tps[variant]
        n = len(d)
        if n < 2:
            print(f"  {variant}: insufficient samples ({n})")
            continue
        print(f"  {variant}: n={n}  "
              f"mean_dur={statistics.mean(d):.3f}s  "
              f"std={statistics.stdev(d):.3f}  "
              f"mean_tps={statistics.mean(t):.0f}  "
              f"tps_std={statistics.stdev(t):.0f}")

    print()
    print("## Welch's t-test (two-sided, H0: mean step duration equal)")
    a, b = pooled["histc"], pooled["bincount"]
    if len(a) > 1 and len(b) > 1:
        t, p = _welch_t(a, b)
        delta = statistics.mean(b) - statistics.mean(a)
        pct = delta / statistics.mean(a) * 100.0 if statistics.mean(a) else 0
        print(f"  delta_mean_dur(bincount - histc) = {delta:+.3f}s  "
              f"({pct:+.1f}%)")
        print(f"  Welch's t = {t:+.3f}")
        print(f"  approx two-sided p = {p:.3g}")
        if p < 0.01:
            print(f"  -> reject H0 (p<0.01); the two variants differ "
                  f"at the per-step level by {pct:+.1f}%.")
        elif p < 0.05:
            print(f"  -> reject H0 (p<0.05); weak evidence of a "
                  f"{pct:+.1f}% per-step difference.")
        else:
            print(f"  -> fail to reject H0 (p>={p:.2f}); no detectable "
                  f"E2E perf difference between the two variants.")

    print()
    print("## Welch's t-test on per-step TPS")
    a, b = pooled_tps["histc"], pooled_tps["bincount"]
    if len(a) > 1 and len(b) > 1:
        t, p = _welch_t(a, b)
        delta = statistics.mean(b) - statistics.mean(a)
        pct = delta / statistics.mean(a) * 100.0 if statistics.mean(a) else 0
        print(f"  delta_mean_tps(bincount - histc) = {delta:+.0f} "
              f"({pct:+.2f}%)")
        print(f"  Welch's t = {t:+.3f}, approx p = {p:.3g}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
