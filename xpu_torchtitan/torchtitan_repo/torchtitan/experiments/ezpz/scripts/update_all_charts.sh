#!/bin/bash --login
# Run all production + eval chart-generating scripts in parallel.
#
# Re-renders:
#   - Per-trajectory production figures (loss / MFU / grad_norm / tokens-vs-time)
#     under docs/production/agpt/{2b,20b}/n{N}/figures/
#   - The all_production_training.{svg,png} cross-chain overview
#   - W&B-pulled per-chain figures (loss/MFU/tokens) for the SophiaG + LR-fork chains
#   - The all_production_evals.svg cross-eval overview
#   - 2B + 20B v1-vs-v2 eval comparison figures
#
# Usage (from repo root):
#   bash torchtitan/experiments/ezpz/scripts/update_all_charts.sh
#
# All scripts run in parallel and write to their own paths; failures in one
# don't block the others. Exit status is 0 iff all six succeeded.

set -o pipefail

cd "$(dirname "$0")/../../../.."   # repo root

PY="${PY:-.venv/bin/python3}"
if [[ ! -x "$PY" ]]; then
    echo "ERROR: python interpreter not found at $PY" >&2
    echo "       set PY env to override (e.g. PY=python3 bash $0)" >&2
    exit 1
fi

# PYTHONPATH=. is required because plot_production_combined.py imports
# from torchtitan.experiments.ezpz.utils.plot_production_wandb — a
# top-level torchtitan import that needs the repo root on sys.path.
export PYTHONPATH=.

# The W&B-sourced production charts (plot_production_wandb.py) fetch run
# histories over the network. On the Aurora/Sunspot login nodes that needs
# the ALCF proxy, or the fetch fails and the charts silently stay stale
# (refresh_all.sh treats the chart step as non-fatal). Export the proxy if
# it isn't already set so an unattended refresh actually regenerates them.
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

declare -A SCRIPTS=(
    # plot_production.py (PBS-.o-file plotter) dropped 2026-06-24: it is
    # DEPRECATED, its default .o files are gone, and no README embeds its
    # output. Use plot_production_wandb.py instead.
    [production_combined]="torchtitan/experiments/ezpz/utils/plot_production_combined.py"
    [production_wandb]="torchtitan/experiments/ezpz/utils/plot_production_wandb.py"
    [evals_combined]="torchtitan/experiments/ezpz/eval/plot_evals_combined.py"
    [evals_2b_overview]="torchtitan/experiments/ezpz/docs/evals/agpt/2b/plot_eval_overview.py"
    [evals_20b_overview]="torchtitan/experiments/ezpz/docs/evals/agpt/20b/plot_eval_overview.py"
    # LR-finder trend + per-GBS figures (2B/80B). Regenerates from the
    # isolated per-GBS CSVs under outputs/lrtrend*/; tolerates missing CSVs
    # (a queued/incomplete GBS just plots fewer curves), so it is safe to run
    # every refresh -- it auto-picks-up new points as reruns land.
    [lr_finder_trend]="torchtitan/experiments/ezpz/scripts/plot_lr_trend.py"
    # Not a chart, but the same "every-time-we-refresh-docs" cadence: rewrite
    # the docs/README.md table Modified column from `git log -1 --format=%cs`
    # for each linked path. Keeps the top-level index honest.
    [docs_readme_dates]="torchtitan/experiments/ezpz/utils/refresh_docs_readme_table.py"
)

declare -A PIDS LOGS
LOGDIR="$(mktemp -d -t update_all_charts.XXXXXX)"
echo "Per-script logs: $LOGDIR"

t0=$SECONDS
for name in "${!SCRIPTS[@]}"; do
    log="$LOGDIR/$name.log"
    LOGS[$name]="$log"
    echo "  [$name] launching: $PY ${SCRIPTS[$name]}"
    "$PY" "${SCRIPTS[$name]}" > "$log" 2>&1 &
    PIDS[$name]=$!
done

fail=0
for name in "${!PIDS[@]}"; do
    if wait "${PIDS[$name]}"; then
        echo "  [$name] OK ($(wc -l < "${LOGS[$name]}") lines)"
    else
        echo "  [$name] FAILED (rc=$?, see ${LOGS[$name]})"
        fail=$((fail + 1))
    fi
done

elapsed=$((SECONDS - t0))
n_files=$(find torchtitan/experiments/ezpz/docs -type f \
    \( -name "*.svg" -o -name "*.png" \) \
    -newer "$LOGDIR" 2>/dev/null | wc -l)
echo ""
echo "=== done in ${elapsed}s — $n_files figure files updated; $fail/${#SCRIPTS[@]} scripts failed ==="
exit $fail
