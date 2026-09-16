#!/bin/bash --login
# refresh_all.sh -- ONE command to bring every production doc + chart current.
#
# After a training dispatch lands new checkpoints, run this. It:
#   1. REPORT  : check_stale_docs.sh -- read-only punch-list (manifest-driven)
#   2. FILL    : fill_trajectory_fields.py -- rewrite the deterministic scalar
#                fields in each leaf README (latest ckpt / cumulative steps /
#                tokens+% / last-updated, and --wandb: loss) from disk truth
#   3. CHARTS  : update_all_charts.sh -- regenerate all figures + the
#                docs/README.md index table (refresh_docs_readme_table.py)
#   4. SUMMARY : one rolled-up report + non-zero exit iff a worker failed
#   5. COMMIT  : (default ON) conventional commit of the changed docs/figures
#                -- pull-first, NEVER push unless --push is given
#
# Single source of truth: utils/trajectories.py. Steps 1-3 all read the
# trajectory list / run-ids / ckpt-dirs from there, so they cannot drift.
#
# Usage:
#   bash torchtitan/experiments/ezpz/scripts/refresh_all.sh            # full: disk + W&B + charts + commit
#   bash .../refresh_all.sh --fast                                     # disk-only (no W&B, no slow charts)
#   bash .../refresh_all.sh --model 2b_v2_256                          # scope to one trajectory
#   bash .../refresh_all.sh --dry-run                                  # preview, write nothing, no commit
#   bash .../refresh_all.sh --no-commit                               # write files, leave staged for review
#   bash .../refresh_all.sh --push                                     # also push (pull-first); off by default
#
# PBS scripts / login-node note: NOT a PBS job -- runs on the login node.
# Do not `set -euo pipefail` (venv activate trips unbound vars).
set -o pipefail

cd "$(dirname "$0")/../../../.."   # repo root
export PYTHONPATH=.
PY="${PY:-.venv/bin/python3}"
[[ -x "$PY" ]] || PY="python3"

# ---- flags ----
WANDB=1            # full by default (disk + W&B + all charts)
DRY_RUN=0
DO_COMMIT=1        # auto-commit by default
DO_PUSH=0          # never push unless asked
MODEL=""
for arg in "$@"; do
    case "$arg" in
        --fast)      WANDB=0 ;;
        --wandb)     WANDB=1 ;;
        --dry-run)   DRY_RUN=1; DO_COMMIT=0 ;;
        --no-commit) DO_COMMIT=0 ;;
        --commit)    DO_COMMIT=1 ;;
        --push)      DO_PUSH=1 ;;
        --model)     : ;;                 # value consumed below
        --model=*)   MODEL="${arg#*=}" ;;
        2b_*|20b_*|80b_*) MODEL="$arg" ;; # bare trajectory key after --model
        -h|--help)
            sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "refresh_all: unknown arg '$arg'" >&2; exit 2 ;;
    esac
done

FILL_ARGS=()
[[ "$DRY_RUN" -eq 1 ]] && FILL_ARGS+=("--dry-run")
[[ "$WANDB"   -eq 1 ]] && FILL_ARGS+=("--wandb")
[[ -n "$MODEL" ]]      && FILL_ARGS+=("--model" "$MODEL")

echo "============================================================"
echo "refresh_all  (wandb=$WANDB dry_run=$DRY_RUN commit=$DO_COMMIT push=$DO_PUSH${MODEL:+ model=$MODEL})"
echo "============================================================"

# ---- 1. staleness punch-list (read-only) ----
echo ""
echo ">>> [1/3] staleness report (check_stale_docs.sh)"
stale_out="$(bash torchtitan/experiments/ezpz/scripts/check_stale_docs.sh 2>&1)"
echo "$stale_out"
stale_summary="$(echo "$stale_out" | grep -E '^=== summary' | tail -1)"

# ---- 2. fill leaf scalar fields ----
echo ""
echo ">>> [2/3] fill trajectory fields (fill_trajectory_fields.py)"
fill_rc=0
fill_out="$("$PY" -m torchtitan.experiments.ezpz.utils.fill_trajectory_fields "${FILL_ARGS[@]}" 2>&1)" || fill_rc=$?
echo "$fill_out"
fill_summary="$(echo "$fill_out" | grep -E '^=== ' | tail -1)"
if [[ "$fill_rc" -ne 0 ]]; then
    echo "!!! field-filler exited $fill_rc -- NOT committing (partial/failed write); review the working tree" >&2
fi

# ---- 3. charts + docs/README.md index table ----
echo ""
echo ">>> [3/3] charts + index table (update_all_charts.sh)"
charts_rc=0
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "    (dry-run: skipping chart regeneration)"
elif [[ "$WANDB" -eq 0 ]]; then
    # Fast path: only the W&B-free workers (eval figures + index-table
    # refresher). Skip the two W&B chart scripts.
    echo "    (--fast: regenerating index table + eval figures only; skipping W&B charts)"
    "$PY" -m torchtitan.experiments.ezpz.utils.refresh_docs_readme_table 2>&1 || charts_rc=$?
else
    bash torchtitan/experiments/ezpz/scripts/update_all_charts.sh 2>&1 || charts_rc=$?
fi

# ---- 4. summary ----
overall_rc=$(( charts_rc != 0 ? charts_rc : fill_rc ))
echo ""
echo "============================================================"
echo "SUMMARY"
echo "  staleness : ${stale_summary:-(none)}"
echo "  fields    : ${fill_summary:-(none)} (rc=$fill_rc)"
echo "  charts    : rc=$charts_rc$( [[ $DRY_RUN -eq 1 ]] && echo ' (dry-run, skipped)' )"
echo "============================================================"

# ---- 5. commit (default on; never push unless --push) ----
if [[ "$DRY_RUN" -eq 1 || "$DO_COMMIT" -eq 0 ]]; then
    [[ "$DO_COMMIT" -eq 0 && "$DRY_RUN" -eq 0 ]] && \
        echo "(--no-commit: changes left in the working tree for review)"
    exit "$overall_rc"
fi

# Never auto-commit a failed/partial field-fill -- a crash may have
# rewritten some-but-not-all READMEs. Leave it for human review.
if [[ "$fill_rc" -ne 0 ]]; then
    echo "(field-filler failed; skipping auto-commit -- review + commit manually)"
    exit "$overall_rc"
fi

# Stage ONLY known generated subpaths (never code, never hand-written docs
# like journal.md / meeting-notes). This keeps the auto-commit's blast
# radius to exactly what the refresh produces.
git add \
    "torchtitan/experiments/ezpz/docs/production/**/README.md" \
    "torchtitan/experiments/ezpz/docs/production/**/figures/*" \
    "torchtitan/experiments/ezpz/docs/production/figures/*" \
    "torchtitan/experiments/ezpz/docs/evals/**/figures/*" \
    "torchtitan/experiments/ezpz/docs/experiments/lr-finder/**/figures/*" \
    "torchtitan/experiments/ezpz/docs/README.md" \
    2>/dev/null

if git diff --cached --quiet 2>/dev/null; then
    echo "(no doc/chart changes to commit)"
    exit "$overall_rc"
fi

# Avoid a churn commit whose ONLY changes are the daily "Last updated:"
# date bump (+ the git-derived Modified column that tracks it). If every
# added/removed line is one of those, skip the commit.
substantive="$(git diff --cached -U0 2>/dev/null \
    | grep -E '^[+-]' \
    | grep -vE '^(\+\+\+|---)' \
    | grep -vE '(Last updated|Modified)' \
    | grep -vE '^[+-]\s*\|.*[0-9]{4}-[0-9]{2}-[0-9]{2}\s*\|' \
    | head -1)"
if [[ -z "$substantive" ]]; then
    echo "(only date/Modified churn staged -- skipping commit; \`git add\` left staged for review)"
    exit "$overall_rc"
fi

echo ""
echo ">>> committing refreshed docs + figures"
git -c core.pager=cat diff --cached --stat | tail -20
git commit -m "docs(ezpz/production): auto-refresh trajectory fields + charts

Generated by refresh_all.sh from the trajectories.py manifest:
${fill_summary:-fields refreshed}
${stale_summary:-staleness checked}" 2>&1 | tail -4

if [[ "$DO_PUSH" -eq 1 ]]; then
    echo ">>> pull (plain) then push"
    git pull 2>&1 | tail -3 && git push 2>&1 | tail -2
else
    echo "(committed locally; not pushed -- pass --push to push, or push manually)"
fi

exit "$overall_rc"
