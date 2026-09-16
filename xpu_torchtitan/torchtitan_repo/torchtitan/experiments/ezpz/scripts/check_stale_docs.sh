#!/bin/bash --login
# Check per-chain production READMEs for staleness vs on-disk state.
#
# For each canonical chain, compares:
#   - latest valid step-N dir on disk (`.metadata` present, not an empty placeholder)
#   - latest step-N mentioned in the chain's README
#   - the README's last-modified date (git log -1)
#
# Also flags chain checkpoint dirs that have NO matching README at all
# (e.g. new chains like the 80B 4N validation).
#
# Prints a punch list. Doesn't auto-edit — README updates need narrative
# judgment (failure-mode interpretation, "next continuation" identification,
# etc.) that's best done manually with the report in hand.
#
# Usage (from repo root):
#   bash torchtitan/experiments/ezpz/scripts/check_stale_docs.sh

set -o pipefail

cd "$(dirname "$0")/../../../.."   # repo root

# Map: ckpt_dir -> README path. SINGLE SOURCE OF TRUTH is the trajectory
# manifest (utils/trajectories.py); this emits the bash assoc-array from
# it, filtered to dirs that exist on disk. To add/move a chain, edit the
# manifest -- never hand-edit a map here (that drift is exactly how the
# relocated 20B-256N path went missing). Falls back to nothing if the
# emitter fails (loop below just reports zero chains).
PY="${PY:-.venv/bin/python3}"
[[ -x "$PY" ]] || PY="python3"
eval "$(PYTHONPATH=. "$PY" -m torchtitan.experiments.ezpz.utils.trajectories --emit stale-map 2>/dev/null)"
if [[ "${#CHAIN_TO_README[@]}" -eq 0 ]]; then
    echo "WARNING: trajectory manifest emitted no chains (emitter failed?)" >&2
fi

stale_count=0
missing_count=0
ok_count=0

echo "=== production README staleness check ($(date +%Y-%m-%d_%H:%M)) ==="
echo ""

for ckpt_dir in "${!CHAIN_TO_README[@]}"; do
    readme="${CHAIN_TO_README[$ckpt_dir]}"
    chain_name="$(basename "$ckpt_dir")"

    if [[ ! -d "$ckpt_dir" ]]; then
        # Chain doesn't exist on disk — skip silently
        continue
    fi

    # Latest VALID step-N (has .metadata + >=1 .distcp; excludes mid-save
    # placeholders + .bak-*). Computed by the SAME helper the field-filler
    # uses (fill_trajectory_fields.largest_valid_step) so the checker and
    # the filler can never disagree about disk truth.
    last_num=$(PYTHONPATH=. "$PY" -c "
import sys
from torchtitan.experiments.ezpz.utils.fill_trajectory_fields import largest_valid_step
s = largest_valid_step('$ckpt_dir')
print(s if s is not None else '')
" 2>/dev/null)
    if [[ -z "$last_num" ]]; then
        echo "  [no ckpts] $chain_name"
        continue
    fi
    last_ckpt="step-${last_num}"
    last_mtime=$(stat -c '%y' "$ckpt_dir/$last_ckpt" 2>/dev/null | head -c 19)

    if [[ ! -f "$readme" ]]; then
        echo "  [MISSING] $chain_name"
        echo "    disk: $last_ckpt ($last_mtime)"
        echo "    no README at: $readme"
        echo ""
        missing_count=$((missing_count + 1))
        continue
    fi

    # Prefer the canonical "**Latest checkpoint:** step-N" field (the one
    # fill_trajectory_fields.py writes), tolerating commas + bold
    # (step-**86,200**). Fall back to a textual max-scan of all step-N
    # mentions only if that field is absent. Strip commas/asterisks so
    # the numeric compare below works.
    readme_step=$(grep -oE '\*\*Latest(\s+\*[a-z]+\*)? checkpoint:\*\*[^|]*' "$readme" 2>/dev/null \
        | grep -oE 'step-\**[0-9,]+' | head -1 | tr -d '*,')
    if [[ -z "$readme_step" ]]; then
        readme_step=$(grep -oE 'step-\**[0-9,]+' "$readme" 2>/dev/null | tr -d '*,' \
            | sort -u | sort -t- -k2 -n | tail -1)
    fi
    readme_git=$(git log -1 --format='%ad' --date=short -- "$readme" 2>/dev/null)

    if [[ -z "$readme_step" ]]; then
        echo "  [no step-N in README] $chain_name"
        echo "    disk: $last_ckpt ($last_mtime)"
        echo "    readme: $readme (git: $readme_git)"
        echo ""
        stale_count=$((stale_count + 1))
        continue
    fi

    disk_num=${last_ckpt#step-}
    readme_num=${readme_step#step-}

    if (( disk_num > readme_num )); then
        delta=$((disk_num - readme_num))
        echo "  [STALE] $chain_name"
        echo "    disk: $last_ckpt ($last_mtime)"
        echo "    readme: $readme_step (git: $readme_git) — $delta steps behind"
        echo "    edit: $readme"
        echo ""
        stale_count=$((stale_count + 1))
    else
        ok_count=$((ok_count + 1))
    fi
done

echo "=== summary: $ok_count up-to-date, $stale_count stale, $missing_count missing-README ==="

# --- Phase 2: 'Last updated:' freshness across ALL docs (manifest-independent) ---
#
# The trajectory check above only covers per-chain READMEs (those with a
# checkpoint dir). Overview/guide pages (e.g. agpt/80b/README.md, the
# top-level production/README.md) carry a '> Last updated: YYYY-MM-DD'
# marker but are NOT in the manifest, so a content edit that forgets to
# bump the marker goes unnoticed (this is exactly how 80b/README.md sat
# at 2026-06-12 while being edited 2026-06-26). Flag any doc whose
# 'Last updated' date is OLDER than that file's last git-commit date.
echo ""
echo "=== 'Last updated:' freshness (all docs with the marker) ==="
lu_stale=0
lu_ok=0
while IFS= read -r doc; do
    marker=$(grep -oE "Last updated:?\*{0,2} *[0-9]{4}-[0-9]{2}-[0-9]{2}" "$doc" 2>/dev/null \
        | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2}" | head -1)
    [[ -z "$marker" ]] && continue
    git_date=$(git log -1 --format='%ad' --date=short -- "$doc" 2>/dev/null)
    [[ -z "$git_date" ]] && continue
    # String compare works for YYYY-MM-DD. Stale = marker predates the
    # file's last commit (content changed after the stamp was last set).
    if [[ "$marker" < "$git_date" ]]; then
        echo "  [STALE] ${doc#torchtitan/experiments/ezpz/}"
        echo "    marker: $marker   last commit: $git_date"
        lu_stale=$((lu_stale + 1))
    else
        lu_ok=$((lu_ok + 1))
    fi
done < <(grep -rlE "^> *\*{0,2}Last updated" torchtitan/experiments/ezpz/docs 2>/dev/null)
echo "=== Last-updated summary: $lu_ok fresh, $lu_stale stale ==="

exit 0
