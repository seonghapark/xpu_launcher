#!/bin/bash
# Statistical A/B comparison of histc vs bincount in MoE routing,
# end-to-end on moe_2b_ep. Runs on a single PBS alloc to keep compile
# cache + node + cluster state shared across variants.
#
# Strategy: strict alternation across 6 trials (H, B, H, B, H, B).
# Compile-cache asymmetry cancels out because both variants are
# "warm" on trials 3+ and both are "cold-after-switch" on trials 1-2.
#
# Each trial: 50 steps, no determinism, LBS=2, checkpointing disabled.
# Per-step times come from the trainer's metric log; the analyzer
# extracts steps 6-50 (drop warmup/compile) and runs a Welch's t-test.

set -u

# Usage: stats_histc_vs_bincount_run_trials.sh <compute-node> [repo-root] [run-dir]
NODE=${1:?missing NODE}
REPO=${2:-/lus/flare/projects/datascience/foremans/projects/saforem2/torchtitan}
RUN_DIR=${3:-$REPO/logs/smoke-40th-sync/stats-runs}
mkdir -p "$RUN_DIR"

MOE_FILE=$REPO/torchtitan/models/common/moe.py
TD_FILE=$REPO/torchtitan/models/common/token_dispatcher.py

# Two variants as exact-match anchors for sed replacement.
HISTC_MOE='        num_tokens_per_expert = torch.histc(\n            selected_experts_indices.view(-1),\n            bins=self.num_experts,\n            min=0,\n            max=self.num_experts,\n        )'
BINCOUNT_MOE='        num_tokens_per_expert = torch.bincount(\n            selected_experts_indices.view(-1),\n            minlength=self.num_experts,\n        )'

# Make sure we start in histc state (upstream baseline).
ensure_histc() {
    /bin/cp "$MOE_FILE" "$MOE_FILE.snapshot"
    /bin/cp "$TD_FILE" "$TD_FILE.snapshot"
}

set_variant() {
    local variant=$1
    if [[ $variant == "histc" ]]; then
        cp "$MOE_FILE.snapshot" "$MOE_FILE"
        cp "$TD_FILE.snapshot" "$TD_FILE"
    elif [[ $variant == "bincount" ]]; then
        # Use python for the replacement to avoid sed escaping headaches.
        python3 - <<PYEOF
import re
for path in ["$MOE_FILE", "$TD_FILE"]:
    src = open("$MOE_FILE.snapshot" if path == "$MOE_FILE" else "$TD_FILE.snapshot").read()
    new = re.sub(
        r'        num_tokens_per_expert = torch\.histc\(\n'
        r'            selected_experts_indices\.view\(-1\),\n'
        r'            bins=self\.num_experts,\n'
        r'            min=0,\n'
        r'            max=self\.num_experts,\n'
        r'        \)',
        '        num_tokens_per_expert = torch.bincount(\n'
        '            selected_experts_indices.view(-1),\n'
        '            minlength=self.num_experts,\n'
        '        )',
        src,
    )
    open(path, 'w').write(new)
PYEOF
    else
        echo "unknown variant: $variant" >&2
        exit 1
    fi
}

run_one() {
    local variant=$1 trial=$2
    local log=$RUN_DIR/trial-$trial-$variant.log
    echo "[trial $trial variant=$variant] -> $log"
    set_variant "$variant"
    ssh "$NODE" "bash --login -c 'cd $REPO && source \$HOME/.ezpz/utils.sh && ezpz_setup_job && ezpz_setup_xpu && source .venv/bin/activate && ezpz launch python3 -m torchtitan.experiments.ezpz.train --module ezpz.moe --config moe_2b_ep --training.steps 50 --checkpoint.no-enable'" > "$log" 2>&1
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  exit=$rc"
        tail -5 "$log"
    fi
}

ensure_histc

# Strict alternation: H B H B H B
ORDER=(histc bincount histc bincount histc bincount)
for i in "${!ORDER[@]}"; do
    run_one "${ORDER[$i]}" "$((i+1))"
done

# Restore upstream-canonical state.
set_variant histc
rm -f "$MOE_FILE.snapshot" "$TD_FILE.snapshot"
echo "DONE — restored histc"
