#!/bin/bash
# Competition sweep: try all speedrun configs and collect final loss.
#
# Usage:
#   ezpz submit -n 2 -q debug -A Aurora_deployment -t 01:00:00 \
#       torchtitan/experiments/ezpz/competition/run_sweep.sh
#
# Or run individual configs:
#   ezpz launch python3 -m torchtitan.experiments.ezpz.train \
#       --module ezpz.agpt --config speedrun_2b_muon

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_FILE="${SCRIPT_DIR}/results.csv"

CONFIGS=(
    speedrun_2b_adamw
    speedrun_2b_muon
    speedrun_2b_sophiag
    speedrun_2b_muon_aggressive
    speedrun_2b_adamw_large_batch
)

echo "config,final_loss,final_grad_norm,wall_time_s,tps_per_gpu" > "$RESULTS_FILE"

for config in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: $config"
    echo "=========================================="

    start_time=$(date +%s)

    # Run training, capture output
    output=$(ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt \
        --config "$config" \
        2>&1) || {
        echo "$config,FAILED,,,," >> "$RESULTS_FILE"
        echo "FAILED: $config"
        continue
    }

    end_time=$(date +%s)
    wall_time=$((end_time - start_time))

    # Extract final metrics from output
    final_loss=$(echo "$output" | grep -oP 'loss:\s*\K[\d.]+' | tail -1)
    final_grad_norm=$(echo "$output" | grep -oP 'grad_norm:\s*\K[\d.]+' | tail -1)
    tps=$(echo "$output" | grep -oP 'tps:\s*\K[\d.]+' | tail -1)

    echo "$config,$final_loss,$final_grad_norm,$wall_time,$tps" >> "$RESULTS_FILE"
    echo "  loss=$final_loss grad_norm=$final_grad_norm wall=${wall_time}s tps=$tps"
done

echo ""
echo "=========================================="
echo "Results saved to: $RESULTS_FILE"
echo "=========================================="
column -t -s',' "$RESULTS_FILE"
