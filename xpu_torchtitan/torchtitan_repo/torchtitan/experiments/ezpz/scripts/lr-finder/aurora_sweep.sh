#!/bin/bash --login
# LR Finder sweep — reproducing Sunspot 80B + 2B/20B experiments on Aurora
# Run inside a 2-node PBS allocation via .ezpz-interactive-launch.sh

set -o pipefail

WORKDIR="/lus/flare/projects/AuroraGPT/foremans/projects/saforem2/torchtitan-ezpz"
cd "$WORKDIR"

OUTDIR="$WORKDIR/lr-finder-aurora-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTDIR"

DATASET="torchtitan/experiments/ezpz/data-lists/aurora/olmo-mix-1124.txt"
STEPS=1000
FRACTION=0.1

run_lr_finder() {
    local model="$1" opt="$2" tp="$3" lbs="$4" compile="$5" gas="${6:-1}"
    local label="${model}_${opt}_gas${gas}"
    local logfile="${OUTDIR}/${label}.log"
    local compile_flag="--compile.enable"
    [[ "$compile" == "off" ]] && compile_flag="--compile.no-enable"

    echo "=== [$(date +%H:%M:%S)] ${label} (TP=${tp}, LBS=${lbs}, GAS=${gas}) ==="

    timeout 1800 python3 -m torchtitan.experiments.ezpz.train \
        --module=ezpz.agpt \
        --config="agpt_${model}" \
        ${compile_flag} \
        --parallelism.tensor_parallel_degree=${tp} \
        --checkpoint.no-enable \
        --dataloader.dataset=blendcorpus \
        --dataloader.dataset-path="${DATASET}" \
        --optimizer="${opt}" \
        --training.local-batch-size=${lbs} \
        --training.gradient-accumulation-steps=${gas} \
        --training.seq-len=8192 \
        --training.steps=${STEPS} \
        --lr_finder.enable \
        --lr_finder.fraction=${FRACTION} \
        --metrics.log_freq=1 \
        2>&1 | tee "${logfile}"

    echo "=== ${label} done (rc=$?) ==="
    echo ""
    # Kill leftovers
    pkill -u "$USER" -f "torchtitan.experiments.ezpz.train" 2>/dev/null || true
    sleep 5
}

echo "=============================================="
echo " LR Finder Sweep — Aurora $(date -Iseconds)"
echo " Output: ${OUTDIR}"
echo "=============================================="
echo ""

# 1. 80B x {AdamW, Muon, SophiaG} — no compile (AC+TP regression)
run_lr_finder "80b" "adamw"   2 1 "off"
run_lr_finder "80b" "muon"    2 1 "off"
run_lr_finder "80b" "sophiag" 2 1 "off"

# 2. 2B x {AdamW, Muon, SophiaG} — compile on
run_lr_finder "2b" "adamw"   1 1 "on"
run_lr_finder "2b" "muon"    1 1 "on"
run_lr_finder "2b" "sophiag" 1 1 "on"

# 3. 20B x {AdamW, Muon, SophiaG} — compile on
run_lr_finder "20b" "adamw"   1 1 "on"
run_lr_finder "20b" "muon"    1 1 "on"
run_lr_finder "20b" "sophiag" 1 1 "on"

# 4. GAS sweep — 2B AdamW
for gas in 4 8 16; do
    run_lr_finder "2b" "adamw" 1 1 "on" $gas
done

# 5. GAS sweep — 20B AdamW
for gas in 4 8 16; do
    run_lr_finder "20b" "adamw" 1 1 "on" $gas
done

echo "=============================================="
echo " All done. Results in ${OUTDIR}"
echo "=============================================="
