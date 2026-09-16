#!/usr/bin/bash
# Learning rate finder sweep across models and optimizers.
#
# Runs the LR finder for each (model, optimizer) combination and saves
# results to outputs/lr_finder/ezpz/<module>/<flavor>/<optimizer>/.
#
# Usage:
#   # Inside a PBS job or interactive session with ezpz available:
#   bash torchtitan/experiments/ezpz/scripts/run_lr_finder_sweep.sh
#
#   # Override defaults:
#   LRF_MODELS="2b 20b" LRF_OPTIMIZERS="adamw muon" LRF_STEPS=500 \
#       bash torchtitan/experiments/ezpz/scripts/run_lr_finder_sweep.sh
#
# Environment variables:
#   LRF_MODULE     — module to use (default: "ezpz.agpt")
#   LRF_MODELS     — space-separated model flavors (default: "2b 20b")
#   LRF_OPTIMIZERS — space-separated optimizers (default: "adamw muon sophiag")
#   LRF_STEPS      — training.steps (finder uses 10%, default: 1000)
#   LRF_INIT_LR    — initial learning rate (default: 1e-6)
#   LRF_MAX_LR     — maximum learning rate (default: 1.0)
#   LRF_FRACTION   — fraction of steps to sweep (default: 0.1)
#   LRF_BETA       — EMA smoothing factor (default: 0.98)
#   LRF_WARMUP     — warmup fraction before sweep (default: 0.0)
#   LRF_SMOOTH     — derivative smoothing fraction (default: 0.05)
#   LRF_GAS        — gradient accumulation steps (default: 1)

set -o pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
LRF_MODULE="${LRF_MODULE:-ezpz.agpt}"
LRF_MODELS="${LRF_MODELS:-2b 20b}"
LRF_OPTIMIZERS="${LRF_OPTIMIZERS:-adamw muon sophiag}"
LRF_STEPS="${LRF_STEPS:-1000}"
LRF_INIT_LR="${LRF_INIT_LR:-1e-6}"
LRF_MAX_LR="${LRF_MAX_LR:-1.0}"
LRF_FRACTION="${LRF_FRACTION:-0.1}"
LRF_BETA="${LRF_BETA:-0.98}"
LRF_WARMUP="${LRF_WARMUP:-0.0}"
LRF_SMOOTH="${LRF_SMOOTH:-0.05}"
LRF_GAS="${LRF_GAS:-1}"

# Derive config prefix from module name (ezpz.agpt -> agpt, ezpz.moe -> moe)
CONFIG_PREFIX="${LRF_MODULE##*.}_"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

set +u
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
if ! command -v ezpz >/dev/null; then
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
fi
set -u

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
TOTAL=0
PASSED=0
FAILED=0
RESULTS=()

echo "============================================================"
echo "LR Finder Sweep"
echo "  Module:     ${LRF_MODULE}"
echo "  Models:     ${LRF_MODELS}"
echo "  Optimizers: ${LRF_OPTIMIZERS}"
echo "  Steps:      ${LRF_STEPS} (finder iters: $(python3 -c "print(max(1,int(${LRF_STEPS}*${LRF_FRACTION})))"))"
echo "  LR range:   ${LRF_INIT_LR} -> ${LRF_MAX_LR}"
echo "  Warmup:     ${LRF_WARMUP}  GAS: ${LRF_GAS}  Smooth: ${LRF_SMOOTH}"
echo "============================================================"

for model in ${LRF_MODELS}; do
    for opt in ${LRF_OPTIMIZERS}; do
        TOTAL=$((TOTAL + 1))
        config="${CONFIG_PREFIX}${model}"

        echo ""
        echo "------------------------------------------------------------"
        echo "[${TOTAL}] ${config} + ${opt}"
        echo "------------------------------------------------------------"

        # Build optimizer args
        opt_args=()
        if [[ "${opt}" != "adamw" ]]; then
            opt_args=("--optimizer" "${opt}")
        fi

        start_seconds=$SECONDS

        # Compute GBS for gradient accumulation
        gas_args=()
        if ((LRF_GAS > 1)); then
            gbs=$((NGPUS * LRF_GAS))
            gas_args=("--training.global_batch_size" "${gbs}")
        fi

        ezpz launch python3 -m torchtitan.experiments.ezpz.train \
            --module "${LRF_MODULE}" \
            --config "${config}" \
            --training.steps "${LRF_STEPS}" \
            --checkpoint.no_enable \
            --lr_finder.enable \
            --lr_finder.init_lr "${LRF_INIT_LR}" \
            --lr_finder.max_lr "${LRF_MAX_LR}" \
            --lr_finder.fraction "${LRF_FRACTION}" \
            --lr_finder.beta "${LRF_BETA}" \
            --lr_finder.warmup_fraction "${LRF_WARMUP}" \
            --lr_finder.smooth_frac "${LRF_SMOOTH}" \
            "${gas_args[@]}" \
            "${opt_args[@]}"
        rc=$?

        elapsed=$((SECONDS - start_seconds))

        if [[ $rc -eq 0 ]]; then
            PASSED=$((PASSED + 1))
            status="OK"
        else
            FAILED=$((FAILED + 1))
            status="FAIL (rc=${rc})"
        fi

        RESULTS+=("| ${config} | ${opt} | ${status} | ${elapsed}s |")
        echo "[${TOTAL}] ${config} + ${opt}: ${status} (${elapsed}s)"
    done
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "LR Finder Sweep Complete"
echo "  Total: ${TOTAL}  Passed: ${PASSED}  Failed: ${FAILED}"
echo "============================================================"
echo ""
echo "| Config | Optimizer | Status | Time |"
echo "|--------|-----------|--------|------|"
for row in "${RESULTS[@]}"; do
    echo "${row}"
done
echo ""
echo "Results saved to: outputs/lr_finder/"
