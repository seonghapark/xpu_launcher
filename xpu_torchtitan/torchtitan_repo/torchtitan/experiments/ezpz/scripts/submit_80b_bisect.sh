#!/bin/bash --login
#PBS -N agpt-80b-bisect
#PBS -l walltime=04:00:00
#PBS -l filesystems=flare:home
#PBS -A datascience
#PBS -q workq
#PBS -j oe

# 80B compile+AC+TP=2 DeviceMesh-in-saved-tensors bug bisect.
#
# Three sequential 10-step smokes on the same allocation:
#   1. agpt_50b_wide  (48 layers) — known-good control
#   2. agpt_70b_wide  (72 layers) — bisect midpoint
#   3. agpt_80b       (84 layers) — known-broken control
#
# Each writes to its own log so the upstream report can cite three
# independent runs. 4N gives TP=2 + dp_shard=24 with enough memory
# headroom (50B_wide on 2N TP=2 was already at 96% memory).
#
# Submit:
#   qsub -l select=4 torchtitan/experiments/ezpz/scripts/submit_80b_bisect.sh

set -o pipefail

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate

# Yeet env to compute nodes (tarball mode if available).
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "yeet-env via tarball: .venv.tar.gz"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "yeet-env via rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

# ---- Bisect run ----
LOG_DIR="logs/agpt-80b-bisect-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

# Configs to bisect: (config_name, n_layers).
CONFIGS=(
    "agpt_50b_wide:48"
    "agpt_70b_wide:72"
    "agpt_80b:84"
)

run_one() {
    local config="$1"
    local n_layers="$2"
    local log="${LOG_DIR}/${config}.log"

    log_message INFO "=========================================="
    log_message INFO "Bisect run: ${config} (n_layers=${n_layers})"
    log_message INFO "Log: ${log}"
    log_message INFO "=========================================="

    # 10 steps is enough to either crash at step 1 (the bug we are
    # hunting fires during the first compile/forward) or confirm the
    # config runs cleanly. Long enough to see TPS/MFU stabilize.
    ezpz launch python3 -m torchtitan.experiments.ezpz.train \
        --module ezpz.agpt \
        --config "${config}" \
        --training.steps 10 \
        --checkpoint.no_enable \
        --metrics.no_enable_wandb \
        2>&1 | tee "${log}"

    local exit_code=${PIPESTATUS[0]}
    log_message INFO "Exit code for ${config}: ${exit_code}"
    return ${exit_code}
}

# Run all three regardless of individual failures — we want a complete
# picture even if the middle one (70B_wide) crashes too.
declare -A RESULTS
for entry in "${CONFIGS[@]}"; do
    config="${entry%%:*}"
    n_layers="${entry##*:}"
    if run_one "${config}" "${n_layers}"; then
        RESULTS["${config}"]="OK"
    else
        RESULTS["${config}"]="FAIL"
    fi
done

# Summary
echo
log_message INFO "=========================================="
log_message INFO "80B bisect summary (${PBS_JOBID%%.*}):"
log_message INFO "=========================================="
for entry in "${CONFIGS[@]}"; do
    config="${entry%%:*}"
    n_layers="${entry##*:}"
    log_message INFO "  ${config} (${n_layers} layers): ${RESULTS[${config}]}"
done
log_message INFO "Logs in: ${LOG_DIR}"
