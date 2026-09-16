#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:15:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1-node Sunspot smoke for the FSDP device-mismatch BASELINE.
# Checks out the commit BEFORE b7ded2dc4 (the untested device-pin fix)
# and runs phase-1 diagnostic + phase-2 train_grpo. The phase-2 output
# is the baseline-bug-still-fires reference to compare phase-2 of
# the WITH-FIX job against.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${SUBMIT_DIR}"
# Check out the commit BEFORE b7ded2dc4 to test the baseline (no fix).
# We need to do this AFTER ezpz_setup_job in case the source isn't
# usable from the compute node before that.
git stash push -m "baseline-test-stash-$(date +%s)" -- torchtitan/experiments/ezpz/rl/train_grpo.py 2>&1 || true
git checkout e3477c717 -- torchtitan/experiments/ezpz/rl/train_grpo.py 2>&1

source .venv/bin/activate
ezpz tar-env
ezpz yeet .venv.tar.gz
deactivate
source /tmp/.venv/bin/activate

LOG_DIR="logs/diag-fsdp-device-baseline-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== BASELINE (b7ded2dc4 reverted): train_grpo with --fsdp full_shard ===" | tee "${LOG_DIR}/baseline.log"
echo "=== HEAD of train_grpo.py: ===" | tee -a "${LOG_DIR}/baseline.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_grpo.py | tee -a "${LOG_DIR}/baseline.log"
echo "" | tee -a "${LOG_DIR}/baseline.log"
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 2 \
    2>&1 | tee -a "${LOG_DIR}/baseline.log" || true

# Restore the with-fix version so the working tree is left clean
git checkout HEAD -- torchtitan/experiments/ezpz/rl/train_grpo.py 2>&1

echo "" | tee -a "${LOG_DIR}/baseline.log"
echo "=== DONE: baseline log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/baseline.log"
