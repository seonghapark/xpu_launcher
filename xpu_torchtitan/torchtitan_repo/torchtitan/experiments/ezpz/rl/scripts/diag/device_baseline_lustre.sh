#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:25:00
#PBS -l filesystems=flare:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
#
# 4-node Sunspot BASELINE smoke (b7ded2dc4 device-pin reverted), sourcing
# venv from /flare/ directly. Paired with scripts/diag/device_lustre.sh for
# a clean A/B on whether the device-pin commit actually helps.

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
# Check out the train_grpo.py from BEFORE b7ded2dc4 to test baseline
git stash push -m "baseline-4n-stash-$(date +%s)" -- torchtitan/experiments/ezpz/rl/train_grpo.py 2>&1 || true
git checkout e3477c717 -- torchtitan/experiments/ezpz/rl/train_grpo.py 2>&1

source .venv/bin/activate
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

LOG_DIR="logs/diag-fsdp-device-baseline-4n-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== BASELINE 4N: train_grpo at HEAD of train_grpo.py = e3477c717 ===" | tee "${LOG_DIR}/baseline.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_grpo.py | tee -a "${LOG_DIR}/baseline.log"
echo "" | tee -a "${LOG_DIR}/baseline.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 2 \
    2>&1 | tee -a "${LOG_DIR}/baseline.log" || true

# Restore HEAD train_grpo.py so working tree is clean
git checkout HEAD -- torchtitan/experiments/ezpz/rl/train_grpo.py 2>&1

echo "" | tee -a "${LOG_DIR}/baseline.log"
echo "=== DONE: baseline log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/baseline.log"
