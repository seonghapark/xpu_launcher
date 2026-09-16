#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:15:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1-node Sunspot smoke for the FSDP device-mismatch diagnostic.
# Runs scripts/diag/device_mismatch.py + train_grpo with --fsdp full_shard
# and captures per-rank diagnostic output for analysis.

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
source .venv/bin/activate
ezpz tar-env
ezpz yeet .venv.tar.gz
deactivate
source /tmp/.venv/bin/activate

LOG_DIR="logs/diag-fsdp-device-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== PHASE 1: diagnostic (per-rank device state) ===" | tee "${LOG_DIR}/phase1.log"
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.scripts.diag.device_mismatch \
    2>&1 | tee -a "${LOG_DIR}/phase1.log" || true

echo "" | tee -a "${LOG_DIR}/phase1.log"
echo "=== PHASE 2: actual train_grpo with --fsdp full_shard ===" | tee "${LOG_DIR}/phase2.log"
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 2 \
    2>&1 | tee -a "${LOG_DIR}/phase2.log" || true

echo "" | tee -a "${LOG_DIR}/phase2.log"
echo "=== DONE: logs in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/phase2.log"
