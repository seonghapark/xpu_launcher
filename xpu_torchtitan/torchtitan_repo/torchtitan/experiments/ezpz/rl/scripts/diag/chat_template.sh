#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
#
# 4N AuroraGPT-2B + gemma chat template smoke (25 steps). Long enough
# to verify the model is NOT echoing the prompt and to capture a real
# completions-table snippet for the docs.

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
python3 -c "import trl; print('trl', trl.__version__)" || { echo "FATAL: trl missing"; exit 1; }

LOG_DIR="logs/diag-chat-template-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 25-step AuroraGPT-2B + gemma chat template ===" | tee "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_grpo.py \
    | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 25 \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE ===" | tee -a "${LOG_DIR}/run.log"
