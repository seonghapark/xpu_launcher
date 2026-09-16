#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:40:00
#PBS -l filesystems=flare:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
#
# 4N Sunspot smoke for train_sft.py with the local AuroraGPT-2B
# checkpoint. Caps at --max_steps 50 so we're not waiting for a real
# epoch — just verifying that:
#   - train_sft.py imports + parses CLI cleanly
#   - SFTTrainer accepts the gemma chat-template fallback
#     (assistant_only_loss=True needs {% generation %} markers — this
#     run tells us whether that's a problem or not)
#   - gsm8k dataset loads via the ALCF proxy
#   - FSDP wraps + a few training steps land
#
# Expected runtime: ~10-15 min wall (init + 50 SFT steps + save).

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

LOG_DIR="logs/diag-sft-aurora2b-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== train_sft.py + AuroraGPT-2B + gsm8k, 4N FSDP full_shard, 50 steps ===" \
    | tee "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset gsm8k \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --bf16 --fsdp full_shard \
    --max_steps 50 \
    --logging_steps 5 \
    --no_save \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
