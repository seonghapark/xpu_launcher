#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l select=32
#PBS -q workq
#PBS -j oe
#
# Resume of 32N SFT job 12468218, which trained cleanly to step 400
# (epoch 1.59) then a worker rank hit a transient ccl::v1::exception
# (signal 6) at ~step 442. Loss / token-accuracy at the kill point
# were healthy (0.20 / 0.934).
#
# This resubmission passes --resume_from_checkpoint pointing at
# checkpoint-400 and otherwise mirrors the original launch exactly
# (same dataset, same LR/scheduler/optim state, so the cosine LR
# decay continues smoothly to 0 over the remaining 356 steps).

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

CKPT_DIR="outputs/sft/aurora2b-sophiag-metamathqa-32n"
RESUME_FROM="${CKPT_DIR}/checkpoint-400"

if [[ ! -d "${RESUME_FROM}" ]]; then
    echo "FATAL: resume checkpoint missing: ${RESUME_FROM}"
    exit 1
fi

LOG_DIR="logs/sft-aurora2b-metamathqa-32n-resume-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== 32N SFT RESUME: AuroraGPT-2B-sophiag-metamathqa from checkpoint-400 ===" \
    | tee "${LOG_DIR}/run.log"
echo "Resuming from: ${RESUME_FROM}" | tee -a "${LOG_DIR}/run.log"
echo "Expected: step 401 -> step 756 (~356 steps, ~10 min training)" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset metamathqa \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --output_dir "${CKPT_DIR}" \
    --resume_from_checkpoint "${RESUME_FROM}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 100 \
    --save_total_limit 8 \
    --report_to wandb \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
