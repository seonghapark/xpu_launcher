#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=02:00:00
#PBS -l filesystems=flare:home
#PBS -l select=36
#PBS -q workq
#PBS -j oe
#
# 32N Sunspot SFT for AuroraGPT-2B on math_alpaca_mix (60% metamathqa
# + 10% gsm8k + 30% alpaca). Broader instruction-following than the
# math-only metamathqa run — should help downstream tasks like
# word_sort that aren't pure arithmetic.
#
# Allocation: select=36 (32 train + 4 spare) for --auto-retry. Past
# 32N SFT runs (12468218, 12468220, 12468222) all hit
# ccl::v1::exception → SIGABRT on a worker rank within ~10-15 min;
# auto-retry's bad-node detection swaps in a spare and continues
# from the latest checkpoint.
#
# Walltime: 2h is enough for ~3 epochs at this scale (interleaved
# dataset is ~990k = 395k/0.4 effective examples via 'all_exhausted'
# stopping). Adjust epochs or walltime up if the run gets killed
# before reaching 3 epochs.
#
# Output: outputs/sft/aurora2b-sophiag-mix-32n/

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

CKPT_DIR="outputs/sft/aurora2b-sophiag-mix-32n"
LOG_DIR="logs/sft-aurora2b-mix-32n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 32N SFT: AuroraGPT-2B + math_alpaca_mix (60/10/30), 3 epochs ===" \
    | tee "${LOG_DIR}/run.log"
echo "Allocation: select=36 (32 train + 4 spare for --auto-retry)" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

ezpz launch --np 384 -ppn 12 --auto-retry --max-failover-retries 3 \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset math_alpaca_mix \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 200 \
    --save_total_limit 8 \
    --report_to wandb \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
