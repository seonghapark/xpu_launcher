#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=01:00:00
#PBS -l filesystems=flare:home
#PBS -l select=32
#PBS -q workq
#PBS -j oe
#
# 32N Sunspot SFT for AuroraGPT-2B-sophiag-gs138650 on metamathqa.
# Goal: produce a checkpoint that's stronger at instruction-following
# than the raw sophiag-138650 ckpt, suitable as a starting point for
# the arithmetic GRPO loop.
#
# Dataset: meta-math/MetaMathQA, ~395k augmented GSM8K+MATH samples.
# At 32N x per_dev_bsz=1 x packing=True ~ 260 steps/epoch.
# 3 epochs => ~780 steps. ETA ~15-25 min training + ~2 min init.
#
# Save: every 100 steps + final. ~8 intermediate ckpts for rollback.
# Each ckpt is ~4 GB (bf16 model + optim state, sharded state dict).

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
LOG_DIR="logs/sft-aurora2b-metamathqa-32n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 32N SFT: AuroraGPT-2B-sophiag-138650 + metamathqa, 3 epochs ===" \
    | tee "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset metamathqa \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --output_dir "${CKPT_DIR}" \
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
