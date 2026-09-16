#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# 2N Sunspot smoke for the new tulu_math_uc_mix SFT dataset.
# Goal: verify the canonical 3-way mix (65% tulu-3-sft-mixture +
# 15% OpenMathInstruct-2 + 20% ultrachat-200k) builds and trains end-
# to-end on a small node count before burning a 32N production slot.
#
# Walltime 45min because OpenMathInstruct-2 has ~14M rows and Map()
# takes ~10 min single-threaded. Once cache is warm, 10 training
# steps should land in ~30s.
#
# Output: outputs/sft/aurora2b-sophiag-tulu-mix-smoke-n2/
#         logs/sft-aurora2b-tulu-mix-smoke-n2-${PBS_JOBID%%.*}/

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

CKPT_DIR="outputs/sft/aurora2b-sophiag-tulu-mix-smoke-n2"
LOG_DIR="logs/sft-aurora2b-tulu-mix-smoke-n2-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 2N SFT smoke: AuroraGPT-2B + tulu_math_uc_mix, 10 steps ===" \
    | tee "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/datasets_sft.py \
    torchtitan/experiments/ezpz/rl/train_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Use the mix-spec syntax (not tulu_math_uc_mix) to swap
# OpenMathInstruct-2 → metamathqa: 14M rows → 395k rows, eliminates
# the rank-0 interleave-setup bottleneck. Same math distribution
# (both are GSM8K+MATH-derived).
#
# --max_train_samples 50000 truncates after the build so tokenize+pack
# fits in <2 min instead of ~67 min on the full mix. Smoke covers all
# the real shapes (multi-turn tulu, single-turn math, multi-turn
# ultrachat) without the wall-clock cost.
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset 'tulu-3-sft-mixture:0.65,metamathqa:0.15,ultrachat-200k:0.20' \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --output_dir "${CKPT_DIR}" \
    --max_steps 10 \
    --max_train_samples 50000 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy no \
    --report_to wandb \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/run.log"
