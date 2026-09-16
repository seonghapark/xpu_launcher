#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=04:00:00
#PBS -l filesystems=flare:home
#PBS -l select=10
#PBS -q workq
#PBS -j oe
#
# 8N Sunspot production GRPO on the SFT'd AuroraGPT-2B
# (`checkpoint-729-hf`, the deliverable from the 2026-06-10
# `tulu_math_uc_mix` recipe). Task is `arithmetic` — combined
# +, -, ×, ÷ on small integers with accuracy + format + length
# rewards. 1000 GRPO steps.
#
# Justification for using the SFT'd checkpoint as the starting
# point (vs the raw pretrained AuroraGPT-2B-sophiag-gs138650): the
# 50-step smoke comparing the two showed an 8× speedup on the
# `sum_digits` task with the SFT'd model — 0.92 mean reward over
# the last 10 steps vs the baseline's 0.12, with the SFT'd model
# already at 28% accuracy cold (step 1) where the baseline was at
# 0%. See
# docs/production/sft/aurora2b/tulu_math_uc_mix/evals/grpo-smoke.md.
#
# Allocation: select=10 (8 train + 2 spare for --auto-retry). The
# Sunspot oneCCL `pidfd_getfd` SIGABRT (the dominant 32N failure
# mode this week) is rarer at 8N but still happens; spares are
# cheap insurance.
#
# Output: outputs/grpo/aurora2b-sft-arithmetic-8n/

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

MODEL=outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf
CKPT_DIR=outputs/grpo/aurora2b-sft-arithmetic-8n
LOG_DIR="logs/grpo-aurora2b-sft-arithmetic-8n-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 8N GRPO: SFT'd AuroraGPT-2B-tulu-mix + arithmetic, 1000 steps ===" \
    | tee "${LOG_DIR}/run.log"
echo "Allocation: select=10 (8 train + 2 spare for --auto-retry)" \
    | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 96 ranks × bsz=1 × num_gens=4 = 384 generations/step" \
    | tee -a "${LOG_DIR}/run.log"
echo "Model: ${MODEL}" | tee -a "${LOG_DIR}/run.log"
echo "Output: ${CKPT_DIR}" | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_grpo.py \
    torchtitan/experiments/ezpz/rl/tasks/arithmetic.py \
    2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Per-step takes ~5s at this scale (matched the smoke), so 1000 steps
# is ~85 min training + ~3 min init/load + per-failover overhead. Plenty
# of headroom in the 4h walltime for 1–2 spare-node rotations.
ezpz launch --np 96 -ppn 12 --auto-retry --max-failover-retries 2 \
    python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task arithmetic \
    --model_name_or_path "${MODEL}" \
    --output_dir "${CKPT_DIR}" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_generations 4 \
    --max_completion_length 64 \
    --temperature 0.7 \
    --max_steps 1000 \
    --learning_rate 1e-6 \
    --beta 0.0 \
    --bf16 --fsdp full_shard \
    --logging_steps 1 \
    --save_strategy steps --save_steps 100 \
    --save_total_limit 5 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

# --resume_from_checkpoint same as the SFT submit: HF Trainer
# auto-detects the latest checkpoint-N/ subdir. Safe on fresh runs
# (no-op if no ckpt yet). Critical for autoretry survival — without
# it, every relaunch starts from GRPO step 0 and loses progress.

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
