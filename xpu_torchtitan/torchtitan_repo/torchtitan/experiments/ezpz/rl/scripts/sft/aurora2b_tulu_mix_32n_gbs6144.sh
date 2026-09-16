#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=12:00:00
#PBS -l filesystems=flare:home
#PBS -l select=36
#PBS -q workq
#PBS -j oe
#
# 32N Sunspot SFT for AuroraGPT-2B on tulu_math_uc_mix (65% tulu-3-sft +
# 15% OpenMathInstruct-2 + 20% ultrachat-200k), 3 epochs at production-
# equivalent GBS=6144 (matches the pre-training GBS that AuroraGPT-2B
# was trained at).
#
# Sizing:
#   NGPUS = 32 nodes × 12 ranks = 384
#   GBS = NGPUS × per_device_train_batch_size × gradient_accumulation_steps
#       = 384 × 2 × 8 = 6144
#   tokens/step = GBS × max_length = 6144 × 1024 = 6.29M
#
# OpenMathInstruct-2 has 14M rows in this mix; `all_exhausted` cycling
# scales the effective interleaved size to ~21M total. 3 epochs at
# GBS=6144 is ~10500 steps. The HF .map output for OpenMathInstruct-2
# is pre-warmed at ~/.cache/huggingface/datasets/nvidia___open_math_instruct-2/
# (39 GB on /home NFS, shared with compute) so rank-0 build hits cache
# and completes in ~20-30 sec instead of ~10 min cold.
#
# Allocation: select=36 (32 train + 4 spare) for --auto-retry. Past
# 32N SFT runs (12468218, 12468220, 12468222) all hit
# ccl::v1::exception → SIGABRT on a worker rank within ~10-15 min;
# auto-retry's bad-node detection swaps in a spare and continues
# from the latest checkpoint. CAVEAT: ezpz#163 (auto-retry _drain
# can die on non-UTF-8 stdout, then the watchdog SIGTERMs healthy
# training after 30 min idle) is still open — watch for that signature.
#
# Output: outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/

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

CKPT_DIR="outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144"
LOG_DIR="logs/sft-aurora2b-tulu-mix-32n-gbs6144-${PBS_JOBID%%.*}"
mkdir -p "${CKPT_DIR}" "${LOG_DIR}"

echo "=== 32N SFT: AuroraGPT-2B + tulu_math_uc_mix, 3 epochs, GBS=6144 ===" \
    | tee "${LOG_DIR}/run.log"
echo "Allocation: select=36 (32 train + 4 spare for --auto-retry)" \
    | tee -a "${LOG_DIR}/run.log"
echo "Sizing: 384 ranks × bsz=2 × gas=8 = GBS=6144, 6.29M tokens/step" \
    | tee -a "${LOG_DIR}/run.log"
git log -1 --oneline -- torchtitan/experiments/ezpz/rl/train_sft.py \
    torchtitan/experiments/ezpz/rl/datasets_sft.py 2>&1 | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Swap OpenMathInstruct-2 (14M rows, ~20 min interleave setup on rank 0,
# blows past the XPU oneCCL barrier timeout even with PyTorch
# `dist.barrier(timeout=30min)` — caught this in job 12468398 where
# all 384 worker ranks crashed with `atl_comm->wait fails with status: 1`
# at the barrier while rank 0 was still building) for metamathqa
# (395k, ~10s build). Same GSM8K+MATH distribution coverage.
ezpz launch --np 384 -ppn 12 --auto-retry --max-failover-retries 3 \
    python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset 'tulu-3-sft-mixture:0.65,metamathqa:0.15,ultrachat-200k:0.20' \
    --model_name_or_path AuroraGPT-2B-sophiag-gs138650 \
    --output_dir "${CKPT_DIR}" \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_length 1024 \
    --bf16 --fsdp full_shard \
    --logging_steps 10 \
    --save_strategy steps --save_steps 100 \
    --save_total_limit 8 \
    --report_to wandb \
    --resume_from_checkpoint "${CKPT_DIR}" \
    2>&1 | tee -a "${LOG_DIR}/run.log" || true

# --resume_from_checkpoint <dir> is HF Trainer's auto-resume hook: if
# <dir> contains a checkpoint-N subdir, training picks up from the
# latest; if not, training starts from scratch. Safe to leave on for
# fresh runs. Critical for ezpz auto-retry: when a bad-node crash
# triggers attempt 2, the relaunch IS a fresh `python3 -m train_sft`
# invocation (not an in-process restart), so without this flag every
# retry attempt restarts from step 0 — wasting all prior progress.
# Caught in job 12468404 attempt 2 after a rank-286 SIGABRT killed
# attempt 1 at step 140; attempt 2 went back to step 10 instead of
# resuming from checkpoint-100.

echo "" | tee -a "${LOG_DIR}/run.log"
echo "=== DONE: log in ${LOG_DIR}/, ckpts in ${CKPT_DIR}/ ===" \
    | tee -a "${LOG_DIR}/run.log"
