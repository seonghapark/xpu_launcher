#!/usr/bin/env bash
#PBS -N gemma_bsz1_1265000_x4310
#PBS -A AuroraGPT
#PBS -q debug
#PBS -k doe
#PBS -j oe
#PBS -l filesystems=home:flare
#PBS -l select=1:host=x4310c4s0b0n0:ncpus=1
#PBS -l walltime=01:00:00

set -euo pipefail

cd /lus/flare/projects/datascience/seonghapark/llm_evaluation/xpu_launcher

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_name="gemma-bsz1-steps1265000-x4310c4s0b0n0-${run_stamp}"
log_dir="/lus/flare/projects/datascience/seonghapark/llm_evaluation/xpu_launcher/outputs/gemma_bsz1_steps1265000_x4310c4s0b0n0_${run_stamp}"
mkdir -p "$log_dir"

echo "hostname=$(hostname)"
echo "PBS_JOBID=${PBS_JOBID:-}"
echo "run_name=$run_name"
echo "log_dir=$log_dir"

LOG_DIR="$log_dir" \
WANDB_RUN_NAME="$run_name" \
BATCH_SIZE=1 \
TRAIN_STEPS=1265000 \
SEQ_LEN=16384 \
./run_gemma_non_torchtitan_xpu.sh single
