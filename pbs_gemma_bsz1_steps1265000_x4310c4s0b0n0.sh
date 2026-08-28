#!/usr/bin/env bash
#PBS -N gemma_bsz4_158000_2n
#PBS -A AuroraGPT
#PBS -q debug
#PBS -k doe
#PBS -j oe
#PBS -l filesystems=home:flare
#PBS -l select=2:ncpus=1
#PBS -l walltime=01:00:00

set -euo pipefail

cd /lus/flare/projects/datascience/seonghapark/llm_evaluation/xpu_launcher

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_name="gemma-pg19-multinews-bsz4-steps158000-2n-${run_stamp}"
log_dir="/lus/flare/projects/datascience/seonghapark/llm_evaluation/xpu_launcher/outputs/gemma_pg19_multinews_bsz4_steps158000_2n_${run_stamp}"
mkdir -p "$log_dir"
hostfile="${log_dir}/hosts.txt"
awk 'NF {print $1}' "${PBS_NODEFILE:?PBS_NODEFILE is required for multi-node training}" | sort -u > "$hostfile"

echo "hostname=$(hostname)"
echo "PBS_JOBID=${PBS_JOBID:-}"
echo "run_name=$run_name"
echo "log_dir=$log_dir"
echo "hostfile=$hostfile"
cat "$hostfile"

LOG_DIR="$log_dir" \
WANDB_RUN_NAME="$run_name" \
DATASET_PATH=pg19,multi_news \
BATCH_SIZE=4 \
TRAIN_STEPS=158000 \
SEQ_LEN=16384 \
./run_gemma_non_torchtitan_xpu.sh multi "$hostfile"
