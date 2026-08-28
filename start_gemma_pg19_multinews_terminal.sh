#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_name="gemma-pg19-multinews-bsz4-steps158000-2n-terminal-${run_stamp}"
log_dir="${SCRIPT_DIR}/outputs/gemma_pg19_multinews_bsz4_steps158000_2n_terminal_${run_stamp}"
hostfile="${log_dir}/hosts.txt"
run_log="${log_dir}/terminal_run.log"

mkdir -p "$log_dir"

if [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]]; then
  awk 'NF {print $1}' "$PBS_NODEFILE" | sort -u > "$hostfile"
else
  printf 'x4319c0s5b0n0\nx4319c0s6b0n0\n' > "$hostfile"
fi

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
nohup ./run_gemma_non_torchtitan_xpu.sh multi "$hostfile" > "$run_log" 2>&1 &

pid=$!
echo "$pid" > "${log_dir}/pid"
echo "started_pid=$pid"
echo "run_log=$run_log"