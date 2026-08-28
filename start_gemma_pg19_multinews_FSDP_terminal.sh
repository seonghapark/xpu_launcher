#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_name="gemma-FSDP-pg19-multinews-bsz4-steps158000-seq16384-2n-${run_stamp}"
log_dir="${SCRIPT_DIR}/outputs/gemma_FSDP_pg19_multinews_bsz4_steps158000_seq16384_2n_${run_stamp}"
hostfile="${log_dir}/hosts.txt"
run_log="${log_dir}/terminal_run.log"

mkdir -p "$log_dir"

if [[ -z "${PBS_NODEFILE:-}" || ! -f "$PBS_NODEFILE" ]]; then
  echo "error: PBS_NODEFILE is not set or does not exist. Run this inside the 2-node qsub interactive terminal." >&2
  exit 1
fi

awk 'NF {print $1}' "$PBS_NODEFILE" | sort -u > "$hostfile"
node_count="$(wc -l < "$hostfile")"
if [[ "$node_count" -ne 2 ]]; then
  echo "error: expected 2 unique nodes in PBS_NODEFILE, got ${node_count}" >&2
  cat "$hostfile" >&2
  exit 1
fi

echo "run_name=$run_name"
echo "log_dir=$log_dir"
echo "hostfile=$hostfile"
cat "$hostfile"

DATASET_MAX_SAMPLES=256 \
LOG_DIR="$log_dir" \
WANDB_RUN_NAME="$run_name" \
DATASET_PATH=pg19,multi_news \
BATCH_SIZE=4 \
TRAIN_STEPS=158000 \
SEQ_LEN=16384 \
./run_gemma_FSDP_xpu.sh multi "$hostfile" -- \
  --micro-batch-size 1 \
  --logit-chunk-size 256 \
  "$@" 2>&1 | tee "$run_log"
