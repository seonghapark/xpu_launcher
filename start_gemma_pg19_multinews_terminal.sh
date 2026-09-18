#!/usr/bin/env bash
#
# Start a 2-node Gemma pg19+multi_news run from inside a qsub interactive
# terminal, then hand the shell back.
#
# Usage:
#   ./start_gemma_pg19_multinews_terminal.sh [extra trainer args...]
#
# Environment:
#   GEMMA_PRESET   run shape, passed through to run_gemma_non_torchtitan_xpu.sh
#                  (default "fsdp"; see presets/gemma_*.env)
#   EXPECTED_NODES number of unique nodes to require in PBS_NODEFILE
#                  (default 2; set to 0 to skip the check)
#   FOLLOW         1 to stay attached and tee the log to this terminal,
#                  0 (default) to detach with nohup and return immediately
#   DRY_RUN        1 to print the launch command instead of running it
#
# The run always writes its own log under outputs/<run>/terminal_run.log, so
# detaching loses nothing; FOLLOW=1 only mirrors it to the terminal as well.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GEMMA_PRESET="${GEMMA_PRESET:-fsdp}"
EXPECTED_NODES="${EXPECTED_NODES:-2}"
FOLLOW="${FOLLOW:-0}"
DRY_RUN="${DRY_RUN:-0}"

run_stamp="$(date +%Y%m%d_%H%M%S)"
run_tag="gemma_${GEMMA_PRESET}_pg19_multinews"
run_name="${run_tag//_/-}-${EXPECTED_NODES}n-${run_stamp}"
log_dir="${SCRIPT_DIR}/outputs/${run_tag}_${EXPECTED_NODES}n_${run_stamp}"
hostfile="${log_dir}/hosts.txt"
run_log="${log_dir}/terminal_run.log"

# This script only makes sense inside a PBS allocation: the hostfile it builds
# is what run_gemma_non_torchtitan_xpu.sh launches across.
if [[ -z "${PBS_NODEFILE:-}" || ! -f "$PBS_NODEFILE" ]]; then
  echo "error: PBS_NODEFILE is not set or does not exist." >&2
  echo "       Run this inside a ${EXPECTED_NODES}-node qsub interactive terminal." >&2
  exit 1
fi

mkdir -p "$log_dir"
awk 'NF {print $1}' "$PBS_NODEFILE" | sort -u > "$hostfile"

node_count="$(wc -l < "$hostfile")"
if [[ "$EXPECTED_NODES" -ne 0 && "$node_count" -ne "$EXPECTED_NODES" ]]; then
  echo "error: expected ${EXPECTED_NODES} unique nodes in PBS_NODEFILE, got ${node_count}" >&2
  cat "$hostfile" >&2
  exit 1
fi

echo "preset=$GEMMA_PRESET"
echo "run_name=$run_name"
echo "log_dir=$log_dir"
echo "hostfile=$hostfile"
cat "$hostfile"

run_env=(
  GEMMA_PRESET="$GEMMA_PRESET"
  LOG_DIR="$log_dir"
  WANDB_RUN_NAME="$run_name"
  DATASET_PATH=pg19,multi_news
)

run_cmd=(./run_gemma_non_torchtitan_xpu.sh multi "$hostfile")
if [[ "$DRY_RUN" == "1" ]]; then
  run_cmd+=(--dry-run)
fi
if [[ $# -gt 0 ]]; then
  run_cmd+=(-- "$@")
fi

# DRY_RUN and FOLLOW both stay in the foreground: there is nothing to detach
# from when the launch command is only being printed.
if [[ "$FOLLOW" == "1" || "$DRY_RUN" == "1" ]]; then
  env "${run_env[@]}" "${run_cmd[@]}" 2>&1 | tee "$run_log"
else
  nohup env "${run_env[@]}" "${run_cmd[@]}" > "$run_log" 2>&1 &
  pid=$!
  echo "$pid" > "${log_dir}/pid"
  echo "started_pid=$pid"
  echo "run_log=$run_log"
fi
