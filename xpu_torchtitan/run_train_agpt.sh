#!/usr/bin/env bash
set -euo pipefail

# AGPT Training Launcher for TorchTitan
# Wrapper around run_train_torchtitan.sh with pre-optimized AGPT parameters
#
# Automatically calculates global batch size, training steps, and other
# parameters based on model size and node count.
#
# Usage:
#   ./run_train_agpt.sh --model 2b --nnodes 8 single
#   ./run_train_agpt.sh --model 2b multi /path/to/hosts.txt
#   ./run_train_agpt.sh --model 80b multi  # auto-detect nodes from PBS_NODEFILE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Capture the MODEL/MODEL_PATH environment convention used by
# run_train_torchtitan.sh (MODEL = path to model assets) before the defaults
# below clobber it; --model / --model-path still take precedence.
ENV_MODEL_PATH="${MODEL:-${MODEL_PATH:-}}"

# Defaults
MODEL=""
MODEL_PATH=""
NNODES=""
TP=""
PP=""
CP=""
LBS=""
GAS=""
SEQ_LEN=""
TRAIN_TOKENS=""
MODE=""
HOSTFILE=""
DRY_RUN=0
LOG_DIR=""
SPARE_NODES=""
SPARE_NODES_PERCENTAGE=""
AUTO_RETRY=""
FAILOVER_PROFILE=""
CHECKPOINT_PATH=""
RESOURCE_MONITOR=""
RESOURCE_INTERVAL=""
RESOURCE_OUTPUT_DIR=""

usage() {
  cat <<'EOF'
AGPT Training Launcher for TorchTitan

Usage:
  run_train_agpt.sh --model {2b|20b|80b} [OPTIONS] {single|multi} [HOSTFILE] [-- <extra args>]

Required:
  --model {2b|20b|80b}      Model size

Mode:
  single                    Single-node training
  multi                     Multi-node training (with optional hostfile)

Optional (Training):
  --nnodes N                Number of nodes (auto-detect from PBS_NODEFILE in PBS jobs)
  --tp N                    Tensor parallelism (override model default)
  --pp N                    Pipeline parallelism (default: 1)
  --cp N                    Context parallelism (default: 1)
  --lbs N                   Local batch size (override model default)
  --gas N                   Gradient accumulation steps (default: 1)
  --seq-len N               Sequence length (default: 8192)
  --train-tokens N          Total token budget (default: 4.67T)

Optional (Paths):
  --model-path PATH         Path to model/tokenizer directory
  --log-dir DIR             Output directory
  --checkpoint PATH         Checkpoint path to resume from

Optional (Monitoring):
  --resource-monitor        Enable resource monitoring
  --resource-interval N     Resource monitor interval in seconds
  --resource-output-dir DIR Resource metrics output directory

Optional (Fault Tolerance):
  --spare-nodes N           Number of spare nodes for failover
  --spare-nodes-percentage N Spare nodes as percentage of total
  --auto-retry {0|1}        Enable auto-retry (default: 1 for multi)
  --failover-profile PROF   Failover strategy profile (default: auto)

Optional (Other):
  --dry-run                 Print command without executing
  --help                    Show this help message

Examples:
  # Single-node 2B training
  ./run_train_agpt.sh --model 2b single

  # Multi-node 2B on 8 nodes with hostfile
  ./run_train_agpt.sh --model 2b --nnodes 8 multi /path/to/hosts.txt

  # In PBS job (auto-detect nodes from PBS_NODEFILE)
  ./run_train_agpt.sh --model 20b multi

  # 80B with custom tensor parallelism
  ./run_train_agpt.sh --model 80b --nnodes 32 --tp 4 multi /path/to/hosts.txt

  # Pass extra training arguments
  ./run_train_agpt.sh --model 2b single -- --training.enable_loss_std_termination

EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

# Detect NNODES from PBS context if available
detect_nnodes() {
  if [[ -n "${NNODES:-}" ]]; then
    return  # Already set
  fi

  if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
    NNODES=$(wc -l < "$PBS_NODEFILE")
  fi
}

# Validate model
validate_model() {
  case "$MODEL" in
    2b|20b|80b) ;;
    *)
      echo "error: invalid model: $MODEL (must be 2b, 20b, or 80b)" >&2
      exit 1
      ;;
  esac
}

# Validate mode
validate_mode() {
  case "$MODE" in
    single|multi) ;;
    *)
      echo "error: invalid mode: $MODE (must be single or multi)" >&2
      exit 1
      ;;
  esac
}

# Parse arguments
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --nnodes)
      NNODES="$2"
      shift 2
      ;;
    --tp)
      TP="$2"
      shift 2
      ;;
    --pp)
      PP="$2"
      shift 2
      ;;
    --cp)
      CP="$2"
      shift 2
      ;;
    --lbs)
      LBS="$2"
      shift 2
      ;;
    --gas)
      GAS="$2"
      shift 2
      ;;
    --seq-len)
      SEQ_LEN="$2"
      shift 2
      ;;
    --train-tokens)
      TRAIN_TOKENS="$2"
      shift 2
      ;;
    --model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --spare-nodes)
      SPARE_NODES="$2"
      shift 2
      ;;
    --spare-nodes-percentage)
      SPARE_NODES_PERCENTAGE="$2"
      shift 2
      ;;
    --auto-retry)
      AUTO_RETRY="$2"
      shift 2
      ;;
    --failover-profile)
      FAILOVER_PROFILE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --resource-monitor)
      RESOURCE_MONITOR=1
      shift
      ;;
    --resource-interval)
      RESOURCE_INTERVAL="$2"
      shift 2
      ;;
    --resource-output-dir)
      RESOURCE_OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS=("$@")
      break
      ;;
    single|multi)
      MODE="$1"
      shift

      # Next positional argument might be hostfile (only in multi mode)
      if [[ "$MODE" == "multi" && $# -gt 0 && "$1" != "--" && ! "$1" =~ ^-- ]]; then
        HOSTFILE="$1"
        shift
      fi

      # Keep parsing: options may follow the mode word, as in
      #   ./run_train_agpt.sh multi --resource-monitor -- --training.steps 400
      # Breaking here would silently discard every one of them.
      ;;
    *)
      echo "error: unknown option or invalid mode: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Fall back to the MODEL/MODEL_PATH environment convention when --model-path
# was not given.
if [[ -z "$MODEL_PATH" && -n "$ENV_MODEL_PATH" ]]; then
  MODEL_PATH="$ENV_MODEL_PATH"
fi

# Infer the model size from the assets path when --model was not given, so the
# env-driven invocation style (MODEL=/path/to/agpt-2b-... ) works here too.
if [[ -z "$MODEL" && -n "$MODEL_PATH" ]]; then
  case "$(basename "$MODEL_PATH")" in
    *-2b-*|*-2b|2b*)    MODEL="2b" ;;
    *-20b-*|*-20b|20b*) MODEL="20b" ;;
    *-80b-*|*-80b|80b*) MODEL="80b" ;;
  esac
  if [[ -n "$MODEL" ]]; then
    echo "[INFO] Inferred model size '$MODEL' from $(basename "$MODEL_PATH")" >&2
  fi
fi

# Validate required arguments
if [[ -z "$MODEL" ]]; then
  echo "error: --model is required (or set MODEL/--model-path to a path whose" >&2
  echo "       name contains the size, e.g. agpt-2b-v2-...)" >&2
  usage
  exit 1
fi

validate_model "$MODEL"

has_extra_arg() {
  local option="$1"
  local arg
  for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "$option" || "$arg" == "$option="* ]]; then
      return 0
    fi
  done
  return 1
}

if [[ -z "$MODE" ]]; then
  echo "error: mode (single or multi) is required" >&2
  usage
  exit 1
fi

validate_mode "$MODE"

# Auto-detect NNODES if in PBS job
detect_nnodes

# Validate NNODES
if [[ -z "$NNODES" && "$MODE" == "multi" ]]; then
  echo "error: nnodes required for multi mode (provide --nnodes or run in PBS job)" >&2
  exit 1
fi

# Set NNODES for single-node mode if not specified
if [[ -z "$NNODES" ]]; then
  NNODES=1
fi

# Build calculator args
CALC_ARGS=("--model" "$MODEL" "--nnodes" "$NNODES")
[[ -n "$TP" ]] && CALC_ARGS+=(--tp "$TP")
[[ -n "$PP" ]] && CALC_ARGS+=(--pp "$PP")
[[ -n "$CP" ]] && CALC_ARGS+=(--cp "$CP")
[[ -n "$LBS" ]] && CALC_ARGS+=(--lbs "$LBS")
[[ -n "$GAS" ]] && CALC_ARGS+=(--gas "$GAS")
[[ -n "$SEQ_LEN" ]] && CALC_ARGS+=(--seq-len "$SEQ_LEN")
[[ -n "$TRAIN_TOKENS" ]] && CALC_ARGS+=(--train-tokens "$TRAIN_TOKENS")

# Call calculator and evaluate environment variables
echo "[INFO] Calculating AGPT parameters for $MODEL on $NNODES nodes..." >&2
eval $("${SCRIPT_DIR}/calculate_agpt_params.sh" "${CALC_ARGS[@]}")

# Validate checkpoint path if provided
if [[ -n "$CHECKPOINT_PATH" ]]; then
  if [[ ! -d "$CHECKPOINT_PATH" ]]; then
    echo "error: checkpoint path not found: $CHECKPOINT_PATH" >&2
    exit 1
  fi
  CHECKPOINT_PATH="$(realpath "$CHECKPOINT_PATH")"
fi

# Preserve the model size (2b|20b|80b): MODEL is later overwritten with the
# model path for run_train_torchtitan.sh, which expects MODEL to be a path.
MODEL_SIZE="$MODEL"

# MODEL_PATH has no useful default: the assets live in per-user, per-checkpoint
# directories whose names carry a step number, so guessing a path only turns a
# clear "tell me where the model is" into a confusing "not found".
if [[ -z "$MODEL_PATH" ]]; then
  echo "error: model assets path is required for agpt-${MODEL_SIZE}" >&2
  echo "       pass --model-path PATH, or set MODEL=/path/to/agpt-${MODEL_SIZE}-..." >&2
  exit 1
fi

# Validate MODEL_PATH exists
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "error: model path not found: $MODEL_PATH" >&2
  exit 1
fi

# Set LOG_DIR with timestamp if not provided. Absolute: run_train_torchtitan.sh
# cd's to TORCHTITAN_ROOT before launching, so a relative path would be created
# here but resolved against a different directory there.
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="${SCRIPT_DIR}/outputs/agpt_${MODEL_SIZE}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$LOG_DIR"

TORCHTITAN_LAUNCH_SCRIPT="${SCRIPT_DIR}/run_train_torchtitan.sh"
if [[ ! -x "$TORCHTITAN_LAUNCH_SCRIPT" ]]; then
  echo "error: torchtitan launch wrapper not executable: $TORCHTITAN_LAUNCH_SCRIPT" >&2
  echo "hint: chmod +x ${TORCHTITAN_LAUNCH_SCRIPT}" >&2
  exit 1
fi

# Build run_train_torchtitan.sh command
TRAIN_CMD=(
  "$TORCHTITAN_LAUNCH_SCRIPT"
  "$MODE"
)

# Add hostfile for multi mode if provided
if [[ "$MODE" == "multi" && -n "$HOSTFILE" ]]; then
  TRAIN_CMD+=("$HOSTFILE")
fi

# Add dry-run flag if requested
if [[ $DRY_RUN -eq 1 ]]; then
  TRAIN_CMD+=(--dry-run)
fi

# Add separator and training args
TRAIN_CMD+=(--)

# Set environment variables for run_train_torchtitan.sh
export MODEL="$MODEL_PATH"
# Caller-supplied MODULE/CONFIG win, so non-default AGPT variants
# (agpt_2b_yarn, *_flex_attn, ...) are reachable through this wrapper.
export MODULE="${MODULE:-agpt}"
export CONFIG="${CONFIG:-agpt_${AGPT_MODEL}}"
export DATASET_NAME="${DATASET_NAME:-$AGPT_DATASET}"
export DATASET_PATH="${DATASET_PATH:-$AGPT_DATASET_PATH}"
export TRAINING_STEPS="$AGPT_TRAINING_STEPS"
export SEQ_LEN="$AGPT_SEQ_LEN"
export LOG_DIR="$LOG_DIR"
export CKPT_FOLDER="$AGPT_CKPT_DIR"
# The calculator decides these; forward them so they reach the training
# command instead of only appearing in the summary. Caller env still wins.
export OPTIMIZER_NAME="${OPTIMIZER_NAME:-$AGPT_OPTIMIZER}"
export LEARNING_RATE="${LEARNING_RATE:-$AGPT_LR}"
export CKPT_INTERVAL="${CKPT_INTERVAL:-$AGPT_CKPT_INTERVAL}"
export CKPT_KEEP_LATEST_K="${CKPT_KEEP_LATEST_K:-$AGPT_CKPT_KEEP_LATEST_K}"

# Set fault tolerance variables if provided
[[ -n "$SPARE_NODES" ]] && export SPARE_NODES="$SPARE_NODES"
[[ -n "$SPARE_NODES_PERCENTAGE" ]] && export SPARE_NODES_PERCENTAGE="$SPARE_NODES_PERCENTAGE"
[[ -n "$AUTO_RETRY" ]] && export AUTO_RETRY="$AUTO_RETRY"
[[ -n "$FAILOVER_PROFILE" ]] && export FAILOVER_PROFILE="$FAILOVER_PROFILE"

# Set checkpoint variable if provided
[[ -n "$CHECKPOINT_PATH" ]] && export CKPT="$CHECKPOINT_PATH"

# Set resource monitoring variables if provided
[[ -n "$RESOURCE_MONITOR" ]] && export RESOURCE_MONITOR="$RESOURCE_MONITOR"
[[ -n "$RESOURCE_INTERVAL" ]] && export RESOURCE_INTERVAL="$RESOURCE_INTERVAL"
[[ -n "$RESOURCE_OUTPUT_DIR" ]] && export RESOURCE_OUTPUT_DIR="$RESOURCE_OUTPUT_DIR"

# Build training-specific arguments. Each default is skipped when the user
# supplied the same flag after `--`, so explicit overrides always win.
# AGPT_GAS is not forwarded: torchtitan has no gradient-accumulation field, and
# the calculator already folds GAS into AGPT_GBS.
TRAINING_ARGS=()
if ! has_extra_arg "--training.steps"; then
  TRAINING_ARGS+=("--training.steps" "$AGPT_TRAINING_STEPS")
fi
if ! has_extra_arg "--training.seq_len"; then
  TRAINING_ARGS+=("--training.seq_len" "$AGPT_SEQ_LEN")
fi
if ! has_extra_arg "--training.local_batch_size"; then
  TRAINING_ARGS+=("--training.local_batch_size" "$AGPT_LBS")
fi
if ! has_extra_arg "--parallelism.tensor_parallel_degree"; then
  TRAINING_ARGS+=("--parallelism.tensor_parallel_degree" "$AGPT_TP")
fi
if ! has_extra_arg "--parallelism.pipeline_parallel_degree"; then
  TRAINING_ARGS+=("--parallelism.pipeline_parallel_degree" "$AGPT_PP")
fi
if ! has_extra_arg "--parallelism.context_parallel_degree"; then
  TRAINING_ARGS+=("--parallelism.context_parallel_degree" "$AGPT_CP")
fi

if [[ ${#TRAINING_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${TRAINING_ARGS[@]}")
fi

# Add any extra arguments passed by user
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

cat >&2 <<EOF
================================================================================
AGPT TRAINING LAUNCH CONFIGURATION
================================================================================

[MODEL]
  Model                  = ${MODEL_SIZE}
  Model Path             = ${MODEL_PATH}

[LAUNCH MODE]
  Mode                   = ${MODE}$( [[ "$MODE" == "multi" ]] && echo " (hostfile=${HOSTFILE:-auto from PBS_NODEFILE})" )
  Nodes                  = ${NNODES}

[PARALLELISM & BATCH]
  Tensor Parallelism     = ${AGPT_TP}
  Pipeline Parallelism   = ${AGPT_PP}
  Context Parallelism    = ${AGPT_CP}
  Local Batch Size       = ${AGPT_LBS}
  Gradient Accumulation  = ${AGPT_GAS}
  Global Batch Size      = ${AGPT_GBS}

[TRAINING HYPERPARAMETERS]
  Training Steps         = ${AGPT_TRAINING_STEPS}
  Sequence Length        = ${AGPT_SEQ_LEN}
  Train Tokens           = ${AGPT_TRAIN_TOKENS}

[SYSTEM & PATHS]
  Log Directory          = ${LOG_DIR}
  Checkpoint Directory   = ${AGPT_CKPT_DIR}
  Checkpoint Load        = ${CHECKPOINT_PATH:-<unset>}

[MONITORING]
  Resource Monitor       = ${RESOURCE_MONITOR:-0}
  Resource Interval      = ${RESOURCE_INTERVAL:-<default>}s
  Resource Output Dir    = ${RESOURCE_OUTPUT_DIR:-<default>}

================================================================================
EOF

printf 'Running: '
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

exec "${TRAIN_CMD[@]}"
