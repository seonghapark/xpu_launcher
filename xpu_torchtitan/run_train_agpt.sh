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

error() {
  echo "Error: $*" >&2
  exit 1
}

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
    *) error "Invalid model: $MODEL (must be 2b, 20b, or 80b)" ;;
  esac
}

# Validate mode
validate_mode() {
  case "$MODE" in
    single|multi) ;;
    *) error "Invalid mode: $MODE (must be single or multi)" ;;
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

      # Collect remaining args as extra training args
      if [[ $# -gt 0 && "$1" == "--" ]]; then
        shift
        EXTRA_ARGS=("$@")
      fi
      break
      ;;
    *)
      error "Unknown option or invalid mode: $1"
      ;;
  esac
done

# Validate required arguments
if [[ -z "$MODEL" ]]; then
  error "--model is required"
fi

validate_model "$MODEL"

if [[ -z "$MODE" ]]; then
  error "MODE (single or multi) is required"
fi

validate_mode "$MODE"

# Auto-detect NNODES if in PBS job
detect_nnodes

# Validate NNODES
if [[ -z "$NNODES" && "$MODE" == "multi" ]]; then
  error "NNODES required for multi mode (provide --nnodes or run in PBS job)"
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

# Set MODEL_PATH default if not provided
if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="${HOME}/models/agpt-${MODEL}"
fi

# Validate MODEL_PATH exists
if [[ ! -d "$MODEL_PATH" ]]; then
  error "Model path not found: $MODEL_PATH"
fi

# Set LOG_DIR with timestamp if not provided
if [[ -z "$LOG_DIR" ]]; then
  LOG_DIR="outputs/agpt_${MODEL}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$LOG_DIR"

# Build run_train_torchtitan.sh command
TRAIN_CMD=(
  "${SCRIPT_DIR}/run_train_torchtitan.sh"
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
export MODULE="agpt"
export CONFIG="agpt_${AGPT_MODEL}"
export TRAINING_STEPS="$AGPT_TRAINING_STEPS"
export SEQ_LEN="$AGPT_SEQ_LEN"
export LOG_DIR="$LOG_DIR"
export CKPT_FOLDER="$AGPT_CKPT_DIR"

# Set fault tolerance variables if provided
[[ -n "$SPARE_NODES" ]] && export SPARE_NODES="$SPARE_NODES"
[[ -n "$SPARE_NODES_PERCENTAGE" ]] && export SPARE_NODES_PERCENTAGE="$SPARE_NODES_PERCENTAGE"
[[ -n "$AUTO_RETRY" ]] && export AUTO_RETRY="$AUTO_RETRY"
[[ -n "$FAILOVER_PROFILE" ]] && export FAILOVER_PROFILE="$FAILOVER_PROFILE"

# Build training-specific arguments
TRAINING_ARGS=(
  "--training.steps" "$AGPT_TRAINING_STEPS"
  "--training.seq_len" "$AGPT_SEQ_LEN"
)

TRAIN_CMD+=("${TRAINING_ARGS[@]}")

# Add any extra arguments passed by user
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

# Print configuration summary
cat >&2 <<EOF
==========================================
AGPT Training Configuration
==========================================
Model:                    $MODEL
Mode:                     $MODE
Nodes:                    $NNODES
Global Batch Size:        $AGPT_GBS
Tensor Parallelism:       $AGPT_TP
Training Steps:           $AGPT_TRAINING_STEPS
Sequence Length:          $AGPT_SEQ_LEN
Model Path:               $MODEL_PATH
Log Directory:            $LOG_DIR
Checkpoint Directory:     $AGPT_CKPT_DIR
==========================================
EOF

# Execute or dry-run
if [[ $DRY_RUN -eq 1 ]]; then
  echo "[DRY-RUN] Would execute:" >&2
  printf '%q ' "${TRAIN_CMD[@]}" >&2
  printf '\n' >&2
else
  echo "[INFO] Executing training command..." >&2
  exec "${TRAIN_CMD[@]}"
fi
