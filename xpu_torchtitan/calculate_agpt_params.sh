#!/usr/bin/env bash
set -euo pipefail

# AGPT Parameter Calculator for TorchTitan Training
# Extracts pre-optimized parameter calculations from train_agpt_*_venv.sh scripts
# Supports models: 2B, 20B, 80B with production-ready defaults
#
# Usage:
#   eval $(./calculate_agpt_params.sh --model 2b --nnodes 8)
#   ./calculate_agpt_params.sh --model 2b --nnodes 8 --dry-run
#   ./calculate_agpt_params.sh --help

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values (global)
MODEL=""
NNODES=""
TP=""
PP="${PP:-1}"
CP="${CP:-1}"
LBS=""
GAS="${GAS:-1}"
SEQ_LEN="${SEQ_LEN:-8192}"
TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"  # 4.67T tokens
DATA_LIST="${DATA_LIST:-olmo-mix-1124}"
OUTPUT_FORMAT="shell"
DRY_RUN=0

# AGPT-specific constants (all models)
OPTIMIZER="sophiag"
LR="2.28e-5"
DATASET="blendcorpus"
DEVICES_PER_NODE=12  # Aurora XPU: 12 devices per node
CKPT_INTERVAL=100
CKPT_KEEP_LATEST_K=0

usage() {
  cat <<'EOF'
AGPT Parameter Calculator for TorchTitan Training

Usage:
  calculate_agpt_params.sh --model {2b|20b|80b} [OPTIONS]

Required:
  --model {2b|20b|80b}      Model size (determines TP/LBS defaults)

Optional:
  --nnodes N                Number of nodes (auto-detect from PBS_NODEFILE if not provided)
  --tp N                    Tensor parallelism (default: model-specific)
  --pp N                    Pipeline parallelism (default: 1)
  --cp N                    Context parallelism (default: 1)
  --lbs N                   Local batch size (default: model-specific)
  --gas N                   Gradient accumulation steps (default: 1)
  --seq-len N               Sequence length (default: 8192)
  --train-tokens N          Total token budget (default: 4.67T)
  --data-list NAME          Dataset list name (default: olmo-mix-1124)
  --output {shell|json}     Output format (default: shell)
  --help                    Show this help message
  --dry-run                 Print calculated values to stdout (don't export)

Output:
  Shell environment variables with AGPT_ prefix for use with run_train_torchtitan.sh
  Example: AGPT_MODEL, AGPT_NNODES, AGPT_GBS, AGPT_TRAINING_STEPS, etc.

Examples:
  # Calculate 2B on 8 nodes
  eval $(./calculate_agpt_params.sh --model 2b --nnodes 8)

  # In PBS job (auto-detect NNODES from PBS_NODEFILE)
  eval $(./calculate_agpt_params.sh --model 20b)

  # 80B with custom tensor parallelism
  eval $(./calculate_agpt_params.sh --model 80b --nnodes 32 --tp 4)

  # Dry-run to see calculated values
  ./calculate_agpt_params.sh --model 2b --nnodes 8 --dry-run

EOF
}

error() {
  echo "Error: $*" >&2
  exit 1
}

validate_positive_int() {
  local var_name="$1"
  local value="$2"

  if ! [[ "$value" =~ ^[0-9]+$ ]] || [[ $value -eq 0 ]]; then
    error "$var_name must be a positive integer, got: $value"
  fi
}

validate_model() {
  local model="$1"
  case "$model" in
    2b|20b|80b) ;;
    *) error "Unknown model '$model' (must be 2b, 20b, or 80b)" ;;
  esac
}

# Get model-specific defaults
get_model_defaults() {
  local model="$1"
  local -n tp_out="$2"
  local -n lbs_out="$3"

  case "$model" in
    2b)
      tp_out=1
      lbs_out=2
      ;;
    20b)
      tp_out=1
      lbs_out=2
      ;;
    80b)
      tp_out=2
      lbs_out=1
      ;;
    *)
      error "Unknown model: $model"
      ;;
  esac
}

# Parse command line arguments
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
    --data-list)
      DATA_LIST="$2"
      shift 2
      ;;
    --output)
      OUTPUT_FORMAT="$2"
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
    *)
      error "Unknown option: $1"
      ;;
  esac
done

# Validate required arguments
if [[ -z "$MODEL" ]]; then
  error "--model is required (2b, 20b, or 80b)"
fi

validate_model "$MODEL"

# Auto-detect or validate NNODES
if [[ -z "$NNODES" ]]; then
  if [[ -n "${PBS_NODEFILE:-}" && -f "${PBS_NODEFILE:-}" ]]; then
    NNODES=$(wc -l < "$PBS_NODEFILE")
  else
    error "--nnodes required (PBS_NODEFILE not found or not in PBS job)"
  fi
fi

validate_positive_int "NNODES" "$NNODES"

# Load model-specific defaults
MODEL_TP_DEFAULT=0
MODEL_LBS_DEFAULT=0
get_model_defaults "$MODEL" MODEL_TP_DEFAULT MODEL_LBS_DEFAULT

# Apply model defaults, then override with user-provided values
TP="${TP:-$MODEL_TP_DEFAULT}"
LBS="${LBS:-$MODEL_LBS_DEFAULT}"

# Validate parallelism parameters
validate_positive_int "TP" "$TP"
validate_positive_int "PP" "$PP"
validate_positive_int "CP" "$CP"
validate_positive_int "LBS" "$LBS"
validate_positive_int "GAS" "$GAS"
validate_positive_int "SEQ_LEN" "$SEQ_LEN"
validate_positive_int "TRAIN_TOKENS" "$TRAIN_TOKENS"

# Calculate derived values
NGPUS=$((NNODES * DEVICES_PER_NODE))

# Validate parallelism doesn't exceed GPU count
PARALLELISM_PRODUCT=$((TP * PP * CP))
if [[ $PARALLELISM_PRODUCT -gt $NGPUS ]]; then
  error "TP×PP×CP ($PARALLELISM_PRODUCT) exceeds total GPUs ($NGPUS)"
fi

# Calculate global batch size
# GBS = (NGPUS * LBS * GAS) / (TP * PP * CP)
GBS=$(( (NGPUS * LBS * GAS) / PARALLELISM_PRODUCT ))

# Calculate training steps from token budget
# TRAINING_STEPS = TRAIN_TOKENS / (GBS * SEQ_LEN)
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))

# Build checkpoint directory name
CKPT_DIR="agpt-${MODEL}-${OPTIMIZER}-${DATA_LIST}-n${NNODES}-gbs${GBS}"

# Output results
output_shell() {
  cat <<EOF
export AGPT_MODEL="$MODEL"
export AGPT_NNODES="$NNODES"
export AGPT_NGPUS="$NGPUS"
export AGPT_TP="$TP"
export AGPT_PP="$PP"
export AGPT_CP="$CP"
export AGPT_LBS="$LBS"
export AGPT_GAS="$GAS"
export AGPT_GBS="$GBS"
export AGPT_SEQ_LEN="$SEQ_LEN"
export AGPT_TRAINING_STEPS="$TRAINING_STEPS"
export AGPT_TRAIN_TOKENS="$TRAIN_TOKENS"
export AGPT_OPTIMIZER="$OPTIMIZER"
export AGPT_LR="$LR"
export AGPT_DATASET="$DATASET"
export AGPT_DATA_LIST="$DATA_LIST"
export AGPT_CKPT_DIR="$CKPT_DIR"
EOF
}

output_json() {
  cat <<EOF
{
  "model": "$MODEL",
  "nnodes": $NNODES,
  "ngpus": $NGPUS,
  "parallelism": {
    "tp": $TP,
    "pp": $PP,
    "cp": $CP
  },
  "batch": {
    "lbs": $LBS,
    "gas": $GAS,
    "gbs": $GBS
  },
  "training": {
    "seq_len": $SEQ_LEN,
    "steps": $TRAINING_STEPS,
    "tokens": $TRAIN_TOKENS
  },
  "optimizer": "$OPTIMIZER",
  "lr": "$LR",
  "dataset": "$DATASET",
  "data_list": "$DATA_LIST",
  "ckpt_dir": "$CKPT_DIR"
}
EOF
}

output_dry_run() {
  cat <<EOF
==========================================
AGPT Parameter Calculation Results
==========================================

Model Configuration:
  Model:                  $MODEL
  Number of Nodes:        $NNODES
  Total GPUs:             $NGPUS

Parallelism:
  Tensor Parallelism:     $TP
  Pipeline Parallelism:   $PP
  Context Parallelism:    $CP

Batch Size:
  Local Batch Size:       $LBS
  Gradient Accumulation:  $GAS
  Global Batch Size:      $GBS

Training:
  Sequence Length:        $SEQ_LEN
  Training Steps:         $TRAINING_STEPS
  Total Tokens:           $TRAIN_TOKENS

Optimizer & Data:
  Optimizer:              $OPTIMIZER
  Learning Rate:          $LR
  Dataset:                $DATASET
  Data List:              $DATA_LIST

Output:
  Checkpoint Directory:   $CKPT_DIR

==========================================
EOF
}

# Generate output
if [[ $DRY_RUN -eq 1 ]]; then
  output_dry_run
else
  case "$OUTPUT_FORMAT" in
    shell)
      output_shell
      ;;
    json)
      output_json
      ;;
    *)
      error "Unknown output format: $OUTPUT_FORMAT"
      ;;
  esac
fi
