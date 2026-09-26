#!/usr/bin/env bash
set -euo pipefail

# Training launcher script with automatic summary generation
# Usage: ./start_training.sh [options]
#
# This script:
# 1. Sets up the training environment
# 2. Starts the 5-hour training job
# 3. Monitors logs
# 4. Generates a summary document when complete

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
MODE="${TRAIN_MODE:-single}"
HOSTFILE="${TRAIN_HOSTFILE:-}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/xpu_torchtitan_$(date +%Y%m%d_%H%M%S)}"
TRAINING_STEPS="${TRAINING_STEPS:-18000}"  # ~5 hours at ~1 step/sec
DRY_RUN="${DRY_RUN:-0}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"  # seconds between checks

# Model configuration
MODEL_PATH="${MODEL_PATH:-}"
MODULE="${MODULE:-llama3}"
CONFIG="${CONFIG:-llama3_debugmodel}"
DATASET_NAME="${DATASET_NAME:-pg19_multinews}"

# Help message
usage() {
    cat <<'EOF'
Usage: ./start_training.sh [OPTIONS]

OPTIONS:
  --mode {single|multi}      Training mode (default: single)
  --hostfile FILE            Hostfile path (required for multi mode)
  --model-path PATH          Model/tokenizer directory (required)
  --module NAME              Config module name (default: llama3)
  --config NAME              Config callable (default: llama3_debugmodel)
  --dataset NAME             Dataset name (default: pg19_multinews)
  --training-steps N         Number of training steps (default: 18000 for ~5 hours)
  --log-dir DIR              Output directory (default: outputs/xpu_torchtitan_<timestamp>)
  --dry-run                  Print command without running
  --help                     Show this help message

ENVIRONMENT:
  All environment variables used by run_train_torchtitan.sh are supported.
  Examples:
    CKPT=/path/to/checkpoint
    SEQ_LEN=16384
    NPROC=36 (optional, auto-detected if not set)

  Note: NPROC_PER_NODE is auto-detected from XPU device count

EXAMPLE:
  ./start_training.sh --mode single \
    --model-path /path/to/llama-3.1-8b \
    --training-steps 5000

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --hostfile)
            HOSTFILE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --module)
            MODULE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dataset)
            DATASET_NAME="$2"
            shift 2
            ;;
        --training-steps)
            TRAINING_STEPS="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
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
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required"
    usage
    exit 1
fi

if [[ "$MODE" == "multi" && -z "$HOSTFILE" ]]; then
    if [[ -z "${PBS_NODEFILE:-}" ]]; then
        echo "Error: multi mode requires --hostfile or PBS_NODEFILE environment variable"
        usage
        exit 1
    fi
fi

# Prepare environment
export MODEL_PATH
export MODULE
export CONFIG
export DATASET_NAME
export TRAINING_STEPS="${TRAINING_STEPS}"
export LOG_DIR="${LOG_DIR}"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "Mode: $MODE"
echo "Model: $MODEL_PATH"
echo "Module/Config: $MODULE/$CONFIG"
echo "Dataset: $DATASET_NAME"
echo "Training Steps: $TRAINING_STEPS (estimated ~5 hours at 1 step/sec)"
echo "Log Directory: $LOG_DIR"
if [[ -n "$HOSTFILE" ]]; then
    echo "Hostfile: $HOSTFILE"
fi
echo ""

# Build training command
TRAIN_ARGS=(
    "--training.steps" "$TRAINING_STEPS"
)

# Add optional arguments if set
if [[ -n "${SEQ_LEN:-}" ]]; then
    TRAIN_ARGS+=("--training.seq_len" "$SEQ_LEN")
fi

if [[ -n "${CKPT:-}" ]]; then
    TRAIN_ARGS+=("--checkpoint.initial_load_path" "$CKPT")
fi

# Run training
if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY RUN] Would execute:"
    LAUNCHER_ROOT="$SCRIPT_DIR"
    if [[ "$MODE" == "single" ]]; then
        echo "$LAUNCHER_ROOT/xpu_torchtitan/run_train_torchtitan.sh single --dry-run -- ${TRAIN_ARGS[@]}"
    else
        echo "$LAUNCHER_ROOT/xpu_torchtitan/run_train_torchtitan.sh multi $HOSTFILE --dry-run -- ${TRAIN_ARGS[@]}"
    fi
    exit 0
fi

echo "Starting training..."
TRAIN_START=$(date +%s)

# Run the training
if [[ "$MODE" == "single" ]]; then
    "$SCRIPT_DIR/xpu_torchtitan/run_train_torchtitan.sh" single -- "${TRAIN_ARGS[@]}"
else
    "$SCRIPT_DIR/xpu_torchtitan/run_train_torchtitan.sh" multi "$HOSTFILE" -- "${TRAIN_ARGS[@]}"
fi

TRAIN_END=$(date +%s)
TRAIN_DURATION=$((TRAIN_END - TRAIN_START))

echo ""
echo "=========================================="
echo "Training Completed"
echo "=========================================="
echo "Duration: $(printf '%02d:%02d:%02d' $((TRAIN_DURATION / 3600)) $((TRAIN_DURATION % 3600 / 60)) $((TRAIN_DURATION % 60)))"
echo "Log directory: $LOG_DIR"

# Generate summary
echo ""
echo "Generating training summary..."
if command -v python3 >/dev/null 2>&1; then
    python3 "$SCRIPT_DIR/train_monitor.py" "$LOG_DIR"
else
    python "$SCRIPT_DIR/train_monitor.py" "$LOG_DIR"
fi

echo ""
echo "Summary generated: $LOG_DIR/TRAINING_SUMMARY.md"
