#!/usr/bin/env bash
set -euo pipefail

# Training launcher with automatic model format conversion
# Converts distcp format models to HF format before training
#
# Usage:
#   ./start_training_with_conversion.sh --model-path /path/to/model [options]
#
# The script automatically:
# 1. Detects model format (distcp or HF)
# 2. Converts distcp → HF if needed
# 3. Starts training with the converted model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINING_SCRIPT="${SCRIPT_DIR}/start_training.sh"
CONVERTER_SCRIPT="${SCRIPT_DIR}/convert_model_format.py"

# Configuration
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_PATH=""
CONVERT_OUTPUT_DIR="${CONVERT_OUTPUT_DIR:-${SCRIPT_DIR}/converted_models}"
AUTO_CONVERT="${AUTO_CONVERT:-1}"
KEEP_CONVERTED="${KEEP_CONVERTED:-1}"
DRY_RUN="${DRY_RUN:-0}"
VERBOSE="${VERBOSE:-0}"

# Pass-through arguments for training script
TRAINING_ARGS=()

usage() {
    cat <<'EOF'
Usage: ./start_training_with_conversion.sh --model-path PATH [OPTIONS]

Starts model training with automatic format conversion (distcp → HF).

REQUIRED:
  --model-path PATH              Path to model (distcp or HF format)

OPTIONS:
  --output-dir DIR               Where to save converted models (default: ./converted_models)
  --no-auto-convert              Don't auto-convert, fail if distcp format
  --delete-converted             Delete converted model after training
  --dry-run                      Show what would be done
  --verbose                      Verbose output
  --help                         Show this help message

TRAINING OPTIONS (passed to start_training.sh):
  --mode {single|multi}          Training mode (default: single)
  --hostfile FILE                Hostfile for multi-node (optional if PBS_NODEFILE set)
  --module NAME                  TorchTitan module (default: llama3)
  --config NAME                  Config callable (default: llama3_debugmodel)
  --dataset NAME                 Dataset (default: pg19_multinews)
  --training-steps N             Training steps (default: 18000)
  --log-dir DIR                  Output directory
  --spare-nodes N                Spare nodes for failover

ENVIRONMENT VARIABLES:
  PYTHON_BIN                     Python binary (default: python3)
  CONVERT_OUTPUT_DIR             Converted models directory
  AUTO_CONVERT                   Auto-convert models (default: 1)
  KEEP_CONVERTED                 Keep converted model after training (default: 1)
  PBS_NODEFILE                   PBS job node list (auto-used for multi-node without --hostfile)

EXAMPLES:
  # Auto-convert distcp model and train (single node)
  ./start_training_with_conversion.sh --model-path /path/to/model.dcp

  # Train HF model without conversion (single node)
  ./start_training_with_conversion.sh --model-path /path/to/model.hf

  # Multi-node in PBS job (uses PBS_NODEFILE automatically)
  ./start_training_with_conversion.sh \
    --model-path /path/to/model \
    --mode multi

  # Multi-node with training parameters
  ./start_training_with_conversion.sh \
    --model-path /path/to/model \
    --mode multi \
    -- --training.steps 400000 --training.seq_len 16384

  # Multi-node with explicit hostfile
  ./start_training_with_conversion.sh \
    --model-path /path/to/model \
    --mode multi \
    --hostfile nodes.txt

  # Preview conversion without training
  ./start_training_with_conversion.sh --model-path /path/to/model --dry-run
EOF
}

# Parse arguments
HOSTFILE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --output-dir)
            CONVERT_OUTPUT_DIR="$2"
            shift 2
            ;;
        --no-auto-convert)
            AUTO_CONVERT=0
            shift
            ;;
        --delete-converted)
            KEEP_CONVERTED=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --hostfile)
            HOSTFILE="$2"
            TRAINING_ARGS+=("$1" "$2")
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        --)
            # Everything after -- goes to training script
            shift
            TRAINING_ARGS+=("--" "$@")
            break
            ;;
        *)
            # Pass through to training script
            TRAINING_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model-path is required"
    usage
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

# Resolve paths
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
CONVERT_OUTPUT_DIR="$(mkdir -p "$CONVERT_OUTPUT_DIR" && cd "$CONVERT_OUTPUT_DIR" && pwd)"

echo "=========================================="
echo "Training with Model Format Conversion"
echo "=========================================="
echo "Model Path:    $MODEL_PATH"
echo "Auto-Convert:  $AUTO_CONVERT"
echo "Output Dir:    $CONVERT_OUTPUT_DIR"
echo ""

# Check if model format converter exists
if [[ ! -f "$CONVERTER_SCRIPT" ]]; then
    echo "Error: Model converter script not found: $CONVERTER_SCRIPT"
    echo "Please ensure convert_model_format.py is in the launcher directory"
    exit 1
fi

# Detect model format
echo "[INFO] Detecting model format..."
FORMAT_CHECK=$("$PYTHON_BIN" "$CONVERTER_SCRIPT" --check-format "$MODEL_PATH" 2>&1 || true)

if echo "$FORMAT_CHECK" | grep -q "Hugging Face format"; then
    echo "[INFO] Model is already in Hugging Face format"
    FINAL_MODEL_PATH="$MODEL_PATH"
    NEEDS_CONVERSION=0
elif echo "$FORMAT_CHECK" | grep -q "DCP format"; then
    echo "[INFO] Model is in DCP format (needs conversion)"
    NEEDS_CONVERSION=1
else
    echo "[WARNING] Could not detect model format"
    NEEDS_CONVERSION=0
    FINAL_MODEL_PATH="$MODEL_PATH"
fi

# Convert if needed
if [[ $NEEDS_CONVERSION -eq 1 ]]; then
    if [[ $AUTO_CONVERT -eq 0 ]]; then
        echo "Error: Model is in distcp format but auto-convert is disabled"
        echo "Use --model-path with HF format model or enable auto-conversion"
        exit 1
    fi

    MODEL_NAME=$(basename "$MODEL_PATH")
    CONVERTED_MODEL="${CONVERT_OUTPUT_DIR}/${MODEL_NAME}_hf"

    echo ""
    echo "=========================================="
    echo "Converting Model Format"
    echo "=========================================="
    echo "From: $MODEL_PATH"
    echo "To:   $CONVERTED_MODEL"
    echo ""

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] Would execute:"
        echo "$PYTHON_BIN $CONVERTER_SCRIPT --dcp \"$MODEL_PATH\" --output \"$CONVERTED_MODEL\""
        if [[ $VERBOSE -eq 1 ]]; then
            echo ""
            echo "[DRY RUN] Then would train with converted model:"
            echo "$TRAINING_SCRIPT --model-path \"$CONVERTED_MODEL\" ${TRAINING_ARGS[@]}"
            echo ""
            if [[ -z "$HOSTFILE" ]] && grep -q "mode.*multi" <<< "${TRAINING_ARGS[@]:-}"; then
                echo "[DRY RUN] Multi-mode without explicit hostfile: will use PBS_NODEFILE if available"
            fi
        fi
        exit 0
    fi

    # Run conversion
    CONVERTER_ARGS=(
        "$CONVERTER_SCRIPT"
        "--dcp" "$MODEL_PATH"
        "--output" "$CONVERTED_MODEL"
    )
    [[ $VERBOSE -eq 1 ]] && CONVERTER_ARGS+=(--verbose)

    echo "[INFO] Running conversion..."
    if ! "$PYTHON_BIN" "${CONVERTER_ARGS[@]}"; then
        echo "Error: Model conversion failed"
        exit 1
    fi

    echo ""
    echo "[INFO] Conversion successful"
    FINAL_MODEL_PATH="$CONVERTED_MODEL"

    # Register for cleanup
    CONVERTED_MODEL_TO_DELETE="$FINAL_MODEL_PATH"
else
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY RUN] Model already in HF format, would execute:"
        echo "$TRAINING_SCRIPT --model-path \"$MODEL_PATH\" ${TRAINING_ARGS[@]}"
        echo ""
        if [[ -z "$HOSTFILE" ]] && grep -q "mode.*multi" <<< "${TRAINING_ARGS[@]:-}"; then
            echo "[DRY RUN] Multi-mode without explicit hostfile: will use PBS_NODEFILE if available"
        fi
        exit 0
    fi
    CONVERTED_MODEL_TO_DELETE=""
fi

# Validate multi-mode requirements
if grep -q "mode.*multi" <<< "${TRAINING_ARGS[@]:-}"; then
    if [[ -z "$HOSTFILE" && -z "${PBS_NODEFILE:-}" ]]; then
        echo "Error: Multi-mode requires either --hostfile or to be run inside a PBS job (PBS_NODEFILE)"
        exit 1
    fi
fi

# Start training
echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Model:         $FINAL_MODEL_PATH"
echo "Training Args: ${TRAINING_ARGS[*]:-<none>}"
if [[ -z "$HOSTFILE" ]] && grep -q "mode.*multi" <<< "${TRAINING_ARGS[@]:-}"; then
    echo "Hostfile:      Using PBS_NODEFILE from job environment"
fi
echo ""

# Trap to cleanup converted model if requested
cleanup() {
    if [[ -n "$CONVERTED_MODEL_TO_DELETE" && $KEEP_CONVERTED -eq 0 ]]; then
        echo ""
        echo "[INFO] Cleaning up converted model: $CONVERTED_MODEL_TO_DELETE"
        rm -rf "$CONVERTED_MODEL_TO_DELETE"
    fi
}
trap cleanup EXIT

# Run training
TRAINING_CMD=(
    "$TRAINING_SCRIPT"
    "--model-path" "$FINAL_MODEL_PATH"
    "${TRAINING_ARGS[@]}"
)

exec "${TRAINING_CMD[@]}"
