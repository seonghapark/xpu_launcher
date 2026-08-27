#!/usr/bin/env bash
set -euo pipefail

# TorchTitan training template on top of xpu_launch/run_train.sh
#
# Examples:
#   ./run_train_torchtitan.sh single --dry-run
#   ./run_train_torchtitan.sh multi /path/to/hosts --dry-run
#   MODULE=llama3 CONFIG=llama3_8b HF_ASSETS_PATH=/path/hf/Llama-3.1-8B \
#     DATASET_NAME=c4 DATASET_PATH=allenai/c4 LOG_DIR=/path/logs \
#     ./run_train_torchtitan.sh multi /path/to/hosts -- --training.steps 2000

usage() {
  cat <<'EOF'
Usage:
  run_train_torchtitan.sh single [--dry-run] [-- <extra torchtitan args...>]
  run_train_torchtitan.sh multi <hostfile> [--dry-run] [-- <extra torchtitan args...>]

Core env variables:
  MODULE              (default: llama3)
  CONFIG              (default: llama3_debugmodel)
  HF_ASSETS_PATH      (default: ../torchtitan/tests/assets/tokenizer)
  DATASET_NAME        (default: c4_test)
  DATASET_PATH        (optional, default: empty)
  LOG_DIR             (default: ../torchtitan/outputs/xpu_torchtitan_<timestamp>)
  CKPT_FOLDER         (default: checkpoint)
  TRAINING_STEPS      (default: 100)
  TORCHTITAN_ROOT     (default: ../torchtitan)

Launch env variables (handled by run_train.sh):
  XPU_CMD, PYTHON_BIN, SCHEDULER, NPROC_PER_NODE, NNODES, NPROC,
  AUTO_RETRY, SPARE_NODES, FAILOVER_PROFILE, HOST_IP_MAP
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCH_SCRIPT="${SCRIPT_DIR}/run_train.sh"

if [[ ! -x "$BASE_LAUNCH_SCRIPT" ]]; then
  echo "error: base launch wrapper not executable: $BASE_LAUNCH_SCRIPT" >&2
  echo "hint: chmod +x ${SCRIPT_DIR}/run_train.sh" >&2
  exit 1
fi

MODE="$1"
shift

DRY_RUN=0
HOSTFILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      break
      ;;
    -* )
      echo "error: unknown option: $1" >&2
      usage
      exit 1
      ;;
    *)
      if [[ "$MODE" == "multi" && -z "$HOSTFILE" ]]; then
        HOSTFILE="$1"
        shift
      else
        break
      fi
      ;;
  esac
done

if [[ "$MODE" == "multi" ]]; then
  if [[ -z "$HOSTFILE" ]]; then
    echo "error: multi mode requires <hostfile>" >&2
    usage
    exit 1
  fi
  if [[ ! -f "$HOSTFILE" ]]; then
    echo "error: hostfile not found: $HOSTFILE" >&2
    exit 1
  fi
  HOSTFILE="$(realpath "$HOSTFILE")"
elif [[ "$MODE" != "single" ]]; then
  echo "error: mode must be 'single' or 'multi'" >&2
  usage
  exit 1
fi

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
  shift
  EXTRA_ARGS=("$@")
fi

TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-${SCRIPT_DIR}/../torchtitan}"
MODULE="${MODULE:-llama3}"
CONFIG="${CONFIG:-llama3_debugmodel}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-${TORCHTITAN_ROOT}/tests/assets/tokenizer}"
DATASET_NAME="${DATASET_NAME:-c4_test}"
DATASET_PATH="${DATASET_PATH:-}"
LOG_DIR="${LOG_DIR:-${TORCHTITAN_ROOT}/outputs/xpu_torchtitan_$(date +%Y%m%d_%H%M%S)}"
CKPT_FOLDER="${CKPT_FOLDER:-checkpoint}"
TRAINING_STEPS="${TRAINING_STEPS:-100}"
if [[ -x "${TORCHTITAN_ROOT}/.venv/bin/python" ]]; then
  TRAIN_PYTHON_BIN_DEFAULT="${TORCHTITAN_ROOT}/.venv/bin/python"
else
  TRAIN_PYTHON_BIN_DEFAULT="python"
fi
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-$TRAIN_PYTHON_BIN_DEFAULT}"

mkdir -p "$LOG_DIR"

TRAIN_CMD=(
  "$TRAIN_PYTHON_BIN" "-m" "torchtitan.train"
  "--module" "$MODULE"
  "--config" "$CONFIG"
  "--hf_assets_path" "$HF_ASSETS_PATH"
  "--dump_folder" "$LOG_DIR"
  "--dataloader.dataset" "$DATASET_NAME"
  "--training.steps" "$TRAINING_STEPS"
  "--checkpoint.enable"
  "--checkpoint.folder" "$CKPT_FOLDER"
)

if [[ -n "$DATASET_PATH" ]]; then
  TRAIN_CMD+=("--dataloader.dataset_path" "$DATASET_PATH")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ ! -d "$TORCHTITAN_ROOT" ]]; then
  echo "error: TORCHTITAN_ROOT not found: $TORCHTITAN_ROOT" >&2
  exit 1
fi

cd "$TORCHTITAN_ROOT"

CMD=("$BASE_LAUNCH_SCRIPT" "$MODE")
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=("--dry-run")
fi
if [[ "$MODE" == "multi" ]]; then
  CMD+=("$HOSTFILE")
fi
CMD+=("--")
CMD+=("${TRAIN_CMD[@]}")

printf 'Running: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
