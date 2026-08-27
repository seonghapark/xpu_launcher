#!/usr/bin/env bash
set -euo pipefail

# Generic (non-TorchTitan) training template on top of xpu_launch/run_train.sh
#
# Examples:
#   ./run_train_non_torchtitan.sh single --dry-run
#   ./run_train_non_torchtitan.sh multi /path/to/hosts --dry-run
#   MODEL_PATH=/path/model DATASET_PATH=/path/data LOG_DIR=/path/logs \
#     ./run_train_non_torchtitan.sh multi /path/to/hosts -- --epochs 3 --lr 1e-4

usage() {
  cat <<'EOF'
Usage:
  run_train_non_torchtitan.sh single [--dry-run] [-- <extra train args...>]
  run_train_non_torchtitan.sh multi <hostfile> [--dry-run] [-- <extra train args...>]

Core env variables:
  TRAIN_ENTRY         (default: train.py)
  PYTHON_BIN          (default: python)
  MODEL_PATH          (default: /path/to/model)
  DATASET_PATH        (default: /path/to/dataset)
  LOG_DIR             (default: ./outputs/xpu_generic_<timestamp>)

The default training command is:
  python ${TRAIN_ENTRY} --model-path ${MODEL_PATH} --dataset-path ${DATASET_PATH} --log-dir ${LOG_DIR}

Launch env variables (handled by run_train.sh):
  XPU_CMD, SCHEDULER, NPROC_PER_NODE, NNODES, NPROC,
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

TRAIN_ENTRY="${TRAIN_ENTRY:-train.py}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/path/to/model}"
DATASET_PATH="${DATASET_PATH:-/path/to/dataset}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/xpu_generic_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"

TRAIN_CMD=(
  "$PYTHON_BIN" "$TRAIN_ENTRY"
  "--model-path" "$MODEL_PATH"
  "--dataset-path" "$DATASET_PATH"
  "--log-dir" "$LOG_DIR"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

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
