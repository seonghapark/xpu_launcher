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
  run_train_torchtitan.sh multi [hostfile] [--dry-run] [-- <extra torchtitan args...>]
                          (hostfile optional inside a PBS job: PBS_NODEFILE is used)

Core env variables:
  MODULE              (default: llama3)
  CONFIG              (default: llama3_debugmodel)
  HF_ASSETS_PATH      (default: ../torchtitan/tests/assets/tokenizer)
  DATASET_NAME        (default: pg19_multinews — PG19+MultiNews interleaved, streaming;
                       use c4_test for the offline bundled smoke sample)
  DATASET_PATH        (optional, default: empty)
  SEQ_LEN             (default: 16384, passed as --training.seq_len)
  LOG_DIR             (default: ../torchtitan/outputs/xpu_torchtitan_<timestamp>)
  CKPT_FOLDER         (default: checkpoint)
  TRAINING_STEPS      (default: 100)
  TORCHTITAN_ROOT     (default: <this script dir>/torchtitan_repo)

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
BASE_LAUNCH_SCRIPT="${SCRIPT_DIR}/../run_train.sh"

if [[ ! -x "$BASE_LAUNCH_SCRIPT" ]]; then
  echo "error: base launch wrapper not executable: $BASE_LAUNCH_SCRIPT" >&2
  echo "hint: chmod +x ${SCRIPT_DIR}/../run_train.sh" >&2
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
    # No hostfile given: run_train.sh derives one from PBS_NODEFILE
    if [[ -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE:-}" ]]; then
      echo "error: multi mode requires <hostfile> (or run inside a PBS job with PBS_NODEFILE)" >&2
      usage
      exit 1
    fi
  else
    if [[ ! -f "$HOSTFILE" ]]; then
      echo "error: hostfile not found: $HOSTFILE" >&2
      exit 1
    fi
    HOSTFILE="$(realpath "$HOSTFILE")"
  fi
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

TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-${SCRIPT_DIR}/torchtitan_repo}"
MODULE="${MODULE:-llama3}"
CONFIG="${CONFIG:-llama3_debugmodel}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-${TORCHTITAN_ROOT}/tests/assets/tokenizer}"
DATASET_NAME="${DATASET_NAME:-pg19_multinews}"
DATASET_PATH="${DATASET_PATH:-}"
LOG_DIR="${LOG_DIR:-${TORCHTITAN_ROOT}/outputs/xpu_torchtitan_$(date +%Y%m%d_%H%M%S)}"
CKPT_FOLDER="${CKPT_FOLDER:-checkpoint}"
TRAINING_STEPS="${TRAINING_STEPS:-100}"
SEQ_LEN="${SEQ_LEN:-16384}"
TRAIN_PYTHON_BIN_DEFAULT="/lus/flare/projects/datascience/seonghapark/venv/bin/python"
if [[ ! -x "$TRAIN_PYTHON_BIN_DEFAULT" ]] && [[ -x "${TORCHTITAN_ROOT}/.venv/bin/python" ]]; then
  TRAIN_PYTHON_BIN_DEFAULT="${TORCHTITAN_ROOT}/.venv/bin/python"
fi
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-$TRAIN_PYTHON_BIN_DEFAULT}"

mkdir -p "$LOG_DIR"

# titan_train.py = ezpz-free entry: FaultTolerantTrainer upgrade (FT_TRAINER=0
# to disable), --optimizer swap, IPEX + xccl split-group XPU workarounds
export TORCHTITAN_ROOT
export FT_TRAINER="${FT_TRAINER:-1}"
TRAIN_CMD=(
  "$TRAIN_PYTHON_BIN" "${SCRIPT_DIR}/titan_train.py"
  "--module" "$MODULE"
  "--config" "$CONFIG"
  "--hf_assets_path" "$HF_ASSETS_PATH"
  "--dump_folder" "$LOG_DIR"
  "--dataloader.dataset" "$DATASET_NAME"
  "--training.steps" "$TRAINING_STEPS"
  "--training.seq_len" "$SEQ_LEN"
  "--checkpoint.enable"
  "--checkpoint.folder" "$CKPT_FOLDER"
)

if [[ -n "${OPTIMIZER:-}" ]]; then
  TRAIN_CMD+=("--optimizer" "$OPTIMIZER")
fi

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

cat >&2 <<EOF
[ARGS] mode              = ${MODE}$( [[ "$MODE" == "multi" ]] && echo " (hostfile=${HOSTFILE:-auto from PBS_NODEFILE})" )
[ARGS] MODULE/CONFIG     = ${MODULE} / ${CONFIG}
[ARGS] DATASET_NAME      = ${DATASET_NAME}
[ARGS] DATASET_PATH      = ${DATASET_PATH:-<unset>}
[ARGS] TRAINING_STEPS    = ${TRAINING_STEPS}
[ARGS] SEQ_LEN           = ${SEQ_LEN}
[ARGS] OPTIMIZER         = ${OPTIMIZER:-<config default>}
[ARGS] FT_TRAINER        = ${FT_TRAINER}
[ARGS] TORCHTITAN_ROOT   = ${TORCHTITAN_ROOT}
[ARGS] HF_ASSETS_PATH    = ${HF_ASSETS_PATH}
[ARGS] LOG_DIR           = ${LOG_DIR}
[ARGS] CKPT_FOLDER       = ${CKPT_FOLDER}
[ARGS] TRAIN_PYTHON_BIN  = ${TRAIN_PYTHON_BIN}
[ARGS] topology          = NNODES=${NNODES:-auto} NPROC_PER_NODE=${NPROC_PER_NODE:-4} NPROC=${NPROC:-auto} SPARE_NODES=${SPARE_NODES:-auto} AUTO_RETRY=${AUTO_RETRY:-1(multi)}
[ARGS] extra train args  = ${EXTRA_ARGS[*]:-<none>}
EOF

CMD=("$BASE_LAUNCH_SCRIPT" "$MODE")
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=("--dry-run")
fi
if [[ "$MODE" == "multi" && -n "$HOSTFILE" ]]; then
  CMD+=("$HOSTFILE")
fi
CMD+=("--")
CMD+=("${TRAIN_CMD[@]}")

printf 'Running: '
printf '%q ' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
