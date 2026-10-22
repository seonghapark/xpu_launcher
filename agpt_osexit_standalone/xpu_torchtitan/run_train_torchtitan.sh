#!/usr/bin/env bash
set -euo pipefail

# TorchTitan training template on top of xpu_launch/run_train.sh
#
# Examples:
#   MODEL=/path/to/hf/model ./run_train_torchtitan.sh single --dry-run
#   MODEL=/path/to/hf/model ./run_train_torchtitan.sh multi /path/to/hosts --dry-run
#   MODEL=/path/to/hf/Llama-3.1-8B MODULE=llama3 CONFIG=llama3_8b \
#     DATASET_NAME=c4 DATASET_PATH=allenai/c4 LOG_DIR=/path/logs \
#     ./run_train_torchtitan.sh multi /path/to/hosts -- --training.steps 2000

usage() {
  cat <<'EOF'
Usage:
  run_train_torchtitan.sh single [--dry-run] [--resource-monitor] [-- <extra torchtitan args...>]
  run_train_torchtitan.sh multi [hostfile] [--dry-run] [--resource-monitor] [-- <extra torchtitan args...>]
                          (hostfile optional inside a PBS job: PBS_NODEFILE is used)

Core env variables:
  MODEL               Path to the model/tokenizer assets directory
                      (forwarded to torchtitan as --hf_assets_path)
  MODEL_PATH          Compatibility alias for MODEL (MODEL takes precedence)
  MODULE              (default: llama3; torchtitan config-registry module)
  CONFIG              (default: llama3_debugmodel; callable in that module's config_registry.py)
  HF_ASSETS_PATH      (optional override; default: MODEL)
  DATASET_NAME        (default: pg19_multinews — PG19+MultiNews interleaved, streaming;
                       use c4_test for the offline bundled smoke sample)
  DATASET_PATH        (optional, default: empty)
  SEQ_LEN             (default: 16384, passed as --training.seq_len)
  LOG_DIR             (default: ../torchtitan/outputs/xpu_torchtitan_<timestamp>)
  CKPT_FOLDER         (default: checkpoint)
  TRAINING_STEPS      (default: 100)
  TORCHTITAN_ROOT     (default: <this script dir>/torchtitan_repo)
  RESOURCE_MONITOR    (default: 0; set to 1 or pass --resource-monitor)
  RESOURCE_INTERVAL   (default: 5 seconds)
  RESOURCE_OUTPUT_DIR (default: LOG_DIR/resource_metrics)
  LOSS_STD_TERMINATION_ENABLED (default: 0; set to 1 to enable early termination on loss convergence)
  LOSS_STD_THRESHOLD  (default: 0.001; threshold for loss std convergence)
  LOSS_STD_WINDOW     (default: 50; window size for computing loss standard deviation)

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
RESOURCE_MONITOR="${RESOURCE_MONITOR:-0}"
RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-5}"
RESOURCE_OUTPUT_DIR="${RESOURCE_OUTPUT_DIR:-}"
LOSS_STD_TERMINATION_ENABLED="${LOSS_STD_TERMINATION_ENABLED:-0}"
LOSS_STD_THRESHOLD="${LOSS_STD_THRESHOLD:-0.001}"
LOSS_STD_WINDOW="${LOSS_STD_WINDOW:-50}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --resource-monitor)
      RESOURCE_MONITOR=1
      shift
      ;;
    --resource-interval)
      if [[ $# -lt 2 ]]; then
        echo "error: --resource-interval requires seconds" >&2
        exit 1
      fi
      RESOURCE_INTERVAL="$2"
      shift 2
      ;;
    --resource-output-dir)
      if [[ $# -lt 2 ]]; then
        echo "error: --resource-output-dir requires a path" >&2
        exit 1
      fi
      RESOURCE_OUTPUT_DIR="$2"
      shift 2
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

TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-${SCRIPT_DIR}/torchtitan_repo}"
# MODEL: required path to model/tokenizer assets, forwarded as --hf_assets_path
MODEL="${MODEL:-${MODEL_PATH:-}}"
if [[ -z "${MODEL:-}" ]]; then
  echo "error: MODEL (or MODEL_PATH) is required: path to the model/tokenizer assets directory" >&2
  echo "       e.g. MODEL=${TORCHTITAN_ROOT}/tests/assets/tokenizer" >&2
  usage
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "error: MODEL path not found: $MODEL" >&2
  exit 1
fi
MODEL="$(realpath "$MODEL")"
MODULE="${MODULE:-llama3}"
CONFIG="${CONFIG:-llama3_debugmodel}"
HF_ASSETS_PATH="${HF_ASSETS_PATH:-${MODEL}}"
# AGPT_* fallbacks let calculate_agpt_params.sh output flow in unchanged; the
# built-in default stays the streaming pg19_multinews corpus.
DATASET_NAME="${DATASET_NAME:-${AGPT_DATASET:-pg19_multinews}}"
DATASET_PATH="${DATASET_PATH:-${AGPT_DATASET_PATH:-}}"
LOG_DIR="${LOG_DIR:-${TORCHTITAN_ROOT}/outputs/xpu_torchtitan_$(date +%Y%m%d_%H%M%S)}"
RESOURCE_OUTPUT_DIR="${RESOURCE_OUTPUT_DIR:-${LOG_DIR}/resource_metrics}"
CKPT_FOLDER="${CKPT_FOLDER:-checkpoint}"
# Checkpoint every CKPT_INTERVAL steps; keep only the CKPT_KEEP_LATEST_K most
# recent ones. 0 keeps every checkpoint. AGPT_* names let the values flow
# straight from calculate_agpt_params.sh.
CKPT_INTERVAL="${CKPT_INTERVAL:-${AGPT_CKPT_INTERVAL:-100}}"
CKPT_KEEP_LATEST_K="${CKPT_KEEP_LATEST_K:-${AGPT_CKPT_KEEP_LATEST_K:-0}}"
TRAINING_STEPS="${TRAINING_STEPS:-100}"
SEQ_LEN="${SEQ_LEN:-16384}"

# Resolve the training interpreter. Order matters: an explicitly activated
# virtualenv first, then a .venv beside the checkout, then the site venv that
# actually has torch+torchtitan installed. A bare python3 from PATH is the last
# resort — on Aurora login nodes that resolves to /usr/bin/python3, which has no
# torch, so preferring it would break every launch. Set TRAIN_PYTHON_BIN or
# TORCHTITAN_VENV to override.
TORCHTITAN_VENV="${TORCHTITAN_VENV:-/lus/flare/projects/datascience/seonghapark/venv}"
resolve_train_python() {
  local candidate
  for candidate in \
    "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}" \
    "${TORCHTITAN_ROOT}/.venv/bin/python" \
    "${SCRIPT_DIR}/.venv/bin/python" \
    "${TORCHTITAN_VENV}/bin/python"
  do
    [[ -n "$candidate" && -x "$candidate" ]] && { echo "$candidate"; return 0; }
  done

  command -v python3 2>/dev/null || echo python3
}
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-$(resolve_train_python)}"

# Optimizer / learning rate. torchtitan has no flat --optimizer.lr: the
# optimizer is a list of parameter groups, each with its own name and kwargs,
# so the flags are indexed (group 0 is AGPT's catch-all ".*" group).
# Unset by default, so the model registry's own choice stands.
OPTIMIZER_NAME="${OPTIMIZER_NAME:-${AGPT_OPTIMIZER:-}}"
LEARNING_RATE="${LEARNING_RATE:-${AGPT_LR:-}}"
OPTIMIZER_PARAM_GROUP="${OPTIMIZER_PARAM_GROUP:-0}"

mkdir -p "$LOG_DIR"

# titan_train.py normalizes MPI/PBS environment variables, installs the XCCL
# workaround when needed, and delegates to torchtitan.train.main().
export TORCHTITAN_ROOT
TRAIN_CMD=(
  "$TRAIN_PYTHON_BIN" "${SCRIPT_DIR}/titan_train.py"
  "--module" "$MODULE"
  "--config" "$CONFIG"
  "--hf_assets_path" "$HF_ASSETS_PATH"
  "--dump_folder" "$LOG_DIR"
  "--dataloader.dataset" "$DATASET_NAME"
  "--checkpoint.enable"
  "--checkpoint.folder" "$CKPT_FOLDER"
)

if ! has_extra_arg "--training.steps"; then
  TRAIN_CMD+=("--training.steps" "$TRAINING_STEPS")
fi
if ! has_extra_arg "--training.seq_len"; then
  TRAIN_CMD+=("--training.seq_len" "$SEQ_LEN")
fi
if ! has_extra_arg "--training.enable_loss_std_termination"; then
  if [[ "$LOSS_STD_TERMINATION_ENABLED" == "1" ]]; then
    TRAIN_CMD+=("--training.enable_loss_std_termination")
  fi
fi
if ! has_extra_arg "--training.loss_std_threshold"; then
  TRAIN_CMD+=("--training.loss_std_threshold=$LOSS_STD_THRESHOLD")
fi
if ! has_extra_arg "--training.loss_std_window"; then
  TRAIN_CMD+=("--training.loss_std_window=$LOSS_STD_WINDOW")
fi

if ! has_extra_arg "--checkpoint.interval"; then
  TRAIN_CMD+=("--checkpoint.interval" "$CKPT_INTERVAL")
fi
# keep_latest_k=0 disables purging entirely (torchtitan only deletes when
# keep_latest_k > 0), i.e. every checkpoint is kept.
if ! has_extra_arg "--checkpoint.keep_latest_k"; then
  TRAIN_CMD+=("--checkpoint.keep_latest_k" "$CKPT_KEEP_LATEST_K")
fi

_OPT_PREFIX="--optimizer.param-groups.${OPTIMIZER_PARAM_GROUP}"
if [[ -n "$OPTIMIZER_NAME" ]] && ! has_extra_arg "${_OPT_PREFIX}.optimizer-name"; then
  TRAIN_CMD+=("${_OPT_PREFIX}.optimizer-name" "$OPTIMIZER_NAME")
fi
if [[ -n "$LEARNING_RATE" ]] && ! has_extra_arg "${_OPT_PREFIX}.optimizer-kwargs.lr"; then
  TRAIN_CMD+=("${_OPT_PREFIX}.optimizer-kwargs.lr" "$LEARNING_RATE")
fi

if [[ -n "$DATASET_PATH" ]]; then
  TRAIN_CMD+=("--dataloader.dataset_path" "$DATASET_PATH")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "$RESOURCE_MONITOR" == "1" ]]; then
  RESOURCE_MONITOR_SCRIPT="${SCRIPT_DIR}/../src/xpu_launch/resource_monitor.py"
  if [[ ! -f "$RESOURCE_MONITOR_SCRIPT" ]]; then
    echo "error: resource monitor not found: $RESOURCE_MONITOR_SCRIPT" >&2
    exit 1
  fi
  TRAIN_CMD=(
    "$TRAIN_PYTHON_BIN" "$RESOURCE_MONITOR_SCRIPT"
    "--output-dir" "$RESOURCE_OUTPUT_DIR"
    "--interval" "$RESOURCE_INTERVAL"
    "${TRAIN_CMD[@]}"
  )
fi

if [[ ! -d "$TORCHTITAN_ROOT" ]]; then
  echo "error: TORCHTITAN_ROOT not found: $TORCHTITAN_ROOT" >&2
  exit 1
fi

cd "$TORCHTITAN_ROOT"

TRAINING_STEPS_NOTE=""
SEQ_LEN_NOTE=""
if has_extra_arg "--training.steps"; then
  TRAINING_STEPS_NOTE=" (overridden by extra args)"
fi
if has_extra_arg "--training.seq_len"; then
  SEQ_LEN_NOTE=" (overridden by extra args)"
fi

cat >&2 <<EOF
================================================================================
TRAINING LAUNCH CONFIGURATION
================================================================================

[LAUNCH MODE]
  Mode                   = ${MODE}$( [[ "$MODE" == "multi" ]] && echo " (hostfile=${HOSTFILE:-auto from PBS_NODEFILE})" )
  Topology               = NNODES=${NNODES:-auto} NPROC_PER_NODE=${NPROC_PER_NODE:-4} NPROC=${NPROC:-auto}
  Auto Retry             = ${AUTO_RETRY:-1(multi)}
  Spare Nodes            = ${SPARE_NODES:-auto}

[MODEL & DATASET]
  Model Path             = ${MODEL}
  Module/Config          = ${MODULE} / ${CONFIG}
  Optimizer              = ${OPTIMIZER_NAME:-<registry default>}
  Learning Rate          = ${LEARNING_RATE:-<registry default>}
  Dataset Name           = ${DATASET_NAME}
  Dataset Path           = ${DATASET_PATH:-<unset>}
  HF Assets Path         = ${HF_ASSETS_PATH}

[TRAINING HYPERPARAMETERS]
  Training Steps         = ${TRAINING_STEPS}${TRAINING_STEPS_NOTE}
  Sequence Length        = ${SEQ_LEN}${SEQ_LEN_NOTE}
  Loss Std Termination   = ${LOSS_STD_TERMINATION_ENABLED} (threshold: ${LOSS_STD_THRESHOLD}, window: ${LOSS_STD_WINDOW})

[SYSTEM & PATHS]
  TorchTitan Root        = ${TORCHTITAN_ROOT}
  Log Directory          = ${LOG_DIR}
  Checkpoint Folder      = ${CKPT_FOLDER}
  Checkpoint Interval    = every ${CKPT_INTERVAL} steps
  Checkpoint Keep Last K = ${CKPT_KEEP_LATEST_K}$([[ ${CKPT_KEEP_LATEST_K} -eq 0 ]] && echo " (keep all)")
  Checkpoint Load        = ${CKPT:-<unset>}
  Python Binary          = ${TRAIN_PYTHON_BIN}

[MONITORING]
  Resource Monitor       = ${RESOURCE_MONITOR}
  Resource Interval      = ${RESOURCE_INTERVAL}s
  Resource Output Dir    = ${RESOURCE_OUTPUT_DIR}

[EXTRA ARGUMENTS]
  Additional Args        = ${EXTRA_ARGS[*]:-<none>}

================================================================================
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
