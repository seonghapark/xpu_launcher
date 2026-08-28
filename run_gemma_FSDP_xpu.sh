#!/usr/bin/env bash
set -euo pipefail

# FSDP2 Gemma train/smoke wrapper using xpu_launcher.
# This script configures Aurora XPU/IPEX/oneCCL-related runtime variables, then
# delegates launch construction to ./run_train_non_torchtitan.sh.
#
# Examples:
#   ./run_gemma_FSDP_xpu.sh single --dry-run
#   ./run_gemma_FSDP_xpu.sh single
#   ./run_gemma_FSDP_xpu.sh multi /path/to/hosts --dry-run

usage() {
  cat <<'EOF'
Usage:
  run_gemma_FSDP_xpu.sh single [--dry-run] [-- <extra train args...>]
  run_gemma_FSDP_xpu.sh multi <hostfile> [--dry-run] [-- <extra train args...>]

Defaults:
  MODEL_PATH=/lus/flare/projects/datascience/seonghapark/torchtitan/assets/hf/gemma-7b
  DATASET_PATH=pg19,multi_news
  DATASET_CACHE_DIR=/lus/flare/projects/datascience/seonghapark/llm_evaluation/evaluation/datasets/hf_cache
  DATASET_MAX_SAMPLES=64
  DATASET_STREAMING=1
  VALIDATION_DATASET_PATH=${DATASET_PATH}
  VALIDATION_DATASET_SPLIT=validation
  VALIDATION_EVERY=1
  LOG_DIR=./outputs/gemma_non_torchtitan_xpu_<timestamp>
  WANDB_PROJECT=xpu-non-torchtitan-gemma
  WANDB_MODE=online
  SAVE_AFTER_MINUTES=45
  CHECKPOINT_DIR=${LOG_DIR}/checkpoints
  TRAIN_STEPS=158000
  SEQ_LEN=16384
  BATCH_SIZE=4
  MICRO_BATCH_SIZE=1
  LOGIT_CHUNK_SIZE=256
  TRAIN_MODE=lm_head
  NPROC_PER_NODE=1

Runtime env configured here:
  LOAD_FRAMEWORKS=1
  CCL_PROCESS_LAUNCHER=hydra
  ZE_FLAT_DEVICE_HIERARCHY=FLAT
  CCL_ATL_TRANSPORT=ofi
  FI_PROVIDER=cxi
  FI_MR_CACHE_MONITOR=memhooks
  TORCH_LLM_ALLREDUCE=1
  ENABLE_SDP_FUSION=1
  PYTHONMULTIPROCESSINGMETHOD=spawn
  VLLM_WORKER_MULTIPROC_METHOD=spawn
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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

USER_CCL_PROCESS_LAUNCHER="${CCL_PROCESS_LAUNCHER:-}"
USER_ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-}"

# Mirror the Aurora runtime setup used by llm_evaluation/run_lm-evaluation-harness_vllm.sh.
if [[ "${LOAD_FRAMEWORKS:-1}" == "1" ]]; then
  set +u
  if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/modules.sh ]]; then
      # shellcheck disable=SC1091
      source /etc/profile.d/modules.sh
    fi
  fi
  if type module >/dev/null 2>&1; then
    module -q load frameworks
  else
    echo "warning: environment module command not available; skipping 'module -q load frameworks'" >&2
  fi
  set -u
fi

# Aurora XPU / IPEX / oneCCL runtime defaults. User-provided values win.
export CCL_PROCESS_LAUNCHER="${USER_CCL_PROCESS_LAUNCHER:-hydra}"
export ZE_FLAT_DEVICE_HIERARCHY="${USER_ZE_FLAT_DEVICE_HIERARCHY:-FLAT}"
export CCL_ATL_TRANSPORT="${CCL_ATL_TRANSPORT:-ofi}"
export FI_PROVIDER="${FI_PROVIDER:-cxi}"
export FI_MR_CACHE_MONITOR="${FI_MR_CACHE_MONITOR:-memhooks}"
export TORCH_LLM_ALLREDUCE="${TORCH_LLM_ALLREDUCE:-1}"
export ENABLE_SDP_FUSION="${ENABLE_SDP_FUSION:-1}"
export PYTHONMULTIPROCESSINGMETHOD="${PYTHONMULTIPROCESSINGMETHOD:-spawn}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-/lus/flare/projects/datascience/seonghapark/llm_evaluation/evaluation/models/cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/lus/flare/projects/datascience/seonghapark/llm_evaluation/evaluation/datasets/hf_cache}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HTTP_PROXY="${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

# Launch defaults. Keep this smoke run conservative unless the caller overrides.
export SCHEDULER="${SCHEDULER:-auto}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export AUTO_RETRY="${AUTO_RETRY:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/lus/flare/projects/datascience/seonghapark/llm_evaluation/venv/bin/python}"
export XPU_CMD="${XPU_CMD:-xpu}"

export TRAIN_ENTRY="${TRAIN_ENTRY:-${SCRIPT_DIR}/non_torchtitan_gemma_train_FSDP.py}"
export MODEL_PATH="${MODEL_PATH:-/lus/flare/projects/datascience/seonghapark/torchtitan/assets/hf/gemma-7b}"
export DATASET_PATH="${DATASET_PATH:-pg19,multi_news}"
export LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/outputs/gemma_FSDP_pg19_multinews_bsz4_steps158000_seq16384_$(date +%Y%m%d_%H%M%S)}"

TRAIN_STEPS="${TRAIN_STEPS:-158000}"
SEQ_LEN="${SEQ_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
LOGIT_CHUNK_SIZE="${LOGIT_CHUNK_SIZE:-256}"
TRAIN_MODE="${TRAIN_MODE:-lm_head}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-xpu}"
LR="${LR:-1e-5}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/lus/flare/projects/datascience/seonghapark/llm_evaluation/evaluation/datasets/hf_cache}"
DATASET_MAX_SAMPLES="${DATASET_MAX_SAMPLES:-256}"
DATASET_STREAMING="${DATASET_STREAMING:-1}"
VALIDATION_DATASET_PATH="${VALIDATION_DATASET_PATH:-$DATASET_PATH}"
VALIDATION_DATASET_SPLIT="${VALIDATION_DATASET_SPLIT:-validation}"
VALIDATION_EVERY="${VALIDATION_EVERY:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-xpu-non-torchtitan-gemma}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_AFTER_MINUTES="${SAVE_AFTER_MINUTES:-45}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${LOG_DIR}/checkpoints}"
DISABLE_TIME_CHECKPOINT="${DISABLE_TIME_CHECKPOINT:-1}"
mkdir -p "$DATASET_CACHE_DIR"

TRAIN_ARGS=(
  "--steps" "$TRAIN_STEPS"
  "--seq-len" "$SEQ_LEN"
  "--batch-size" "$BATCH_SIZE"
  "--micro-batch-size" "$MICRO_BATCH_SIZE"
  "--logit-chunk-size" "$LOGIT_CHUNK_SIZE"
  "--train-mode" "$TRAIN_MODE"
  "--dtype" "$DTYPE"
  "--device" "$DEVICE"
  "--lr" "$LR"
  "--dataset-split" "$DATASET_SPLIT"
  "--dataset-cache-dir" "$DATASET_CACHE_DIR"
  "--dataset-max-samples" "$DATASET_MAX_SAMPLES"
  "--validation-dataset-path" "$VALIDATION_DATASET_PATH"
  "--validation-dataset-split" "$VALIDATION_DATASET_SPLIT"
  "--validation-every" "$VALIDATION_EVERY"
  "--wandb-project" "$WANDB_PROJECT"
  "--wandb-mode" "$WANDB_MODE"
  "--save-after-minutes" "$SAVE_AFTER_MINUTES"
  "--checkpoint-dir" "$CHECKPOINT_DIR"
)

if [[ "$DATASET_STREAMING" == "1" ]]; then
  TRAIN_ARGS+=("--dataset-streaming")
fi

if [[ "${DISABLE_TIME_CHECKPOINT:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--disable-time-checkpoint")
fi

if [[ "${DISABLE_VALIDATION:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--disable-validation")
fi

if [[ "${DISABLE_WANDB:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--disable-wandb")
fi

if [[ "${REQUIRE_CCL:-0}" == "1" ]]; then
  TRAIN_ARGS+=("--require-ccl")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

CMD=("${SCRIPT_DIR}/run_train_non_torchtitan.sh" "$MODE")
if [[ "$MODE" == "multi" ]]; then
  CMD+=("$HOSTFILE")
fi
if [[ "$DRY_RUN" == "1" ]]; then
  CMD+=("--dry-run")
fi
CMD+=("--")
CMD+=("${TRAIN_ARGS[@]}")

printf 'Runtime env summary:\n'
printf '  PYTHON_BIN=%s\n' "$PYTHON_BIN"
printf '  MODEL_PATH=%s\n' "$MODEL_PATH"
printf '  DATASET_PATH=%s DATASET_SPLIT=%s DATASET_STREAMING=%s\n' "$DATASET_PATH" "$DATASET_SPLIT" "$DATASET_STREAMING"
printf '  DATASET_CACHE_DIR=%s\n' "$DATASET_CACHE_DIR"
printf '  VALIDATION_DATASET_PATH=%s VALIDATION_DATASET_SPLIT=%s VALIDATION_EVERY=%s DISABLE_VALIDATION=%s\n' "$VALIDATION_DATASET_PATH" "$VALIDATION_DATASET_SPLIT" "$VALIDATION_EVERY" "${DISABLE_VALIDATION:-0}"
printf '  WANDB_PROJECT=%s WANDB_MODE=%s DISABLE_WANDB=%s\n' "$WANDB_PROJECT" "$WANDB_MODE" "${DISABLE_WANDB:-0}"
printf '  LOG_DIR=%s\n' "$LOG_DIR"
printf '  SAVE_AFTER_MINUTES=%s CHECKPOINT_DIR=%s DISABLE_TIME_CHECKPOINT=%s\n' "$SAVE_AFTER_MINUTES" "$CHECKPOINT_DIR" "${DISABLE_TIME_CHECKPOINT:-0}"
printf '  SCHEDULER=%s NPROC_PER_NODE=%s AUTO_RETRY=%s\n' "$SCHEDULER" "$NPROC_PER_NODE" "$AUTO_RETRY"
printf '  LOAD_FRAMEWORKS=%s CCL_PROCESS_LAUNCHER=%s\n' "${LOAD_FRAMEWORKS:-1}" "$CCL_PROCESS_LAUNCHER"
printf '  ZE_FLAT_DEVICE_HIERARCHY=%s CCL_ATL_TRANSPORT=%s FI_PROVIDER=%s\n' "$ZE_FLAT_DEVICE_HIERARCHY" "$CCL_ATL_TRANSPORT" "$FI_PROVIDER"
printf '  PYTHONMULTIPROCESSINGMETHOD=%s VLLM_WORKER_MULTIPROC_METHOD=%s\n' "$PYTHONMULTIPROCESSINGMETHOD" "$VLLM_WORKER_MULTIPROC_METHOD"
printf 'Running: '
printf '%q ' "${CMD[@]}"
printf '\n'

cd "$SCRIPT_DIR"
exec "${CMD[@]}"
