#!/usr/bin/env bash
set -euo pipefail

# TorchTitan Gemma FSDP2 wrapper using xpu_launcher.
# Builds a small local SFT-style JSONL from PG19 + MultiNews, then launches
# torchtitan.models.gemma.train through the existing xpu launch wrapper.

usage() {
  cat <<'EOF'
Usage:
  run_gemma_torchtitan_xpu.sh single [--dry-run] [-- <extra torchtitan args...>]
  run_gemma_torchtitan_xpu.sh multi <hostfile> [--dry-run] [-- <extra torchtitan args...>]

Defaults:
  TORCHTITAN_ROOT=<this script dir>/torchtitan_repo (self-contained copy)
  MODEL_PATH=${TORCHTITAN_ROOT}/assets/hf/gemma-7b
  DATASET_PATH=pg19,multi_news
  DATASET_SPLIT=train
  DATASET_MAX_SAMPLES=256
  DATASET_CACHE_DIR=/lus/flare/projects/datascience/seonghapark/llm_evaluation/evaluation/datasets/hf_cache
  NUM_EPOCHS=10
  SEQ_LEN=16384
  BATCH_SIZE=4               effective local batch via gradient accumulation
  PER_DEVICE_BATCH_SIZE=1
  GRADIENT_ACCUMULATION_STEPS=${BATCH_SIZE}
  NPROC_PER_NODE=1

Example in a 2-node qsub interactive shell:
  awk 'NF {print $1}' "$PBS_NODEFILE" | sort -u > hosts.txt
  ./run_gemma_torchtitan_xpu.sh multi hosts.txt
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

USER_CCL_PROCESS_LAUNCHER="${CCL_PROCESS_LAUNCHER:-}"
USER_ZE_FLAT_DEVICE_HIERARCHY="${ZE_FLAT_DEVICE_HIERARCHY:-}"

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

export SCHEDULER="${SCHEDULER:-auto}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export AUTO_RETRY="${AUTO_RETRY:-0}"
export PYTHON_BIN="${PYTHON_BIN:-/lus/flare/projects/datascience/seonghapark/venv/bin/python}"
export XPU_CMD="${XPU_CMD:-xpu}"

TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-${SCRIPT_DIR}/torchtitan_repo}"
MODEL_PATH="${MODEL_PATH:-${TORCHTITAN_ROOT}/assets/hf/gemma-7b}"
TRAIN_PYTHON_BIN="${TRAIN_PYTHON_BIN:-${PYTHON_BIN}}"
TYRO_SITE_PACKAGES="${TYRO_SITE_PACKAGES:-/lus/flare/projects/datascience/seonghapark/venv/lib/python3.12/site-packages}"
DATASET_PATH="${DATASET_PATH:-pg19,multi_news}"
DATASET_SPLIT="${DATASET_SPLIT:-train}"
DATASET_CACHE_DIR="${DATASET_CACHE_DIR:-/lus/flare/projects/datascience/seonghapark/evaluation/datasets/hf_cache}"
DATASET_MAX_SAMPLES="${DATASET_MAX_SAMPLES:-256}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
SEQ_LEN="${SEQ_LEN:-16384}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$BATCH_SIZE}"
LR="${LR:-1e-5}"
DTYPE="${DTYPE:-bfloat16}"
PARAM_DTYPE="${PARAM_DTYPE:-bfloat16}"
REDUCE_DTYPE="${REDUCE_DTYPE:-float32}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.02}"
EVAL_INTERVAL="${EVAL_INTERVAL:-200}"
EVAL_MAX_BATCHES="${EVAL_MAX_BATCHES:-0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-torchtitan-gemma-pg19-multinews}"
WANDB_MODE="${WANDB_MODE:-online}"
ENABLE_WANDB="${ENABLE_WANDB:-1}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/../outputs/gemma_torchtitan_pg19_multinews_bsz${BATCH_SIZE}_epochs${NUM_EPOCHS}_seq${SEQ_LEN}_$(date +%Y%m%d_%H%M%S)}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-gemma-torchtitan-pg19-multinews-bsz${BATCH_SIZE}-epochs${NUM_EPOCHS}-seq${SEQ_LEN}}"
DATASET_LOCAL_PATH="${DATASET_LOCAL_PATH:-${LOG_DIR}/pg19_multinews_sft.jsonl}"

if [[ ! -d "$TORCHTITAN_ROOT" ]]; then
  echo "error: TORCHTITAN_ROOT not found: $TORCHTITAN_ROOT" >&2
  exit 1
fi
# TRAIN_PYTHON_BIN may be a compute-node-only build; only require it when executing
if [[ "$DRY_RUN" != "1" && ! -x "$TRAIN_PYTHON_BIN" ]]; then
  echo "error: TRAIN_PYTHON_BIN not executable: $TRAIN_PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$LOG_DIR" "$DATASET_CACHE_DIR"

if [[ "$DRY_RUN" != "1" && ! -f "$DATASET_LOCAL_PATH" ]]; then
  DATASET_PATH="$DATASET_PATH" \
  DATASET_SPLIT="$DATASET_SPLIT" \
  DATASET_CACHE_DIR="$DATASET_CACHE_DIR" \
  DATASET_MAX_SAMPLES="$DATASET_MAX_SAMPLES" \
  DATASET_LOCAL_PATH="$DATASET_LOCAL_PATH" \
  "$TRAIN_PYTHON_BIN" - <<'PY'
import json
import os
from itertools import islice

from datasets import load_dataset

aliases = {
    "multinews": "Awesome075/multi_news_parquet",
    "multi-news": "Awesome075/multi_news_parquet",
    "multi_news": "Awesome075/multi_news_parquet",
    "pg19": "emozilla/pg19",
    "pg-19": "emozilla/pg19",
}
text_fields = {
    "Awesome075/multi_news_parquet": ("document", "summary"),
    "emozilla/pg19": ("text",),
    "multi_news": ("document", "summary"),
    "pg19": ("text",),
}

def text_from_sample(dataset_name, sample):
    chunks = []
    for field in text_fields.get(dataset_name, ("text", "document", "summary")):
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            chunks.append(value.strip())
    if chunks:
        return "\n\n".join(chunks)
    for value in sample.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""

dataset_names = []
for raw_name in os.environ["DATASET_PATH"].split(","):
    raw_name = raw_name.strip()
    if raw_name:
        dataset_names.append(aliases.get(raw_name.lower(), raw_name))

out_path = os.environ["DATASET_LOCAL_PATH"]
os.makedirs(os.path.dirname(out_path), exist_ok=True)
max_samples = max(1, int(os.environ["DATASET_MAX_SAMPLES"]))
written = 0
with open(out_path, "w", encoding="utf-8") as handle:
    for dataset_name in dataset_names:
        dataset = load_dataset(
            dataset_name,
            split=os.environ["DATASET_SPLIT"],
            cache_dir=os.environ["DATASET_CACHE_DIR"],
            streaming=True,
        )
        for sample in islice(iter(dataset), max_samples):
            text = text_from_sample(dataset_name, dict(sample))
            if not text:
                continue
            record = {"instruction": "Continue the text.", "input": "", "output": text}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
print(f"wrote {written} records to {out_path}", flush=True)
PY
fi

TORCHTITAN_BOOTSTRAP_CODE="import runpy, sys; sys.path.insert(0, '${TORCHTITAN_ROOT}'); sys.path.append('${TYRO_SITE_PACKAGES}'); runpy.run_module('torchtitan.models.gemma.train', run_name='__main__')"

TRAIN_CMD=(
  "$TRAIN_PYTHON_BIN" "-c" "$TORCHTITAN_BOOTSTRAP_CODE"
  "--model_name_or_path" "$MODEL_PATH"
  "--dataset_local_path" "$DATASET_LOCAL_PATH"
  "--max_seq_len" "$SEQ_LEN"
  "--per_device_batch_size" "$PER_DEVICE_BATCH_SIZE"
  "--gradient_accumulation_steps" "$GRADIENT_ACCUMULATION_STEPS"
  "--num_epochs" "$NUM_EPOCHS"
  "--lr" "$LR"
  "--dtype" "$DTYPE"
  "--param_dtype" "$PARAM_DTYPE"
  "--reduce_dtype" "$REDUCE_DTYPE"
  "--output_dir" "$LOG_DIR"
  "--validation_split" "$VALIDATION_SPLIT"
  "--eval_interval" "$EVAL_INTERVAL"
  "--eval_max_batches" "$EVAL_MAX_BATCHES"
  "--save_interval" "$SAVE_INTERVAL"
  "--wandb_project" "$WANDB_PROJECT"
  "--wandb_run_name" "$WANDB_RUN_NAME"
  "--wandb_mode" "$WANDB_MODE"
)

if [[ "$ENABLE_WANDB" == "1" ]]; then
  TRAIN_CMD+=("--enable_wandb")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

printf 'Runtime env summary:\n'
printf '  TORCHTITAN_ROOT=%s\n' "$TORCHTITAN_ROOT"
printf '  TRAIN_PYTHON_BIN=%s\n' "$TRAIN_PYTHON_BIN"
printf '  MODEL_PATH=%s\n' "$MODEL_PATH"
printf '  DATASET_PATH=%s DATASET_SPLIT=%s DATASET_MAX_SAMPLES=%s\n' "$DATASET_PATH" "$DATASET_SPLIT" "$DATASET_MAX_SAMPLES"
printf '  DATASET_LOCAL_PATH=%s\n' "$DATASET_LOCAL_PATH"
printf '  LOG_DIR=%s\n' "$LOG_DIR"
printf '  NUM_EPOCHS=%s SEQ_LEN=%s BATCH_SIZE=%s PER_DEVICE_BATCH_SIZE=%s GRADIENT_ACCUMULATION_STEPS=%s\n' "$NUM_EPOCHS" "$SEQ_LEN" "$BATCH_SIZE" "$PER_DEVICE_BATCH_SIZE" "$GRADIENT_ACCUMULATION_STEPS"
printf '  SCHEDULER=%s NPROC_PER_NODE=%s AUTO_RETRY=%s\n' "$SCHEDULER" "$NPROC_PER_NODE" "$AUTO_RETRY"

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

# run_train.sh resolves ./.venv and src/cli relative to the repo root
cd "${SCRIPT_DIR}/.."
exec "${CMD[@]}"