#!/usr/bin/env bash
# Lmod's module() breaks under set -u, so only -eo pipefail
set -eo pipefail

# run_agpt_torchtitan_xpu.sh
#
# aGPT training on Aurora XPU using TorchTitan (compat experiment wrapper).
# Wraps `torchtitan.experiments.ezpz.train` (torchtitan-side package path;
# renaming it requires a torchtitan change), which trains the aGPT models
# (agpt_2b / agpt_20b, ...) registered in
# torchtitan/experiments/ezpz/agpt/config_registry.py with FSDP2 sharding,
# DCP checkpoints, and the blendcorpus data pipeline.
#
# Run inside a PBS job shell (xpu launch reads PBS_NODEFILE for multi-node).
#
# Usage:
#   ./run_agpt_torchtitan_xpu.sh [--dry-run] [-- <extra torchtitan args...>]
#
# Environment variables:
#   MODEL             aGPT flavor: 2b | 20b | debugmodel | ...  Default: 2b
#                     (resolves to CONFIG=ezpz_agpt_${MODEL})
#   CONFIG            Config registry entry, overrides MODEL.  Default: ezpz_agpt_${MODEL}
#   TORCHTITAN_ROOT   TorchTitan tree.  Default: <this script dir>/torchtitan_repo
#   VENV              Virtualenv with torchtitan and its deps.  Default: ${TORCHTITAN_ROOT}/.venv
#   DATA_FILE_LIST    blendcorpus data file list.
#                     Default: torchtitan/experiments/ezpz/data-lists/aurora/books.txt
#   CHECKPOINT_DIR    DCP checkpoint folder.  Default: outputs/checkpoints/aGPT-${MODEL}-<datalist>
#   STEPS             Training steps (--training.steps).  Default: unset (config default)
#   SEQ_LEN           Sequence length (--training.seq_len).  Default: unset (config default)
#   GLOBAL_BATCH_SIZE Global batch size (--training.global_batch_size).  Default: unset
#   COMM_MODE         fake_backend | local_tensor for dry validation without XPUs.
#
# Examples:
#   MODEL=2b STEPS=10 ./run_agpt_torchtitan_xpu.sh
#   MODEL=20b DATA_FILE_LIST=/path/to/list.txt ./run_agpt_torchtitan_xpu.sh -- --training.seq_len 8192
#   MODEL=2b COMM_MODE=fake_backend ./run_agpt_torchtitan_xpu.sh   # config validation only

usage() { sed -n '3,37p' "$0"; }

# Resolve before any cd; BASH_SOURCE may be relative
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "error: unknown option: $1 (pass torchtitan args after --)" >&2; exit 1 ;;
  esac
done

TORCHTITAN_ROOT="${TORCHTITAN_ROOT:-${SCRIPT_DIR}/torchtitan_repo}"
VENV="${VENV:-${TORCHTITAN_ROOT}/.venv}"
MODEL="${MODEL:-2b}"
MODULE="${MODULE:-ezpz.agpt}"
CONFIG="${CONFIG:-ezpz_agpt_${MODEL}}"
DATA_FILE_LIST="${DATA_FILE_LIST:-${TORCHTITAN_ROOT}/torchtitan/experiments/ezpz/data-lists/aurora/books.txt}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-outputs/checkpoints/aGPT-${MODEL}-$(basename "${DATA_FILE_LIST%.txt}")}"
COMM_MODE="${COMM_MODE:-}"

[[ -d "${TORCHTITAN_ROOT}" ]] || { echo "error: TORCHTITAN_ROOT not found: ${TORCHTITAN_ROOT}" >&2; exit 1; }
[[ -f "${VENV}/bin/activate" ]] || { echo "error: venv not found: ${VENV}" >&2; exit 1; }
[[ -f "${DATA_FILE_LIST}" ]] || { echo "error: DATA_FILE_LIST not found: ${DATA_FILE_LIST}" >&2; exit 1; }
if [[ -z "${COMM_MODE}" && ( -z "${PBS_NODEFILE:-}" || ! -f "${PBS_NODEFILE:-}" ) ]]; then
  echo "error: PBS_NODEFILE is unset; run inside a PBS job (or set COMM_MODE=fake_backend)" >&2
  exit 1
fi

module -q load frameworks 2>/dev/null || true
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

export CCL_PROCESS_LAUNCHER=hydra
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

cd "${TORCHTITAN_ROOT}"

TRAIN_ARGS=(
  -m torchtitan.experiments.ezpz.train
  --debug.print_config
  --module "${MODULE}"
  --config "${CONFIG}"
  --training.dataset_path "${DATA_FILE_LIST}"
  --checkpoint.folder "${CHECKPOINT_DIR}"
)
[[ -n "${STEPS:-}" ]]             && TRAIN_ARGS+=(--training.steps "${STEPS}")
[[ -n "${SEQ_LEN:-}" ]]           && TRAIN_ARGS+=(--training.seq_len "${SEQ_LEN}")
[[ -n "${GLOBAL_BATCH_SIZE:-}" ]] && TRAIN_ARGS+=(--training.global_batch_size "${GLOBAL_BATCH_SIZE}")
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

echo "[INFO] TORCHTITAN_ROOT = ${TORCHTITAN_ROOT}"
echo "[INFO] VENV            = ${VENV}"
echo "[INFO] MODEL/CONFIG    = ${MODEL} / ${CONFIG}"
echo "[INFO] DATA_FILE_LIST  = ${DATA_FILE_LIST}"
echo "[INFO] CHECKPOINT_DIR  = ${CHECKPOINT_DIR}"

if [[ -n "${COMM_MODE}" ]]; then
  # Single-process validation mode: no launcher, no XPU communication
  CMD=(python3 "${TRAIN_ARGS[@]}" --comm.mode="${COMM_MODE}" --training.steps 1)
else
  # A venv may carry a stale `xpu` entrypoint; verify it works before trusting it
  if command -v xpu >/dev/null 2>&1 && xpu --version >/dev/null 2>&1; then
    XPU_INVOKE=(xpu)
  elif [[ -d "${LAUNCHER_ROOT}/src/cli" ]]; then
    export PYTHONPATH="${LAUNCHER_ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
    XPU_INVOKE=(python3 -m cli)
  else
    echo "error: xpu CLI not found (pip install -e ${LAUNCHER_ROOT})" >&2
    exit 1
  fi
  CMD=("${XPU_INVOKE[@]}" launch -- python3 "${TRAIN_ARGS[@]}")
fi

echo "[INFO] ${CMD[*]}"
if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[INFO] dry run; not executing"
  exit 0
fi
exec "${CMD[@]}"
