#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <wandb_name> [extra torchtitan args ...]"
  exit 1
fi

WAND_NAME="$1"
shift || true
EXTRA_ARGS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# Lmod scripts used by ezpz env setup can reference unset vars (e.g. ZSH_EVAL_CONTEXT),
# so temporarily disable nounset during environment initialization.
set +u
source <(curl -fsSL https://bit.ly/ezpz-utils)
ezpz_setup_env
set -u

export ZE_FLAT_DEVICE_HIERARCHY="FLAT"

RUN_SLUG="$(echo "${WAND_NAME}" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9._-' '-')"
RUN_ROOT="${REPO_ROOT}/outputs/moe_runs/${RUN_SLUG}"
CKPT_ROOT="${RUN_ROOT}/checkpoints"
mkdir -p "${RUN_ROOT}" "${CKPT_ROOT}"

export WANDB_PROJECT="${WANDB_PROJECT:-torchtitan.moe_runs}"
export WANDB_TEAM="${WANDB_TEAM:-moe_experiments}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-${WAND_NAME}}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_SLUG}}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-deepseek_v3_moe_ep12}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

export TT_CONFIG_JSON="${TT_CONFIG_JSON:-${SCRIPT_DIR}/deepseek_v3_10b2b_ep12_2nodes_smoke.json}"

if [[ -z "${HF_ASSETS_PATH:-}" ]]; then
  CANDIDATES=(
    "${REPO_ROOT}/assets/hf/gemma-7b"
    "${REPO_ROOT}/assets/hf/deepseek-moe-16b-base"
    "${REPO_ROOT}/assets/hf/DeepSeek-V3.1-Base"
  )
  for p in "${CANDIDATES[@]}"; do
    if [[ -d "${p}" ]]; then
      HF_ASSETS_PATH="${p}"
      break
    fi
  done
fi

if [[ -z "${HF_ASSETS_PATH:-}" || ! -d "${HF_ASSETS_PATH}" ]]; then
  echo "No valid tokenizer/assets path found."
  echo "Set HF_ASSETS_PATH or download tokenizer assets first, for example:"
  echo "  python3 scripts/download_hf_assets.py --repo_id google/gemma-7b --assets tokenizer"
  exit 1
fi

if [[ ! -f "${HF_ASSETS_PATH}/tokenizer.json" && ! -f "${HF_ASSETS_PATH}/tokenizer.model" ]]; then
  echo "HF_ASSETS_PATH=${HF_ASSETS_PATH} does not contain tokenizer.json or tokenizer.model"
  echo "Please point HF_ASSETS_PATH at a valid tokenizer assets directory."
  exit 1
fi
export HF_ASSETS_PATH

echo "Repo root: ${REPO_ROOT}"
echo "Using TT_CONFIG_JSON=${TT_CONFIG_JSON}"
echo "Using HF_ASSETS_PATH=${HF_ASSETS_PATH}"
echo "Run slug: ${RUN_SLUG}"
echo "Dump folder: ${RUN_ROOT}"
echo "Checkpoint folder: ${CKPT_ROOT}"

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
  --module deepseek_v3 \
  --config deepseek_v3_10b_2b_ep12_from_json \
  --dump-folder "${RUN_ROOT}" \
  --hf-assets-path "${HF_ASSETS_PATH}" \
  --checkpoint.enable \
  --checkpoint.folder checkpoints \
  --checkpoint.load_step -1 \
  --metrics.enable_wandb \
  --debug.print-config \
  --debug.save-config-file "configs/effective_config.json" \
  "${EXTRA_ARGS[@]}"
