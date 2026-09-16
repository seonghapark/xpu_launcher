#!/bin/bash --login
#
# Consolidate an FSDP-sharded SFTTrainer checkpoint into a flat HF
# format so train_grpo (or anything else using from_pretrained) can
# load it.
#
# SFTTrainer saves checkpoints as:
#   checkpoint-N/
#     pytorch_model_fsdp_0/    <- FSDP sharded distcp shards
#     optimizer_0/             <- optimizer state (dropped here)
#     rng_state_*.pth          <- per-rank RNG (dropped here)
#     trainer_state.json       <- HF Trainer state (dropped here)
#
# We need:
#   checkpoint-N-hf/
#     config.json              <- from the original model dir
#     tokenizer*               <- from the original model dir
#     model.safetensors        <- merged from pytorch_model_fsdp_0/
#
# Usage:
#   bash scripts/consolidate_sft_ckpt.sh \
#     outputs/sft/aurora2b-sophiag-metamathqa-32n/checkpoint-400 \
#     AuroraGPT-2B-sophiag-gs138650

set -e -o pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <sft_checkpoint_dir> <original_model_dir>"
    echo "  e.g. $0 outputs/sft/.../checkpoint-400 AuroraGPT-2B-sophiag-gs138650"
    exit 1
fi

CKPT_DIR="$1"
SRC_MODEL="$2"
OUT_DIR="${CKPT_DIR}-hf"

if [[ ! -d "${CKPT_DIR}/pytorch_model_fsdp_0" ]]; then
    echo "ERROR: ${CKPT_DIR}/pytorch_model_fsdp_0 not found"
    exit 1
fi
if [[ ! -f "${SRC_MODEL}/config.json" ]]; then
    echo "ERROR: ${SRC_MODEL}/config.json not found"
    exit 1
fi

source .venv/bin/activate

echo "[consolidate] merging FSDP shards from ${CKPT_DIR}/pytorch_model_fsdp_0"
accelerate merge-weights "${CKPT_DIR}/pytorch_model_fsdp_0" "${OUT_DIR}"

echo "[consolidate] copying config + tokenizer from ${SRC_MODEL}"
cp "${SRC_MODEL}/config.json" "${OUT_DIR}/"
cp "${SRC_MODEL}"/tokenizer* "${OUT_DIR}/" 2>/dev/null || true
cp "${SRC_MODEL}/special_tokens_map.json" "${OUT_DIR}/" 2>/dev/null || true

echo "[consolidate] done -> ${OUT_DIR}"
ls -la "${OUT_DIR}/"
