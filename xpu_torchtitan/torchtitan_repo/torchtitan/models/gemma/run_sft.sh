#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Instruction fine-tune google/gemma-7b via HF Transformers + FSDP2.
#
# Single-node launch via `torchrun --nproc_per_node=$NGPU`.
#
# Overrides:
#   NGPU        Per-node process count (default: 6).
#   LOG_RANK    Ranks whose stdout to tee to the terminal (default "0").
#   MODEL_PATH  HF model path or hub id (default: ./assets/hf/gemma-7b[/main]).
#
# Examples:
#   ./torchtitan/models/gemma/run_sft.sh \
#       --dataset_name AI-MO/NuminaMath-CoT \
#       --instruction_key problem --output_key solution \
#       --output_dir outputs/gemma-7b-numina
#
#   NGPU=12 ./torchtitan/models/gemma/run_sft.sh ...
set -eo pipefail

LOG_RANK=${LOG_RANK:-0}

NGPU="${NGPU:-6}"
MODEL_PATH="${MODEL_PATH:-"./assets/hf/gemma-7b"}"
echo "[run_sft.sh] NGPU=${NGPU}  MODEL_PATH=${MODEL_PATH}"

PYTORCH_ALLOC_CONF="expandable_segments:True" \
torchrun \
    --nproc_per_node="${NGPU}" \
    --rdzv_backend c10d \
    --rdzv_endpoint="localhost:0" \
    --local-ranks-filter "${LOG_RANK}" \
    --role rank --tee 3 \
    -m torchtitan.models.gemma.train \
    --model_name_or_path "${MODEL_PATH}" \
    "$@"