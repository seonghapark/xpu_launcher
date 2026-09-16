#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# Instruction fine-tune google/gemma-7b via HF Transformers + FSDP2.
#
# The launcher auto-detects the device count (XPU on Aurora, else CUDA) and
# picks a launcher:
#   * If `ezpz` is on PATH AND a PBS/SLURM job env is present -> `ezpz launch`
#     (correct for multi-node Aurora / ALCF).
#   * Otherwise -> `torchrun --nproc_per_node=<detected>` (single-node).
#
# Overrides:
#   NGPU        Force the per-node process count.
#   LAUNCHER    "ezpz" | "torchrun" | "auto" (default "auto").
#   LOG_RANK    Ranks whose stdout to tee to the terminal (default "0").
#   MODEL_PATH  HF model path or hub id (default: ./assets/hf/gemma-7b[/main]).
#
# Examples:
#   ./torchtitan/models/gemma/run_sft.sh \
#       --dataset_name AI-MO/NuminaMath-CoT \
#       --instruction_key problem --output_key solution \
#       --output_dir outputs/gemma-7b-numina
#
#   NGPU=12 LAUNCHER=torchrun ./torchtitan/models/gemma/run_sft.sh ...
set -eo pipefail

LOG_RANK=${LOG_RANK:-0}
LAUNCHER=${LAUNCHER:-auto}

# ---------------------------------------------------------------------------
# Auto-detect device count (XPU on Aurora, else CUDA)
# ---------------------------------------------------------------------------
_detect_ngpu() {
    python3 - <<'PY' 2>/dev/null
try:
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        print(torch.xpu.device_count())
    elif torch.cuda.is_available():
        print(torch.cuda.device_count())
    else:
        print("no-gpu")
except Exception:
    print("ngpu-detect-failed")
PY
}

NGPU=6
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