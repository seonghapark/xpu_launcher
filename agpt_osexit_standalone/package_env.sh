#!/usr/bin/env bash

AGPT_PACKAGE_ROOT=${AGPT_PACKAGE_ROOT:-/lus/flare/projects/datascience/seonghapark/agpt_osexit_standalone}
AGPT_PACKAGE_ROOT=$(realpath "$AGPT_PACKAGE_ROOT") || return 1
AGPT_INITIAL_DCP=${AGPT_INITIAL_DCP:-/lus/flare/projects/AuroraGPT/foremans/runs/agpt-2b-v2/torchtitan-ezpz/outputs/checkpoints/agpt-2b-sophiag-olmo-mix-1124-n256-gbs6144/step-49900}
if [[ ! -r "$AGPT_INITIAL_DCP/.metadata" ]]; then
  echo "FATAL: unreadable DCP metadata: $AGPT_INITIAL_DCP/.metadata" >&2
  return 1
fi
export AGPT_PACKAGE_ROOT AGPT_INITIAL_DCP
export TORCHTITAN_ROOT="$AGPT_PACKAGE_ROOT/xpu_torchtitan/torchtitan_repo"
export PYTHONPATH="$AGPT_PACKAGE_ROOT/src:$TORCHTITAN_ROOT"
export XPU_CMD="$AGPT_PACKAGE_ROOT/bin/xpu"
export XPU_VENV=/lus/flare/projects/datascience/seonghapark/venv
export TORCHTITAN_VENV=$XPU_VENV
export EVAL_BASE="$AGPT_PACKAGE_ROOT/evaluation"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
