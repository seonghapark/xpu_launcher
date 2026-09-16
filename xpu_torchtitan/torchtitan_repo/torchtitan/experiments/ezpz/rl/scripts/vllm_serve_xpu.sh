#!/bin/bash --login
#
# Launch a TRL-wrapped vLLM OpenAI-compat server on a single XPU tile.
# Designed to be backgrounded by a PBS submit script before the GRPO
# trainer launches.
#
# Why this wrapper exists:
#   - vLLM-XPU lives in venvs/vllm-test/ (torch 2.12.0+xpu) because
#     vllm-xpu-kernels' pre-built wheels link against 2.12. The trainer
#     lives in .venv/ (torch 2.13). Two venvs is the Option B sibling-
#     venv strategy from docs/rl/vllm-xpu-investigation.md.
#   - `trl vllm-serve` is installed in .venv/. We point PYTHONPATH at
#     .venv/'s trl package while running from venvs/vllm-test/, so vllm
#     itself comes from the torch-2.12 venv but TRL's server wrapper
#     comes from the torch-2.13 venv. ABI-compatible — `trl/scripts/
#     vllm_serve.py` only imports vllm + torch + transformers at module
#     level, and all three are present in vllm-test (transformers via
#     vllm's own deps).
#
# Usage:
#   MODEL=/path/to/hf/checkpoint PORT=8000 \
#       torchtitan/experiments/ezpz/rl/scripts/vllm_serve_xpu.sh
#
# Env knobs:
#   MODEL                 required — HF checkpoint dir (or model name)
#   PORT                  default 8000
#   TP                    default 1 (tensor_parallel_size)
#   GPU_MEM_UTIL          default 0.5
#   ENFORCE_EAGER         default true (--enforce_eager True)
#   REPO_ROOT             default $(git rev-parse --show-toplevel)

set -o pipefail

MODEL="${MODEL:?MODEL env var required (HF checkpoint dir)}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
ENFORCE_EAGER="${ENFORCE_EAGER:-true}"
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"

VLLM_VENV="${REPO_ROOT}/venvs/vllm-test"
TRAINER_VENV="${REPO_ROOT}/.venv"

test -d "${VLLM_VENV}" || { echo "FATAL: ${VLLM_VENV} not found"; exit 1; }
test -d "${TRAINER_VENV}" || { echo "FATAL: ${TRAINER_VENV} not found"; exit 1; }

# Use .venv's trl (the vllm-test venv doesn't have it).
TRL_PYTHONPATH="${TRAINER_VENV}/lib/python3.14/site-packages"

# Aurora-specific env (mirrors scripts/train_moe_*_venv.sh)
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

# Activate the vllm-test venv so `python` is the torch-2.12 one.
source "${VLLM_VENV}/bin/activate"

# Sanity: torch + vllm versions and XPU visibility.
python -c "
import torch, vllm
print(f'serve: torch={torch.__version__} vllm={vllm.__version__} xpu_count={torch.xpu.device_count() if hasattr(torch, \"xpu\") else 0}')
" || exit 1

# Run TRL's vllm_serve script using .venv's TRL implementation.
exec env PYTHONPATH="${TRL_PYTHONPATH}:${PYTHONPATH:-}" \
    python -m trl.scripts.vllm_serve \
    --model "${MODEL}" \
    --tensor_parallel_size "${TP}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --gpu_memory_utilization "${GPU_MEM_UTIL}" \
    --enforce_eager "${ENFORCE_EAGER}"
