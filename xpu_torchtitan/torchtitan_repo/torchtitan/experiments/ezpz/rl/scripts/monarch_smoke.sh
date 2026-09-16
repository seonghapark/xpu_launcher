#!/bin/bash --login
#PBS -A datascience
#PBS -N monarch-actor-smoke
#PBS -l walltime=00:15:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Runs torchtitan/experiments/ezpz/rl/scripts/monarch_smoke.py inside
# the `venvs/rl-vllm/` venv (py3.12 + torch 2.12+xpu + monarch +
# torchstore + vllm-xpu). Single node, single tile is enough for the
# Phase 1 smoke (import + 2-actor message-pass + torchstore transport
# import).
#
# Gates the Monarch+vLLM RL actor work — if this fails, the
# upstream VLLMGenerator / PolicyTrainer adaptation can't proceed
# until the underlying framework works on XPU.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
cd "${SUBMIT_DIR}"

LOG_DIR="logs/monarch-smoke-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

# Tarball the rl-vllm venv too so it's available on the compute node /tmp.
# (Match the .venv yeet pattern from production submit scripts.)
source venvs/rl-vllm/bin/activate
echo "venv torch: $(python -c 'import torch; print(torch.__version__)')" | tee "${LOG_DIR}/run.log"
python torchtitan/experiments/ezpz/rl/scripts/monarch_smoke.py 2>&1 | tee -a "${LOG_DIR}/run.log"
