#!/bin/bash --login
#PBS -A datascience
#PBS -N vllm-xpu-bare
#PBS -l walltime=00:15:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Standalone vLLM-XPU smoke (no TRL wrapper). Replays the 2026-06-10
# end-to-end verification from venvs/rl-vllm/ (py3.12 + torch 2.12 +
# vllm 0.22 + vllm-xpu-kernels + triton-xpu 3.7.1 + monarch +
# torchstore + trl + transformers + accelerate + datasets).
#
# **Critical**: scrub CCL_*/FI_* env vars before invoking python.
# `ezpz_setup_env` exports CCL_PROCESS_LAUNCHER=pmix, CCL_OP_SYNC=1,
# FI_PROVIDER=cxi,tcp;ofi_rxm, plus many FI_CXI_* settings. These are
# correct for ezpz/mpiexec training, but vLLM's EngineCore subprocess
# is launched via multiprocessing.spawn (no MPI), so:
#   - The CCL_PROCESS_LAUNCHER=pmix override makes oneCCL look for a
#     PMIx context that doesn't exist.
#   - The `cxi` OFI provider needs a Slingshot NIC handle that only
#     mpiexec-bootstrapped processes have. `fi_getinfo` returns 0
#     providers and ATL init fails with "can't find suitable provider".
# Stripping all of them lets oneCCL fall back to a working default
# (tcp via libfabric).
#
# See `docs/rl/vllm-xpu-current-status.md` for the full debug chain
# that landed on this fix.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
# vLLM uses ZMQ IPC for its EngineCore IPC, but Linux's
# sockaddr_un.sun_path is limited to 107 chars. PBS-default TMPDIR
# (/var/tmp/pbs.<long-jobid>) exceeds that once vLLM appends a UUID.
mkdir -p "/tmp/vllm-${USER}"
export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

# IMPORTANT: scrub the polluted env AFTER ezpz_setup_job has set up
# PBS / nodefile vars but BEFORE we invoke vLLM. The unsets here
# don't affect ezpz_setup_job's bookkeeping.
unset CCL_OP_SYNC CCL_PROCESS_LAUNCHER CCL_ATL_TRANSPORT CCL_OFI_PROVIDER
unset FI_PROVIDER FI_LOG_LEVEL FI_LOG_PROV FI_LOG_LOCATION
unset FI_CXI_DEFAULT_CQ_SIZE FI_CXI_DEFAULT_TX_SIZE FI_CXI_OFLOW_BUF_COUNT
unset FI_CXI_OFLOW_BUF_SIZE FI_CXI_RDZV_EAGER_SIZE FI_CXI_RDZV_THRESHOLD
unset FI_CXI_REQ_BUF_MAX_CACHED FI_CXI_REQ_BUF_MIN_POSTED FI_CXI_REQ_BUF_SIZE
unset FI_CXI_RX_MATCH_MODE FI_MR_CACHE_MAX_COUNT FI_MR_CACHE_MAX_SIZE

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/vllm-xpu-bare-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"

MODEL="${MODEL:-outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf}"

echo "=== vllm-xpu bare smoke (rl-vllm venv, env-scrubbed) ===" | tee "${LOG_DIR}/run.log"
echo "MODEL=${MODEL}" | tee -a "${LOG_DIR}/run.log"
date | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"
echo "Residual CCL_/FI_ env (should be CCL_ROOT only):" | tee -a "${LOG_DIR}/run.log"
env | grep -iE "^(CCL_|FI_)" | sort | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

# Bare vLLM via plain python — NO `ezpz launch` wrapper. vLLM's
# EngineCore subprocess spawns itself; an outer mpiexec adds no value
# at TP=1 and actively breaks the PMIx state oneCCL ends up in.
#
# Uses venvs/rl-vllm/ (py3.12), the unified Monarch+vLLM venv. Both
# torchmonarch and triton-xpu==3.7.1 ship cp312 wheels, so a single
# Python version covers actor framework + vLLM worker + trainer
# (no need for the prior 2-venv split with rl-actors/vllm-test).
MODEL="${MODEL}" "${SUBMIT_DIR}/venvs/rl-vllm/bin/python" \
    "${SUBMIT_DIR}/torchtitan/experiments/ezpz/rl/scripts/vllm_xpu_bare_smoke.py" \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "" | tee -a "${LOG_DIR}/run.log"
echo "VERDICT: complete" | tee -a "${LOG_DIR}/run.log"
