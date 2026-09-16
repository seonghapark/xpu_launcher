#!/bin/bash --login
#PBS -A datascience
#PBS -N vllm-xpu-noccl
#PBS -l walltime=00:15:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Same script as vllm_xpu_vllmtest_replay.sh but WITHOUT the CCL env
# overrides (CCL_ATL_TRANSPORT, CCL_PROCESS_LAUNCHER, CCL_OP_SYNC).
#
# Why: replay job 12468750 failed identically to the rl-actors smokes
# even though it used py3.14 / vllm-test. The EngineCore subprocess
# died silently right after these CCL warnings:
#     CCL_WARN| value of CCL_OP_SYNC changed to be 1 (default:0)
#     CCL_WARN| value of CCL_PROCESS_LAUNCHER changed to be pmix (default:hydra)
#
# The 2026-06-10 working invocation did NOT set these. Test whether
# removing them restores the working path.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
# NB: NO CCL_* overrides — let oneCCL use its defaults
# (atl_ofi transport, hydra process launcher, OP_SYNC=0).
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128
mkdir -p "/tmp/vllm-${USER}"
export TMPDIR="/tmp/vllm-${USER}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="logs/vllm-xpu-noccl-${JOBID_SHORT:-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "${LOG_DIR}"

MODEL="${MODEL:-outputs/sft/aurora2b-sophiag-tulu-mix-32n-gbs6144/checkpoint-729-hf}"

echo "=== vllm-xpu no-CCL-overrides (py3.14 venv, original 0610 recipe) ===" | tee "${LOG_DIR}/run.log"
echo "MODEL=${MODEL}" | tee -a "${LOG_DIR}/run.log"
date | tee -a "${LOG_DIR}/run.log"
echo "" | tee -a "${LOG_DIR}/run.log"

NP="${NP:-1}"
PPN="${PPN:-1}"
MODEL="${MODEL}" "${SUBMIT_DIR}/.venv/bin/ezpz" launch --np "${NP}" -ppn "${PPN}" \
    "${SUBMIT_DIR}/venvs/vllm-test/bin/python" \
    "${SUBMIT_DIR}/torchtitan/experiments/ezpz/rl/scripts/vllm_xpu_vllmtest_replay.py" \
    2>&1 | tee -a "${LOG_DIR}/run.log"

echo "" | tee -a "${LOG_DIR}/run.log"
echo "VERDICT: complete" | tee -a "${LOG_DIR}/run.log"
