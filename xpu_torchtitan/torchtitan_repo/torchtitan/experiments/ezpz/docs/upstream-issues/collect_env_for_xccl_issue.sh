#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:15:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# Run python -m torch.utils.collect_env on a Sunspot compute node so
# we have an authoritative environment block to paste into the
# pytorch/pytorch issue body for the xccl supportsSplitting bug.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${SUBMIT_DIR}"
source .venv/bin/activate
ezpz tar-env
ezpz yeet .venv.tar.gz
deactivate
source /tmp/.venv/bin/activate

LOG="logs/collect-env-${PBS_JOBID%%.*}.log"
mkdir -p logs
echo "=== python -m torch.utils.collect_env (Sunspot compute node) ===" | tee "$LOG"
python3 -m torch.utils.collect_env 2>&1 | tee -a "$LOG"
echo "=== done ===" | tee -a "$LOG"
