#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:15:00
#PBS -l filesystems=flare:home
#PBS -l select=1
#PBS -q workq
#PBS -j oe
#
# 1-node Sunspot PBS smoke that runs repro_xccl_supports_splitting.py
# and tees the output. Use the resulting log as evidence on the
# pytorch/pytorch issue.

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

LOG_DIR="logs/repro-xccl-supports-splitting-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"

# The repro script lives at
# torchtitan/experiments/ezpz/docs/upstream-issues/repro_xccl_supports_splitting.py
# but its dotted-module name has hyphens in it, so we invoke it via
# file path under ezpz launch instead of -m.
ezpz launch python3 \
    torchtitan/experiments/ezpz/docs/upstream-issues/repro_xccl_supports_splitting.py \
    2>&1 | tee "${LOG_FILE}" || true

# Print the script's verdict line for quick eyeballing
echo
echo "=== repro verdict ==="
grep -E "Layer [12] .*(REPRODUCED|NOT REPRODUCED)" "${LOG_FILE}" || echo "(no verdict line found)"
