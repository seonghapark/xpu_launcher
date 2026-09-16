#!/bin/bash --login
#PBS -N agpt-50b-wide-2n-t213
#PBS -l walltime=01:30:00
#PBS -l filesystems=flare:home
#PBS -A datascience
#PBS -q workq
#PBS -j oe

# Disambiguation experiment for the DeviceMesh-in-saved-tensors crash.
#
# Background: in the 4N bisect job 12465952, all three configs
# (agpt_50b_wide / agpt_70b_wide / agpt_80b) crashed with the
# tensors_saved_with_vc_check AssertionError on torch 2.13. But the
# same agpt_50b_wide config ran 10/10 steps cleanly on May 3 at 2N on
# torch 2.10 (aurora_frameworks-2025.3.1). Two variables changed
# between those two runs: node count (2N → 4N, dp_shard 12 → 24) and
# torch version (2.10 → 2.13).
#
# This job pins one variable: agpt_50b_wide on 2N (matching May 3)
# but on torch 2.13 (matching the 4N crash). The result decides which
# variable is the actual trigger:
#
#   crashes → torch 2.13 alone is enough (and the May 3 success was
#             only because torch 2.10 didn't have the failing assertion
#             in this code path)
#   runs    → 4N (or dp_shard=24) is the trigger; depth-sensitivity
#             would need its own re-test on 4N + torch 2.10
#
# Submit:
#   qsub -l select=2 torchtitan/experiments/ezpz/scripts/submit_50b_wide_2n_t213.sh

set -o pipefail

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate

# Yeet env to compute nodes (tarball mode if available).
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "yeet-env via tarball: .venv.tar.gz"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "yeet-env via rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

LOG_DIR="logs/agpt-50b-wide-2n-t213-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"
log="${LOG_DIR}/run.log"

log_message INFO "=========================================="
log_message INFO "agpt_50b_wide @ 2N @ torch 2.13"
log_message INFO "torch version:"
python3 -c "import torch; print(torch.__version__)" 2>&1 | tee -a "${log}"
log_message INFO "Log: ${log}"
log_message INFO "=========================================="

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config agpt_50b_wide \
    --training.steps 10 \
    --checkpoint.no_enable \
    --metrics.no_enable_wandb \
    2>&1 | tee -a "${log}"

# Note: ezpz launch wraps mpiexec and returns 0 even when mpiexec
# exits 143; grep the log for the assertion to determine pass/fail.
if grep -q "tensors_saved_with_vc_check" "${log}"; then
    log_message INFO "RESULT: CRASHED — torch 2.13 alone is enough to trigger the bug"
elif grep -q "step: *10" "${log}"; then
    log_message INFO "RESULT: SUCCESS — bug needs 4N (or larger) to trigger"
else
    log_message INFO "RESULT: INCONCLUSIVE — neither assertion nor step:10 in log"
fi
