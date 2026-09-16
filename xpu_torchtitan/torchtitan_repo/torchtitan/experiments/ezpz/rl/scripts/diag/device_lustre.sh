#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:25:00
#PBS -l filesystems=flare:home
#PBS -l select=4
#PBS -q workq
#PBS -j oe
#
# 4-node Sunspot diagnostic for the FSDP `Inconsistent compute device
# and device_id` bug, sourcing the venv DIRECTLY from /flare/ instead of
# tarballing + yeeting it. Saves 3-5min tarball-build + the silently-
# stale-tarball foot-gun (June 6 tarball missing trl bit us already).
#
# On Sunspot /flare/ behaves well enough that compute-node direct-import
# is acceptable. Don't do this on Aurora /flare with high concurrency.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${SUBMIT_DIR}"
# Direct activate from lustre — no tar-env, no yeet
source .venv/bin/activate

# Sanity-check trl is importable BEFORE wasting compute-node time
python3 -c "import trl; print('trl', trl.__version__)" || {
    echo "FATAL: trl not in .venv — abort"; exit 1
}

LOG_DIR="logs/diag-fsdp-device-4n-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"

echo "=== PHASE 1: diagnostic (per-rank device state, 48 ranks) ===" | tee "${LOG_DIR}/phase1.log"
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.scripts.diag.device_mismatch \
    2>&1 | tee -a "${LOG_DIR}/phase1.log" || true

echo "" | tee -a "${LOG_DIR}/phase2.log"
echo "=== PHASE 2: train_grpo --fsdp full_shard (Qwen3-0.6B) ===" | tee "${LOG_DIR}/phase2.log"
ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_grpo \
    --task sum_digits \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --bf16 --beta 0.0 --fsdp full_shard --max_steps 2 \
    2>&1 | tee -a "${LOG_DIR}/phase2.log" || true

echo "" | tee -a "${LOG_DIR}/phase2.log"
echo "=== DONE: logs in ${LOG_DIR}/ ===" | tee -a "${LOG_DIR}/phase2.log"
