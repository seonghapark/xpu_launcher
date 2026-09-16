#!/bin/bash --login
#PBS -N agpt-80b-no-compile-t213
#PBS -l walltime=01:30:00
#PBS -l filesystems=flare:home
#PBS -A datascience
#PBS -q workq
#PBS -j oe

# Smoke test: can we actually train 80B v2 on torch 2.13?
#
# Today's bisect (jobs 12465952 + 12465962) confirmed the AOT autograd
# DeviceMesh-in-saved-tensors crash fires on every 80B-family config
# at every node count we have tested ON TORCH 2.13 — but only with
# compile + AC + TP=2 all on. Drop compile and the assertion path
# never runs.
#
# This job tests the natural next-best 80B v2 production candidate:
#   agpt_80b at TP=2, AC=full, compile=OFF, AdamW LR=1e-6, fp32 master
# on torch 2.13. If it survives ~10-20 steps with sane loss/grad_norm
# and no NaN, we have a path forward for 80B v2 production. If it
# crashes/NaNs/OOMs, we need a different approach (smaller LR, more
# nodes, smaller model variant, etc.).
#
# 4N is chosen so dp_shard=24 cuts the per-rank param+optim memory
# in half vs 2N (dp_shard=12) — earlier 80B compile=OFF runs at TP=2
# topped out at ~95% memory; 4N should give comfortable headroom for
# activations and grad accumulation.
#
# Submit:
#   qsub -l select=4 torchtitan/experiments/ezpz/scripts/submit_80b_no_compile_t213.sh

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

# `ezpz_setup_job` overwrites $PBS_O_WORKDIR with the script's initial
# cwd ($HOME under default qsub). Stash the real submit dir first.
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${SUBMIT_DIR}"
source .venv/bin/activate

# `ezpz yeet-env` is deprecated in ezpz 0.18.x; explicit tar-env + yeet
# is the canonical sequence. Build the tarball fresh so the broadcast
# reflects the active .venv.
ezpz tar-env
ezpz yeet .venv.tar.gz
deactivate
source /tmp/.venv/bin/activate

LOG_DIR="logs/agpt-80b-no-compile-t213-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"
log="${LOG_DIR}/run.log"

log_message INFO "=========================================="
log_message INFO "agpt_80b — compile=OFF, AC=full, TP=2, AdamW LR=1e-6, torch 2.13, 4N"
python3 -c "import torch; print(f'torch: {torch.__version__}')" 2>&1 | tee -a "${log}"
log_message INFO "Log: ${log}"
log_message INFO "=========================================="

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module ezpz.agpt \
    --config agpt_80b \
    --training.steps 20 \
    --checkpoint.no_enable \
    --metrics.no_enable_wandb \
    --compile.no_enable \
    --optimizer.lr 1e-6 \
    2>&1 | tee -a "${log}"

# ezpz launch wraps mpiexec and returns 0 even when mpiexec exits 143.
# Detect outcome by grepping the log for terminal signatures.
if grep -q "tensors_saved_with_vc_check" "${log}"; then
    log_message INFO "RESULT: CRASHED — DeviceMesh assertion fired (compile=OFF should bypass — investigate)"
elif grep -q "loss: nan" "${log}" || grep -qE "loss: *nan|grad_norm: *nan" "${log}"; then
    log_message INFO "RESULT: NaN — LR=1e-6 still too aggressive on torch 2.13"
elif grep -q "UR_RESULT_ERROR_OUT_OF_RESOURCES\|out of memory" "${log}"; then
    log_message INFO "RESULT: OOM — need more nodes or smaller LBS"
elif grep -qE "step: *20" "${log}"; then
    log_message INFO "RESULT: SUCCESS — 80B v2 trains on torch 2.13 with compile=OFF"
elif grep -qE "step: *(1[0-9])" "${log}"; then
    log_message INFO "RESULT: PARTIAL — got past step 10, see log for stop reason"
else
    log_message INFO "RESULT: INCONCLUSIVE — see log"
fi
