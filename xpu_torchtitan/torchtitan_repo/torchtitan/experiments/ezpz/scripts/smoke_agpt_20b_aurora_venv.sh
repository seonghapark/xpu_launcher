#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N smoke-agpt-20b-v2
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -l select=4
#PBS -q debug-scaling
#PBS -j oe

# Smoke test for the agpt-20b-v2 clone:
# - validates the new torch 2.13 + xpu venv builds and runs end-to-end at 20B scale
# - validates the fp32-master fix actually unsticks RMSNorm.weight
# - 4 nodes, 30 steps, one checkpoint saved at the end (no compile)
#
# Post-run check: load `norm.weight` and confirm it's no longer all-1.0.

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
# Aurora compute nodes need the ALCF proxy for any outbound HTTP (W&B, HF Hub).
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# Source ezpz-utils (cached locally to avoid bit.ly redirect hangs).
EZPZ_UTILS="$(dirname "$(realpath "$0")")/../../../.ezpz-utils-cache/ezpz-utils.sh"
EZPZ_UTILS="$(realpath "$EZPZ_UTILS" 2>/dev/null || echo "")"
if [[ -z "$EZPZ_UTILS" || ! -f "$EZPZ_UTILS" ]]; then
    EZPZ_UTILS="${PBS_O_WORKDIR:-.}/.ezpz-utils-cache/ezpz-utils.sh"
fi
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi
ezpz_setup_job

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    log_message INFO "yeet-env via tarball: .venv.tar.gz"
    ezpz yeet-env --src .venv.tar.gz
else
    log_message INFO "yeet-env via rsync (.venv.tar.gz not present)"
    ezpz yeet-env
fi
deactivate
source /tmp/.venv/bin/activate

# ---- Configuration ----
MODEL="20b"
NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
SEQ_LEN="${SEQ_LEN:-8192}"
LBS="${LBS:-1}"
GBS=$(( NGPUS * LBS ))
TRAINING_STEPS="${TRAINING_STEPS:-30}"
OPTIMIZER="${OPTIMIZER:-sophiag}"
LR="${LR:-2.28e-5}"

DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

# Distinct smoke ckpt dir so it doesn't collide with future production runs.
CKPT_DIR="checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}-smoke"
DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "SMOKE TEST: ${MODEL} on ${TRAINING_STEPS} steps (Aurora venv, fp32 master, no compile)"
log_message INFO "-------------------------------------------"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "NNODES: ${NNODES}  WORLD_SIZE: ${WORLD_SIZE:-${NGPUS}}  GBS: ${GBS}"
log_message INFO "Checkpoint dir: ${CKPT_DIR}"
log_message INFO "==========================================="

# ---- Launch ----
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}" \
    --compile.no-enable \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${TRAINING_STEPS}" \
    --checkpoint.keep-latest-k=2 \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-async}" \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --debug.print-config \
    --optimizer="${OPTIMIZER}" \
    --optimizer.lr="${LR}" \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"
