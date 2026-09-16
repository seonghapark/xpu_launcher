#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -l walltime=06:00:00
#PBS -l filesystems=flare:home
#PBS -q workq
#PBS -j oe

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${PBS_O_WORKDIR:-$(pwd)}"
source .venv/bin/activate
# Prefer tarball broadcast over per-file rsync — single ~3GB sequential read
# per node beats thousands of small-file rsyncs at scale (Lustre metadata).
# Falls back to plain yeet-env if tarball isn't built yet.
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
MODEL="2b"
NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-2}"
GAS="${GAS:-1}"
GBS=$(( NGPUS * LBS * GAS / (TP * PP * CP) ))

TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))

OPTIMIZER="${OPTIMIZER:-sophiag}"
LR="${LR:-2.28e-5}"

DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

CKPT_KEEP_LATEST_K="${CKPT_KEEP_LATEST_K:-0}"
CKPT_INTERVAL=100
CKPT_DIR="checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}"

DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "Training ${MODEL} on ${TRAIN_TOKENS} tokens"
log_message INFO "-------------------------------------------"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "NNODES: ${NNODES}"
log_message INFO "WORLD_SIZE: ${WORLD_SIZE:-${NGPUS}}"
log_message INFO "OPTIMIZER: ${OPTIMIZER}"
log_message INFO "LR: ${LR}"
log_message INFO "Local batch size (LBS): ${LBS}"
log_message INFO "Gradient accumulation steps (GAS): ${GAS}"
log_message INFO "Global batch size (GBS): ${GBS}"
log_message INFO "Training steps calculated as: ${TRAINING_STEPS}"
log_message INFO "DATASET_PATH: ${DFL}"
log_message INFO "Checkpoint directory: ${CKPT_DIR}"
log_message INFO "==========================================="

# ---- Launch ----
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}" \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${CKPT_INTERVAL}" \
    --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
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
