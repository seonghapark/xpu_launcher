#!/bin/bash --login
#PBS -N agpt-20b-sophiag-olmo-mix-n1024
#PBS -l select=1024
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -A AuroraGPT
#PBS -q prod
#PBS -k doe
#PBS -j oe

# ---- Environment ----
cd "${PBS_O_WORKDIR}" || exit 1

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_env
# Kill stale palsd processes from previous runs, but spare our own process tree
_my_pids=$(ps -o pid= --ppid $$ 2>/dev/null | tr '\n' '|')
_stale_palsd=$(ps aux | grep -E "$USER.+palsd" | grep -v grep | grep -v -E "^\S+\s+($$|${_my_pids%|})\s" | awk '{print $2}')
if [[ -n "$_stale_palsd" ]]; then
    log_message INFO "Killing stale palsd processes: $_stale_palsd"
    echo "$_stale_palsd" | xargs -r kill 2>/dev/null || true
fi
unset _my_pids _stale_palsd


# ---- Configuration ----
MODEL="20b"
NNODES="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-1}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-1}"
GAS="${GAS:-1}"
GBS=$(( NGPUS * LBS * GAS / (TP * PP * CP) ))

TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
# 4,673,780,159,710 tokens / (12288 * 8192) = 46,429 steps
TRAINING_STEPS=$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))

OPTIMIZER="${OPTIMIZER:-sophiag}"
LR="${LR:-2.28e-5}"

DFL_PARENT="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)"
DFL_NAME="${DFL_NAME:-olmo-mix-1124}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

CKPT_KEEP_LATEST_K="${CKPT_KEEP_LATEST_K:-0}"
CKPT_INTERVAL=100
CKPT_DIR="checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}"

dcp=".cache/${DFL_NAME}/index-cache"
DATA_CACHE_PATH="${CKPT_DIR}/${dcp}"

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

if ! command -v ezpz >/dev/null; then
    uv pip install --no-cache --link-mode=copy "git+https://github.com/saforem2/ezpz"
fi

# ---- Launch ----
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}" \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${CKPT_INTERVAL}" \
    --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
    --checkpoint.no-last-save-model-only \
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
