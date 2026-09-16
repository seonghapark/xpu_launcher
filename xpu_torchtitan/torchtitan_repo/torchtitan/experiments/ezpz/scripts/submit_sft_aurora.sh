#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-2b-sft
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
# select= overridden via qsub -l select=N
#PBS -q prod
#PBS -j oe
#
# SFT for the AuroraGPT-2B SophiaG checkpoint via TRL's SFTTrainer +
# train_sft.py companion. Designed to produce a stronger starting
# point for the GRPO arithmetic task — the pretrained checkpoint has
# weak instruction-following on its own.
#
# Usage:
#   qsub -l select=8 -l walltime=06:00:00 \
#       -v SFT_DATASET=gsm8k,NUM_EPOCHS=3 \
#       torchtitan/experiments/ezpz/scripts/submit_sft_aurora.sh
#
# Env knobs (all optional, with sensible defaults):
#   SFT_DATASET             default: gsm8k (also: metamathqa)
#   MODEL_PATH              default: AuroraGPT-2B-sophiag checkpoint on /flare
#   NUM_EPOCHS              default: 3
#   LR                      default: 2e-5 (standard SFT LR for 2B llama)
#   PER_DEV_BSZ             default: 1 (FSDP-full_shard fits this safely)
#   GAS                     default: 4 (effective bsz per rank = 4)
#   MAX_LENGTH              default: 1024 (bump to 2048 for metamathqa)
#   CKPT_DIR                default: outputs/sft/{SFT_DATASET}-{NNODES}n
#   FAILOVER_MAX_RETRIES    default: 2 (passed through to failover_run)

set -o pipefail

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then
    echo "ERROR: NHOSTS_TRAIN env var required (use qsub -v NHOSTS_TRAIN=N)"
    echo "       e.g. for a 4-train + 0-spare 4-node allocation:"
    echo "       qsub -l select=4 -v NHOSTS_TRAIN=4 ..."
    exit 1
fi

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

# Stash the failover helpers if they're being used; fall back to a
# plain ezpz launch if failover_run isn't available (single-node smoke).
if command -v failover_init >/dev/null 2>&1; then
    failover_init "$NHOSTS_TRAIN" || exit 1
    LAUNCH_WRAPPER=(
        failover_run
        FAILOVER_IDLE_TIMEOUT="${PREFLIGHT_IDLE_TIMEOUT:-600}"
        FAILOVER_MAX_RETRIES="${FAILOVER_MAX_RETRIES:-2}"
    )
else
    LAUNCH_WRAPPER=()
fi

cd "${SUBMIT_DIR}"
# Use lustre venv directly — see docs/rl/README.md for why we prefer
# this over tarball + yeet on Sunspot/Aurora (flare/flare both behave
# well with concurrent direct-import at the 4-16 node scale typical
# for SFT runs).
source .venv/bin/activate

python3 -c "import trl; print('trl', trl.__version__)" || {
    echo "FATAL: trl not in .venv — abort"; exit 1
}

# --- Defaults ---------------------------------------------------------------
SFT_DATASET="${SFT_DATASET:-gsm8k}"
MODEL_PATH="${MODEL_PATH:-/flare/AuroraGPT/AuroraGPT-v1/Experiments/AuroraGPT-2B/public/sophiag/hf/global_step138650}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-2e-5}"
PER_DEV_BSZ="${PER_DEV_BSZ:-1}"
GAS="${GAS:-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
NNODES="${NHOSTS_TRAIN}"
CKPT_DIR="${CKPT_DIR:-outputs/sft/${SFT_DATASET}-${NNODES}n}"

# Sanity-check the model path exists before burning compute on init
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "ERROR: MODEL_PATH does not exist: ${MODEL_PATH}"
    echo "       (this script defaults to the AuroraGPT-2B SophiaG ckpt on /flare;"
    echo "        override MODEL_PATH= to point at a different model)"
    exit 1
fi

log_message INFO "==========================================="
log_message INFO "SFT for AuroraGPT-2B (${NNODES} nodes)"
log_message INFO "-------------------------------------------"
log_message INFO "MODEL_PATH:             ${MODEL_PATH}"
log_message INFO "SFT_DATASET:            ${SFT_DATASET}"
log_message INFO "NUM_EPOCHS:             ${NUM_EPOCHS}"
log_message INFO "LR:                     ${LR}"
log_message INFO "PER_DEV_BSZ * GAS:      ${PER_DEV_BSZ} * ${GAS}"
log_message INFO "MAX_LENGTH:             ${MAX_LENGTH}"
log_message INFO "CKPT_DIR:               ${CKPT_DIR}"
log_message INFO "PBS_JOBID:              ${PBS_JOBID}"
log_message INFO "==========================================="

mkdir -p "${CKPT_DIR}"

# --- Launch -----------------------------------------------------------------
"${LAUNCH_WRAPPER[@]}" ezpz launch python3 -m torchtitan.experiments.ezpz.rl.train_sft \
    --sft_dataset="${SFT_DATASET}" \
    --model_name_or_path="${MODEL_PATH}" \
    --output_dir="${CKPT_DIR}" \
    --num_train_epochs="${NUM_EPOCHS}" \
    --learning_rate="${LR}" \
    --per_device_train_batch_size="${PER_DEV_BSZ}" \
    --gradient_accumulation_steps="${GAS}" \
    --max_length="${MAX_LENGTH}" \
    --bf16 \
    --fsdp full_shard \
    --gradient_checkpointing \
    --report_to wandb \
    --logging_steps=10 \
    --save_strategy=no
