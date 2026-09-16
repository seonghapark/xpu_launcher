#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N agpt-80b-autoretry
#PBS -l walltime=12:00:00
#PBS -l filesystems=home:flare
#PBS -q prod
#PBS -j oe
# select= overridden via qsub -l select=N (e.g. 64 = 62 train + 2 spare)

# Native-auto-retry variant for 80B v2 production training.
#
# Uses `ezpz launch --auto-retry` EXCLUSIVELY for bad-node failover (no
# scripts/failover_lib.sh). See submit_agpt_2b_autoretry.sh and
# docs/guides/bad-node-failover.md (current native path).
#
# DEFAULT CONFIG = the confirmed-stable corner (2026-06-24 investigation,
# docs/production/agpt/80b/README.md): TP=4, LBS=1, AdamW LR=1e-6,
# bf16-compute / fp32-master (the agpt_80b builder default -- no dtype
# flag), AC=full, compile=OFF. This SUPERSEDES the old TP=2 failover
# default, which NaNs at production GBS.
#
# Why TP=4/LBS=1: the grad-path NaN has two triggers -- LBS>1, and large
# dp_degree (= NGPUS/TP). The safe corner is LBS=1 AND dp_degree <= ~186;
# you reach a target GBS by adding GAS (sequential microbatches), NOT by
# raising LBS or dp_degree. TP=4 helps only because it halves dp_degree at
# fixed NGPUS. CAVEAT: validated to 30 steps / GBS=372 / 62N; the safe
# corner is NOT yet known to scale past ~62N (at 64N+, dp_degree = NGPUS/4
# alone exceeds 186 and GAS cannot lower it). The underlying TP/LBS
# grad-path overflow is an open upstream-worthy bug. This script WARNS when
# dp_degree > 186 so you don't silently enter the NaN regime.
#
# Submit (Aurora, 62 active + 2 spare):
#   qsub -l select=64 -l walltime=12:00:00 -v NHOSTS_TRAIN=62 \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_80b_autoretry.sh
# Sunspot (override the #PBS directives; qsub flags beat #PBS):
#   qsub -A datascience -q workq -l filesystems=flare:home \
#     -l select=64 -v NHOSTS_TRAIN=62 \
#     torchtitan/experiments/ezpz/scripts/submit_agpt_80b_autoretry.sh
#
# Required: NHOSTS_TRAIN. Optional: MAX_FAILOVER_RETRIES (default
# unbounded; consider 2 -- each 80B retry pays ~5-15 min model-build +
# dataloader-init), IDLE_TIMEOUT (1800), GBS (default keeps dp_degree
# within the safe corner), TP/LBS/OPTIMIZER/LR/DATASET/ACKPT_MODE/...

if [[ -z "${NHOSTS_TRAIN:-}" ]]; then
    echo "ERROR: NHOSTS_TRAIN env var required (use qsub -v NHOSTS_TRAIN=N)"
    exit 1
fi

# ---- Common Intel-XPU runtime env (machine-agnostic bits) ----
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

# ---- Source ezpz-utils ----
EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

cd "${PBS_O_WORKDIR:-$(pwd)}"

# ---- Environment setup + venv broadcast ----
# `ezpz launch --auto-retry` does NOT broadcast the venv to spare nodes.
# We leave PBS_NODEFILE whole (ezpz splits internally from --nproc) and
# yeet .venv.tar.gz to the full allocation, so any swapped-in spare
# already has /tmp/.venv.
ezpz_load_modules
ezpz_setup_job
source .venv/bin/activate
if [[ -f .venv.tar.gz ]]; then
    ezpz yeet --src .venv.tar.gz
else
    ezpz yeet
fi
deactivate
source /tmp/.venv/bin/activate

# Kill stale palsd processes from previous runs.
_my_pids=$(ps -o pid= --ppid $$ 2>/dev/null | tr '\n' '|')
_stale_palsd=$(ps aux | grep -E "$USER.+palsd" | grep -v grep | grep -v -E "^\S+\s+($$|${_my_pids%|})\s" | awk '{print $2}')
if [[ -n "$_stale_palsd" ]]; then
    log_message INFO "Killing stale palsd processes: $_stale_palsd"
    echo "$_stale_palsd" | xargs -r kill 2>/dev/null || true
fi
unset _my_pids _stale_palsd

# ---- Active-rank arithmetic ----
# ezpz_setup_job sees the FULL allocation; we need ACTIVE ranks only.
# Active count is constant across retries (in-place swap by index).
PPN="${NGPU_PER_HOST:-12}"
NGPUS_ACTIVE=$(( NHOSTS_TRAIN * PPN ))   # == --nproc passed to ezpz launch
NNODES="$NHOSTS_TRAIN"                    # for CKPT_DIR naming

# ---- Configuration ----
# Defaults = confirmed-stable TP=4/LBS=1/AdamW/bf16 corner (see header).
MODEL="80b"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-4}"
PP="${PP:-1}"
CP="${CP:-1}"
LBS="${LBS:-1}"
GAS="${GAS:-1}"

# dp_degree = NGPUS / (TP*PP*CP) is the data-parallel rank count -- the
# grad-path NaN trigger. Compute it so we can (a) default GBS to the safe
# value and (b) warn if it exceeds the validated ceiling.
DP_DEGREE=$(( NGPUS_ACTIVE / (TP * PP * CP) ))
# GBS defaults to dp_degree * LBS * GAS (i.e. GAS=1 -> GBS=dp_degree).
# To hit a larger token target, raise GAS (sequential microbatches) which
# scales GBS WITHOUT touching dp_degree -- the safe lever. Overridable.
GBS="${GBS:-$(( DP_DEGREE * LBS * GAS ))}"

# NaN-regime guard. The safe corner is dp_degree <= ~186 (validated at 62N,
# TP=4). Past that, the grad-path overflow is a known, unfixed bug -- warn
# loudly rather than let it silently NaN a multi-hour job.
if (( DP_DEGREE > 186 )); then
    log_message WARN "==========================================================="
    log_message WARN "dp_degree=${DP_DEGREE} (= NGPUS/${TP}/${PP}/${CP}) EXCEEDS the"
    log_message WARN "validated safe ceiling (~186). The 80B grad-path NaN trigger"
    log_message WARN "is large dp_degree; GAS cannot lower it. This config is"
    log_message WARN "NaN-prone and NOT validated. Raise TP, drop NHOSTS_TRAIN, or"
    log_message WARN "see docs/production/agpt/80b/README.md before proceeding."
    log_message WARN "==========================================================="
fi

TRAIN_TOKENS="${TRAIN_TOKENS:-4673780159710}"
TRAINING_STEPS="${TRAINING_STEPS:-$(( TRAIN_TOKENS / (GBS * SEQ_LEN) ))}"

# 80B uses AdamW LR=1e-6 -- SophiaG/Muon are known broken at dim=9216
# (bf16 overflow in Hessian/Newton-Schulz). LR=1.1e-5 NaNs at production
# GBS; LR=1e-6 was stable.
OPTIMIZER="${OPTIMIZER:-adamw}"
LR="${LR:-1e-6}"
# Linear-warmup length (steps). Default 200 matches the 2B/20B v2 configs.
# NOTE: the scheduler clamps warmup to total_steps when warmup > total, so
# short smoke/sim runs (steps < 200) are effectively warming up the whole
# time (effective LR = LR * step/total). Lower this to exercise the full
# nominal LR within a short run (e.g. WARMUP_STEPS=5 for an LR probe).
WARMUP_STEPS="${WARMUP_STEPS:-200}"

# LR-scheduler decay-ratio. The agpt base config defaults to a WSD decay
# (decay_ratio=0.8: stable then linear decay over the last 80% of steps).
# Set DECAY_RATIO=0 for warmup-then-CONSTANT (no decay) -- needed for
# continued pre-training (CPT) past the nominal token budget, where the LR
# must stay flat after warmup. Default empty = leave the config default
# untouched (decay as before). When set, passed as
# --lr-scheduler.decay-ratio=${DECAY_RATIO}.
DECAY_RATIO="${DECAY_RATIO:-}"

# Validation: run the EzpzValidator on the blendcorpus validation split
# every VALIDATOR_FREQ steps for VALIDATOR_STEPS iters. On by default;
# set VALIDATOR_FREQ to a huge number or pass --validator.no-enable to skip.
# NOTE: 80B stays on the plain `agpt_80b` flavor (complex RoPE) -- it runs
# compile OFF, where the `_real` cos_sin RoPE win (a torch.compile lowering
# optimization) does not apply, and agpt_80b is the numerically-validated
# config. So no CONFIG_SUFFIX=_real default here, unlike 2B/20B.
VALIDATOR_FREQ="${VALIDATOR_FREQ:-100}"
VALIDATOR_STEPS="${VALIDATOR_STEPS:-10}"

# Activation checkpoint is required at 80B to fit in tile memory.
# AC is a tyro subcommand union (57th sync, PR #3674): the
# `activation-checkpoint:<policy>` token is positional and must be passed
# LAST (after every --flag and "$@").
ACKPT_MODE="${ACKPT_MODE:-full}"
case "${ACKPT_MODE}" in
    none)      ACKPT_SUBCOMMAND="activation-checkpoint:none" ;;
    full)      ACKPT_SUBCOMMAND="activation-checkpoint:full" ;;
    selective) ACKPT_SUBCOMMAND="activation-checkpoint:selective" ;;
    *) echo "Unknown ACKPT_MODE: ${ACKPT_MODE}" >&2; exit 1 ;;
esac

# Per-machine data default (see submit_agpt_2b_autoretry.sh for the
# Sunspot /gila-not-mounted rationale). DFL_NAME stays overridable.
MACHINE="$(ezpz_get_machine_name)"
case "$MACHINE" in
    aurora)  DEFAULT_DFL_NAME="olmo-mix-1124" ;;
    sunspot) DEFAULT_DFL_NAME="books" ;;
    *)       DEFAULT_DFL_NAME="books" ;;
esac
DFL_PARENT="torchtitan/experiments/ezpz/data-lists/${MACHINE}"
DFL_NAME="${DFL_NAME:-$DEFAULT_DFL_NAME}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"

# DATASET selects the dataloader path: blendcorpus (default) or an HF
# <org>/<repo> for streaming (machines without the curated mix).
DATASET="${DATASET:-blendcorpus}"

# CKPT_KEEP_LATEST_K is HARDCODED to 0 (= keep all). DO NOT change.
# keep_latest_k>0 makes torchtitan delete every prior step-* dir on save,
# irreversibly (lost ~334 2B chain ckpts on 2026-05-25 to one accidental
# override, job 8505252). For rotation use a fresh CKPT_DIR on a separate
# experiment -- not the canonical chain.
if [[ -n "${CKPT_KEEP_LATEST_K:-}" && "${CKPT_KEEP_LATEST_K}" != "0" ]]; then
    echo "ERROR: CKPT_KEEP_LATEST_K=${CKPT_KEEP_LATEST_K} is set but this script hardcodes it to 0." >&2
    echo "       Setting keep_latest_k>0 will destroy older ckpts in the canonical chain." >&2
    echo "       If you really want rotation, use a fresh CKPT_DIR override on a separate experiment." >&2
    exit 1
fi
CKPT_KEEP_LATEST_K=0
CKPT_INTERVAL="${CKPT_INTERVAL:-100}"
if [[ "${DATASET}" == "blendcorpus" ]]; then
    _CKPT_DATASET_SLUG="${DFL_NAME}"
else
    _CKPT_DATASET_SLUG="$(echo "${DATASET}" | tr '/' '-')"
fi
CKPT_DIR="${CKPT_DIR:-checkpoints/agpt-${MODEL}-${OPTIMIZER}-${_CKPT_DATASET_SLUG}-n${NNODES}-gbs${GBS}}"
DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "Training ${MODEL} v2 (native --auto-retry, ${NNODES} active nodes)"
log_message INFO "-------------------------------------------"
log_message INFO "machine: ${MACHINE}"
log_message INFO "PBS allocation: $(wc -l < "${PBS_NODEFILE}") nodes total"
log_message INFO "  active (NHOSTS_TRAIN): ${NHOSTS_TRAIN}  spare: --spare-nodes auto (rest)"
log_message INFO "--nproc (active ranks): ${NGPUS_ACTIVE}  -ppn: ${PPN}"
log_message INFO "MAX_FAILOVER_RETRIES: ${MAX_FAILOVER_RETRIES:-unbounded}"
log_message INFO "IDLE_TIMEOUT: ${IDLE_TIMEOUT:-1800}"
log_message INFO "TRAINING_STEPS: ${TRAINING_STEPS}"
log_message INFO "PBS_JOBID: ${PBS_JOBID}"
log_message INFO "OPTIMIZER: ${OPTIMIZER}"
log_message INFO "LR: ${LR}  (warmup_steps: ${WARMUP_STEPS})"
if [[ -n "${DECAY_RATIO}" ]]; then
    if [[ "${DECAY_RATIO}" == "0" || "${DECAY_RATIO}" == "0.0" ]]; then
        log_message INFO "LR schedule: warmup then CONSTANT (decay_ratio=${DECAY_RATIO}, no decay -- CPT mode)"
    else
        log_message INFO "LR schedule: WSD (decay_ratio=${DECAY_RATIO})"
    fi
else
    log_message INFO "LR schedule: config default (WSD decay_ratio=0.8)"
fi
log_message INFO "TP: ${TP}, LBS: ${LBS}, GAS: ${GAS}, dp_degree: ${DP_DEGREE}, AC: ${ACKPT_MODE}, compile: OFF"
log_message INFO "GBS: ${GBS}"
log_message INFO "DATASET: ${DATASET}"
log_message INFO "Checkpoint directory: ${CKPT_DIR}"
log_message INFO "==========================================="

# Build dataloader flag set based on DATASET shape. The validator reads
# the validation split of the same corpus, so for blendcorpus we point its
# dataloader at the same $DFL; for HF streaming it inherits the config
# default (no explicit path).
if [[ "${DATASET}" == "blendcorpus" ]]; then
    DATALOADER_FLAGS=(
        "--dataloader.dataset=blendcorpus"
        "--dataloader.dataset-path=${DFL}"
        "--dataloader.data-cache-path=${DATA_CACHE_PATH}"
        "--dataloader.num-workers=2"
    )
    VALIDATOR_DATA_FLAGS=(
        "--validator.dataloader.dataset-path=${DFL}"
        "--validator.dataloader.data-cache-path=${DATA_CACHE_PATH}"
    )
else
    DATALOADER_FLAGS=(
        "--dataloader.dataset=${DATASET}"
        "--dataloader.num-workers=2"
    )
    VALIDATOR_DATA_FLAGS=()
fi

# Validator flag set, gated by VALIDATOR_ENABLE (default 1 = on). Set
# VALIDATOR_ENABLE=0 to skip validation entirely (no --validator.enable,
# no validator dataloader build). Useful when validation isn't the point
# of a run (e.g. batch-size/throughput studies) or to sidestep a validator
# collective issue at scale.
if [[ "${VALIDATOR_ENABLE:-1}" == "1" ]]; then
    VALIDATOR_FLAGS=(
        "--validator.enable"
        "--validator.freq=${VALIDATOR_FREQ}"
        "--validator.steps=${VALIDATOR_STEPS}"
        "${VALIDATOR_DATA_FLAGS[@]}"
    )
else
    VALIDATOR_FLAGS=("--validator.no-enable")
fi

# ---- Launch with native auto-retry ----
# No preflight: ezpz's STUCK_PRE_TRAINING guard bails (without burning
# spares) on a twice-zero-progress init crash. Key 80B-specific flags:
#   --parallelism.tensor-parallel-degree=${TP}  (TP>1 required for 80B)
#   --parallelism.expert-parallel-degree=1      (dense, no MoE)
#   --parallelism.data-parallel-{replicate=1,shard=-1}  (pure FSDP)
#   --compile.no-enable                         (compile crashes on torch 2.13)
#   activation-checkpoint:<policy>              (positional, passed LAST)
#
# Build the optional --max-failover-retries as an ARRAY (an inline
# ${VAR:+--flag "$VAR"} collapses to a single argv token argparse rejects).
mfr_args=()
[[ -n "${MAX_FAILOVER_RETRIES:-}" ]] && mfr_args=(--max-failover-retries "${MAX_FAILOVER_RETRIES}")

# Optional --lr-scheduler.decay-ratio (ARRAY for the same reason). Empty
# DECAY_RATIO leaves the config's base decay_ratio (0.8 WSD) untouched;
# DECAY_RATIO=0 yields warmup-then-constant (no decay) for CPT.
lr_sched_args=()
[[ -n "${DECAY_RATIO}" ]] && lr_sched_args=(--lr-scheduler.decay-ratio="${DECAY_RATIO}")

ezpz launch \
    --nproc "${NGPUS_ACTIVE}" \
    --nproc_per_node "${PPN}" \
    --auto-retry \
    --spare-nodes auto \
    --timeout "${IDLE_TIMEOUT:-1800}" \
    "${mfr_args[@]}" \
    -- \
    python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}${CONFIG_SUFFIX:-}" \
    --checkpoint.enable \
    --checkpoint.folder="${CKPT_DIR}" \
    --checkpoint.interval="${CKPT_INTERVAL}" \
    --checkpoint.keep-latest-k="${CKPT_KEEP_LATEST_K}" \
    --checkpoint.no-last-save-model-only \
    --checkpoint.async-mode="${CHECKPOINT_ASYNC_MODE:-disabled}" \
    "${DATALOADER_FLAGS[@]}" \
    "${VALIDATOR_FLAGS[@]}" \
    --debug.print-config \
    --optimizer="${OPTIMIZER}" \
    --optimizer.lr="${LR}" \
    --lr-scheduler.warmup-steps="${WARMUP_STEPS}" \
    "${lr_sched_args[@]}" \
    --parallelism.tensor-parallel-degree="${TP}" \
    --parallelism.expert-parallel-degree=1 \
    --parallelism.data-parallel-replicate-degree=1 \
    --parallelism.data-parallel-shard-degree=-1 \
    --compile.no-enable \
    --training.local-batch-size="${LBS}" \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@" \
    "${ACKPT_SUBCOMMAND}"
