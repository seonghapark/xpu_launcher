#!/bin/bash --login
#PBS -A datascience
#PBS -N blendcorpus-prewarm
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -q workq
#PBS -j oe
# select= is small (e.g. 2): a low-rank job to BUILD the blendcorpus index
# cache serially before a large production run reads it.

# Pre-warm the blendcorpus index cache for a given (model, data, GBS, N)
# BEFORE launching a large run, so the big run finds the cache already
# built ("loading", not "building") and never exercises the index-build
# path at scale.
#
# WHY: blendcorpus builds the index/shuffle/.npy cache on global rank 0,
# then historically "waited" with only DP-group + PP-group barriers before
# every rank np.loads it. Those subgroup barriers do NOT gate ranks whose
# tensor-parallel coordinate != 0 against rank 0 (their subgroups exclude
# rank 0), so at TP>1 ~(1 - 1/TP) of ranks race ahead and np.load the
# .npy mid-write -> FileNotFoundError / "EOF: reading magic string" /
# "mmap length is greater than file size". Observed on agpt_80b TP=4 @
# 744 ranks (jobs 12469548: 3 ranks raced per-corpus shuffle_idx;
# 12469550: 558 ranks raced the blendable index).
#
# The root-cause fix is a global barrier in deps/blendcorpus
# (blendcorpus commit debfff5). This pre-warm step is DEFENSE-IN-DEPTH:
# build the cache once at small N (where the race window is negligible),
# so even an unpatched/older blendcorpus copy is safe for the big run.
# Cheap + idempotent: if the cache already exists it just loads and exits.
#
# Usage -- run with the SAME vars as the target run so the cache path
# matches (CKPT_DIR / DFL_NAME / GBS must resolve identically):
#   qsub -l select=2 -v MODEL=80b,NHOSTS_TRAIN=62,GAS=2,OPTIMIZER=adamw \
#       torchtitan/experiments/ezpz/scripts/prewarm_blendcorpus_cache.sh
#
# Then submit the real run (e.g. submit_agpt_80b_autoretry.sh) with the
# matching -v vars; it will hit the warm cache.
#
# Required: MODEL (2b|20b|80b). Optional: NHOSTS_TRAIN (default = this
# job's node count), GAS, OPTIMIZER, TP, LBS, DFL_NAME, SEQ_LEN, CKPT_DIR
# -- all must match the target run for the cache hash + path to align.

set -o pipefail

# ---- Common Intel-XPU runtime env ----
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"
export ftp_proxy="${ftp_proxy:-http://proxy.alcf.anl.gov:3128}"
export no_proxy="${no_proxy:-localhost,127.0.0.1,*.alcf.anl.gov,*.aurora.alcf.anl.gov}"

EZPZ_UTILS="${PBS_O_WORKDIR:-$PWD}/.ezpz-utils-cache/ezpz-utils.sh"
if [[ -f "$EZPZ_UTILS" ]]; then
    source "$EZPZ_UTILS"
else
    source <(curl -fsSL --max-time 30 https://bit.ly/ezpz-utils)
fi

cd "${PBS_O_WORKDIR:-$(pwd)}"

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

PPN="${NGPU_PER_HOST:-12}"
# Warm at the FULL node count of THIS (small) job -- a few ranks is enough
# to build the cache; the point is just to not be at production scale.
NHOSTS_WARM="${NHOSTS:-$(wc -l < "${PBS_NODEFILE}")}"
NGPUS_WARM=$(( NHOSTS_WARM * PPN ))

# ---- Reproduce the target run's cache path EXACTLY ----
# The cache path is CKPT_DIR/.cache/<DFL_NAME>/index-cache, and CKPT_DIR
# (when not overridden) embeds n<NNODES> + gbs<GBS>. To match the target
# run's path, pass the SAME MODEL/NHOSTS_TRAIN/GAS/OPTIMIZER/etc here, OR
# pass CKPT_DIR explicitly. NNODES/GBS below mirror the target, NOT this
# warm job's small size.
MODEL="${MODEL:?set MODEL=2b|20b|80b to match the target run}"
SEQ_LEN="${SEQ_LEN:-8192}"
TP="${TP:-$( [[ "$MODEL" == "80b" ]] && echo 4 || echo 1 )}"
PP="${PP:-1}"; CP="${CP:-1}"
LBS="${LBS:-$( [[ "$MODEL" == "80b" ]] && echo 1 || echo 2 )}"
GAS="${GAS:-1}"
# The index cache hash keys on num_samples = GBS * train_iters, so the
# prewarm MUST use the same training.steps the target run will pass, or
# the hash won't match and the target re-builds cold. Default 1 (cheapest)
# is fine when the target also runs 1 step; set TRAINING_STEPS to the
# target's step count otherwise (e.g. an LR-finder sweeping 15 steps).
TRAINING_STEPS="${TRAINING_STEPS:-1}"
OPTIMIZER="${OPTIMIZER:-$( [[ "$MODEL" == "80b" ]] && echo adamw || echo sophiag )}"
# NNODES/GBS mirror the TARGET run (so the path matches), via NHOSTS_TRAIN.
NHOSTS_TRAIN="${NHOSTS_TRAIN:-$NHOSTS_WARM}"
NGPUS_TARGET=$(( NHOSTS_TRAIN * PPN ))
if [[ "$MODEL" == "80b" ]]; then
    DP_DEGREE=$(( NGPUS_TARGET / (TP * PP * CP) ))
    GBS="${GBS:-$(( DP_DEGREE * LBS * GAS ))}"
else
    GBS="${GBS:-$(( NGPUS_TARGET * LBS * GAS / (TP * PP * CP) ))}"
fi
NNODES="$NHOSTS_TRAIN"

MACHINE="$(ezpz_get_machine_name)"
case "$MACHINE" in
    aurora)  DEFAULT_DFL_NAME="olmo-mix-1124" ;;
    *)       DEFAULT_DFL_NAME="books" ;;
esac
DFL_PARENT="torchtitan/experiments/ezpz/data-lists/${MACHINE}"
DFL_NAME="${DFL_NAME:-$DEFAULT_DFL_NAME}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"
CKPT_DIR="${CKPT_DIR:-checkpoints/agpt-${MODEL}-${OPTIMIZER}-${DFL_NAME}-n${NNODES}-gbs${GBS}}"
DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "blendcorpus cache PRE-WARM (small-N build)"
log_message INFO "  machine: ${MACHINE}  warm ranks: ${NGPUS_WARM} (${NHOSTS_WARM} nodes)"
log_message INFO "  target run: ${MODEL} n${NNODES} gbs${GBS} -> cache path:"
log_message INFO "    ${DATA_CACHE_PATH}"
log_message INFO "  data list: ${DFL}"
log_message INFO "==========================================="

# Build the cache by running 1 training step at the warm (small) rank
# count, checkpoint + compile OFF (we only want the dataloader to build
# the index). Same dataset/data-cache-path the target run will use.
#
# Also enable the validator so its VALIDATION-SPLIT index is built into the
# same cache dir: the real run enables the validator, whose dataloader
# builds a SEPARATE (validation) index. If we don't build it here, the real
# run builds it cold at full scale -- the exact race that crashed job
# 12469584. --validator.freq=1 forces one validation pass during the single
# training step so the val index gets written.
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}" \
    --checkpoint.no-enable \
    --compile.no-enable \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --dataloader.num-workers=2 \
    --validator.enable \
    --validator.freq=1 \
    --validator.steps=1 \
    --validator.dataloader.dataset-path="${DFL}" \
    --validator.dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"

log_message INFO "pre-warm done -- cache at ${DATA_CACHE_PATH} is now built."
log_message INFO "submit the real run with matching MODEL/NHOSTS_TRAIN/GAS/OPTIMIZER."
