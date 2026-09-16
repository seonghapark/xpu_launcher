#!/bin/bash --login
#PBS -A AuroraGPT
#PBS -N blendcorpus-prewarm-1rank
#PBS -l walltime=00:60:00
#PBS -l filesystems=home:flare
#PBS -q debug
#PBS -j oe
# select=1 -- a SINGLE-RANK index-cache build (no concurrent readers ->
# no race), to pre-build the blendcorpus index cache before a large run.

# Why single-rank (vs scripts/prewarm_blendcorpus_cache.sh):
#
# The blendcorpus `_build_index_mappings` path (gpt_dataset.py) builds the
# per-corpus shuffle/sample/doc index on rank 0, then ALL ranks np.load it
# with NO barrier between write and read. At TP>1 the ranks whose TP
# coordinate != 0 race ahead and mmap the file mid-write
# (EOFError / "mmap length > file size" / "invalid load key '\x00'").
# blendcorpus 74b09fd deliberately does NOT barrier this path (a barrier
# there fires a data-dependent count across ranks -> oneCCL hang), so the
# documented mitigation is to pre-build the cache. But the existing
# all-ranks prewarm can ITSELF race at TP>1.
#
# A SINGLE rank has no concurrent readers, so the write-then-read is
# trivially safe. The index cache is model-independent -- its hash is
# `desc = prefix + num_samples + seq_length + seed` where
# `num_samples = global_batch_size * train_iters` (gpt_dataset.py:51,121),
# with NO model term -- so we build it with the tiny `agpt_debugmodel`
# (fits on one GPU) and it produces the byte-identical cache a full 80B
# run at the same GBS/STEPS/SEQ_LEN will load.
#
# Usage -- pass the SAME GBS / TRAINING_STEPS / SEQ_LEN / CKPT_DIR the
# target run resolves to, so the cache path + hash match exactly:
#   qsub -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home:flare \
#     -v GBS=12,TRAINING_STEPS=20,SEQ_LEN=8192,\
# CKPT_DIR=checkpoints/agpt-80b-adamw-olmo-mix-1124-n4-gbs12 \
#     torchtitan/experiments/ezpz/scripts/prewarm_blendcorpus_singlerank.sh
#
# Required: GBS, CKPT_DIR (must equal the target run resolved values).
# Optional: TRAINING_STEPS (default 20), SEQ_LEN (8192), DFL_NAME
# (olmo-mix-1124 on Aurora), DATASET (blendcorpus).

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

# ---- Required matching params ----
GBS="${GBS:?set GBS to match the target run (cache hash uses GBS*STEPS)}"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR to the target run resolved checkpoint dir}"
TRAINING_STEPS="${TRAINING_STEPS:-20}"
SEQ_LEN="${SEQ_LEN:-8192}"
DATASET="${DATASET:-blendcorpus}"

MACHINE="$(ezpz_get_machine_name)"
case "$MACHINE" in
    aurora)  DEFAULT_DFL_NAME="olmo-mix-1124" ;;
    *)       DEFAULT_DFL_NAME="books" ;;
esac
DFL_PARENT="torchtitan/experiments/ezpz/data-lists/${MACHINE}"
DFL_NAME="${DFL_NAME:-$DEFAULT_DFL_NAME}"
DFL="${DFL_PARENT}/${DFL_NAME}.txt"
DATA_CACHE_PATH="${CKPT_DIR}/.cache/${DFL_NAME}/index-cache"

log_message INFO "==========================================="
log_message INFO "SINGLE-RANK blendcorpus index pre-warm"
log_message INFO "  cache path : ${DATA_CACHE_PATH}"
log_message INFO "  GBS=${GBS}  STEPS=${TRAINING_STEPS}  SEQ_LEN=${SEQ_LEN}"
log_message INFO "  num_samples (=GBS*STEPS) = $(( GBS * TRAINING_STEPS ))"
log_message INFO "  data list  : ${DFL}"
log_message INFO "  --nproc 1 (no concurrent readers -> race-free build)"
log_message INFO "==========================================="

# Single rank: --nproc 1. agpt_debugmodel is tiny (1-GPU) and the cache is
# model-independent. Checkpoint + compile off; validator off (eval_iters=0
# matches a VALIDATOR_ENABLE=0 target; if your target enables the
# validator, add --validator.enable + --validator.steps to also build the
# validation-split index here).
ezpz launch --nproc 1 --nproc_per_node 1 \
    -- \
    python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config=agpt_debugmodel \
    --checkpoint.no-enable \
    --compile.no-enable \
    --dataloader.dataset="${DATASET}" \
    --dataloader.dataset-path="${DFL}" \
    --dataloader.data-cache-path="${DATA_CACHE_PATH}" \
    --dataloader.num-workers=2 \
    --validator.no-enable \
    --training.global-batch-size="${GBS}" \
    --training.seq-len="${SEQ_LEN}" \
    --training.steps="${TRAINING_STEPS}" \
    "$@"

log_message INFO "single-rank pre-warm done -- cache at ${DATA_CACHE_PATH}"
log_message INFO "submit the real run with the SAME GBS/STEPS/SEQ_LEN/CKPT_DIR."
