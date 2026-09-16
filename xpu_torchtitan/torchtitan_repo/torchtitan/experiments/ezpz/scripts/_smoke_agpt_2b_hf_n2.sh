#!/bin/bash --login
#PBS -A datascience
#PBS -N smoke-agpt-2b-hf
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# 2N Sunspot smoke for the agpt_2b config with an HF-streamed dataset
# (eliplutchok/fineweb-small-sample, already in ~/.cache). Validates
# the rank-0 prefetch + barrier hook in torchtitan/experiments/ezpz/
# datasets.py landed by commit 1ea2dce76 (49th upstream sync).
#
# Pairs with the bitwise smoke at 12468309 (blendcorpus path) and the
# moe smoke (separate launch). 10 steps is enough to surface a
# rank-0-doesn't-prefetch / 429-storm regression.

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"
source .venv/bin/activate

JOBID_SHORT="${PBS_JOBID%%.*}"
LOG_DIR="${SUBMIT_DIR}/logs/smoke-agpt-2b-hf-${JOBID_SHORT}"
mkdir -p "${LOG_DIR}"

# Same memory-fit overrides as bitwise_sync_check.sh — chunked CE + full AC
# keep peak memory under the 64 GB tile budget without changing numerics.
#
# NOTE: `--dataloader.num-workers=0` is REQUIRED for HF-streamed datasets.
# With N>0, each rank spawns N independent worker processes that each
# iterate the dataset; on tiny samples (e.g. eliplutchok/fineweb-small-sample)
# the workers race ahead of training and spam "re-loop epoch K" warnings
# (job 12468322 logged 2.4M re-loops in 30min and never trained a step).
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config=agpt_2b_chunkedce \
    --compile.no-enable \
    --checkpoint.no-enable \
    --dataloader.dataset=eliplutchok/fineweb-small-sample \
    --dataloader.num-workers=0 \
    --training.local-batch-size=1 \
    --training.seq-len=8192 \
    --training.steps=10 \
    --optimizer=sophiag \
    --optimizer.lr=2.28e-5 \
    activation-checkpoint:full \
    2>&1 | tee "${LOG_DIR}/run.log"
