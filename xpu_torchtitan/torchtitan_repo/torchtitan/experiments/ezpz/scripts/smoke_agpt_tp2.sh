#!/bin/bash --login
#PBS -A datascience
#PBS -N smoke-agpt-tp2
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare:home
#PBS -l select=2
#PBS -q workq
#PBS -j oe
#
# Small TP=2 sanity smoke for `agpt_${MODEL}` configs. Runs a few
# `--debug.seed=42 --debug.deterministic` steps and prints the
# step-1 loss for comparison against a known-good baseline.
#
# Originally written to validate the removal of the
# `loss.full_tensor()` workaround in ezpz/trainer.py + validator.py
# (after upstream PR #3159 made the `_dist_reduce` DTensor path
# correct via `to_local()`). Generalized to smoke any TP > 1 agpt
# change since:
#
#  - Loss reporting under TP > 1 was historically the canary
#    (off-by-`dp_world_size` if the DTensor reduction path is wrong).
#  - For `agpt_2b` the known-good step-1 loss is ~12.94 nats.
#    Anything that looks like `~12.94 / dp_world_size` (≈1.08 at
#    NGPUS=24) is the regression signature.
#
# Usage from a login node:
#   qsub -l select=2 -v MODEL=2b,STEPS=3 \
#       torchtitan/experiments/ezpz/scripts/smoke_agpt_tp2.sh
#
# Env knobs:
#   MODEL    default `2b` — picks `agpt_${MODEL}_chunkedce`
#   STEPS    default 3
#   SEED     default 42 (matches CLAUDE.md convention)

set -o pipefail

module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR
export http_proxy="${http_proxy:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="${https_proxy:-http://proxy.alcf.anl.gov:3128}"

MODEL="${MODEL:-2b}"
STEPS="${STEPS:-3}"
SEED="${SEED:-42}"

SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"
source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job
cd "${SUBMIT_DIR}"

source .venv/bin/activate
ezpz tar-env
ezpz yeet .venv.tar.gz
deactivate
source /tmp/.venv/bin/activate

LOG_DIR="logs/smoke-agpt-${MODEL}-tp2-${PBS_JOBID%%.*}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run.log"

NNODES="$(wc -l < "${PBS_NODEFILE}")"
NGPUS=$(( NNODES * 12 ))
# TP=2 -> dp_world = NGPUS/2 -> GBS = dp_world * LBS = NGPUS/2 (LBS=1)
GBS=$(( NGPUS / 2 ))

echo "smoke: NNODES=${NNODES} NGPUS=${NGPUS} MODEL=${MODEL} TP=2 GBS=${GBS}" | tee "${LOG_FILE}"

ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="agpt_${MODEL}_chunkedce" \
    --parallelism.tensor-parallel-degree=2 \
    --compile.no-enable \
    --checkpoint.no-enable \
    --dataloader.dataset=blendcorpus \
    --dataloader.dataset-path="torchtitan/experiments/ezpz/data-lists/$(ezpz_get_machine_name)/books.txt" \
    --debug.seed="${SEED}" \
    --debug.deterministic \
    --training.local-batch-size=1 \
    --training.global-batch-size="${GBS}" \
    --training.seq-len=8192 \
    --training.steps="${STEPS}" \
    --optimizer=sophiag \
    --optimizer.lr=2.28e-5 \
    activation-checkpoint:full \
    2>&1 | tee -a "${LOG_FILE}"

echo "---" | tee -a "${LOG_FILE}"
echo "expected step-1 loss ~12.94 for agpt_2b (regression signature: loss/dp_world)" | tee -a "${LOG_FILE}"
echo "observed step-1 loss:" | tee -a "${LOG_FILE}"
sed 's/\x1b\[[0-9;]*m//g' "${LOG_FILE}" \
    | grep -oE 'step:\s+1\s+loss:\s+[0-9.]+' \
    | head -1 \
    | tee -a "${LOG_FILE}"
