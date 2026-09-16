#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=03:00:00
#PBS -l filesystems=flare:home
#PBS -q workq
#PBS -j oe
#
# Competition run: submit a single speedrun config.
#
# Usage:
#   qsub -l select=2 -N speedrun_2b_muon -v CONFIG=speedrun_2b_muon \
#       torchtitan/experiments/ezpz/competition/submit_run.sh

CONFIG="${CONFIG:?CONFIG env var must be set (e.g. speedrun_2b_muon)}"

# ---- Environment (torch 2.13+ .venv) ----
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${PBS_O_WORKDIR}"
source .venv/bin/activate
ezpz yeet-env
deactivate
source /tmp/.venv/bin/activate

# ---- Launch ----
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.agpt \
    --config="$CONFIG" \
    --debug.print-config
