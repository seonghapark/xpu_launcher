#!/bin/bash --login
#PBS -A datascience
#PBS -l walltime=00:45:00
#PBS -l filesystems=flare:home
#PBS -q workq
#PBS -j oe
#
# moe smoke test submission script. Mirrors competition/submit_run.sh
# but for moe configs.
#
# Usage:
#   qsub -l select=2 -N smoke_moe -v CONFIG=moe_2b_ep,STEPS=10 \
#       torchtitan/experiments/ezpz/scripts/submit_moe_smoke.sh
#
# Optional env vars:
#   CONFIG      Required. moe config registry function name (e.g. moe_2b_ep).
#   STEPS       Number of training steps (default 10).
#   LOG_DIR     Override log destination (default logs/smoke-moe-${CONFIG}).
#   EXTRA_ARGS  Extra CLI args passed through to train.py.

CONFIG="${CONFIG:?CONFIG env var must be set (e.g. moe_2b_ep)}"
STEPS="${STEPS:-10}"
LOG_DIR="${LOG_DIR:-logs/smoke-moe-${CONFIG}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---- Environment (torch 2.13+ .venv) ----
# Same env order as the production scripts/submit_agpt_*_aurora_venv.sh.
# oneapi/release/2025.3.1 provides the Python the .venv's shebang binds
# against; do not also `module load python` (changes PATH ordering and
# breaks .venv/bin/ezpz on compute nodes).
module load oneapi/release/2025.3.1 hdf5 pti-gpu
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export CCL_PROCESS_LAUNCHER=pmix
export CCL_OP_SYNC=1
export ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"
export TORCH_CPP_LOG_LEVEL=ERROR

# `ezpz_setup_job` (in ~/.ezpz/utils.sh) overwrites $PBS_O_WORKDIR with
# the *current* cwd when the two don't match — which inside qsub's
# default $HOME is exactly the case. After that override, `cd
# $PBS_O_WORKDIR` would cd to $HOME and `source .venv/bin/activate`
# resolves to ~/.venv, NOT the repo's .venv. Stash the real submit dir
# under SUBMIT_DIR before sourcing the utils so we can cd to the right
# place regardless of what ezpz_setup_job mutates.
SUBMIT_DIR="${PBS_O_WORKDIR:-$(pwd)}"

source <(curl -fsSL https://bit.ly/ezpz-utils) && ezpz_setup_job

cd "${SUBMIT_DIR}"
echo "=== smoke[diag] cwd=$(pwd)"
echo "=== smoke[diag] before activate: which python3=$(which python3 2>/dev/null) which ezpz=$(which ezpz 2>/dev/null)"
echo "=== smoke[diag] .venv/bin/activate exists? $(test -f .venv/bin/activate && echo yes || echo NO)"
echo "=== smoke[diag] .venv/bin/python3 resolves to: $(readlink -f .venv/bin/python3 2>/dev/null)"
source .venv/bin/activate
echo "=== smoke[diag] after activate:  which python3=$(which python3 2>/dev/null) which ezpz=$(which ezpz 2>/dev/null)"

# Rebuild + broadcast the venv tarball. `ezpz tar-env` regenerates
# .venv.tar.gz from the active .venv (so the broadcast reflects
# whatever was last installed locally); `ezpz yeet .venv.tar.gz`
# tar-broadcasts it to /tmp/.venv on every compute node. `ezpz
# yeet-env` is the deprecated alias.
ezpz tar-env
ezpz yeet .venv.tar.gz
deactivate
echo "=== smoke[diag] before /tmp/.venv activate: /tmp/.venv/bin/activate exists? $(test -f /tmp/.venv/bin/activate && echo yes || echo NO)"
source /tmp/.venv/bin/activate
echo "=== smoke[diag] after /tmp/.venv activate: which python3=$(which python3) which ezpz=$(which ezpz)"

# ---- Launch ----
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${CONFIG}-$(date +%Y%m%d-%H%M%S).log"

# shellcheck disable=SC2086  # intentional word-split on EXTRA_ARGS
ezpz launch python3 -m torchtitan.experiments.ezpz.train \
    --module=ezpz.moe \
    --config="$CONFIG" \
    --training.steps="$STEPS" \
    --checkpoint.no-enable \
    --compile.no-enable \
    --debug.print-config \
    ${EXTRA_ARGS} \
    2>&1 | tee "${LOG_FILE}"
